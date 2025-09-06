import os, time, argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import SalObjDataset
from net.MBAF_PVNet import MBAF_PVNet_VGG  # 切换到 VGG 版


torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
torch.cuda.set_device(0)

def iou_loss_from_logits(logits, gt, eps=1e-6):
    prob = logits.sigmoid()
    inter = (prob * gt).sum(dim=(2,3))
    union = prob.sum(dim=(2,3)) + gt.sum(dim=(2,3)) - inter
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()

def ssim_loss_from_logits(logits, gt, C1=0.01**2, C2=0.03**2):
    x = logits.sigmoid()
    y = gt
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x*x, 3, 1, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y*y, 3, 1, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-6)
    return (1.0 - ssim_map).mean()

def build_loader(image_root, gt_root, trainsize, batchsize, num_workers=4, shuffle=True, drop_last=True, indices=None):
    ds = SalObjDataset(image_root, gt_root, trainsize=trainsize)
    if indices is not None:
        ds = Subset(ds, indices)
    loader = DataLoader(
        ds, batch_size=batchsize, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    return ds, loader

def main():
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--gt_root',    type=str, required=True)
    parser.add_argument('--val_image_root', type=str, default='')
    parser.add_argument('--val_gt_root',    type=str, default='')
    parser.add_argument('--trainsize',  type=int, default=512)
    parser.add_argument('--valsize',    type=int, default=512)
    parser.add_argument('--batchsize',  type=int, default=8)

    # 训练
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--wd',         type=float, default=1e-4)
    parser.add_argument('--save_dir',   type=str, default='models/miaf_vgg')
    parser.add_argument('--use_amp',    action='store_true')
    parser.add_argument('--resume',     type=str, default='')  # 继续训练
    parser.add_argument('--learnable_loss', action='store_true', help='使用 e^{-s}·L + s 加权')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Loader
    train_ds, train_loader = build_loader(
        args.image_root, args.gt_root, args.trainsize, args.batchsize,
        num_workers=max(1, min(8, os.cpu_count() or 1)), shuffle=True, drop_last=True
    )
    val_loader = None
    if args.val_image_root and args.val_gt_root:
        _, val_loader = build_loader(
            args.val_image_root, args.val_gt_root, args.valsize, batchsize=8,
            num_workers=max(1, min(8, os.cpu_count() or 1)), shuffle=False, drop_last=False
        )

    # Model
    model = MBAF_PVNet_VGG().cuda()
    model = model.to(memory_format=torch.channels_last)

    bce_loss_fn = nn.BCEWithLogitsLoss()

    if args.learnable_loss:
        sB = torch.nn.Parameter(torch.zeros((), device='cuda'))
        sI = torch.nn.Parameter(torch.zeros((), device='cuda'))
        sS = torch.nn.Parameter(torch.zeros((), device='cuda'))
        loss_params = [sB, sI, sS]
    else:
        sB = sI = sS = None
        loss_params = []

    # Optimizer / Scheduler
    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
        {'params': loss_params, 'lr': args.lr, 'weight_decay': 0.0}
    ])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Resume
    start_epoch = 1
    best_f1 = -1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        if 'optimizer' in ckpt: opt.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt: sch.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_f1 = ckpt.get('best_f1', best_f1)
        print(f'[Resume] from {args.resume}, epoch={start_epoch}, best_f1={best_f1:.4f}')

    # 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_meter = {'bce':0.0, 'iou':0.0, 'ssim':0.0, 'total':0.0}
        n_samples = 0

        for images, gts in train_loader:
            images = images.to('cuda', non_blocking=True).to(memory_format=torch.channels_last)
            gts    = gts.to('cuda', non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                # 前向：返回 s1..s5 logits（及 p1..p5，但训练用 logits 更稳）
                s1, s2, s3, s4, s5, _, _, _, _, _ = model(images)

                # 统一上采样到 GT 尺寸
                to_size = gts.shape[2:]
                logits = [
                    F.interpolate(s1, size=to_size, mode='bilinear', align_corners=False),
                    F.interpolate(s2, size=to_size, mode='bilinear', align_corners=False),
                    F.interpolate(s3, size=to_size, mode='bilinear', align_corners=False),
                    F.interpolate(s4, size=to_size, mode='bilinear', align_corners=False),
                    F.interpolate(s5, size=to_size, mode='bilinear', align_corners=False),
                ]
                # 多尺度权重（可调整）
                ms_w = [1.00, 0.80, 0.60, 0.40, 0.20]

                # 先分别聚合三类损失，再（可选）用 e^{-s}·L + s 组合
                bce_sum = sum(w * bce_loss_fn(lg, gts) for w, lg in zip(ms_w, logits))
                iou_sum = sum(w * iou_loss_from_logits(lg, gts) for w, lg in zip(ms_w, logits))
                ssim_sum= sum(w * ssim_loss_from_logits(lg, gts) for w, lg in zip(ms_w, logits))

                if args.learnable_loss:
                    loss = torch.exp(-sB) * bce_sum + sB \
                         + torch.exp(-sI) * iou_sum + sI \
                         + torch.exp(-sS) * ssim_sum + sS
                else:
                    loss = bce_sum + iou_sum + ssim_sum

            scaler.scale(loss).backward()
            # 可选：梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt)
            scaler.update()

            bs = images.size(0)
            n_samples += bs
            loss_meter['bce']   += bce_sum.item() * bs
            loss_meter['iou']   += iou_sum.item() * bs
            loss_meter['ssim']  += ssim_sum.item() * bs
            loss_meter['total'] += loss.item() * bs

        sch.step()

        # 打印 epoch 汇总
        for k in loss_meter:
            loss_meter[k] /= max(1, n_samples)
        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"LR={sch.get_last_lr()[0]:.6f}  "
              f"Loss={loss_meter['total']:.4f}  "
              f"BCE={loss_meter['bce']:.4f}  IoU={loss_meter['iou']:.4f}  SSIM={loss_meter['ssim']:.4f}")

        # 简单验证（可选）
        cur_f1 = -1
        if val_loader is not None:
            model.eval()
            TP=FP=FN=0
            with torch.no_grad():
                for images, gts in val_loader:
                    images = images.to('cuda', non_blocking=True).to(memory_format=torch.channels_last)
                    gts    = gts.to('cuda', non_blocking=True).float()
                    s1, s2, s3, s4, s5, _, _, _, _, _ = model(images)
                    prob = F.interpolate(s1, size=gts.shape[2:], mode='bilinear', align_corners=False).sigmoid()
                    pred = (prob >= 0.5).to(torch.bool)
                    gt   = (gts  >= 0.5).to(torch.bool)
                    TP += (pred & gt).sum().item()
                    FP += (pred & ~gt).sum().item()
                    FN += (~pred & gt).sum().item()
            precision = TP / (TP + FP + 1e-6)
            recall    = TP / (TP + FN + 1e-6)
            cur_f1    = 2*precision*recall/(precision+recall+1e-6)
            print(f"  [Val] P={precision:.4f} R={recall:.4f} F1={cur_f1:.4f}")

        # 保存
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': sch.state_dict(),
            'best_f1': best_f1 if cur_f1 < 0 else max(best_f1, cur_f1)
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pth'))
        if cur_f1 >= 0 and cur_f1 > best_f1:
            best_f1 = cur_f1
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))
            print("  [Saved] best.pth updated.")

    print("Training done:", datetime.now())

if __name__ == '__main__':
    print("Train start:", datetime.now())
    main()
