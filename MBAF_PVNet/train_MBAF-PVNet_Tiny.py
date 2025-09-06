import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import os
from datetime import datetime
from tqdm import tqdm
from model.MBAF_PVNet_mobilev2_tiny import MBAF_PVNet_MobileNetV2  # 请确保模型路径正确
from utils import clip_gradient, adjust_lr  # 请确保 utils 中有这些方法
from data import get_loader  # 确保 get_loader 已正确导入
import torch
from torchvision import transforms


LOSS_MODE = "BIS"   # 第一次改为-I    第二次改为-B   第三次 改为-S


# 配置训练参数
class Config:
    def __init__(self):
        self.epochs = 100
        self.lr = 5e-3
        self.batch_size = 16
        self.trainsize = 512
        self.clip = 0.5
        self.decay_rate = 0.1
        self.decay_epoch = 40
        self.pretrained_weights = '/tmp/pycharm_project_854/mobilenet_v2-b0353104.pth'  # 修改为你自己的路径
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# 加载数据集
image_root = '/root/autodl-tmp/datasetB/images/'
gt_root = '/root/autodl-tmp/datasetB/masks/'

# 获取数据集
train_loader = get_loader(image_root, gt_root, batchsize=config.batch_size, trainsize=config.trainsize, shuffle=True)
val_loader = get_loader(image_root, gt_root, batchsize=config.batch_size, trainsize=config.trainsize, shuffle=False)

# 初始化模型
model = MBAF_PVNet_MobileNetV2(channel=3)  # 如果需要改变 channel 参数可以修改
model.to(config.device)

# 加载预训练权重
if os.path.exists(config.pretrained_weights):
    pretrained_weights = torch.load(config.pretrained_weights)
    model.mobilenet_v2.load_state_dict(pretrained_weights)
else:
    print(f"预训练权重文件 {config.pretrained_weights} 不存在，请检查路径。")
    exit(1)

# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# ================== [仅此处新增/修改：损失函数与可学习权重] ==================
# BCE（对 logits 使用）
CE = nn.BCEWithLogitsLoss()

# IoU 损失（Soft IoU over logits）
class SoftIoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1,2,3))
        union = (probs + targets - probs*targets).sum(dim=(1,2,3))
        iou = (inter + self.eps) / (union + self.eps)
        return 1.0 - iou.mean()

IoU = SoftIoULoss()

# SSIM 损失（单通道，窗口11×11，高斯近似；对 logits 先 sigmoid）
# 简洁实现，稳定可导；如后续接入 piq/pytorch-msssim，可替换此实现。
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = C1
        self.C2 = C2
        # 生成高斯核（在 forward 时根据设备/通道扩展）
        gauss = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-(gauss - (window_size-1)/2)**2 / (2*sigma**2))
        gauss = (gauss / gauss.sum()).unsqueeze(0)  # [1, W]
        window_2d = gauss.t() @ gauss               # [W, W]
        self.register_buffer('win', window_2d)

    def _filter(self, x):
        # x: [B,1,H,W]
        B, C, H, W = x.size()
        w = self.win.to(x.dtype).unsqueeze(0).unsqueeze(0)  # [1,1,W,W]
        return F.conv2d(x, w, padding=self.window_size//2, groups=1)

    def forward(self, logits, targets):
        x = torch.sigmoid(logits)
        y = targets
        # 保证是单通道
        if x.size(1) != 1:
            x = x[:, :1]
        if y.size(1) != 1:
            y = y[:, :1]
        mu_x = self._filter(x)
        mu_y = self._filter(y)
        sigma_x = self._filter(x*x) - mu_x*mu_x
        sigma_y = self._filter(y*y) - mu_y*mu_y
        sigma_xy = self._filter(x*y) - mu_x*mu_y
        ssim_map = ((2*mu_x*mu_y + self.C1) * (2*sigma_xy + self.C2)) / \
                   ((mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2))
        return 1.0 - ssim_map.mean()

# SSIM = SSIMLoss()
SSIM = SSIMLoss().to(config.device)

# 可学习的同方差不确定性参数 s_B, s_I, s_S（标量）
s_b = torch.nn.Parameter(torch.zeros(1, device=config.device))
s_i = torch.nn.Parameter(torch.zeros(1, device=config.device))
s_s = torch.nn.Parameter(torch.zeros(1, device=config.device))
# 把它们加入优化器（不改变网络结构与FLOPs）
optimizer.add_param_group({'params': [s_b, s_i, s_s]})

def weighted_loss(bce, iou, ssim, mode: str):
    """
    Kendall & Gal (CVPR'18) 同方差不确定性加权：
    L = sum_{t in active}  exp(-s_t)*L_t + s_t
    mode: "BIS" | "-S" | "-I" | "-B"
    """
    loss = 0.0
    terms = []
    if mode in ("BIS", "-S", "-I", "-B"):
        if mode != "-B":
            terms.append(torch.exp(-s_b)*bce + s_b)
        if mode != "-I":
            terms.append(torch.exp(-s_i)*iou + s_i)
        if mode != "-S":
            terms.append(torch.exp(-s_s)*ssim + s_s)
    else:
        # 默认回退到基础组
        terms.append(torch.exp(-s_b)*bce + s_b)
        terms.append(torch.exp(-s_i)*iou + s_i)
        terms.append(torch.exp(-s_s)*ssim + s_s)
    for t in terms:
        loss = loss + t
    return loss
# =====================================================

# ======= 新增的尺寸对齐工具函数（仅用于 Loss/Acc 计算，不改模型） =======
def up_to_gt(x, gts):
    # 将预测张量双线性插值到与 GT 相同的空间尺寸
    return F.interpolate(x, size=gts.shape[-2:], mode='bilinear', align_corners=False)
# =====================================================================

# 准确度计算（保持接口不变；内部对齐到 GT）
def calculate_accuracy(preds, gts):
    preds = up_to_gt(preds, gts)
    preds = torch.sigmoid(preds) > 0.5
    correct = (preds == gts).sum().item()
    accuracy = correct / gts.numel()
    return accuracy

# 训练函数
def train_one_epoch(train_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    for i, (images, gts) in enumerate(train_loader):
        images = images.to(config.device)
        gts = gts.to(config.device)

        optimizer.zero_grad()

        # 获取模型输出
        s1, s2, s3, s4, s5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = model(images)

        # ======== [仅此处修改：自动加权的 B+I+S，多尺度平均；先对齐到 GT；只用 logits] ========
        logits_native = [s1, s2, s3, s4, s5]
        logits = [up_to_gt(s, gts) for s in logits_native]  # 对齐到 gts 的 H×W
        gts = gts.float()

        bce = sum(CE(s, gts) for s in logits) / len(logits)
        iou = sum(IoU(s, gts) for s in logits) / len(logits)
        ssm = sum(SSIM(s, gts) for s in logits) / len(logits)
        loss = weighted_loss(bce=bce, iou=iou, ssim=ssm, mode=LOSS_MODE)
        # =====================================================================

        loss.backward()
        clip_gradient(optimizer, config.clip)
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率（使用上采样后的 s1）
        s1_up = logits[0]
        accuracy = calculate_accuracy(s1_up, gts)
        total_accuracy += accuracy
        total_samples += 1

        if i % 20 == 0:
            # 打印当前等效权重（exp(-s)）
            wb = torch.exp(-s_b).item()
            wi = torch.exp(-s_i).item()
            ws = torch.exp(-s_s).item()
            print(f"Epoch [{epoch}/{config.epochs}], Step [{i}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Acc: {accuracy:.4f}, "
                  f"[w_BCE={wb:.3f}, w_IoU={wi:.3f}, w_SSIM={ws:.3f}]")

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    print(f"Epoch [{epoch}/{config.epochs}] - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

# 验证函数
def validate(val_loader, model):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for images, gts in val_loader:
            images = images.to(config.device)
            gts = gts.to(config.device)

            # 获取模型输出
            s1, s2, s3, s4, s5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = model(images)

            # ======== [仅此处修改：验证同样使用自动加权的 B+I+S；先对齐到 GT] ========
            logits_native = [s1, s2, s3, s4, s5]
            logits = [up_to_gt(s, gts) for s in logits_native]
            gts = gts.float()

            bce = sum(CE(s, gts) for s in logits) / len(logits)
            iou = sum(IoU(s, gts) for s in logits) / len(logits)
            ssm = sum(SSIM(s, gts) for s in logits) / len(logits)
            loss = weighted_loss(bce=bce, iou=iou, ssim=ssm, mode=LOSS_MODE)
            # ================================================================

            total_loss += loss.item()

            # 计算准确率（使用上采样后的 s1）
            s1_up = logits[0]
            accuracy = calculate_accuracy(s1_up, gts)
            total_accuracy += accuracy
            total_samples += 1

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    # 同步打印当前等效权重，便于记录“权重演化”
    wb = torch.exp(-s_b).item()
    wi = torch.exp(-s_i).item()
    ws = torch.exp(-s_s).item()
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f} "
          f"[w_BCE={wb:.3f}, w_IoU={wi:.3f}, w_SSIM={ws:.3f}]")
    return avg_loss, avg_accuracy

# 保存模型（不变）
def save_model(model, epoch):
    save_path = '/tmp/pycharm_project_854/models/small_data/'
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}ACCoNet_MobileNetV2.pth.{epoch}"
    torch.save(model.state_dict(), filename)
    print(f"Model saved at {filename}")

# 训练过程（不变）
for epoch in range(1, config.epochs + 1):
    adjust_lr(optimizer, config.lr, epoch, config.decay_rate, config.decay_epoch)
    train_one_epoch(train_loader, model, optimizer, epoch)

    if epoch % 5 == 0:
        val_loss, val_accuracy = validate(val_loader, model)
        save_model(model, epoch)

        print("Testing final model on test set...")
        # 这里你可以执行测试集的评估
