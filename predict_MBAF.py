import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import imageio

from model.MBAF-PVNet_VGG_models import MBAF_PVNet_VGG
from data import test_dataset

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--dataset_root', type=str, default='/tmp/pycharm_project_729/PV_dataset',
                    help='root directory of dataset')
parser.add_argument('--dataset_name', type=str, default='test', help='name of the dataset subfolder')
parser.add_argument('--weight_path', type=str, default='/tmp/pycharm_project_729/models/ACCoNet_VGG/ACCoNet_VGG.pth.40',
                    help='path to model weights')
opt = parser.parse_args()

# 选择模型
model = MBAF_PVNet_VGG()
model.load_state_dict(torch.load(opt.weight_path))
model.cuda()
model.eval()

# 结果保存路径
save_path = os.path.join('./results1','VGG', opt.dataset_name)
os.makedirs(save_path, exist_ok=True)

# 构造图像和标签路径
image_root = os.path.join(opt.dataset_root, opt.dataset_name, 'image/')
gt_root = os.path.join(opt.dataset_root, opt.dataset_name, 'GT/')

print(f"Testing on dataset: {opt.dataset_name}")

# 加载数据
test_loader = test_dataset(image_root, gt_root, opt.testsize)

time_sum = 0

# --------------------【修改部分1：初始化指标】--------------------
iou_total = 0
pa_total = 0
precision_total = 0
recall_total = 0
fscore_total = 0
metric_count = 0
# --------------------------------------------------------------

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()

    if gt is not None:
        gt = np.asarray(gt, np.float32)
        gt = (gt > 127).astype(np.uint8)
        h, w = gt.shape
    else:
        h, w = opt.testsize, opt.testsize

    image = image.cuda()
    time_start = time.time()
    res, *_ = model(image)
    time_end = time.time()
    time_sum += (time_end - time_start)
    res = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
    res = torch.sigmoid(res).data.cpu().numpy().squeeze()

    pred_binary = (res > 0.5).astype(np.uint8)

    if gt is not None:
        intersection = np.logical_and(pred_binary, gt).sum()
        union = np.logical_or(pred_binary, gt).sum()
        iou = intersection / (union + 1e-8)
        iou_total += iou

        correct = (pred_binary == gt).sum()
        total = gt.size
        pa = correct / (total + 1e-8)
        pa_total += pa

        tp = ((pred_binary == 1) & (gt == 1)).sum()
        fp = ((pred_binary == 1) & (gt == 0)).sum()
        fn = ((pred_binary == 0) & (gt == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        fscore = (2 * precision * recall) / (precision + recall + 1e-8)

        precision_total += precision
        recall_total += recall
        fscore_total += fscore

        metric_count += 1

    save_file = os.path.join(save_path, name)
    imageio.imwrite(save_file, (pred_binary * 255).astype(np.uint8))

    print(f"Saved mask: {save_file}")

    if i == test_loader.size - 1:
        print(f'Average time per image: {time_sum / test_loader.size:.5f} seconds')
        print(f'Average speed: {test_loader.size / time_sum:.4f} fps')

if metric_count > 0:
    miou = iou_total / metric_count
    mpa = pa_total / metric_count
    mean_precision = precision_total / metric_count
    mean_recall = recall_total / metric_count
    mean_fscore = fscore_total / metric_count

    print(f"\nEvaluation Results on dataset '{opt.dataset_name}':")
    print(f"  - mIoU: {miou:.4f}")
    print(f"  - mPA: {mpa:.4f}")
    print(f"  - Precision: {mean_precision:.4f}")
    print(f"  - Recall: {mean_recall:.4f}")
    print(f"  - F-score: {mean_fscore:.4f}")
else:
    print("\nNo ground truth labels found; skipping metrics calculation.")