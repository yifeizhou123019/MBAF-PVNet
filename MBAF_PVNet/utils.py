import numpy as np

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*decay
        print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, init_lr*decay))
# utils.py
def compute_metrics_numpy(pred, gt, threshold=0.5):
    """
    计算单张二值分割图的 IoU、Precision、Recall、F-score、Pixel Accuracy
    pred: numpy array，[H, W]，float in [0,1]，模型输出概率图
    gt:   numpy array，[H, W]，0/1 的真实二值标签
    threshold: 二值化阈值（默认 0.5）

    返回：
        iou, precision, recall, f_score, pixel_accuracy
    """
    # 二值化
    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin   = (gt   > 0.5).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union        = np.logical_or(pred_bin,  gt_bin).sum()
    tp = intersection
    fp = ((pred_bin == 1) & (gt_bin == 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin == 1)).sum()
    tn = ((pred_bin == 0) & (gt_bin == 0)).sum()

    iou       = intersection / (union + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f_score   = 2 * precision * recall / (precision + recall + 1e-8)
    pa        = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return iou, precision, recall, f_score, pa

