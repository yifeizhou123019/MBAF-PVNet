# MBAF-PVNet (VGG) · 训练与使用说明

本工程包含 **MBAF-PVNet（VGG骨干）** 与 **MBAF-PVNet-Tiny（MobileNetV2骨干）** 两个实现：
- **REAM 可学习角度**：`net/MBAF_PVNet.py` 中的 `RotEquiAttention` 支持将旋转角度作为 `nn.Parameter` 学习；
- **BIS 自主学习权重融合**：训练脚本 `train_MBAF_PVNet.py` 提供 `--learnable_loss`，自动学习 BCE/IoU/SSIM 的权重；
- **多尺度深监督**：对 s1..s5 五尺度 logits 进行加权监督；
- **高性能实践**：支持 AMP（混合精度）、channels_last、Cosine LR、断点 `--resume` 恢复等。

---

## 目录结构（节选）
```text
MBAF_PVNet/
  net/
    MBAF_PVNet.py           # VGG骨干 + REAM（可学习角度）
    MBAF_PVNet_tiny.py      # MobileNetV2 Tiny 版本
    mobilenetv2.py, vgg.py  # 骨干网络定义
  data.py                   # 数据集与DataLoader（SalObjDataset）
  train_MBAF_PVNet.py       # VGG版训练脚本（BIS自适应权重，可AMP）
  train_MBAF-PVNet_Tiny.py  # Tiny版本
  utils.py

```

## 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.10（建议 1.12+ / 2.x）
- torchvision
- Pillow, numpy

安装示例：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # 按你的CUDA选择
pip install pillow numpy
# 可选
pip install thop fvcore
```

## 数据准备

`data.SalObjDataset` 期望如下目录：
```
<dataset_root>/images/*.jpg
<dataset_root>/masks/*.png|*.jpg
```
- 输入图片以 **.jpg** 为主；
- 掩码以灰度图读取（`convert('L')`），`ToTensor()` 后范围为 **[0,1]**（若原始是 0/255 将被归一化）。
- 训练/验证需分别指定 `--image_root/--gt_root` 与 `--val_image_root/--val_gt_root`，目录末尾记得带 `/`。

## 训练（VGG版，推荐）

脚本：`train_MBAF_PVNet.py`

可选参数一览：
```text
--image_root (str, required)      训练图片目录
--gt_root (str, required)         训练掩码目录
--val_image_root (str, default='')    验证图片目录（可留空）
--val_gt_root (str, default='')       验证掩码目录（可留空）
--trainsize (int, default=512)    输入尺寸（训练时会resize）
--valsize (int, default=512)      验证时resize尺寸
--batchsize (int, default=8)
--epochs (int, default=100)
--lr (float, default=1e-3)
--wd (float, default=1e-4)
--save_dir (str, default='models/miaf_vgg')    权重保存目录
--use_amp (flag)                  开启自动混合精度
--resume (str, default='')        断点恢复（last.pth/best.pth均可）
--learnable_loss (flag)           开启BIS自适应权重融合：L=Σ e^(-s_i)·L_i + Σ s_i
```

### 快速开始
```bash
python train_MBAF_PVNet.py   --image_root /data/train/images/   --gt_root    /data/train/masks/    --val_image_root /data/val/images/   --val_gt_root    /data/val/masks/    --trainsize 512 --valsize 512 --batchsize 8   --epochs 100 --lr 1e-3 --wd 1e-4   --save_dir models/miaf_vgg   --use_amp --learnable_loss
```

### 训练细节
- **深监督**：`ms_w = [1.0, 0.8, 0.6, 0.4, 0.2]` 对应 s1..s5；
- **组合损失**：
  - `BCEWithLogitsLoss`（对 logits 计算，内部自带 sigmoid）
  - `IoU Loss`（对 sigmoid 概率计算 IoU）
  - `SSIM Loss`（对概率图计算 SSIM）
- **BIS 自适应权重**（可选）：学习 `sB,sI,sS`，得到 `λ_i = exp(-s_i)`；
- **验证指标**：若提供验证集，逐 epoch 报告 `P/R/F1`（前景为正类）；
- **模型保存**：`save_dir` 下保存 `last.pth` 与 `best.pth`（按验证 F1 选优）。


## REAM 可学习角度（VGG版）

`net/MBAF_PVNet.py` → `class RotEquiAttention`：
- 构造参数：`angles=(0,45,90,135)`, `learnable=True`；
- 内部以**弧度**形式保存为 `nn.Parameter`，梯度可经 `affine_grid/grid_sample` 反向传播；
- 若需固定角度不学习，可在实例化时传 `learnable=False`，或训练前对 `angles` 关闭梯度：
  ```python
  for p in model.rot_equi_att.angles.parameters():
      p.requires_grad_(False)
  ```

## Tiny 版本（可选）
- 参考 `net/MBAF_PVNet_tiny.py` 与 `train_MBAF-PVNet_Tiny.py`；
- 该脚本为较早的配置式写法（非 argparse），请按文件内注释调整路径与参数。

## 推理/导出（示例代码）

如需简易推理，可参考：
```python
import torch, torch.nn.functional as F, os
from PIL import Image
from torchvision import transforms
from net.MBAF_PVNet import MBAF_PVNet_VGG

model = MBAF_PVNet_VGG().cuda().eval()
ckpt = torch.load('models/miaf_vgg/best.pth', map_location='cpu')
state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
model.load_state_dict(state, strict=False)

to_tensor = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open('/path/to/img.jpg').convert('RGB')
x = to_tensor(img).unsqueeze(0).cuda()
with torch.no_grad():
    s1, s2, s3, s4, s5, *_ = model(x)
    prob = torch.sigmoid(F.interpolate(s1, size=(512,512), mode='bilinear', align_corners=False))

mask = (prob[0,0].cpu().numpy()*255).astype('uint8')
Image.fromarray(mask).save('pred.png')