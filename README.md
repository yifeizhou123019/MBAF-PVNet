# MBAF-PVNet

基于多分支注意力融合的图像分割网络

## 项目结构

```
MBAFNet/
├── train_MBAF_PVNet.py    # 训练脚本
├── predict_MBAF.py        # 预测脚本  
├── data.py               # 数据加载和预处理
├── utils.py              # 工具函数
├── model/
│   ├── MBAF-PVNet_VGG_models.py  # 模型定义
│   └── vgg.py            # VGG骨干网络
└── iou/                  # IoU计算模块
```

## 环境要求

- Python 3.6+
- PyTorch
- torchvision
- numpy
- PIL
- imageio

## 使用方法

### 训练

```bash
python train_MBAF_PVNet.py --epoch 100 --lr 1e-4 --batchsize 8 --trainsize 256
```

参数说明：
- `--epoch`: 训练轮数 (默认: 100)
- `--lr`: 学习率 (默认: 1e-4)
- `--batchsize`: 批次大小 (默认: 8)
- `--trainsize`: 输入图像尺寸 (默认: 256)
- `--clip`: 梯度裁剪阈值 (默认: 0.5)
- `--decay_rate`: 学习率衰减率 (默认: 0.1)
- `--decay_epoch`: 学习率衰减周期 (默认: 40)

### 预测

```bash
python predict_MBAF.py --testsize 256 --dataset_root /path/to/dataset --weight_path /path/to/model.pth
```

参数说明：
- `--testsize`: 测试图像尺寸 (默认: 256)
- `--dataset_root`: 数据集根目录
- `--dataset_name`: 数据集子文件夹名称 (默认: test)
- `--weight_path`: 模型权重路径

## 数据集格式

```
dataset/
├── JPEGImages/          # 原始图像
│   ├── image1.jpg
│   └── ...
└── SegmentationClass/   # 分割标签
    ├── image1.png
    └── ...
```

## 模型特点

- 基于VGG骨干网络
- 多分支注意力融合机制
- 旋转等变注意力模块
- 支持数据增强（翻转、旋转、颜色变换等）

## 评估指标

- IoU (Intersection over Union)
- Precision
- Recall  
- F-score
- Pixel Accuracy
