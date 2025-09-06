import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained_weights_path=None, num_classes=1000):
        super(MobileNetV2Backbone, self).__init__()

        # 初始化 MobileNetV2
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=False)

        if pretrained_weights_path:
            # 加载预训练的权重
            pretrain_weights = torch.load(pretrained_weights_path)
            self.mobilenet_v2.load_state_dict(pretrain_weights)

        # 只保留 MobileNetV2 的特征提取部分（不包括分类头）
        self.backbone = self.mobilenet_v2.features

        # 如果需要分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenet_v2.last_channel, num_classes)
        )

    def forward(self, x):
        # 获取特征提取部分的输出
        x = self.backbone(x)

        # 如果需要分类头
        # x = x.mean([2, 3])  # 全局平均池化
        # x = self.classifier(x)  # 分类头（如果需要）

        return x

