"""
医学图像分类模型定义
使用预训练的ResNet18进行迁移学习
"""

import torch
import torch.nn as nn
from torchvision import models


class MedicalImageCNN(nn.Module):
    """
    基于ResNet18的医学图像分类模型
    使用预训练权重进行迁移学习
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        # 加载预训练的ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        """提取特征（不经过分类器）"""
        # 获取除最后一层外的所有层
        modules = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        features = feature_extractor(x)
        return features.view(features.size(0), -1)


class SimpleMedicalCNN(nn.Module):
    """
    简单的CNN模型（不使用预训练）
    适用于资源受限的环境
    """

    def __init__(self, num_classes: int = 2, input_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    model_type: str = "resnet18", num_classes: int = 2, pretrained: bool = True
) -> nn.Module:
    """
    创建模型

    Args:
        model_type: 模型类型 ('resnet18', 'simple')
        num_classes: 类别数量
        pretrained: 是否使用预训练权重

    Returns:
        模型实例
    """
    if model_type == "resnet18":
        return MedicalImageCNN(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "simple":
        return SimpleMedicalCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试模型
    model = MedicalImageCNN(num_classes=2)
    print(f"ResNet18-based model parameters: {count_parameters(model):,}")

    simple_model = SimpleMedicalCNN(num_classes=2)
    print(f"Simple CNN parameters: {count_parameters(simple_model):,}")

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
