# 彩色图像生成

设计并训练一个生成网络，生成彩色图像。可以选择使用GAN、VAE或其他生成模型，目标是从随机噪声生成与训练数据分布相似的彩色图像。

## 数据文件

CIFAR-10 是一个用于图像分类任务的广泛使用的数据集，包含 10 个不同类别的彩色图片。每个类别包含 6000 张图片，总共 60000 张图片，大小为 32x32 像素，分为 50000 张训练图片和 10000 张测试图片。这个数据集在机器学习和计算机视觉任务中非常常见。

CIFAR-10 的类别包括：飞机（airplane）、汽车（automobile）、鸟类（bird）、猫（cat）、鹿（deer）、狗（dog）、青蛙（frog）、马（horse）、船（ship）、卡车（truck）。分析结果时，既可以给出总体性能，又可以按类型进行分析。

![](https://github.com/Hiuyee124/Machine-Learning-Class-Assignment25/raw/main/4-CIFAR10-Example.png)

CIFAR-10 数据集可以通过以下代码获取：

```py
from torchvision.datasets import CIFAR10
dataset = CIFAR10(root='./dataset/CIFARdata', download=True, transform=transforms.ToTensor())
```

## 性能指标

主观指标：生成图像质量主观评价，对比数据集中的真实图像。报告中可以给出随着训练迭代轮数增加，生成图像结果的变化情况。

客观指标：使用Inception Score（IS）和Frechet Inception Distance（FID）等评价指标，分析生成图像的质量。

```
# 安装torch-fidelity库
pip install torch-fidelity

# 用uv安装
uv add torch-fidelity
```

```py
import torch_fidelity
def fidelity_metric(genereated_images_path, real_images_path):
"""
使用fidelity package计算所有的生成相关的指标，输入生成图像路径和真实图像路径
isc: inception score
kid: kernel inception distance
fid: frechet inception distance
"""
  metrics_dict = torch_fidelity.calculate_metrics(
    input1=genereated_images_path,
    input2=real_images_path,
    cuda=True,
    isc=True,
    fid=True,
    kid=True,
    verbose=False
  )
  return metrics_dict
```

## 报告要求

实验报告以学号+名字命名，提交pdf文件到网盘对应文件夹下，实验报告需要包含：

1. 问题描述
2. 实验模型原理和概述
3. 实验模型结构和参数
4. 实验结果分析（包含训练集和测试集里的测试结果）,要求列举出一些失败案例并分析，分析指标提供越多，图表分析越详尽得分会越高。
5. 总结