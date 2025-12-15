# 图像显著性预测

在彩色图像中，预测人眼容易关注的区域（输出显著性图），属于回归问题。

## 数据文件

- 训练数据集（共1600幅待检测图像及1600幅对应的显著图）：以jpg格式存储在 `dataset/3-Saliency-TrainSet.zip` 中。
- 测试数据集（共400幅待检测图像及400幅对应的显著图）：以jpg格式存储在 `dataset/3-Saliency-TestSet.zip` 中。

每个数据集中，待检测图像为人眼直接观察的彩色图像，保存在Stimuli文件夹；对应的显著图(即ground truth)为相同尺寸的灰度图像，颜色越亮的区域代表显著性越强，保存在FIXATIONMAPS文件夹。考虑到图像内容可能对结果产生影响，每个数据集都包括20种不同类型的图像，存放在20个文件夹中（如Action，Affective，Art……），因此分析结果时，既可以给出总体性能，又可以按类型进行分析。

注意读取数据集时要有从压缩包中解压后读取的能力。训练时仅读取训练数据集，测试时再读取测试数据集。

数据集文件压缩包目录示例：

```
3-Saliency-TestSet
├── FIXATIONMAPS
│   ├── Action
│   ├── Affective
│   ├── Art
│   ├── BlackWhite
│   ├── Cartoon
│   ├── Fractal
│   ├── Indoor
│   ├── Inverted
│   ├── Jumbled
│   ├── LineDrawing
│   ├── LowResolution
│   ├── Noisy
│   ├── Object
│   ├── OutdoorManMade
│   ├── OutdoorNatural
│   ├── Pattern
│   ├── Random
│   ├── Satelite
│   ├── Sketch
│   └── Social
└── Stimuli
    ├── Action
    ├── Affective
    ├── Art
    ├── BlackWhite
    ├── Cartoon
    ├── Fractal
    ├── Indoor
    ├── Inverted
    ├── Jumbled
    ├── LineDrawing
    ├── LowResolution
    ├── Noisy
    ├── Object
    ├── OutdoorManMade
    ├── OutdoorNatural
    ├── Pattern
    ├── Random
    ├── Satelite
    ├── Sketch
    └── Social
```

## 性能指标

- 主观指标：预测显著图与ground truth显著图主观上对比。
    - 生成少量案例，将预测显著图和 ground truth 显著图分别重叠到原图上进行可视化对比。
- 客观指标：相关系数（CC）、KL散度（指标函数在metric.py文件中，可直接调用，内有使用说明），或其他衡量显著性图像相似程度的指标等可以自行添加。
    - 如MAE绝对误差、SIM相似度等。

## 报告要求

实验报告以学号+名字命名，提交pdf文件到网盘对应文件夹下，实验报告需要包含：

1. 问题描述
2. 实验模型原理和概述
3. 实验模型结构和参数
4. 实验结果分析（包含训练集和测试集里的测试结果）,要求列举出一些失败案例并分析，分析指标提供越多，图表分析越详尽得分会越高。
5. 总结