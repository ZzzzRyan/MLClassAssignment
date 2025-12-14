# 医学图像分类任务

## 最新更新 (改进版)

**主要改进**：
- ✅ 增加验证集划分，避免在测试集上调参
- ✅ 改用类别权重损失，替代加权采样
- ✅ 添加标签平滑防止过拟合
- ✅ 优化模型架构（降低Dropout，添加BatchNorm）
- ✅ 实现微调策略（warmup + 解冻）
- ✅ 添加早停机制
- ✅ 改进学习率调度（ReduceLROnPlateau）

详细改进说明请查看 `IMPROVEMENTS.md`

## 项目结构

```
2_medimage/
├── dataset_utils.py    # 数据集加载和预处理
├── model.py            # 模型定义（ResNet18迁移学习）
├── train.py            # 训练脚本
├── evaluation.py       # 评估指标和可视化
├── README.md           # 本说明文件
├── models/             # 保存的模型
└── reports/            # 报告和图表
    ├── figures/        # ROC曲线、混淆矩阵等
    └── examples/       # 预测示例图像
```

## 快速开始

### 1. 运行训练

```bash
cd MLClassAssignment
python 2_medimage/train.py
```

### 2. 自定义参数

```bash
python 2_medimage/train.py \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --model-type resnet18 \
    --pretrained
```

### 可用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-zip` | `dataset/2-MedImage-TrainSet.zip` | 训练集路径 |
| `--test-zip` | `dataset/2-MedImage-TestSet.zip` | 测试集路径 |
| `--epochs` | 40 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--lr` | 3e-4 | 学习率 |
| `--val-split` | 0.15 | 验证集比例 |
| `--early-stop-patience` | 10 | 早停patience |
| `--label-smoothing` | 0.1 | 标签平滑系数 |
| `--warmup-epochs` | 5 | 预热轮数（仅训练分类器）|
| `--model-type` | `resnet18` | 模型类型 (`resnet18`, `simple`) |
| `--pretrained` | True | 使用预训练权重 |
| `--augmentation` | True | 启用数据增强 |
| `--out` | `2_medimage/models/best_model.pth` | 模型保存路径 |
| `--report-dir` | `2_medimage/reports` | 报告保存目录 |

## 输出结果

训练完成后，将生成以下文件：

1. **模型文件**: `models/best_model.pth`
2. **训练曲线**: `reports/figures/training_curves.png`
3. **ROC曲线**: `reports/figures/roc_curve.png`
4. **混淆矩阵**: `reports/figures/confusion_matrix.png`
5. **PR曲线**: `reports/figures/pr_curve.png`
6. **预测结果**: `reports/predictions.txt`
7. **示例图像**: `reports/examples/`

## 性能指标

程序会输出以下性能指标：

- **Accuracy**: 分类准确度
- **Precision**: 精确率（预测为患病中真正患病的比例）
- **Recall/Sensitivity**: 召回率/灵敏度（真正患病被检出的比例）
- **Specificity**: 特异度（真正正常被正确判断的比例）
- **F1 Score**: F1分数
- **F2 Score**: F2分数（更重视召回率）
- **AUC**: ROC曲线下面积

## 预测结果格式

`predictions.txt` 文件格式（兼容原MATLAB代码）：

```
序号    真实标签    预测正确性    患病概率
1       0           1             0.123456
2       1           1             0.876543
...
```

- 第一列：测试样本序号
- 第二列：真实标签（1=患病，0=正常）
- 第三列：预测是否正确（1=正确，0=错误）
- 第四列：模型预测该样本患病的概率

## 技术细节

### 模型架构

使用 **ResNet18** 进行迁移学习：
- 预训练于 ImageNet 数据集
- 优化后的分类器：
  - Dropout(0.3) + Linear(512→128) + BatchNorm + Dropout(0.2) + Linear(128→2)
  - 相比原版降低了过拟合风险

### 训练策略

**两阶段微调**：
1. **Warmup阶段** (前5轮)：冻结backbone，只训练分类器
2. **Fine-tuning阶段** (之后)：解冻全部层，降低学习率

**正则化技术**：
- Label Smoothing (0.1)
- Dropout (0.3, 0.2)
- Batch Normalization
- Weight Decay (1e-4)
- 早停机制

### 数据增强

训练时启用以下数据增强（强度适中）：
- 随机水平翻转 (p=0.5)
- 随机垂直翻转 (p=0.3)
- 随机旋转 (±10度)
- 颜色抖动 (0.15)
- 随机平移 (±5%)

### 类别不平衡处理

使用 **类别权重的交叉熵损失**：
```python
class_weights = total / (n_classes * class_counts)
loss = F.cross_entropy(logits, labels, weight=class_weights)
```

相比加权采样更温和，减少过拟合。

### 优化策略

- 优化器：AdamW (weight_decay=1e-4)
- 学习率调度：ReduceLROnPlateau (基于验证集AUC)
- 早停：patience=10，基于验证集AUC
