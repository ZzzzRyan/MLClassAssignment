# 训练改进说明

## 问题诊断

从初始训练结果发现以下问题：

1. **严重过拟合**
   - 训练准确率：99.69%
   - 测试准确率：72.8%
   - 差距过大，说明模型在训练集上过拟合

2. **预测偏向性**
   - Recall: 93% (召回率高)
   - Precision: 60.4% (精确率低)
   - 模型倾向于预测为患病，导致假阳性过多 (FP=61)

3. **类别不平衡**
   - 训练集：normal:993 vs disease:646
   - 加权采样可能导致过度补偿

## 改进措施

### 1. 增加验证集 ✓

**改进前**：直接在测试集上选择最佳模型（数据泄漏）

**改进后**：
```python
# 从训练集划分15%作为验证集
train_idx, val_idx = train_test_split(
    np.arange(len(y_train)),
    test_size=0.15,
    stratify=y_train,
    random_state=42
)
```

**优点**：
- 避免在测试集上调参
- 更准确的模型选择
- 更真实的性能评估

### 2. 类别平衡策略改进 ✓

**改进前**：使用加权随机采样器（WeightedRandomSampler）

**改进后**：使用类别权重的交叉熵损失
```python
# 计算类别权重
class_weights = total / (n_classes * class_counts)

# 在损失函数中使用
loss = F.cross_entropy(logits, yb, weight=class_weights)
```

**优点**：
- 更温和的平衡策略
- 减少过采样带来的过拟合
- 保持数据分布的自然性

### 3. 标签平滑 ✓

**新增**：Label Smoothing (0.1)
```python
loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
```

**优点**：
- 防止模型过于自信
- 提高泛化能力
- 减少过拟合

### 4. 模型架构优化 ✓

**改进前**：
```python
nn.Dropout(0.5)
nn.Linear(512, 256)
nn.Dropout(0.3)
nn.Linear(256, 2)
```

**改进后**：
```python
nn.Dropout(0.3)          # 降低dropout
nn.Linear(512, 128)      # 减少隐藏层大小
nn.BatchNorm1d(128)      # 添加批归一化
nn.Dropout(0.2)
nn.Linear(128, 2)
```

**优点**：
- 降低Dropout率防止欠拟合
- 添加BatchNorm提高训练稳定性
- 减小模型容量降低过拟合风险

### 5. 微调策略 ✓

**新增**：分阶段微调
```python
# 第1-5轮：只训练分类器，冻结backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# 第6轮后：解冻全部层，降低学习率
for param in model.backbone.parameters():
    param.requires_grad = True
optimizer = AdamW(model.parameters(), lr=lr * 0.1)
```

**优点**：
- 避免破坏预训练权重
- 先适应新任务，再精细调整
- 更稳定的训练过程

### 6. 学习率调度改进 ✓

**改进前**：Cosine Annealing（固定衰减）

**改进后**：ReduceLROnPlateau（自适应）
```python
scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
scheduler.step(val_auc)
```

**优点**：
- 根据验证集性能自适应调整
- 避免过早或过晚衰减
- 更灵活的训练策略

### 7. 早停机制 ✓

**新增**：Early Stopping
```python
if epochs_no_improve >= patience:
    print("Early stopping triggered")
    break
```

**优点**：
- 防止过拟合
- 节省训练时间
- 自动选择最佳epoch

### 8. 数据增强调整 ✓

**改进前**：
```python
RandomRotation(15)
ColorJitter(0.2, 0.2, 0.2)
RandomAffine(translate=(0.1, 0.1))
```

**改进后**：
```python
RandomRotation(10)          # 减小旋转角度
ColorJitter(0.15, 0.15, 0.15)  # 减小颜色抖动
RandomAffine(translate=(0.05, 0.05))  # 减小平移
```

**优点**：
- 避免过度增强导致的噪声
- 保持医学图像特征
- 更真实的训练数据

### 9. 超参数优化 ✓

**参数调整**：
- `batch_size`: 32 → 16 (更稳定的梯度)
- `lr`: 1e-4 → 3e-4 (适当提高初始学习率)
- `epochs`: 20 → 40 (配合早停)
- 新增 `warmup_epochs`: 5
- 新增 `early_stop_patience`: 10

## 预期效果

改进后预期达到：

1. **减少过拟合**
   - 训练/测试准确率差距 < 10%
   - 更稳定的训练曲线

2. **平衡的预测**
   - Precision 和 Recall 更平衡
   - 减少假阳性

3. **更高的AUC**
   - 目标 AUC > 0.88
   - 更好的分类能力

4. **更强的泛化**
   - 测试准确率提升至 75-80%
   - 更可靠的临床应用

## 使用方法

```bash
# 使用默认改进参数
python 2_medimage/train.py

# 自定义参数
python 2_medimage/train.py \
    --epochs 40 \
    --batch-size 16 \
    --lr 3e-4 \
    --val-split 0.15 \
    --warmup-epochs 5 \
    --early-stop-patience 10 \
    --label-smoothing 0.1
```

## 进一步改进建议

如果效果仍不理想，可以考虑：

1. **更强的模型**：ResNet34/50
2. **集成学习**：多模型投票
3. **测试时增强**：TTA (Test Time Augmentation)
4. **焦点损失**：Focal Loss 处理难分样本
5. **混合精度训练**：提高训练效率
6. **学习率预热**：Warmup策略
