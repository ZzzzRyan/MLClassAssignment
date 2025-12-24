# CIFAR-10 彩色图像生成实验

基于DCGAN（Deep Convolutional Generative Adversarial Networks）实现的CIFAR-10彩色图像生成任务。

## 模型概述

### DCGAN原理

DCGAN是一种深度卷积生成对抗网络，由两个神经网络组成：
- **生成器 (Generator)**: 从随机噪声向量生成逼真的图像
- **判别器 (Discriminator)**: 判断图像是真实的还是生成的

两个网络通过对抗训练相互博弈：
- 生成器试图生成越来越逼真的图像来"欺骗"判别器
- 判别器试图更准确地区分真实图像和生成图像

### 网络架构

**生成器结构**:
- 输入: 100维噪声向量 (latent vector)
- 使用转置卷积进行上采样: 1×1 → 4×4 → 8×8 → 16×16 → 32×32
- 使用BatchNorm和ReLU激活
- 输出层使用Tanh激活，输出范围[-1, 1]
- 参数量: 约1.5M

**判别器结构**:
- 输入: 32×32×3 RGB图像
- 使用标准卷积进行下采样: 32×32 → 16×16 → 8×8 → 4×4 → 1×1
- 使用BatchNorm和LeakyReLU激活
- 输出层使用Sigmoid激活，输出0-1之间的真假概率
- 参数量: 约1.4M

### 训练策略

- 损失函数: Binary Cross Entropy (BCE)
- 优化器: Adam (lr=0.0002, beta1=0.5)
- 训练技巧:
  - 权重初始化: 使用均值0、标准差0.02的正态分布
  - 标签平滑: 真实标签=1, 假标签=0
  - 交替训练判别器和生成器

## 使用方法

### 环境配置

```bash
# 安装依赖（已在pyproject.toml中配置）
uv sync
```

### 训练模型

```bash
# 运行训练脚本
uv run 4_CIFAR10/train_dcgan.py
```

训练过程会自动：
1. 下载CIFAR-10数据集（如果未下载）
2. 训练DCGAN模型
3. 定期保存生成样本（每500个batch）
4. 每个epoch保存检查点
5. 训练结束后生成评估图像
6. 计算IS、FID、KID指标

### 配置参数

可在 `train_dcgan.py` 中的 `Config` 类修改训练参数：

```python
class Config:
    # 模型参数
    latent_dim = 100      # 噪声维度
    ngf = 64              # 生成器特征数
    ndf = 64              # 判别器特征数

    # 训练参数
    batch_size = 128      # 批大小
    num_epochs = 100      # 训练轮数
    lr = 0.0002           # 学习率
    beta1 = 0.5           # Adam参数

    # 评估参数
    num_eval_images = 10000  # 生成图像数量
```

## 输出文件

训练完成后会在 `4_CIFAR10/outputs/` 目录下生成：

```
outputs/
├── samples/                    # 训练过程中的生成样本
│   ├── epoch_0_step_0.png
│   ├── epoch_0_step_500.png
│   └── ...
├── generated_images/           # 用于评估的生成图像
│   ├── gen_00000.png
│   └── ...
├── real_images/               # 用于对比的真实图像
│   ├── real_00000.png
│   └── ...
├── training_losses.png        # 训练损失曲线
└── metrics.txt                # 评估指标结果

checkpoints/
├── checkpoint_epoch_0.pth
├── checkpoint_epoch_1.pth
├── ...
└── latest.pth                 # 最新检查点
```

## 评估指标

### Inception Score (IS)
- 衡量生成图像的质量和多样性
- 分数越高越好（理想值>10）
- 计算方式: 基于Inception-v3网络的预测分布

### Fréchet Inception Distance (FID)
- 衡量生成图像与真实图像分布的距离
- 分数越低越好（理想值<50）
- 计算方式: 比较真实图像和生成图像在Inception特征空间的分布差异

### Kernel Inception Distance (KID)
- FID的无偏版本
- 分数越低越好
- 对样本数量的敏感度较低

## 实验结果分析

训练建议：
1. **前期训练**（0-20 epochs）：图像模糊，主要学习颜色和基本形状
2. **中期训练**（20-60 epochs）：图像逐渐清晰，可以识别物体类别
3. **后期训练**（60-100 epochs）：图像质量稳定，细节更丰富

常见问题：
- **模式崩溃 (Mode Collapse)**：生成器只生成少数几种图像
  - 解决方法：调整学习率，增加判别器训练频率
- **训练不稳定**：损失剧烈波动
  - 解决方法：降低学习率，使用标签平滑

## 改进方向

1. **模型改进**:
   - 使用WGAN-GP替代标准GAN
   - 增加Self-Attention层
   - 使用Progressive Growing策略

2. **训练技巧**:
   - 使用Spectral Normalization
   - Two Time-Scale Update Rule
   - 标签平滑和单边标签平滑

3. **评估扩展**:
   - 按类别分别评估生成质量
   - 使用LPIPS进行感知相似度评估
   - 人工主观评价

## 代码特点

- **单文件实现**：所有功能集中在一个文件中，便于理解和修改
- **配置类管理**：使用Config类统一管理所有超参数
- **完整流程**：包含训练、生成、评估的完整pipeline
- **进度可视化**：使用tqdm显示训练进度，定期保存样本观察效果
- **自动评估**：训练结束后自动计算IS、FID、KID指标
