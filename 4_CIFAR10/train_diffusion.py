"""
CIFAR-10 彩色图像生成 - Diffusion Model实现
使用DDPM (Denoising Diffusion Probabilistic Model)生成32x32彩色图像
"""

import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm


# ==================== 配置 ====================
class Config:
    # 数据相关
    data_root = "./dataset/CIFARdata"
    image_size = 32
    num_channels = 3

    # Diffusion相关
    timesteps = 1000  # 扩散步数
    beta_start = 0.0001  # 噪声调度起始值
    beta_end = 0.02  # 噪声调度结束值

    # 模型相关
    model_channels = 128  # 基础通道数

    # 训练相关
    batch_size = 128
    num_epochs = 200
    lr = 2e-4

    # 输出相关
    output_dir = "./4_CIFAR10/outputs_diffusion"
    checkpoint_dir = "./4_CIFAR10/checkpoints_diffusion"
    sample_interval = 500
    num_sample_images = 64

    # 评估相关
    num_eval_images = 10000

    device = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 扩散过程工具函数 ====================
def linear_beta_schedule(timesteps, beta_start, beta_end):
    """线性噪声调度"""
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_params(timesteps, beta_start, beta_end):
    """计算扩散过程所需的参数"""
    betas = linear_beta_schedule(timesteps, beta_start, beta_end)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }


# ==================== U-Net模型 ====================
class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DownBlock(nn.Module):
    """下采样块"""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        h = h + time_emb[..., None, None]
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.downsample(h)


class UpBlock(nn.Module):
    """上采样块（带skip connection）"""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.upsample = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        h = h + time_emb[..., None, None]
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.upsample(h)


class MiddleBlock(nn.Module):
    """中间处理块（不改变尺寸）"""

    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(channels)
        self.bnorm2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        h = h + time_emb[..., None, None]
        h = self.bnorm2(self.relu(self.conv2(h)))
        return h


class SimpleUNet(nn.Module):
    """U-Net用于去噪，保持较深的架构"""

    def __init__(self, image_channels=3, model_channels=128, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # 初始卷积
        self.conv0 = nn.Conv2d(image_channels, model_channels, 3, padding=1)

        # 编码器（下采样路径）
        self.down1 = DownBlock(
            model_channels, model_channels, time_emb_dim
        )  # 32->16
        self.down2 = DownBlock(
            model_channels, model_channels * 2, time_emb_dim
        )  # 16->8
        self.down3 = DownBlock(
            model_channels * 2, model_channels * 4, time_emb_dim
        )  # 8->4

        # 瓶颈层
        self.middle = MiddleBlock(model_channels * 4, time_emb_dim)

        # 解码器（上采样路径）
        self.up1 = UpBlock(
            model_channels * 4 + model_channels * 4,
            model_channels * 2,
            time_emb_dim,
        )  # 4->8
        self.up2 = UpBlock(
            model_channels * 2 + model_channels * 2,
            model_channels,
            time_emb_dim,
        )  # 8->16
        self.up3 = UpBlock(
            model_channels + model_channels, model_channels, time_emb_dim
        )  # 16->32

        # 输出
        self.outc = nn.Conv2d(model_channels, image_channels, 1)

    def forward(self, x, timestep):
        # 时间编码
        t = self.time_mlp(timestep)

        # 初始卷积
        x0 = self.conv0(x)  # [B, 128, 32, 32]

        # 编码器（保存skip连接）
        d1 = self.down1(x0, t)  # [B, 128, 16, 16]
        d2 = self.down2(d1, t)  # [B, 256, 8, 8]
        d3 = self.down3(d2, t)  # [B, 512, 4, 4]

        # 瓶颈
        m = self.middle(d3, t)  # [B, 512, 4, 4]

        # 解码器（使用skip连接）
        u1 = self.up1(torch.cat([m, d3], dim=1), t)  # [B, 256, 8, 8]
        u2 = self.up2(torch.cat([u1, d2], dim=1), t)  # [B, 128, 16, 16]
        u3 = self.up3(torch.cat([u2, d1], dim=1), t)  # [B, 128, 32, 32]

        # 输出
        output = self.outc(u3)  # [B, 3, 32, 32]
        return output


# ==================== Diffusion模型 ====================
class DiffusionModel:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 计算扩散参数
        params = get_diffusion_params(
            config.timesteps, config.beta_start, config.beta_end
        )
        for k, v in params.items():
            setattr(self, k, v.to(self.device))

        # 创建模型
        self.model = SimpleUNet(
            image_channels=config.num_channels,
            model_channels=config.model_channels,
        ).to(self.device)

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：给图像添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][
            :, None, None, None
        ]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[
            t
        ][:, None, None, None]

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses(self, x_start, t, noise=None):
        """计算损失：预测噪声与真实噪声的差异"""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """反向去噪过程：单步去噪"""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[
            t
        ][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]

        # 预测噪声
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][
                :, None, None, None
            ]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size):
        """生成图像：从纯噪声开始逐步去噪"""
        shape = (
            batch_size,
            self.config.num_channels,
            self.config.image_size,
            self.config.image_size,
        )

        # 从纯噪声开始
        img = torch.randn(shape, device=self.device)

        # 逐步去噪
        for i in tqdm(
            reversed(range(0, self.config.timesteps)),
            desc="Sampling",
            total=self.config.timesteps,
            leave=False,
        ):
            t = torch.full(
                (batch_size,), i, device=self.device, dtype=torch.long
            )
            img = self.p_sample(img, t, i)

        return img


# ==================== 训练和评估 ====================
def train(config):
    """训练diffusion模型"""
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 加载数据
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # 归一化到[-1, 1]
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 创建模型
    diffusion = DiffusionModel(config)
    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=config.lr)

    # 训练循环
    global_step = 0
    losses_history = []

    for epoch in range(config.num_epochs):
        diffusion.model.train()
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(config.device)
            batch_size = images.shape[0]

            # 随机选择时间步
            t = torch.randint(
                0, config.timesteps, (batch_size,), device=config.device
            ).long()

            # 计算损失
            loss = diffusion.p_losses(images, t)

            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

            # 定期保存样本
            if global_step % config.sample_interval == 0:
                diffusion.model.eval()
                samples = diffusion.sample(config.num_sample_images)
                samples = (samples + 1) / 2  # 反归一化到[0, 1]
                save_image(
                    samples,
                    os.path.join(
                        config.output_dir, f"sample_step_{global_step}.png"
                    ),
                    nrow=8,
                )
                diffusion.model.train()

            global_step += 1

        # 记录epoch损失
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses_history.append(avg_loss)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": diffusion.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(config.output_dir, "loss_curve.png"))
    plt.close()

    return diffusion


def generate_eval_images(diffusion, config):
    """生成用于评估的图像"""
    eval_dir = os.path.join(config.output_dir, "eval_images")
    os.makedirs(eval_dir, exist_ok=True)

    diffusion.model.eval()
    num_batches = config.num_eval_images // config.batch_size

    for i in tqdm(range(num_batches), desc="Generating evaluation images"):
        samples = diffusion.sample(config.batch_size)
        samples = (samples + 1) / 2  # 反归一化到[0, 1]

        for j in range(samples.shape[0]):
            img_idx = i * config.batch_size + j
            save_image(
                samples[j], os.path.join(eval_dir, f"{img_idx:05d}.png")
            )

    print(f"Generated {config.num_eval_images} images in {eval_dir}")
    return eval_dir


def evaluate_fidelity(generated_dir, real_dir, config):
    """计算FID、IS等评估指标"""
    try:
        import torch_fidelity

        metrics = torch_fidelity.calculate_metrics(
            input1=generated_dir,
            input2=real_dir,
            cuda=torch.cuda.is_available(),
            isc=True,
            fid=True,
            kid=True,
            verbose=True,
        )
        print("\n=== Evaluation Metrics ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None


def prepare_real_images(config):
    """准备真实图像用于评估"""
    real_dir = os.path.join(config.output_dir, "real_images")
    if (
        os.path.exists(real_dir)
        and len(os.listdir(real_dir)) >= config.num_eval_images
    ):
        print(f"Real images already exist in {real_dir}")
        return real_dir

    os.makedirs(real_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=False, download=True, transform=transform
    )

    for i in tqdm(
        range(min(config.num_eval_images, len(dataset))),
        desc="Saving real images",
    ):
        img, _ = dataset[i]
        save_image(img, os.path.join(real_dir, f"{i:05d}.png"))

    print(f"Saved real images in {real_dir}")
    return real_dir


# ==================== 主函数 ====================
def main():
    config = Config()
    print(f"Using device: {config.device}")
    print(f"Training Diffusion Model on CIFAR-10")
    print(f"Timesteps: {config.timesteps}, Epochs: {config.num_epochs}")

    # 训练
    diffusion = train(config)

    # 生成评估图像
    print("\nGenerating images for evaluation...")
    generated_dir = generate_eval_images(diffusion, config)

    # 准备真实图像
    print("\nPreparing real images...")
    real_dir = prepare_real_images(config)

    # 评估
    print("\nEvaluating generated images...")
    evaluate_fidelity(generated_dir, real_dir, config)

    print("\nTraining and evaluation completed!")


if __name__ == "__main__":
    main()
