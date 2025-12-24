"""
CIFAR-10 彩色图像生成 - DCGAN实现
使用深度卷积生成对抗网络生成32x32彩色图像
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch_fidelity
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


class Config:
    """配置参数"""

    # 数据相关
    data_root = "./dataset/CIFARdata"
    image_size = 32
    num_channels = 3
    num_classes = 10

    # 模型相关
    latent_dim = 100  # 噪声维度
    ngf = 64  # 生成器特征图数量
    ndf = 64  # 判别器特征图数量

    # 训练相关
    batch_size = 128
    num_epochs = 100
    lr = 0.0002
    beta1 = 0.5  # Adam优化器参数

    # 输出相关
    output_dir = "./4_CIFAR10/outputs"
    checkpoint_dir = "./4_CIFAR10/checkpoints"
    sample_interval = 500  # 每多少个batch保存一次样本
    num_sample_images = 64  # 保存样本数量

    # 评估相关
    eval_batch_size = 100
    num_eval_images = 10000  # 用于评估的生成图像数量

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"


class Generator(nn.Module):
    """生成器网络 - DCGAN架构"""

    def __init__(self, latent_dim=100, ngf=64, num_channels=3):
        super(Generator, self).__init__()

        # 输入是latent_dim维的噪声向量
        # 通过转置卷积逐步上采样到32x32x3
        self.main = nn.Sequential(
            # 输入: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态: (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态: (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 状态: ngf x 16 x 16
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 输出: num_channels x 32 x 32
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """判别器网络 - DCGAN架构"""

    def __init__(self, num_channels=3, ndf=64):
        super(Discriminator, self).__init__()

        # 输入是32x32x3的图像
        # 通过卷积逐步下采样并输出真假判别
        self.main = nn.Sequential(
            # 输入: num_channels x 32 x 32
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: ndf x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # 输出: 1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


def weights_init(m):
    """初始化网络权重"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGANTrainer:
    """DCGAN训练器"""

    def __init__(self, config):
        self.config = config

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
        os.makedirs(
            os.path.join(config.output_dir, "real_images"), exist_ok=True
        )
        os.makedirs(
            os.path.join(config.output_dir, "generated_images"), exist_ok=True
        )

        # 初始化模型
        self.generator = Generator(
            config.latent_dim, config.ngf, config.num_channels
        ).to(config.device)
        self.discriminator = Discriminator(config.num_channels, config.ndf).to(
            config.device
        )

        # 初始化权重
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # 优化器
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.lr,
            betas=(config.beta1, 0.999),
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr,
            betas=(config.beta1, 0.999),
        )

        # 损失函数
        self.criterion = nn.BCELoss()

        # 固定噪声用于可视化训练进度
        self.fixed_noise = torch.randn(
            config.num_sample_images,
            config.latent_dim,
            1,
            1,
            device=config.device,
        )

        # 训练历史
        self.g_losses = []
        self.d_losses = []

    def train(self, dataloader):
        """训练DCGAN"""
        print(f"开始训练，使用设备: {self.config.device}")
        print(
            f"生成器参数量: {sum(p.numel() for p in self.generator.parameters()):,}"
        )
        print(
            f"判别器参数量: {sum(p.numel() for p in self.discriminator.parameters()):,}"
        )

        global_step = 0

        for epoch in range(self.config.num_epochs):
            pbar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )

            for i, (real_images, _) in enumerate(pbar):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.config.device)

                # 真实和假标签
                real_labels = torch.ones(batch_size, device=self.config.device)
                fake_labels = torch.zeros(
                    batch_size, device=self.config.device
                )

                # ==================== 训练判别器 ====================
                self.discriminator.zero_grad()

                # 真实图像
                real_outputs = self.discriminator(real_images)
                d_loss_real = self.criterion(real_outputs, real_labels)

                # 生成假图像
                noise = torch.randn(
                    batch_size,
                    self.config.latent_dim,
                    1,
                    1,
                    device=self.config.device,
                )
                fake_images = self.generator(noise)

                # 假图像
                fake_outputs = self.discriminator(fake_images.detach())
                d_loss_fake = self.criterion(fake_outputs, fake_labels)

                # 判别器总损失
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_d.step()

                # ==================== 训练生成器 ====================
                self.generator.zero_grad()

                # 生成器希望判别器将假图像判为真
                fake_outputs = self.discriminator(fake_images)
                g_loss = self.criterion(fake_outputs, real_labels)

                g_loss.backward()
                self.optimizer_g.step()

                # 记录损失
                self.g_losses.append(g_loss.item())
                self.d_losses.append(d_loss.item())

                # 更新进度条
                pbar.set_postfix(
                    {
                        "D_loss": f"{d_loss.item():.4f}",
                        "G_loss": f"{g_loss.item():.4f}",
                        "D(x)": f"{real_outputs.mean().item():.4f}",
                        "D(G(z))": f"{fake_outputs.mean().item():.4f}",
                    }
                )

                # 定期保存样本
                if global_step % self.config.sample_interval == 0:
                    self.save_samples(epoch, global_step)

                global_step += 1

            # 每个epoch结束保存模型
            self.save_checkpoint(epoch)

        # 训练结束，绘制损失曲线
        self.plot_losses()
        print("训练完成！")

    def save_samples(self, epoch, step):
        """保存生成的样本图像"""
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
            # 反归一化: [-1, 1] -> [0, 1]
            fake_images = (fake_images + 1) / 2

            # 保存图像网格
            grid = make_grid(fake_images, nrow=8, padding=2, normalize=False)
            save_path = os.path.join(
                self.config.output_dir,
                "samples",
                f"epoch_{epoch}_step_{step}.png",
            )
            save_image(grid, save_path)
        self.generator.train()

    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "g_losses": self.g_losses,
            "d_losses": self.d_losses,
        }
        path = os.path.join(
            self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, path)

        # 同时保存为最新的检查点
        latest_path = os.path.join(self.config.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"]
        )
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        self.g_losses = checkpoint["g_losses"]
        self.d_losses = checkpoint["d_losses"]
        return checkpoint["epoch"]

    def plot_losses(self):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label="Generator Loss", alpha=0.7)
        plt.plot(self.d_losses, label="Discriminator Loss", alpha=0.7)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(self.config.output_dir, "training_losses.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def generate_images(self, num_images=10000):
        """生成指定数量的图像用于评估"""
        print(f"生成 {num_images} 张图像用于评估...")
        self.generator.eval()

        gen_dir = os.path.join(self.config.output_dir, "generated_images")
        # 清空目录
        for f in os.listdir(gen_dir):
            os.remove(os.path.join(gen_dir, f))

        with torch.no_grad():
            for i in tqdm(range(0, num_images, self.config.eval_batch_size)):
                batch_size = min(self.config.eval_batch_size, num_images - i)
                noise = torch.randn(
                    batch_size,
                    self.config.latent_dim,
                    1,
                    1,
                    device=self.config.device,
                )
                fake_images = self.generator(noise)
                # 反归一化: [-1, 1] -> [0, 1]
                fake_images = (fake_images + 1) / 2

                # 保存每张图像
                for j in range(batch_size):
                    save_image(
                        fake_images[j],
                        os.path.join(gen_dir, f"gen_{i + j:05d}.png"),
                    )

        self.generator.train()
        print(f"图像已保存到 {gen_dir}")

    def save_real_images(self, dataloader, num_images=10000):
        """保存真实图像用于FID等指标计算"""
        print(f"保存 {num_images} 张真实图像用于评估...")
        real_dir = os.path.join(self.config.output_dir, "real_images")

        # 清空目录
        for f in os.listdir(real_dir):
            os.remove(os.path.join(real_dir, f))

        count = 0
        for images, _ in tqdm(dataloader):
            for i in range(images.size(0)):
                if count >= num_images:
                    break
                # 反归一化: [-1, 1] -> [0, 1]
                img = (images[i] + 1) / 2
                save_image(
                    img, os.path.join(real_dir, f"real_{count:05d}.png")
                )
                count += 1
            if count >= num_images:
                break

        print(f"真实图像已保存到 {real_dir}")


def evaluate_metrics(config):
    """计算评估指标"""
    print("\n计算评估指标...")

    gen_dir = os.path.join(config.output_dir, "generated_images")
    real_dir = os.path.join(config.output_dir, "real_images")

    try:
        metrics = torch_fidelity.calculate_metrics(
            input1=gen_dir,
            input2=real_dir,
            cuda=torch.cuda.is_available(),
            isc=True,
            fid=True,
            kid=True,
            verbose=False,
        )

        print("\n========== 评估结果 ==========")
        print(
            f"Inception Score (IS): {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}"
        )
        print(
            f"Fréchet Inception Distance (FID): {metrics['frechet_inception_distance']:.4f}"
        )
        print(
            f"Kernel Inception Distance (KID): {metrics['kernel_inception_distance_mean']:.6f} ± {metrics['kernel_inception_distance_std']:.6f}"
        )
        print("==============================\n")

        # 保存结果到文件
        with open(os.path.join(config.output_dir, "metrics.txt"), "w") as f:
            f.write("========== 评估结果 ==========\n")
            f.write(
                f"Inception Score (IS): {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}\n"
            )
            f.write(
                f"Fréchet Inception Distance (FID): {metrics['frechet_inception_distance']:.4f}\n"
            )
            f.write(
                f"Kernel Inception Distance (KID): {metrics['kernel_inception_distance_mean']:.6f} ± {metrics['kernel_inception_distance_std']:.6f}\n"
            )
            f.write("==============================\n")

        return metrics
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return None


def main():
    """主函数"""
    config = Config()

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # 归一化到[-1, 1]
        ]
    )

    # 加载数据集
    print("加载CIFAR-10数据集...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 创建训练器
    trainer = DCGANTrainer(config)

    # 训练模型
    trainer.train(train_loader)

    # 生成评估图像
    trainer.generate_images(config.num_eval_images)

    # 保存真实图像用于对比
    trainer.save_real_images(train_loader, config.num_eval_images)

    # 计算评估指标
    evaluate_metrics(config)


if __name__ == "__main__":
    main()
