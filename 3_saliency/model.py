"""
Model definitions for saliency prediction.

We use a light UNet-style decoder with a ResNet34 encoder backbone. The encoder
starts from ImageNet weights to speed up convergence; the decoder is trained from
scratch. Output is a single-channel saliency map logit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch, out_ch, kernel_size=k, stride=1, padding=p, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed (in case of odd sizes)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y or diff_x:
            x = nn.functional.pad(x, (0, diff_x, 0, diff_y))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetSaliency(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)

        # Encoder layers (keeping original stride/padding)
        self.inc = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Decoder layers
        self.up1 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up3 = UpBlock(128, 64, 64)
        self.up4 = UpBlock(64, 64, 32)

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x) -> torch.Tensor:
        # Encoder
        x0 = self.inc(x)  # C64
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)  # C64
        x2 = self.layer2(x1)  # C128
        x3 = self.layer3(x2)  # C256
        x4 = self.layer4(x3)  # C512

        # Decoder with skip connections
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3, x0)
        out = self.out_conv(d4)
        # Upsample to input spatial size to avoid size mismatch
        out = F.interpolate(
            out, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return out  # logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(
    checkpoint_path: str, map_location: str | None = None
) -> UNetSaliency:
    model = UNetSaliency(pretrained=False)
    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.eval()
    return model


if __name__ == "__main__":
    m = UNetSaliency(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    print("Output shape", y.shape, "params", count_parameters(m))
