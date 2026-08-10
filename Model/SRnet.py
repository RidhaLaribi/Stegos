"""
SRNet: Deep Residual Network for Steganalysis of Digital Images
Reference: Boroumand, Chen, Fridrich (2019), IEEE TIFS.

Implemented in PyTorch. This version is adapted to work on RGB images
(e.g. 256x256x3) in addition to grayscale. The original paper operates on
single-channel grayscale patches; here the first layer simply accepts 3
input channels and learns separate noise-residual-like filters per color
channel, and everything downstream is unchanged.

Architecture overview (operates on image patches, e.g. 256x256x3 for RGB):
  - Layer type 1 (x2):  Conv - BN - ReLU                              (no residual, no pooling)
  - Layer type 2 (x5):  Conv - BN - ReLU - Conv - BN, + identity skip  (no pooling)
  - Layer type 3 (x4):  Conv - BN - ReLU - Conv - BN - AvgPool(3,2),
                        + 1x1 conv/BN projection skip                 (spatial downsampling)
  - Layer type 4 (x1):  Conv - BN - ReLU - Conv - BN - GlobalAvgPool
  - Fully connected layer -> 2 classes (cover / stego)

No max-pooling is used anywhere in the network (a deliberate design choice in the
original paper, since max-pooling discards the small-amplitude embedding signal
that steganalysis depends on).
"""

import torch
import torch.nn as nn


class Type1(nn.Module):
    """Conv - BN - ReLU. Used for the first two layers of the network."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Type2(nn.Module):
    """Conv - BN - ReLU - Conv - BN, with an identity residual connection.

    Channel count and spatial size are unchanged, so the skip connection is a
    plain identity add.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class Type3(nn.Module):
    """Conv - BN - ReLU - Conv - BN - AvgPool, with a projected (strided 1x1 conv)
    residual connection to match the downsampled spatial size / new channel count.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Projection shortcut: 1x1 conv with matching stride, to align both
        # channel depth and spatial resolution with the main path.
        self.proj_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        residual = self.proj_bn(self.proj_conv(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        return out + residual


class Type4(nn.Module):
    """Conv - BN - ReLU - Conv - BN - GlobalAveragePool. Final feature-extraction layer."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.global_pool(out)
        return out


class SRNet(nn.Module):
    """Full SRNet model.

    Args:
        in_channels: number of input image channels. Defaults to 3 for RGB
            images; pass 1 to use single-channel grayscale patches instead
            (the setting used in the original paper).
        num_classes: 2 for cover-vs-stego binary detection.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()

        # --- Type 1 layers (no residual, no pooling): Layers 1-2 ---
        self.layer1 = Type1(in_channels, 64)
        self.layer2 = Type1(64, 16)

        # --- Type 2 layers (identity residual, no pooling): Layers 3-7 ---
        self.layer3 = Type2(16)
        self.layer4 = Type2(16)
        self.layer5 = Type2(16)
        self.layer6 = Type2(16)
        self.layer7 = Type2(16)

        # --- Type 3 layers (projected residual + downsampling): Layers 8-11 ---
        self.layer8 = Type3(16, 16)
        self.layer9 = Type3(16, 64)
        self.layer10 = Type3(64, 128)
        self.layer11 = Type3(128, 256)

        # --- Type 4 layer (global pooling): Layer 12 ---
        self.layer12 = Type4(256, 512)

        # --- Classifier ---
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        x = self.layer12(x)          # -> (N, 512, 1, 1)
        x = torch.flatten(x, 1)      # -> (N, 512)
        x = self.fc(x)               # -> (N, num_classes) logits
        return x


if __name__ == "__main__":
    # Sanity check on RGB input (batch of 4, 256x256, 3 channels).
    model_rgb = SRNet(in_channels=3, num_classes=2)
    dummy_rgb = torch.randn(4, 3, 256, 256)
    logits_rgb = model_rgb(dummy_rgb)

    n_params_rgb = sum(p.numel() for p in model_rgb.parameters())
    print("RGB  -> output shape:", logits_rgb.shape)   # expected: torch.Size([4, 2])
    print("RGB  -> total parameters:", f"{n_params_rgb:,}")

    # Sanity check on grayscale input still works, for backward compatibility.
    model_gray = SRNet(in_channels=1, num_classes=2)
    dummy_gray = torch.randn(2, 1, 256, 256)
    logits_gray = model_gray(dummy_gray)

    n_params_gray = sum(p.numel() for p in model_gray.parameters())
    print("Gray -> output shape:", logits_gray.shape)  # expected: torch.Size([2, 2])
    print("Gray -> total parameters:", f"{n_params_gray:,}")