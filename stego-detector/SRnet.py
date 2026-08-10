"""
SRNet: Deep Residual Network for Steganalysis of Digital Images
Reference: Boroumand, Chen, Fridrich (2019), IEEE TIFS.

This file contains THREE models:

  - `SRNet`             : the original 2019 architecture, unchanged, kept
                           here for comparison / reproducibility.
  - `SRNet2036`          : a modernized redesign, built specifically for
                           512x512 RGB inputs, described below.
  - `SRNet2036ViTHybrid` : SRNet2036's CNN backbone feeding into a proper
                           multi-layer Transformer encoder (a real ViT,
                           just tokenizing residual features instead of
                           raw pixels -- see that class's docstring for
                           why a *raw-pixel* ViT is a bad fit here).

-------------------------------------------------------------------------
A quick honesty note on the name "2036": I don't have any actual knowledge
of architectures from 2036 -- that's ten years past my training cutoff.
What follows is the strongest steganalysis backbone I can build today by
combining real, dated, published techniques (2016-2022) that weren't
available or weren't yet standard when the original 2019 SRNet paper was
written, plus one structural change specific to making a 512x512-native
network actually trainable in reasonable time. Think of it as "SRNet if
its authors had access to five more years of the deep learning toolbox,"
not literal future knowledge.
-------------------------------------------------------------------------

WHAT CHANGED AND WHY (SRNet2036 vs. original SRNet):

1. Fixed high-pass "residual" filters in the stem, alongside learned ones.
   The original SRNet's first layer has to *learn* to compute a noise
   residual map from scratch. We instead hand it a small bank of
   classic, non-trainable high-pass filters up front -- including the KV
   kernel (Kodovsky & Fridrich's 2nd-order high-pass filter, used to
   initialize the first conv layer of Xu-Net and several later
   steganalysis CNNs) plus a Laplacian and edge filters. This is the same
   "fixed + learned" hybrid stem idea used in Ye-Net, Yedroudj-Net, and
   GBRAS-Net. It gives the network a running start on exactly the signal
   it's trying to detect, instead of re-deriving it purely from gradients.

2. Downsample EARLY, right after the stem, before the expensive residual
   stack. The original network runs 5 residual blocks at full input
   resolution before downsampling at all. At 256x256 (the resolution the
   original paper targets) that's expensive but tolerable; at 512x512
   (4x the pixels) it becomes the single biggest compute/memory cost in
   the network -- almost certainly why your earlier runs took 25-70
   minutes per epoch. The noise-residual signal steganalysis depends on
   is captured by the stem's first 1-2 conv layers (that's the whole
   point of those layers); once it's captured, the deeper blocks are
   aggregating statistics over that residual map, and don't need to keep
   re-processing it at full spatial resolution. So: fixed+learned filters
   run at full 512x512 resolution (cheap: only ~2 conv layers, few
   channels), then we downsample once (avg-pool, never max-pool -- same
   reasoning as the original paper: max-pooling would throw away exactly
   the small-amplitude signal we're trying to keep) before the deep
   residual stack.

3. Depthwise-separable convolutions in the residual/downsampling blocks
   (MobileNet-style: a per-channel 3x3 "depthwise" conv, then a 1x1
   "pointwise" conv to mix channels), instead of full dense 3x3
   convolutions. Same receptive field and channel count, a fraction of
   the FLOPs. Combined with (2), this is what makes a 512x512-native
   network practical to train on a single GPU.

4. GroupNorm instead of BatchNorm. 512x512 activations are memory-heavy,
   which typically forces small batch sizes. BatchNorm's running
   statistics get noisy and unreliable at small batch sizes; GroupNorm
   (Wu & He, 2018) normalizes within each sample and is batch-size
   independent, so training stays stable regardless of how small you have
   to set --batch-size for a given GPU.

5. ECA channel attention (Wang et al., CVPR 2020) after each residual
   block. A near-free (few dozen parameters) mechanism that lets each
   block re-weight its output channels by importance, without the
   channel-reduction bottleneck of classic Squeeze-and-Excitation (which
   the ECA paper showed loses information).

6. A small bottleneck self-attention stage (à la BoTNet / CoAtNet-style
   hybrid CNN-Transformer backbones), applied only at the network's
   lowest-resolution stage where it's cheap (e.g. 16x16 = 256 tokens).
   Convolutions only ever look at local neighborhoods. Some steganography
   embedding strategies are content-adaptive and leave behind subtle
   *global* consistency shifts -- e.g. the statistical relationship
   between a smooth region and a busy region changes in a way no single
   local patch reveals by itself. Self-attention at the bottleneck lets
   every spatial location directly compare itself against every other
   location's residual statistics.

7. Stochastic depth / DropPath (Huang et al., 2016) on every residual
   branch, GELU instead of ReLU, and a small MLP classifier head with
   dropout -- standard modern regularization/activation choices (also
   used throughout ConvNeXt) that tend to help once you're at this depth.

Everything above is a drop-in `nn.Module` with the same interface as the
original (`SRNet2036(in_channels=3, num_classes=2)`, forward pass takes
(N, C, H, W) and returns (N, num_classes) logits), and, like the
original, uses AdaptiveAvgPool2d at the end so it isn't hard-locked to any
one input resolution -- but it's been designed and dimensioned with
512x512 specifically in mind.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Original 2019 SRNet, kept for reference / comparison.
# =============================================================================

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
    """Conv - BN - ReLU - Conv - BN, with an identity residual connection."""

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
    """Conv - BN - ReLU - Conv - BN - AvgPool, with a projected residual connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.proj_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        residual = self.proj_bn(self.proj_conv(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        return out + residual


class Type4(nn.Module):
    """Conv - BN - ReLU - Conv - BN - GlobalAveragePool."""

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
    """The original architecture, unchanged. Kept for reference/comparison."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.layer1 = Type1(in_channels, 64)
        self.layer2 = Type1(64, 16)
        self.layer3 = Type2(16)
        self.layer4 = Type2(16)
        self.layer5 = Type2(16)
        self.layer6 = Type2(16)
        self.layer7 = Type2(16)
        self.layer8 = Type3(16, 16)
        self.layer9 = Type3(16, 64)
        self.layer10 = Type3(64, 128)
        self.layer11 = Type3(128, 256)
        self.layer12 = Type4(256, 512)
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
        x = self.layer12(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =============================================================================
# SRNet2036: modernized redesign, built for 512x512.
# =============================================================================

def make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with the largest group count <= max_groups that evenly
    divides `channels` (GroupNorm requires channels % groups == 0)."""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class FixedHighPassFilters(nn.Module):
    """
    A small bank of classic, hand-designed, NON-TRAINABLE high-pass
    filters, applied depthwise (independently per input channel).

    Includes the KV kernel (Kodovsky & Fridrich's 2nd-order high-pass
    filter -- the same kernel used to initialize Xu-Net's first conv
    layer in the steganalysis literature), a Laplacian, horizontal /
    vertical Sobel edge filters, and two diagonal derivative filters.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        kv = torch.tensor([
            [-1,  2,  -2,  2, -1],
            [ 2, -6,   8, -6,  2],
            [-2,  8, -12,  8, -2],
            [ 2, -6,   8, -6,  2],
            [-1,  2,  -2,  2, -1],
        ], dtype=torch.float32) / 12.0

        laplacian = torch.tensor([
            [0.,  1., 0.],
            [1., -4., 1.],
            [0.,  1., 0.],
        ])
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.],
        ])
        sobel_y = sobel_x.t().contiguous()
        diag1 = torch.tensor([
            [ 2., -1.,  0.],
            [-1.,  0.,  1.],
            [ 0.,  1., -2.],
        ])
        diag2 = torch.flip(diag1, dims=[1]).contiguous()

        kernels_3x3 = [laplacian, sobel_x, sobel_y, diag1, diag2]
        kernels_5x5 = [F.pad(k, (1, 1, 1, 1)) for k in kernels_3x3] + [kv]

        self.num_filters = len(kernels_5x5)  # 6
        weight = torch.stack(kernels_5x5).unsqueeze(1)          # (6, 1, 5, 5)
        weight = weight.repeat(in_channels, 1, 1, 1)             # (6*in_ch, 1, 5, 5)
        # ^ contiguous per-channel blocks of 6, matching conv2d's groups semantics.

        self.register_buffer("weight", weight)
        self.in_channels = in_channels
        self.out_channels = in_channels * self.num_filters

    def forward(self, x):
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=2, groups=self.in_channels)


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020). Global-average-pool,
    then a tiny 1D conv across channels, then a sigmoid gate. Adds only a
    handful of parameters -- no channel-reduction bottleneck like classic
    Squeeze-and-Excite (which the ECA paper showed loses information)."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)   # (N, 1, C)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (N, C, 1, 1)
        return x * y


class DropPath(nn.Module):
    """Stochastic depth (Huang et al., 2016): randomly zeroes an entire
    residual branch per-sample during training. Standard regularizer in
    modern deep residual / ConvNeXt-style networks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class Type1Modern(nn.Module):
    """Conv - GroupNorm - GELU. Modernized stem block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.norm = make_group_norm(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Type2Modern(nn.Module):
    """Depthwise-separable identity-residual block (channels & spatial size
    unchanged), with ECA channel attention and stochastic depth."""

    def __init__(self, channels: int, drop_path: float = 0.0):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm1 = make_group_norm(channels)
        self.act = nn.GELU()
        self.dw2 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm2 = make_group_norm(channels)
        self.eca = ECA(channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.pw1(self.dw1(x))))
        out = self.norm2(self.pw2(self.dw2(out)))
        out = self.eca(out)
        return residual + self.drop_path(out)


class Type3Modern(nn.Module):
    """Downsampling block: depthwise-separable Conv-GN-GELU x2, then
    AvgPool2d (never max-pool -- would discard the small-amplitude
    embedding signal, same reasoning as the original SRNet paper). The
    projection shortcut also uses AvgPool2d (not a strided conv) to stay
    consistent with that same "never destructively subsample" philosophy."""

    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.dw1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.pw1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm1 = make_group_norm(out_ch)
        self.act = nn.GELU()
        self.dw2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch, bias=False)
        self.pw2 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.norm2 = make_group_norm(out_ch)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.eca = ECA(out_ch)
        self.drop_path = DropPath(drop_path)

        self.proj_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.proj_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.proj_norm = make_group_norm(out_ch)

    def forward(self, x):
        residual = self.proj_norm(self.proj_conv(self.proj_pool(x)))
        out = self.act(self.norm1(self.pw1(self.dw1(x))))
        out = self.norm2(self.pw2(self.dw2(out)))
        out = self.pool(out)
        out = self.eca(out)
        return residual + self.drop_path(out)


class Type4Modern(nn.Module):
    """Conv - GN - GELU - Conv - GN - ECA - GlobalAveragePool. Final
    feature-extraction stage; spatial size is small here so full dense
    convs (not depthwise-separable) are still cheap."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.norm1 = make_group_norm(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = make_group_norm(out_ch)
        self.eca = ECA(out_ch)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.eca(out)
        return self.global_pool(out)


class BottleneckSelfAttention(nn.Module):
    """Lightweight multi-head self-attention over spatial positions,
    applied only at the network's lowest-resolution stage (cheap: e.g.
    16x16 = 256 tokens). Lets every location directly compare its residual
    statistics against every other location's, which a purely
    convolutional stack can only approximate after many more layers.
    Pattern follows hybrid CNN-Transformer backbones like BoTNet
    (Srinivas et al., 2021) and CoAtNet (Dai et al., 2021): convolutions
    early for local/high-frequency structure, attention late for
    global/low-frequency structure."""

    def __init__(self, channels: int, num_heads: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)  # 1 group == normalize over all channels, like LayerNorm
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        n, c, h, w = x.shape
        residual = x
        y = self.norm(x).flatten(2).transpose(1, 2)   # (N, H*W, C)
        y, _ = self.mha(y, y, y, need_weights=False)
        y = y.transpose(1, 2).reshape(n, c, h, w)
        return residual + self.drop_path(y)


class SRNet2036(nn.Module):
    """
    Modernized SRNet, redesigned for 512x512 RGB input. See the module
    docstring above for the full rationale behind each change.

    Args:
        in_channels: 3 for RGB (also works with 1 for grayscale).
        num_classes: 2 for cover-vs-stego binary detection.
        drop_path_rate: max stochastic-depth drop probability, linearly
            ramped across the identity-residual stack (0 disables it).
        attn_heads: number of attention heads in the bottleneck
            self-attention stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        drop_path_rate: float = 0.1,
        attn_heads: int = 4,
    ):
        super().__init__()

        # --- Stem: fixed high-pass filters + learned conv, full native res ---
        self.fixed_filters = FixedHighPassFilters(in_channels)
        fixed_ch = self.fixed_filters.out_channels
        learned_ch = max(64 - fixed_ch, 16)
        self.learned_stem = nn.Sequential(
            nn.Conv2d(in_channels, learned_ch, 3, 1, 1, bias=False),
            make_group_norm(learned_ch),
            nn.GELU(),
        )
        self.stem_fuse = Type1Modern(fixed_ch + learned_ch, 64)
        self.layer2 = Type1Modern(64, 16)

        # --- Early downsample, BEFORE the expensive residual stack. ---
        self.early_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # --- Identity-residual stack ---
        n_type2 = 6
        dpr = [drop_path_rate * i / max(n_type2 - 1, 1) for i in range(n_type2)]
        self.type2_blocks = nn.ModuleList(
            [Type2Modern(16, drop_path=dpr[i]) for i in range(n_type2)]
        )

        # --- Downsampling stack ---
        self.down1 = Type3Modern(16, 64, drop_path=drop_path_rate)
        self.down2 = Type3Modern(64, 128, drop_path=drop_path_rate)
        self.down3 = Type3Modern(128, 256, drop_path=drop_path_rate)
        self.down4 = Type3Modern(256, 384, drop_path=drop_path_rate)

        # --- Bottleneck self-attention ---
        self.bottleneck_attn = BottleneckSelfAttention(
            384, num_heads=attn_heads, drop_path=drop_path_rate
        )

        # --- Final feature extraction + classifier ---
        self.layer_final = Type4Modern(384, 512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fixed = self.fixed_filters(x)
        learned = self.learned_stem(x)
        x = torch.cat([fixed, learned], dim=1)
        x = self.stem_fuse(x)
        x = self.layer2(x)

        x = self.early_pool(x)

        for blk in self.type2_blocks:
            x = blk(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.bottleneck_attn(x)

        x = self.layer_final(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Standard pre-LN Transformer encoder block, same recipe as ViT/BERT:
    LayerNorm -> multi-head self-attention -> residual, LayerNorm -> MLP
    (GELU) -> residual. Stochastic depth on both branches."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path1(y)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class SRNet2036ViTHybrid(nn.Module):
    """
    'What if we use a ViT' -- done in a way that doesn't fight the physics
    of the problem.

    A vanilla/raw-pixel ViT tokenizes the *input image* with a large-stride
    patch embedding (typically a 16x16-stride conv). That's a linear
    average over every pixel inside each patch -- exactly the kind of
    operation this project has been careful to avoid everywhere else (it's
    why the training script uses Crop instead of Resize, and AvgPool
    instead of MaxPool with deliberate care about what gets destroyed).
    Steganographic embedding is a tiny, single-pixel-scale perturbation;
    a 16x16 patch embedding would blur most of that signal away before the
    transformer ever sees it. On top of that, plain ViTs have very weak
    built-in locality / translation-equivariance bias, so they typically
    need hundreds of thousands to millions of training images (or transfer
    from a large pretrained checkpoint) to match a CNN's accuracy -- this
    dataset (16k train images) is small by ViT standards, and any
    pretrained ViT weights would be for natural-image classification, not
    residual-noise statistics, so they're not obviously useful here either.

    So: instead of tokenizing raw pixels, this hybrid reuses SRNet2036's
    CNN backbone (fixed high-pass filters + learned stem + early
    downsample + depthwise-separable residual/downsampling stack) to first
    turn the image into a (384, 16, 16) *residual feature map*. By the
    time we get there, each spatial position already summarizes a small
    neighborhood's worth of noise-residual statistics -- THAT is what gets
    tokenized (256 tokens + one learnable [CLS] token, with learnable
    positional embeddings) and fed through several real pre-LN Transformer
    encoder blocks. This is the actual "ViT part" of the model, just
    applied after the CNN has done the local, high-frequency-signal-
    preserving work instead of before it. Same "convolutions early,
    attention late" pattern as BoTNet/CoAtNet -- just with a proper
    multi-layer transformer instead of SRNet2036's single bottleneck
    attention block, so if you want more genuine "ViT-ness," this is it.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- CNN backbone (identical design to SRNet2036's, up to the
        # bottleneck feature map -- see that class's docstring for the
        # rationale behind each piece) ---
        self.fixed_filters = FixedHighPassFilters(in_channels)
        fixed_ch = self.fixed_filters.out_channels
        learned_ch = max(64 - fixed_ch, 16)
        self.learned_stem = nn.Sequential(
            nn.Conv2d(in_channels, learned_ch, 3, 1, 1, bias=False),
            make_group_norm(learned_ch),
            nn.GELU(),
        )
        self.stem_fuse = Type1Modern(fixed_ch + learned_ch, 64)
        self.layer2 = Type1Modern(64, 16)
        self.early_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        n_type2 = 4
        dpr_cnn = [drop_path_rate * i / max(n_type2 - 1, 1) for i in range(n_type2)]
        self.type2_blocks = nn.ModuleList(
            [Type2Modern(16, drop_path=dpr_cnn[i]) for i in range(n_type2)]
        )

        self.down1 = Type3Modern(16, 64, drop_path=drop_path_rate)
        self.down2 = Type3Modern(64, 128, drop_path=drop_path_rate)
        self.down3 = Type3Modern(128, 256, drop_path=drop_path_rate)
        self.down4 = Type3Modern(256, 384, drop_path=drop_path_rate)
        # 512x512 input -> (N, 384, 16, 16) here -> 256 spatial tokens

        # --- Transformer encoder over the CNN's residual feature map ---
        embed_dim = 384
        base_grid = 16  # token-grid size for a 512x512 input
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, base_grid * base_grid + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        dpr_t = [drop_path_rate * i / max(transformer_depth - 1, 1) for i in range(transformer_depth)]
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, transformer_heads, drop_path=dpr_t[i], dropout=dropout
            )
            for i in range(transformer_depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _positional_embedding(self, h: int, w: int) -> torch.Tensor:
        """Bicubically resizes the learned positional embedding grid if the
        input resolution doesn't produce a 16x16 token grid (e.g. if you
        train/eval at something other than 512x512), same technique
        standard ViT checkpoints use to support multiple resolutions."""
        n_tokens = h * w
        if n_tokens == self.pos_embed.shape[1] - 1:
            return self.pos_embed
        cls_pe = self.pos_embed[:, :1]
        grid_pe = self.pos_embed[:, 1:]
        orig_hw = int(round(grid_pe.shape[1] ** 0.5))
        grid_pe = grid_pe.reshape(1, orig_hw, orig_hw, -1).permute(0, 3, 1, 2)
        grid_pe = F.interpolate(grid_pe, size=(h, w), mode="bicubic", align_corners=False)
        grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat([cls_pe, grid_pe], dim=1)

    def forward(self, x):
        fixed = self.fixed_filters(x)
        learned = self.learned_stem(x)
        x = torch.cat([fixed, learned], dim=1)
        x = self.stem_fuse(x)
        x = self.layer2(x)
        x = self.early_pool(x)

        for blk in self.type2_blocks:
            x = blk(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)  # (N, 384, H, W)

        n, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)               # (N, H*W, C)
        cls = self.cls_token.expand(n, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)             # (N, H*W+1, C)
        tokens = tokens + self._positional_embedding(h, w)
        tokens = self.pos_drop(tokens)

        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]
        return self.classifier(cls_out)


if __name__ == "__main__":
    # --- Original, at its native 256x256 ---
    model_orig = SRNet(in_channels=3, num_classes=2)
    dummy_256 = torch.randn(2, 3, 256, 256)
    out_orig = model_orig(dummy_256)
    n_params_orig = sum(p.numel() for p in model_orig.parameters())
    print("SRNet (original)      @256x256 -> output:", out_orig.shape,
          "| params:", f"{n_params_orig:,}")

    # --- SRNet2036, at 512x512 (its intended resolution) ---
    model_new = SRNet2036(in_channels=3, num_classes=2)
    dummy_512 = torch.randn(2, 3, 512, 512)
    out_new = model_new(dummy_512)
    n_params_new = sum(p.numel() for p in model_new.parameters())
    print("SRNet2036              @512x512 -> output:", out_new.shape,
          "| params:", f"{n_params_new:,}")

    # --- SRNet2036 also still works at other resolutions (adaptive pooling) ---
    dummy_256b = torch.randn(2, 3, 256, 256)
    out_new_256 = model_new(dummy_256b)
    print("SRNet2036              @256x256 -> output:", out_new_256.shape)

    # --- CNN-Transformer hybrid, at 512x512 ---
    model_hybrid = SRNet2036ViTHybrid(in_channels=3, num_classes=2)
    out_hybrid = model_hybrid(dummy_512)
    n_params_hybrid = sum(p.numel() for p in model_hybrid.parameters())
    print("SRNet2036ViTHybrid      @512x512 -> output:", out_hybrid.shape,
          "| params:", f"{n_params_hybrid:,}")

    # --- and at a non-512 resolution, to confirm pos-embed interpolation works ---
    out_hybrid_256 = model_hybrid(dummy_256b)
    print("SRNet2036ViTHybrid      @256x256 -> output:", out_hybrid_256.shape)
