import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchjpeg.codec



# ============================================================
# Configuration
# ============================================================

PIXEL_FEATURE_DIM = 256
DCT_FEATURE_DIM = 256

DCT_COMPONENTS = 3
DCT_FREQUENCIES = 64

DCT_INPUT_CHANNELS = DCT_COMPONENTS * DCT_FREQUENCIES


# ============================================================
# JPEG DCT extraction
# ============================================================

def read_jpeg_dct(
    path: str,
    dequantize: bool = False,
    log_transform: bool = True,
) -> torch.Tensor:
    """
    Read the actual JPEG-domain quantized DCT coefficients.

    Returns:
        Tensor of shape:

            [192, H/8, W/8]

        for a color JPEG:

            3 components × 64 DCT frequencies

    The 64 coefficients of each 8x8 block become channels.

    Component ordering:

        channels 0..63       -> Y
        channels 64..127     -> Cb
        channels 128..191    -> Cr
    """

    (
        dimensions,
        quantization,
        y_coefficients,
        cbcr_coefficients,
    ) = torchjpeg.codec.read_coefficients(path)

    # --------------------------------------------------------
    # Y
    #
    # [1, block_h, block_w, 8, 8]
    # --------------------------------------------------------

    y = y_coefficients.float()

    y = y.squeeze(0)

    # [block_h, block_w, 8, 8]
    block_h = y.shape[0]
    block_w = y.shape[1]

    # [block_h, block_w, 64]
    y = y.reshape(block_h, block_w, 64)

    # [64, block_h, block_w]
    y = y.permute(2, 0, 1).contiguous()

    components = [y]

    # --------------------------------------------------------
    # Cb / Cr
    # --------------------------------------------------------

    if cbcr_coefficients is None:
        raise ValueError(
            f"Expected a color JPEG with Y/Cb/Cr components, "
            f"but {path} appears to be grayscale."
        )

    cbcr = cbcr_coefficients.float()

    # Expected:
    #
    # [2, chroma_block_h, chroma_block_w, 8, 8]
    #

    if cbcr.ndim != 5 or cbcr.shape[0] != 2:
        raise ValueError(
            f"Unexpected CbCr coefficient shape for {path}: "
            f"{tuple(cbcr.shape)}"
        )

    chroma_h = cbcr.shape[1]
    chroma_w = cbcr.shape[2]

    cbcr = cbcr.reshape(
        2,
        chroma_h,
        chroma_w,
        64
    )

    # [2, 64, chroma_h, chroma_w]
    cbcr = cbcr.permute(0, 3, 1, 2).contiguous()

    # --------------------------------------------------------
    # Chroma subsampling
    #
    # Y might be 32x32 while Cb/Cr might be 16x16
    # for 4:2:0 JPEG.
    #
    # We spatially align chroma coefficient maps to the
    # Y block grid using nearest-neighbor interpolation.
    #
    # This does NOT recompute the DCT.
    # It only aligns already-stored JPEG coefficient blocks.
    # --------------------------------------------------------

    if cbcr.shape[-2:] != y.shape[-2:]:
        cbcr = F.interpolate(
            cbcr,
            size=y.shape[-2:],
            mode="nearest",
        )

    cb = cbcr[0]
    cr = cbcr[1]

    components.extend([cb, cr])

    # --------------------------------------------------------
    # [3, 64, H/8, W/8]
    # --------------------------------------------------------

    dct = torch.stack(components, dim=0)

    # --------------------------------------------------------
    # [192, H/8, W/8]
    # --------------------------------------------------------

    dct = dct.reshape(
        DCT_COMPONENTS * DCT_FREQUENCIES,
        dct.shape[-2],
        dct.shape[-1],
    )

    # --------------------------------------------------------
    # Optional dequantization
    #
    # The default experiment DOES NOT dequantize.
    #
    # JPEG stores quantized coefficients.
    # Steghide's JPEG modifications are therefore directly
    # represented in this tensor.
    #
    # The option exists for controlled experiments.
    # --------------------------------------------------------

    if dequantize:

        q = quantization.float()

        # Usually:
        #
        # [3, 8, 8]
        #
        # Convert each 8x8 table into 64 channels.

        if q.shape[0] < 3:
            raise ValueError(
                f"Unexpected JPEG quantization table shape: "
                f"{tuple(q.shape)}"
            )

        q = q[:3].reshape(3, 64)

        q = q.reshape(3, 64, 1, 1)

        dct = dct.reshape(
            3,
            64,
            dct.shape[-2],
            dct.shape[-1],
        )

        dct = dct * q

        dct = dct.reshape(
            192,
            dct.shape[-2],
            dct.shape[-1],
        )

    # --------------------------------------------------------
    # Signed log transform
    #
    # Keeps:
    #
    # sign
    # zero
    # relative coefficient changes
    #
    # while reducing the enormous scale difference between
    # DC and AC coefficients.
    # --------------------------------------------------------

    if log_transform:
        dct = torch.sign(dct) * torch.log1p(torch.abs(dct))

    return dct.contiguous()


# ============================================================
# Pixel branch
# ============================================================

class PixelBranch(nn.Module):
    """
    CNN operating directly on RGB pixels.

    Input:
        [B, 3, 256, 256]

    Output:
        [B, PIXEL_FEATURE_DIM]
    """

    def __init__(
        self,
        feature_dim: int = PIXEL_FEATURE_DIM,
    ):
        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Keep early spatial resolution high.
            # ------------------------------------------------

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 128 x 128
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 64 x 64
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 32 x 32
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256,
                feature_dim,
            ),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.projection(x)

        return x


# ============================================================
# DCT branch
# ============================================================

class DCTBranch(nn.Module):
    """
    CNN operating on JPEG DCT coefficients.

    Expected input:

        [B, 192, 32, 32]

    for a normal 256x256 color JPEG.

    192 =

        64 Y frequencies
        64 Cb frequencies
        64 Cr frequencies
    """

    def __init__(
        self,
        input_channels: int = DCT_INPUT_CHANNELS,
        feature_dim: int = DCT_FEATURE_DIM,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                input_channels,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 16 x 16
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 8 x 8
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256,
                feature_dim,
            ),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.projection(x)

        return x


# ============================================================
# Fusion model
# ============================================================

class PixelDCTNet(nn.Module):
    """
    Pixel + JPEG-DCT dual-branch steganalysis model.

    Pixel branch:
        RGB pixels

    DCT branch:
        actual JPEG quantized DCT coefficients

    Fusion:
        concatenation

    Classifier:
        small MLP

    Output:
        [B, 1]

    No sigmoid is applied inside the model.
    """

    def __init__(
        self,
        pixel_feature_dim: int = PIXEL_FEATURE_DIM,
        dct_feature_dim: int = DCT_FEATURE_DIM,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.pixel_branch = PixelBranch(
            feature_dim=pixel_feature_dim,
        )

        self.dct_branch = DCTBranch(
            input_channels=DCT_INPUT_CHANNELS,
            feature_dim=dct_feature_dim,
        )

        fused_dim = (
            pixel_feature_dim +
            dct_feature_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                fused_dim,
                512,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                512,
                256,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                256,
                1,
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
        dct: torch.Tensor,
        return_features: bool = False,
    ):

        pixel_features = self.pixel_branch(image)

        dct_features = self.dct_branch(dct)

        fused = torch.cat(
            [
                pixel_features,
                dct_features,
            ],
            dim=1,
        )

        logits = self.classifier(fused)

        if return_features:

            return {
                "logits": logits,
                "pixel_features": pixel_features,
                "dct_features": dct_features,
                "fused_features": fused,
            }

        return logits


# ============================================================
# Parameter counter
# ============================================================

def count_parameters(model: nn.Module) -> int:

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )