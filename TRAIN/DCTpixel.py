"""
train.py

Training script for PixelDCTNet — a dual-branch (RGB pixel + JPEG DCT
coefficient) steganalysis classifier.

Expected data layout
---------------------
data_root/
    cover/
        img0001.jpg
        img0002.jpg
        ...
    stego/
        img0001.jpg
        img0002.jpg
        ...

Every image must be a JPEG (steghide-style embedding operates on JPEG
coefficients, so re-saving as PNG etc. would destroy the signal).

Label convention:
    cover -> 0
    stego -> 1

Usage
-----
python train.py \
    --data_root /path/to/data \
    --epochs 30 \
    --batch_size 32 \
    --lr 3e-4 \
    --out_dir ./checkpoints
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from sklearn.metrics import roc_auc_score

from Model.DctPixel  import PixelDCTNet, read_jpeg_dct, count_parameters


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class SteganalysisDataset(Dataset):
    """
    Loads matched cover/stego JPEGs, returning:
        (pixel_tensor [3,H,W], dct_tensor [192,H/8,W/8], label [1])
    """

    IMG_EXTENSIONS = (".jpg", ".jpeg")

    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        dequantize: bool = False,
        log_transform: bool = True,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.dequantize = dequantize
        self.log_transform = log_transform
        self.augment = augment

        self.samples: List[Tuple[Path, int]] = []

        cover_dir = self.data_root / "cover"
        stego_dir = self.data_root / "stego"

        if not cover_dir.is_dir() or not stego_dir.is_dir():
            raise FileNotFoundError(
                f"Expected '{cover_dir}' and '{stego_dir}' to exist. "
                f"data_root must contain 'cover/' and 'stego/' subfolders."
            )

        for label, folder in ((0, cover_dir), (1, stego_dir)):
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append((path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No JPEG images found under {self.data_root}")

        # Pixel-domain normalization (ImageNet stats work fine as a default;
        # swap for dataset-specific stats if you compute them).
        pixel_tf = [transforms.Resize((image_size, image_size))]

        if augment:
            # NOTE: only geometry-preserving, label-safe augmentations here.
            # Anything that re-encodes the JPEG (e.g. random JPEG quality,
            # blur) would destroy the DCT-domain stego signal, since the
            # DCT tensor is read straight from the *file on disk*, not
            # derived from the augmented pixel tensor.
            pixel_tf.append(transforms.RandomHorizontalFlip(p=0.5))

        pixel_tf += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        self.pixel_transform = transforms.Compose(pixel_tf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ---- pixel branch input ----
        with Image.open(path) as im:
            im = im.convert("RGB")
            pixel_tensor = self.pixel_transform(im)

        # ---- DCT branch input (read directly from the JPEG file) ----
        dct_tensor = read_jpeg_dct(
            str(path),
            dequantize=self.dequantize,
            log_transform=self.log_transform,
        )

        # Ensure a consistent DCT spatial size across the batch. If images
        # aren't all the same resolution, center-crop/pad the DCT grid to
        # image_size/8.
        target = self.image_size // 8
        dct_tensor = self._fit_spatial(dct_tensor, target)

        label_tensor = torch.tensor([label], dtype=torch.float32)

        return pixel_tensor, dct_tensor, label_tensor

    @staticmethod
    def _fit_spatial(t: torch.Tensor, target: int) -> torch.Tensor:
        """Center-crop or zero-pad the last two dims of t to (target, target)."""
        c, h, w = t.shape

        # crop
        if h > target:
            top = (h - target) // 2
            t = t[:, top:top + target, :]
        if w > target:
            left = (w - target) // 2
            t = t[:, :, left:left + target]

        c, h, w = t.shape

        # pad
        pad_h = max(0, target - h)
        pad_w = max(0, target - w)
        if pad_h > 0 or pad_w > 0:
            t = torch.nn.functional.pad(
                t,
                (0, pad_w, 0, pad_h),
                mode="constant",
                value=0.0,
            )

        return t


# ============================================================
# Train / eval loops
# ============================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None,
    use_amp: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_labels = []
    all_probs = []
    correct = 0
    n_samples = 0

    torch.set_grad_enabled(is_train)

    for pixel, dct, label in loader:
        pixel = pixel.to(device, non_blocking=True)
        dct = dct.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(pixel, dct)
            loss = criterion(logits, label)

        if is_train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = label.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        probs = torch.sigmoid(logits.detach()).cpu().numpy().reshape(-1)
        labels_np = label.detach().cpu().numpy().reshape(-1)

        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())

        preds = (probs >= 0.5).astype(np.float32)
        correct += (preds == labels_np).sum()

    torch.set_grad_enabled(True)

    avg_loss = total_loss / max(n_samples, 1)
    accuracy = correct / max(n_samples, 1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # happens if a batch/epoch has only one class present
        auc = float("nan")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train PixelDCTNet steganalysis model")

    parser.add_argument("--data_root", type=str, required=True,
                         help="Directory containing cover/ and stego/ subfolders")
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dequantize", action="store_true",
                         help="Multiply DCT coefficients by the JPEG quant table")
    parser.add_argument("--no_log_transform", action="store_true",
                         help="Disable signed log1p transform on DCT coefficients")
    parser.add_argument("--augment", action="store_true",
                         help="Enable horizontal-flip augmentation on the pixel branch")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7,
                         help="Early stopping patience (epochs without val AUC improvement)")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint to resume from")

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    full_dataset = SteganalysisDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        dequantize=args.dequantize,
        log_transform=not args.no_log_transform,
        augment=args.augment,
    )

    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val

    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Dataset: {len(full_dataset)} images "
          f"({n_train} train / {n_val} val)")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model / optimizer / loss
    # --------------------------------------------------------

    model = PixelDCTNet().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best_val_auc = -1.0
    epochs_without_improvement = 0

    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_auc = ckpt.get("best_val_auc", -1.0)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler, use_amp=args.amp,
        )

        val_metrics = run_epoch(
            model, val_loader, criterion, device,
            optimizer=None, scaler=None, use_amp=args.amp,
        )

        scheduler.step(val_metrics["auc"])

        dt = time.time() - t0

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss {train_metrics['loss']:.4f} "
            f"train_acc {train_metrics['accuracy']:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} "
            f"val_acc {val_metrics['accuracy']:.4f} "
            f"val_auc {val_metrics['auc']:.4f} | "
            f"{dt:.1f}s"
        )

        # ---- checkpointing ----
        is_best = val_metrics["auc"] > best_val_auc

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "best_val_auc": max(best_val_auc, val_metrics["auc"]),
            "args": vars(args),
        }

        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))

        if is_best:
            best_val_auc = val_metrics["auc"]
            epochs_without_improvement = 0
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> new best val AUC: {best_val_auc:.4f} (saved best.pt)")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"No val AUC improvement for {args.patience} epochs. "
                f"Stopping early at epoch {epoch + 1}."
            )
            break

    print(f"Training complete. Best val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()