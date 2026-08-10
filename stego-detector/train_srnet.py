import argparse
import time
import os
import sys
sys.path.append('/content/drive/MyDrive/SRnet')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from SRnet import SRNet2036, SRNet2036ViTHybrid

MODEL_REGISTRY = {
    "cnn": SRNet2036,
    "vit": SRNet2036ViTHybrid,
}


def get_dataloaders(args):
    # Train transform: light, steganalysis-safe augmentation only.
    # We avoid RandomCrop / ColorJitter / heavy geometric transforms because
    # they can destroy or shift the subtle embedding residuals SRNet learns
    # to detect. Horizontal + vertical flip and 90-degree rotations are safe
    # (they preserve local pixel statistics) and are standard in steganalysis
    # literature (they effectively give you the 8 D4 symmetries "for free").
    # IMPORTANT: we use Crop, never Resize, to change spatial dimensions.
    # Resize() interpolates between neighboring pixels, which blends/smooths
    # exactly the high-frequency embedding residuals SRNet is trying to
    # detect -- this can destroy or heavily distort the payload's statistical
    # trace. Crop takes a sub-region with pixel values completely untouched,
    # so any embedding trace inside the cropped region is 100% preserved.
    # pad_if_needed guards against source images smaller than image_size.
    train_transform = transforms.Compose([
        transforms.RandomCrop(
            (args.image_size, args.image_size), pad_if_needed=True, padding_mode="reflect"
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.ToTensor(),
    ])

    # Val/Test: NO augmentation, and NO randomness -- CenterCrop is
    # deterministic, so results are reproducible and comparable across runs.
    eval_transform = transforms.Compose([
        transforms.CenterCrop((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(root=args.train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(root=args.val_dir, transform=eval_transform)
    test_ds = datasets.ImageFolder(root=args.test_dir, transform=eval_transform)

    print(f"Train: {len(train_ds)} images | class_to_idx={train_ds.class_to_idx}")
    print(f"Val:   {len(val_ds)} images | class_to_idx={val_ds.class_to_idx}")
    print(f"Test:  {len(test_ds)} images | class_to_idx={test_ds.class_to_idx}")

    # Sanity check: class-to-index mapping must match across splits, otherwise
    # labels 0/1 mean different things in train vs val/test and all metrics
    # become meaningless.
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, \
        "class_to_idx mismatch between splits! Check folder names (clean/stego) in each split."

    # ------------------------------------------------------------------
    # Class-balance check. A model that predicts only the majority class
    # will get a fixed, unmoving accuracy equal to that class's proportion
    # in the split -- exactly the "frozen val_acc" symptom you get from
    # class-imbalanced training data. Print counts here so this is obvious
    # from the very first line of logs instead of discovered 20 epochs in.
    # ------------------------------------------------------------------
    def _print_class_counts(name, ds):
        counts = [0] * len(ds.classes)
        for _, label in ds.samples:
            counts[label] += 1
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        total = sum(counts)
        parts = ", ".join(
            f"{idx_to_class[i]}={c} ({100.0 * c / total:.1f}%)"
            for i, c in enumerate(counts)
        )
        print(f"{name} class counts: {parts}")
        return counts

    train_counts = _print_class_counts("Train", train_ds)
    _print_class_counts("Val", val_ds)
    _print_class_counts("Test", test_ds)

    max_frac = max(train_counts) / sum(train_counts)
    if max_frac > 0.6:
        print(
            f"WARNING: training set is imbalanced (majority class = "
            f"{100.0 * max_frac:.1f}%). A collapsed model that always "
            f"predicts the majority class would score ~{100.0 * max_frac:.1f}% "
            f"accuracy without learning anything. Class weighting / "
            f"weighted sampling is enabled below to counter this."
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx, train_counts


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=2, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # Per-class correct/total, so we can detect majority-class collapse
    # directly (overall accuracy alone can't tell you this).
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for c in range(num_classes):
            mask = labels == c
            class_total[c] += mask.sum().item()
            class_correct[c] += (preds[mask] == labels[mask]).sum().item()
    model.train()
    per_class_acc = [
        100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else float("nan")
        for c in range(num_classes)
    ]
    return running_loss / total, 100.0 * correct / total, per_class_acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_to_idx, train_counts = get_dataloaders(args)

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(
        in_channels=3, num_classes=2,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_cls.__name__} (--model {args.model}) | trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Class-weighted loss: if the training set is imbalanced, weight each
    # class inversely to its frequency so the model can't get a low loss
    # just by always predicting the majority class. Weights are normalized
    # so they average to 1 (keeps the loss scale comparable to unweighted
    # CrossEntropyLoss, which matters since ReduceLROnPlateau watches the
    # absolute val_loss value).
    # ------------------------------------------------------------------
    class_counts_t = torch.tensor(train_counts, dtype=torch.float)
    class_weights = class_counts_t.sum() / (len(class_counts_t) * class_counts_t)
    class_weights = class_weights.to(device)
    print(f"Using class weights (by class index): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW (Adamax -> AdamW): pairs better with GroupNorm/GELU/DropPath
    # than plain Adamax does. Weight decay is excluded on norm and bias
    # parameters (standard practice -- decaying those tends to hurt rather
    # than help) and on the fixed, non-trainable high-pass filter buffers
    # (which have requires_grad=False and are skipped automatically since
    # they're buffers, not parameters, but we double-guard here anyway).
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    # Mixed precision: SRNet2036 is deeper (attention + more blocks) than
    # the original and runs at 512x512, so AMP (bfloat16/float16 autocast
    # + gradient scaling) meaningfully cuts memory and wall-clock time,
    # which is what actually makes 512x512 practical to train on a single
    # GPU. Only enabled when running on CUDA.
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # LR schedule: cut the LR when val loss stalls. SRNet training in the
    # original paper is known to be slow / plateau-prone, so a plateau
    # scheduler is more robust here than a fixed decay.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    start_epoch = 1
    best_val_acc = -1.0
    epochs_no_improve = 0

    # ------------------------------------------------------------------
    # Resume from checkpoint if one already exists, otherwise start fresh.
    # We resume from the "last" checkpoint (not "best") since it holds the
    # most recent optimizer/scheduler state, which is what you need to
    # continue training exactly where you left off. best_val_acc is
    # recovered separately from the "best" checkpoint if present, so the
    # "new best" logic keeps working correctly after resuming.
    # ------------------------------------------------------------------
    last_ckpt_path = f"{args.checkpoint_prefix}_last.pt"
    best_ckpt_path = f"{args.checkpoint_prefix}_best.pt"

    if os.path.exists(last_ckpt_path):
        print(f"Found existing checkpoint at {last_ckpt_path}. Resuming training...")
        ckpt = torch.load(last_ckpt_path, map_location=device)

        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Checkpoint at {last_ckpt_path} does not match the current "
                f"'{args.model}' model architecture ({model_cls.__name__}). "
                f"It was likely saved with a different --model choice or an "
                f"older architecture. Delete the old checkpoint files, point "
                f"--checkpoint-prefix at a new path, or pass the matching "
                f"--model to resume this checkpoint. Original error: {e}"
            )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_amp and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = ckpt["epoch"] + 1

        if ckpt.get("class_to_idx") is not None and ckpt["class_to_idx"] != class_to_idx:
            print(
                "WARNING: class_to_idx in checkpoint does not match current "
                "dataset's class_to_idx. Proceeding anyway, but double-check "
                "your dataset folders."
            )

        # Recover best_val_acc so we don't accidentally overwrite a better
        # checkpoint with a worse one right after resuming.
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            best_val_acc = best_ckpt.get("val_acc", -1.0)
        else:
            best_val_acc = ckpt.get("val_acc", -1.0)

        print(
            f"Resumed from epoch {ckpt['epoch']}. Continuing at epoch "
            f"{start_epoch}. best_val_acc so far = {best_val_acc:.2f}%"
        )
    else:
        print("No existing checkpoint found. Starting training from scratch.")

    model.train()

    if start_epoch > args.epochs:
        print(
            f"start_epoch ({start_epoch}) is already beyond --epochs "
            f"({args.epochs}). Nothing to train. Increase --epochs to continue."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            # Gradient clipping: SRNet's first layers use non-trainable /
            # near-linear high-pass filters and can produce large gradients
            # early in training; clipping stabilizes this. Must unscale
            # first when AMP is on, or the clip threshold is meaningless
            # (gradients are scaled up by the loss-scale factor at this point).
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % args.log_every == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"[{batch_idx}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}"
                )

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        val_loss, val_acc, val_per_class_acc = evaluate(
            model, val_loader, criterion, device, use_amp=use_amp
        )
        scheduler.step(val_loss)

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        per_class_str = ", ".join(
            f"{idx_to_class[i]}={acc:.2f}%" for i, acc in enumerate(val_per_class_acc)
        )
        print(
            f"== Epoch {epoch}/{args.epochs} done in {elapsed:.1f}s "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"| val per-class: {per_class_str} "
            f"| lr={current_lr:.2e} =="
        )
        # If one class's accuracy is near 0% while the other is near 100%,
        # the model has collapsed to always predicting a single class --
        # flag it loudly instead of letting it hide inside the overall
        # accuracy number.
        if any(acc < 5.0 for acc in val_per_class_acc):
            print(
                "WARNING: at least one class has near-0% val accuracy. "
                "The model may have collapsed to predicting a single class."
            )

        # Always save a "last epoch" checkpoint (useful to resume training).
        last_ckpt_dict = {
            "epoch": epoch,
            "model_name": args.model,
            "image_size": args.image_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_to_idx": class_to_idx,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        if use_amp:
            last_ckpt_dict["scaler_state_dict"] = scaler.state_dict()
        torch.save(last_ckpt_dict, last_ckpt_path)

        # Save a SEPARATE "best" checkpoint, selected on val accuracy (not
        # train accuracy!). This is what you should actually use at test
        # time / deployment, since train_acc alone can be misleading
        # (overfitting) especially with 512x512 images and a small model.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            print("check-ponting")
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "image_size": args.image_size,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_to_idx": class_to_idx,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_ckpt_path,
            )
            print(f"  -> New best val_acc={val_acc:.2f}%. Saved {best_ckpt_path}")
        else:
            epochs_no_improve += 1

        # Early stopping to avoid wasting compute once the model stops
        # improving on validation data.
        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"No val improvement for {epochs_no_improve} epochs. "
                f"Early stopping at epoch {epoch}."
            )
            break

    print("Training complete. Loading best checkpoint for final test evaluation...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc, test_per_class_acc = evaluate(
        model, test_loader, criterion, device, use_amp=use_amp
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    per_class_str = ", ".join(
        f"{idx_to_class[i]}={acc:.2f}%" for i, acc in enumerate(test_per_class_acc)
    )
    print(
        f"FINAL TEST RESULT (best checkpoint, epoch {best_ckpt['epoch']}): "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% "
        f"| per-class: {per_class_str}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRNet on an RGB clean/stego dataset")
    parser.add_argument(
        "--model", type=str, default="cnn", choices=list(MODEL_REGISTRY.keys()),
        help="'cnn' = SRNet2036 (recommended default, more data-efficient). "
             "'vit' = SRNet2036ViTHybrid (CNN backbone + multi-layer Transformer "
             "encoder; more capacity, more compute, wants more data to shine).",
    )
    parser.add_argument(
        "--train-dir", type=str, default="/content/drive/MyDrive/SRnet/dataset/train/train",
    )
    parser.add_argument(
        "--val-dir", type=str, default="/content/drive/MyDrive/SRnet/dataset/val/val",
    )
    parser.add_argument(
        "--test-dir", type=str, default="/content/drive/MyDrive/SRnet/dataset/test/test",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Crop size (NOT resize). If < source image size, a random/center "
             "crop is taken -- pixel values are never interpolated, so "
             "embedding traces stay intact.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05,
                         help="AdamW weight decay (applied only to conv/linear "
                              "weights, not norm/bias params -- see optimizer setup).")
    parser.add_argument("--drop-path-rate", type=float, default=0.1,
                         help="Max stochastic-depth drop probability in SRNet2036's "
                              "residual stack. 0 disables it.")
    parser.add_argument("--amp", action="store_true", default=True,
                         help="Use automatic mixed precision (recommended at 512x512 "
                              "on CUDA; ignored on CPU). Pass --no-amp to disable.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=6,
                         help="Stop if val_acc doesn't improve for N epochs. 0 disables it.")
    parser.add_argument(
        "--checkpoint-prefix", type=str, default="/content/drive/MyDrive/SRnet/srnet2036",
        help="Prefix for saved checkpoint files -> <prefix>_best.pt / <prefix>_last.pt. "
             "Defaults to a NEW prefix (srnet2036, not srnet) since checkpoints from "
             "the original SRNet architecture are not compatible with SRNet2036.",
    )
    return parser.parse_args([])


if __name__ == "__main__":
    args = parse_args()
    train(args)
