import argparse
import time
import sys
sys.path.append('/content/drive/MyDrive/SRnet')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Model.SRnet import SRNet


def get_dataloaders(args):
    
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

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return running_loss / total, 100.0 * correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(args)

    model = SRNet(in_channels=3, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)

    # LR schedule: cut the LR when val loss stalls. SRNet training in the
    # original paper is known to be slow / plateau-prone, so a plateau
    # scheduler is more robust here than a fixed decay.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = -1.0
    epochs_no_improve = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            # Gradient clipping: SRNet's first layers use non-trainable /
            # near-linear high-pass filters and can produce large gradients
            # early in training; clipping stabilizes this.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

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

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"== Epoch {epoch}/{args.epochs} done in {elapsed:.1f}s "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"| lr={current_lr:.2e} =="
        )

        # Always save a "last epoch" checkpoint (useful to resume training).
        last_ckpt_path = f"{args.checkpoint_prefix}_last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "class_to_idx": class_to_idx,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            last_ckpt_path,
        )

        # Save a SEPARATE "best" checkpoint, selected on val accuracy (not
        # train accuracy!). This is what you should actually use at test
        # time / deployment, since train_acc alone can be misleading
        # (overfitting) especially with 512x512 images and a small model.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_ckpt_path = f"{args.checkpoint_prefix}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
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
    best_ckpt = torch.load(f"{args.checkpoint_prefix}_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(
        f"FINAL TEST RESULT (best checkpoint, epoch {best_ckpt['epoch']}): "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRNet on an RGB clean/stego dataset")
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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=6,
                         help="Stop if val_acc doesn't improve for N epochs. 0 disables it.")
    parser.add_argument(
        "--checkpoint-prefix", type=str, default="srnet",
        help="Prefix for saved checkpoint files -> srnet_best.pt / srnet_last.pt",
    )
    return parser.parse_args([])


if __name__ == "__main__":
    args = parse_args()
    train(args)