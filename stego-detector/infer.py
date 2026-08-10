"""
Inference script for the steganalysis detector.

Loads a checkpoint saved by train_srnet.py, auto-detects which model
architecture and image size it was trained with (stored in the checkpoint
itself), and classifies one image or a whole folder of images as
clean/stego, with a stego probability.

Usage:
    # single image
    python3 infer.py --checkpoint srnet2036_best.pt --input path/to/image.png

    # a whole folder (recurses into subfolders)
    python3 infer.py --checkpoint srnet2036_best.pt --input path/to/folder --csv results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from SRnet import SRNet2036, SRNet2036ViTHybrid

MODEL_REGISTRY = {
    "cnn": SRNet2036,
    "vit": SRNet2036ViTHybrid,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_name = ckpt.get("model_name", "cnn")  # fall back to 'cnn' for older checkpoints
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Checkpoint records model_name='{model_name}', which isn't one of "
            f"{list(MODEL_REGISTRY.keys())}. Was this checkpoint saved by a "
            f"different version of train_srnet.py?"
        )
    image_size = ckpt.get("image_size", 512)
    class_to_idx = ckpt.get("class_to_idx", {"clean": 0, "stego": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(in_channels=3, num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"Loaded {model_cls.__name__} from {checkpoint_path} "
        f"(epoch {ckpt.get('epoch', '?')}, "
        f"val_acc={ckpt.get('val_acc', float('nan')):.2f}%, "
        f"trained at image_size={image_size})"
    )
    return model, image_size, idx_to_class


def build_transform(image_size: int):
    # Deterministic center crop, matching the eval_transform used during
    # training -- no resize (interpolation would blur the same residual
    # signal this whole project is built around preserving), no
    # augmentation/randomness.
    return transforms.Compose([
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def predict_one(model, transform, image_path: Path, device, idx_to_class):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"path": str(image_path), "error": str(e)}

    x = transform(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    stego_idx = None
    for idx, name in idx_to_class.items():
        if name.lower() == "stego":
            stego_idx = idx
            break
    if stego_idx is None:
        stego_idx = 1  # fall back to convention: index 1 = stego

    pred_idx = int(probs.argmax().item())
    return {
        "path": str(image_path),
        "prediction": idx_to_class.get(pred_idx, str(pred_idx)),
        "stego_probability": float(probs[stego_idx]),
        "confidence": float(probs[pred_idx]),
    }


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(description="Run the steganalysis detector on image(s).")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a _best.pt / _last.pt checkpoint from train_srnet.py")
    parser.add_argument("--input", type=str, required=True,
                         help="A single image file, or a folder to scan recursively.")
    parser.add_argument("--csv", type=str, default=None,
                         help="Optional path to write results as CSV.")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Stego-probability threshold for the printed verdict "
                              "(doesn't affect the reported probabilities themselves).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size, idx_to_class = load_model(args.checkpoint, device)
    transform = build_transform(image_size)

    input_path = Path(args.input)
    images = collect_images(input_path)
    if not images:
        print(f"No images found at {input_path}", file=sys.stderr)
        sys.exit(1)

    results = []
    for img_path in images:
        result = predict_one(model, transform, img_path, device, idx_to_class)
        results.append(result)
        if "error" in result:
            print(f"[ERROR] {result['path']}: {result['error']}")
        else:
            verdict = "STEGO" if result["stego_probability"] >= args.threshold else "clean"
            print(
                f"{result['path']}: {verdict}  "
                f"(stego_probability={result['stego_probability']:.4f}, "
                f"model_prediction={result['prediction']}, "
                f"confidence={result['confidence']:.4f})"
            )

    if args.csv:
        fieldnames = ["path", "prediction", "stego_probability", "confidence", "error"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nWrote {len(results)} results to {args.csv}")


if __name__ == "__main__":
    main()
