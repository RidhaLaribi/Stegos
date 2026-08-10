# Stego Image Detector

A deep-learning steganalysis tool: given an image, predicts whether it
contains hidden (steganographically embedded) data ("stego") or not
("clean"). Binary classifier, RGB images, built around 512x512 input.

This is a **defensive/forensic** tool — it detects hidden data, it does
not embed any.

## What's in here

| File | What it does |
|---|---|
| `SRnet.py` | Three model architectures (see below) |
| `train_srnet.py` | Training script: dataset loading, class-imbalance handling, mixed precision, checkpointing, resume-from-checkpoint, early stopping |
| `infer.py` | Run a trained checkpoint on a single image or a whole folder |
| `requirements.txt` | Python dependencies |

## Models

Three model classes live in `SRnet.py`, selectable in training via `--model`:

- **`SRNet` (reference only, not used by default)** — the original 2019
  SRNet architecture (Boroumand, Chen, Fridrich, IEEE TIFS), kept for
  comparison. Designed for 256x256 input; not used by `train_srnet.py`.

- **`SRNet2036`** (`--model cnn`, **default, recommended**) — a modernized
  redesign built specifically for 512x512:
  - Fixed, non-trainable high-pass filters (including the classic KV
    kernel) fused with a learned conv branch in the stem, so the network
    starts from a real noise-residual signal instead of learning one from
    scratch.
  - Downsamples *early*, right after the stem, before the expensive
    residual stack — the single biggest lever for making 512x512
    actually trainable in reasonable time.
  - Depthwise-separable convolutions, GroupNorm (robust at the small
    batch sizes 512x512 forces you into), ECA channel attention,
    a bottleneck self-attention stage, stochastic depth, GELU.
  - ~5.4M parameters.

- **`SRNet2036ViTHybrid`** (`--model vit`) — the same CNN backbone, but
  instead of one attention layer at the bottleneck, feeds the resulting
  residual feature map (256 tokens + a `[CLS]` token) into a real
  multi-layer Transformer encoder (learnable positional embeddings,
  several pre-LN transformer blocks). More capacity, more compute, and
  wants more training data to pay off than the CNN option — a *raw-pixel*
  ViT was deliberately avoided (see the docstring in `SRnet.py`): naive
  patch embedding averages away the exact small-amplitude signal
  steganalysis depends on.

If you're not sure which to use: start with `cnn`. It's smaller, faster,
more data-efficient, and closer to what's proven in the steganalysis
literature. Try `vit` if you have a larger dataset and want to see if the
extra global-attention capacity helps.

## Expected dataset layout

```
dataset/
  train/
    clean/   *.png / *.jpg / ...
    stego/
  val/
    clean/
    stego/
  test/
    clean/
    stego/
```
Standard `torchvision.datasets.ImageFolder` layout — one subfolder per
class, class names must match across train/val/test.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python3 train_srnet.py \
    --model cnn \
    --train-dir dataset/train \
    --val-dir dataset/val \
    --test-dir dataset/test \
    --image-size 512 \
    --batch-size 20 \
    --epochs 30 \
    --checkpoint-prefix checkpoints/srnet2036
```

Key behavior:
- **Class imbalance**: prints per-split class counts up front and
  automatically weights the loss inversely to class frequency if any
  class exceeds 60% of the training set. Per-class validation accuracy is
  printed every epoch (not just overall accuracy) — this is how you catch
  a model that's collapsed to predicting a single class, since overall
  accuracy alone can hide that.
- **Resumable**: if `<prefix>_last.pt` already exists, training resumes
  from it automatically (model, optimizer, scheduler, AMP scaler state,
  epoch count) — just re-run the same command. Delete the checkpoint
  files (or point `--checkpoint-prefix` elsewhere) to start over.
- **Mixed precision** (`--amp`, on by default on CUDA) — cuts memory and
  time meaningfully at 512x512. Disable with `--no-amp` if you hit
  numerical issues.
- If you hit GPU out-of-memory, lower `--batch-size` first.
- `--early-stop-patience N` (default 6) stops training if val accuracy
  hasn't improved in N epochs; `0` disables it.

Two checkpoints are written: `<prefix>_last.pt` (every epoch, for
resuming) and `<prefix>_best.pt` (only on new best val accuracy — this is
the one to actually deploy). Both record which `--model` and
`--image-size` they were trained with, so `infer.py` can load them
without you having to remember.

## Inference

```bash
# single image
python3 infer.py --checkpoint checkpoints/srnet2036_best.pt --input some_image.png

# a whole folder (recurses into subfolders)
python3 infer.py --checkpoint checkpoints/srnet2036_best.pt --input some_folder/ --csv results.csv
```

Output per image: predicted class, stego probability, and confidence.
`--threshold` (default 0.5) controls the printed clean/STEGO verdict
without changing the reported probabilities, so you can re-threshold
after the fact if you want a different precision/recall trade-off.

## Notes / things worth knowing before you run this at scale

- **Preprocessing never resizes or interpolates.** Training uses random
  crop (`pad_if_needed`, reflect padding) + flips + 90-degree rotations;
  eval/inference use a deterministic center crop. `Resize()` is
  deliberately avoided everywhere in this project, including here —
  interpolating pixels would blur out the same small-amplitude embedding
  signal the whole architecture is built to preserve.
- **A model stuck at a fixed accuracy every epoch, with per-class
  accuracy near 0% for one class**, means it's collapsed to predicting a
  single class — almost always a class-imbalance symptom (the automatic
  class weighting above is meant to prevent this, but it's still worth
  watching the per-class numbers, especially early in a new dataset).
- Changing `--model` (cnn vs vit) or upgrading the architecture in
  `SRnet.py` invalidates old checkpoints (different `state_dict` shapes)
  — use a new `--checkpoint-prefix` rather than trying to resume across
  an architecture change.
