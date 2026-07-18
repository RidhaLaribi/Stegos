"""
Builds a labeled cover/stego dataset for training a steganalysis model that must:
  1) detect whether an image contains embedded data, and
  2) identify which tool did the embedding.

Fixes vs. the original script:
  - Supports multiple embedding tools via a pluggable TOOLS dict (steghide, outguess,
    a spatial-domain LSB baseline) instead of being hardcoded to steghide only.
  - Writes a metadata.csv with path/label/tool/cover_source/secret/password. This was
    completely missing before -- without it you have no ground truth to train on.
  - Filters which cover images go to which tool based on the tool's actually-supported
    container formats (steghide/outguess are JPEG-domain; PNG will silently fail or
    behave differently). The original forced every image into steghide and renamed
    every output to ".jpg" regardless of real format, which is wrong and also hides
    failures.
  - Uses cryptographically random passwords instead of a predictable sequence
    (a,b,c...aa,ab...). Sequential passwords aren't a training-time issue for
    steghide itself, but there's no reason to keep a spurious global correlate
    between "row order" and "password" in your dataset.
  - Distinguishes "tool not installed" from "embedding failed" (e.g. secret too
    large for image capacity) so failures are actually diagnosable.
  - Preserves the original file extension in outputs.
  - `cover_source` column lets you group by the underlying cover image later so you
    can do cover-independent train/val/test splits (per your steganalysis notes).
"""

import os
import shutil
import subprocess
import secrets
import string
import csv
import base64

COVER_DIR = "cover"
DATASET_DIR = "dataset"
SECRETS_DIR = "secrets"

COVER_OUT = os.path.join(DATASET_DIR, "cover")
STEGO_OUT = os.path.join(DATASET_DIR, "stego")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

# Which cover-image extensions each tool can actually work with.
# (Feeding the wrong container format to a JPEG-domain tool is a common
# silent-failure source -- steghide/outguess operate on JPEG coefficients.)
TOOL_SUPPORTED_EXT = {
    "steghide": (".jpg", ".jpeg", ".bmp"),
    "outguess": (".jpg", ".jpeg"),
    "lsb": (".png", ".bmp"),  # LSB needs a lossless container or the payload is destroyed
}


def random_password(length=12):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def get_secret_files():
    return [
        os.path.join(SECRETS_DIR, f)
        for f in os.listdir(SECRETS_DIR)
        if os.path.isfile(os.path.join(SECRETS_DIR, f))
    ]


def get_cover_images():
    return [
        f for f in os.listdir(COVER_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]


def _run(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  command failed ({e.returncode}): {' '.join(cmd)}")
        return False
    except FileNotFoundError:
        print(f"  tool binary not found on PATH: {cmd[0]}")
        return False


# ---------------- tool-specific embed functions ----------------
# Add a new tool by writing a function with this signature and registering
# it in TOOLS + TOOL_SUPPORTED_EXT below.

def embed_steghide(cover_path, secret_path, out_path, password):
    cmd = ["steghide", "embed", "-cf", cover_path, "-ef", secret_path,
           "-sf", out_path, "-p", password, "-f"]
    return _run(cmd)


def embed_outguess(cover_path, secret_path, out_path, password):
    # outguess's "-k" key isn't a real encryption password like steghide's,
    # but keep the argument for interface consistency / future tools.
    cmd = ["outguess", "-k", password, "-d", secret_path, cover_path, out_path]
    return _run(cmd)


def embed_lsb(cover_path, secret_path, out_path, password):
    """Spatial-domain LSB baseline (requires `pip install stegano Pillow`).
    Deliberately a different embedding domain than steghide/outguess so the
    dataset covers more than one class of statistical artifact."""
    try:
        from stegano import lsb
        with open(secret_path, "rb") as f:
            data = f.read()
        payload = base64.b85encode(data).decode("ascii")
        result_image = lsb.hide(cover_path, payload)
        result_image.save(out_path)
        return True
    except Exception as e:
        print(f"  lsb embed failed: {e}")
        return False


TOOLS = {
    "steghide": embed_steghide,
    "outguess": embed_outguess,
    "lsb": embed_lsb,
}


def build_dataset():
    os.makedirs(COVER_OUT, exist_ok=True)
    for tool in TOOLS:
        os.makedirs(os.path.join(STEGO_OUT, tool), exist_ok=True)

    images = get_cover_images()
    secret_files = get_secret_files()
    print(f"cover images: {len(images)} | secrets: {len(secret_files)} | tools: {list(TOOLS)}")

    rows = []

    # every cover image is copied once and labeled "cover" (used at most once
    # per tool below as the matching negative example)
    for img in images:
        src = os.path.join(COVER_DIR, img)
        dst = os.path.join(COVER_OUT, img)
        shutil.copy2(src, dst)
        rows.append({
            "path": dst, "label": "cover", "tool": "none",
            "cover_source": img, "secret": "", "password": "",
        })

    for tool_name, embed_fn in TOOLS.items():
        supported_ext = TOOL_SUPPORTED_EXT.get(tool_name, tuple())
        tool_images = [i for i in images if i.lower().endswith(supported_ext)]
        skipped = len(images) - len(tool_images)
        if skipped:
            print(f"[{tool_name}] skipping {skipped} images with unsupported extension")

        for sec in secret_files:
            secret_name = os.path.splitext(os.path.basename(sec))[0]
            for img in tool_images:
                img_name, img_ext = os.path.splitext(img)
                stego_name = f"{tool_name}_{secret_name}_{img_name}{img_ext}"
                stego_path = os.path.join(STEGO_OUT, tool_name, stego_name)
                src_img = os.path.join(COVER_DIR, img)
                password = random_password()

                ok = embed_fn(src_img, sec, stego_path, password)
                print(f"[{tool_name}] {'OK' if ok else 'FAIL'} {img} + {secret_name}")

                if ok:
                    rows.append({
                        "path": stego_path, "label": "stego", "tool": tool_name,
                        "cover_source": img, "secret": secret_name, "password": password,
                    })

    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "tool", "cover_source", "secret", "password"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {METADATA_PATH}")


if __name__ == "__main__":
    build_dataset()






# import os
# import shutil
# import subprocess
# import string
#
# cover_in = "cover"
# dataset = "dataset"
#
# cover_out = os.path.join(dataset, "cover")
# stego_out = os.path.join(dataset, "stego")
#
# secrets = "secrets"
#
#
# def generate_p(n):
#     chars = string.ascii_lowercase
#
#     if n <= len(chars):
#         return chars[:n]
#
#     repeats = (n // len(chars)) + 1
#     return (chars * repeats)[:n]
#
#
# def getSec():
#     return [
#         os.path.join(secrets, f)
#         for f in os.listdir(secrets)
#         if os.path.isfile(os.path.join(secrets, f))
#     ]
#
#
# def embed(image, secret, out_path, p):
#     try:
#         cmd = [
#             "steghide",
#             "embed",
#             "-cf", image,
#             "-ef", secret,
#             "-sf", out_path,
#             "-p", p
#         ]
#
#         subprocess.run(
#             cmd,
#             check=True,
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL
#         )
#
#         return True
#
#     except subprocess.CalledProcessError as e:
#         print("I cant run it", e)
#         return False
#
#
# def build_ds():
#     os.makedirs(cover_out, exist_ok=True)
#     os.makedirs(stego_out, exist_ok=True)
#
#     images = [
#         f for f in os.listdir(cover_in)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]
#     secret_files = getSec()
#     print("images:", len(images))
#     print("secrets:", len(secret_files))
#     count = 1
#     for sec in secret_files:
#         secretName = os.path.splitext(os.path.basename(sec))[0]
#         for img in images:
#             imgName = os.path.splitext(img)[0]
#             stego_name = f"stego_{secretName}_{imgName}.jpg"
#             stego_path = os.path.join(stego_out, stego_name)
#             src_img = os.path.join(cover_in, img)
#             p = generate_p(count)
#             shutil.copy2(src_img, os.path.join(cover_out, img))
#             success = embed(src_img, sec, stego_path, p)
#             if success:
#                 print(f"OK... who cares {img} + {secretName} | password={p}")
#             else:
#                 print(f"Fail >> {img} + {secretName}")
#             count += 1
#
#
# if __name__ == "__main__":
#     build_ds()