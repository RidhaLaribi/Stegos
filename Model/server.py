import logging
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import io
import os
import tempfile
import asyncio
from fastapi import FastAPI, UploadFile, File
from PIL import Image

from SRnet import SRNet

import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("srnet_last.pt", map_location=device)

model = SRNet()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

classes = ["clean", "stego"]


def predict(image: Image.Image, me=0):
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        confidence, pred = probs.max(1)

        if me == 1 and confidence.item() <= 0.7:
            # /is/it/ only: when the model is unsure, flip the naive
            # argmax label rather than returning it as-is.
            p = 1 if pred.item() == 0 else 0
        else:
            p = pred.item()

    result = {
        "prediction": classes[p],
        "p": p,
        "confidence": float(confidence.item())
    }
    logger.info("Received POST: %s", result)

    return result


app = FastAPI()


@app.get("/")
def status():
    return {"status": 200}


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/is/it/")
async def its(image: UploadFile = File(...)):
    """Model-only endpoint with the confidence-flip heuristic.
    Kept for testing; NOT called by the current mitmproxy client."""
    content = await image.read()
    img = Image.open(io.BytesIO(content))
    img_rgb = img.convert("RGB")

    result = predict(img_rgb, 1)

    return {
        "source": "model",
        "result": result
    }


# --- Production endpoint used by the proxy -----------------------------
STEGSEEK_CONFIRM_THRESHOLD = 0.7


@app.post("/isnt/it/")
async def isit(image: UploadFile = File(...)):
    """
    Decision logic:
      1. The deep learning model always runs first.
      2. If the file is JPEG/JPG AND the model predicts "stego" AND its
         confidence in that prediction is below the threshold, StegSeek
         is used to confirm (or fail to confirm) the verdict.
      3. Any other case (non-JPEG, or a confident JPEG call either way)
         returns the model's verdict directly, no StegSeek involved.
    """
    logger.info("connected")
    content = await image.read()

    img = Image.open(io.BytesIO(content))

    is_jpeg = img.format in ["JPEG", "MPO"] or (
        image.content_type
        and image.content_type.lower() in ["image/jpeg", "image/jpg"]
    )

    file_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(file_path, "wb") as f:
        f.write(content)

    img_rgb = img.convert("RGB")
    result = predict(img_rgb)

    needs_confirmation = (
        is_jpeg
        and result["prediction"] == "stego"
        and result["confidence"] < STEGSEEK_CONFIRM_THRESHOLD
    )

    if not needs_confirmation:
        # Non-JPEG, or a JPEG the model is confident about either way:
        # return the model's verdict as-is.
        return {
            "source": "model",
            "result": result
        }

    # Low-confidence JPEG "stego" call: confirm with StegSeek
    tmp_path = None

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # Convert Windows path -> WSL path
        path_process = await asyncio.create_subprocess_exec(
            "wsl",
            "wslpath",
            "-a",
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        wsl_path_bytes, path_stderr = await path_process.communicate()

        if path_process.returncode != 0:
            return {
                "source": "model",
                "result": result,
                "confirmation": "error",
                "message": "Could not convert Windows path to WSL path",
                "stderr": path_stderr.decode(errors="ignore").strip()
            }

        wsl_path = wsl_path_bytes.decode().strip()

        # Run StegSeek
        process = await asyncio.create_subprocess_exec(
            "wsl",
            "stegseek",
            "--seed",
            wsl_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            t = 50
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=t
            )
        except asyncio.TimeoutError:
            process.kill()
            return {
                "source": "model",
                "result": result,
                "confirmation": "timeout",
                "message": f"StegSeek search exceeded {t} seconds; returning model verdict only"
            }

        # StegSeek return code 0 = seed found = confirmed steghide payload
        confirmed = process.returncode == 0

        return {
            "source": "model+stegseek",
            "result": result,
            "confirmed_stego": confirmed
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
