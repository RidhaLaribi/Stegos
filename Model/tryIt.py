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
from fastapi import FastAPI,UploadFile,File
import io


import torch
from PIL import Image
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


def predict(image: Image.Image,me=0):
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        p=6
        confidence, pred = probs.max(1)

        if me==1 and confidence.item()<=0.7 :
            if pred.item() == 0 : p=1
            else : p=0
        else : p=pred.item()
    result ={
        "prediction": classes[p],
        "p":p,
        "confidence": float(confidence.item())
    }
    logger.info("Received POST: %s", result)

    return result

app = FastAPI()

@app.get("/")
def status():
    return {"status":200}


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



@app.post ("/is/it/")
async def its(image: UploadFile = File(...)):
    content= await image.read()
    img = Image.open(io.BytesIO(content))
    img_rgb = img.convert("RGB")

    result = predict(img_rgb,1)

    return {
        "source": "model",
        "result": result
    }

@app.post("/isnt/it/")
async def isit(image: UploadFile = File(...)):
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

    if is_jpeg:
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

                    "detected": False,
                    "status": "error",
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
                t=50
                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=t
                )

            except asyncio.TimeoutError:
                process.kill()
                # await process.wait()
                img_rgb = img.convert("RGB")

                result = predict(img_rgb)

                return {
                    "source": "model",
                    "message": f" search exceeded {t} seconds",
                    "result": result
                }

                return {
                    "detected": False,
                    "status": "timeout",
                    "message": f" search exceeded {t} seconds"
                }

            # StegSeek return code 0 = seed found
            if process.returncode == 0:
                return {
                    "detected": True,
                }

            return {
                "source": "stegseek",
                "detected": False,
                "status": "not_found"
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    else:
        img_rgb = img.convert("RGB")

        result = predict(img_rgb)

        return {
            "source": "model",
            "result": result
        }


#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load("srnet_epoch1.pt", map_location=device)
#
# model = SRNet()
# model.load_state_dict(checkpoint["model_state_dict"])
# model.to(device)
# model.eval()
# print(device)
#
# import cv2
# import torch
# import numpy as np
#
# image = cv2.imread("img_1.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Convert to float32 and normalize to [0,1]
# image = image.astype(np.float32) / 255.0
#
# # HWC -> CHW
# image = np.transpose(image, (2, 0, 1))
#
# # Convert to tensor and add batch dimension
# image = torch.from_numpy(image).unsqueeze(0).to(device)
#
# with torch.no_grad():
#     output = model(image)
#
# prediction = output.argmax(dim=1).item()
#
# print("Prediction:", output)

# model = SRNet()
# model.load_state_dict(torch.load("srnet_epoh1.pt", map_location=device))
# model.to(device)
# model.eval()
#
# # Image preprocessing (must match training)
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
# ])

# # Load image
# image = Image.open("test_image.png").convert("RGB")
#
# # Transform image
# image = transform(image)
#
# # Add batch dimension
# image = image.unsqueeze(0)   # Shape: (1, 3, 256, 256)
#
# # Move to device
# image = image.to(device)
#
# # Inference
# with torch.no_grad():
#     output = model(image)
#
# print(output)