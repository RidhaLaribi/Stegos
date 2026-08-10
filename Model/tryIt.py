import logging
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




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


def predict(image: Image.Image):
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        p=6
        confidence, pred = probs.max(1)
        if confidence.item()<=0.549 :
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
@app.post("/isnt/it/")
async def isit(image:UploadFile=File()):
    content = await image.read()
    img =Image.open(io.BytesIO(content)).convert("RGB")
    result = predict(img)

    return result
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