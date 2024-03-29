import io
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import timm
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from fastapi import HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from google.cloud import storage
sys.path.append(os.path.normcase(os.getcwd()))
from src.models.model import timm_model

def get_data():
    if not os.path.exists('models/model.pt'):
        os.system('dvc pull models/ -R --force')
        # log.info("Data pulled from dvc")
        storage_client = storage.Client()
        bucket = storage_client.bucket("mri-model")
        blob = bucket.blob("models/model.pt")
        blob.download_to_filename("models/model.pt")


get_data()

# Create a counter metric to track image classification requests
image_classification_requests = Counter('image_classification_requests_total', 'Total number of image classification requests')

app = FastAPI()
CLASS_LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
def load_model(model_path):
    model = timm_model()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {k.partition('model.')[2]: state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

# load model
model_path = os.path.abspath('models/model.pt')
try:
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

def process_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((86, 86)),
    ])

    image = transform(image)
    image_array = np.array(image)
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    return image_tensor

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Increment the image_classification_requests counter
        image_classification_requests.inc()

        if not file.content_type or file.content_type.split("/")[0] != "image":
            raise HTTPException(status_code=400, detail="Invalid file type. Only images are supported.")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = process_image(image)
        output = model(processed_image)
        _, predicted = torch.max(output.data, 1)

        # Map the predicted class index to the corresponding label
        predicted_label = CLASS_LABELS[predicted.item()]

        return JSONResponse(content={"class": predicted_label, "class_label": predicted.item()}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Instrument the FastAPI application
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    get_data()
    uvicorn.run(app, host="0.0.0.0", port=8000)
