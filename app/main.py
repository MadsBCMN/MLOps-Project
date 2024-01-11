import io
import os
import sys
sys.path.append(os.path.normcase(os.getcwd()))
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import timm
from fastapi import HTTPException

### MODEL ####
# Global variables
MODEL_NAME = 'resnet18'
NUM_CLASSES = 4

def timm_model() -> nn.Module:
    """
    Creates a custom model based on timm's resnet18 with modified output layer.

    Returns:
        nn.Module: Custom model.
    """
    # Load pre-trained resnet18 model
    model = timm.create_model(MODEL_NAME, pretrained=True, in_chans=1)

    # Modify the output layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model
#####################

app = FastAPI()
CLASS_LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
def load_model(model_path):
    model = timm_model()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {k.partition('model.')[2]: state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

model_path = "../models/model.pt"
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
        if not file.content_type or file.content_type.split("/")[0] != "image":
            raise HTTPException(status_code=400, detail="Invalid file type. Only images are supported.")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = process_image(image)
        output = model(processed_image)
        _, predicted = torch.max(output.data, 1)

        # Map the predicted class index to the corresponding label
        predicted_label = CLASS_LABELS[predicted.item()]

        return JSONResponse(content={"class": predicted_label}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
