from src.models.model import timm_model
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import Optional
import torch
import timm
import numpy as np

app = FastAPI()

def load_model(model_path):
    model = timm_model()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {k.partition('model.')[2]: state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

MODEL_PATH = "models/model.pt"
MODEL = load_model(MODEL_PATH)

class InputData(BaseModel):
    feature1: float
    feature2: float

class OutputData(BaseModel):
    prediction: str

@app.post("/predict", response_model=OutputData)
async def predict(data: InputData):
    try:

        input_features = [data.feature1, data.feature2]

        with torch.no_grad():
            input_tensor = torch.tensor(input_features, dtype=torch.float32).view(1, -1)
            output = MODEL(input_tensor)

        predicted_class = torch.argmax(output).item()

        class_mapping = {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3"}
        prediction_result = class_mapping[predicted_class]

        return {"prediction": prediction_result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )

@app.post("/predict_image", response_model=OutputData)
async def predict_image(file: UploadFile = File(...)):
    try:

        with open("uploaded_image.jpg", "wb") as f:
            f.write(file.file.read())


        prediction_result = "Class 0"

        return {"prediction": prediction_result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during image prediction: {str(e)}"
        )

# Root endpoint
@app.get("/")
async def read_root():
    return {"Hello": "World"}
