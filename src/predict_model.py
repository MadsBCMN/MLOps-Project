import torch
import torch.nn as nn
import os
import sys
from torchvision import transforms
from PIL import Image
import timm
from models.model import timm_model
import numpy as np



def load_model(model_path):
    model = timm_model()
    # Load the saved model weights
    state_dict = torch.load("models/model.pt")
    state_dict = {k.partition('model.')[2]:state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((86, 86)),
    ])

    image = Image.open(image_path)
    image = transform(image)
    image_array = np.array(image)  # Convert to numpy array
    image_tensor = torch.tensor(image_array, dtype=torch.float32)  # Convert to tensor
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return image_tensor

def predict(model, image_folder):
    predictions = []
    for f in os.listdir(image_folder):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, f)
            image = process_image(image_path)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            predictions.append(predicted.item())
    return predictions

if __name__ == '__main__':

    model_path = "../models/model.pt"  # Path to the saved model
    image_folder = "../data/example_images"  # Path to the folder with images

    model = load_model(model_path)

    predictions = predict(model, image_folder)

    for i, pred in enumerate(predictions):
        print(f'Image {i}: Class {pred}')
