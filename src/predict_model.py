import os
import sys
from typing import List
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from models.model import timm_model
import logging
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.normcase(os.getcwd()))
CLASS_LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
def load_model(model_path: str) -> nn.Module:
    """
    Load a pre-trained model from the specified path.

    Parameters:
    - model_path (str): Path to the saved model.

    Returns:
    - model (nn.Module): Loaded model.
    """
    model = timm_model()

    # Load the saved model weights
    state_dict = torch.load(model_path)
    state_dict = {k.partition('model.')[2]: state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def process_image(image_path: str) -> torch.Tensor:
    """
    Process an image for prediction.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - image_tensor (torch.Tensor): Processed image tensor.
    """
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


def predict(model: nn.Module, image_folder: str) -> List[int]:
    """
    Make predictions on images in the specified folder using the given model.

    Parameters:
    - model (nn.Module): Trained model.
    - image_folder (str): Path to the folder with images.

    Returns:
    - predictions (List[int]): List of predicted classes.
    """
    predictions = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = process_image(image_path)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            predicted_label = CLASS_LABELS[predicted.item()]
            predictions.append((filename, predicted_label))
    return predictions


if __name__ == '__main__':
    os.system('dvc pull models --force')
    # log.info("Data pulled from dvc")
    storage_client = storage.Client()
    bucket = storage_client.bucket("dtumlops_data_bucket")
    blob = bucket.blob("models/model.pt")
    blob.download_to_filename("models/model.pt")
    # log.info("Model saved to gcs")

    model_path = "models/model.pt"  # Path to the saved model
    image_folder = "data/example_images"  # Path to the folder with images

    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        sys.exit(1)

    try:
        predictions = predict(model, image_folder)
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        sys.exit(1)

    for filename, pred in predictions:
        logger.info(f'Image {filename}: Class {pred}')
