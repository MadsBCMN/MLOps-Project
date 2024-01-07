import torch
import torch.nn as nn
import os
import sys
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)
sys.path.append(_PROJECT_ROOT)
from src.models.model import timm_model

def load_model(model_path):
    model = timm_model()
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def test_model():
    model = load_model("models/model.pt")
    assert model(torch.rand(1,1,86,86)).size() == (1,4) , "The output shape is not correct. Expected four categories"