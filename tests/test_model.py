import torch
import torch.nn as nn
import os
import sys
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
sys.path.append(os.path.normcase(os.getcwd()))
from src.models.model import timm_model
from src.data.config import image_size

def load_model(model_path):
    model = timm_model()
    # Load the saved model weights
    state_dict = torch.load("models/model.pt")
    state_dict = {k.partition('model.')[2]:state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def test_model():
    model = load_model("models/model.pt")
    assert model(torch.rand(1,1,image_size[0],image_size[1])).size() == (1,4) , "The output shape is not correct. Expected four categories"
    
test_model()