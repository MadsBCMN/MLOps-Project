import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import timm

model_name = 'resnet18'
num_classes = 4

# Load the pretrained model
model = timm.create_model(model_name, pretrained=True, in_chans=1)

# Modify the classifier (assuming the last layer is named 'fc')
model.fc = nn.Linear(model.fc.in_features, num_classes)
