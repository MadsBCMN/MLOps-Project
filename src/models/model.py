import torch.nn as nn
import timm

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
