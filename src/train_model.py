import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import timm_model
from data.dataloader import dataloader
from omegaconf import OmegaConf
import hydra
import wandb

# Set the working directory to the current directory
os.chdir(sys.path[0])

# setup logging
log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config.yaml")
def train(config: OmegaConf) -> None:
    """
    Train the model using the provided configuration and model.py.

    Args:
        cfg (OmegaConf): Configuration parameters.

    Returns:
        None
    """
    hparams = config
    run = wandb.init(project="MLOps_project", config=OmegaConf.to_container(hparams, resolve=True, throw_on_missing=True))
    torch.manual_seed(hparams["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm_model()
    model.to(device)

    # Setup wandb model logging
    wandb.watch(model, log_freq=100)

    # Load your datasets
    train_dataset, test_dataset = dataloader()
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    for epoch in range(hparams["n_epochs"]):
        # Training Phase
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            run.log({"loss": loss.item()})
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        log.info(f"Epoch [{epoch+1}/{hparams['n_epochs']}], Loss: {total_loss/len(train_loader)}")

        # Evaluation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

        accuracy = 100 * correct / total
        run.log({"accuracy": accuracy})
        log.info(f'Accuracy on the test set after epoch {epoch+1}: {accuracy}%')

    torch.save(model.state_dict(), '../models/model.pt')
    run.log_model(path='../models/model.pt', name="resnet18")
    log.info("Model saved")

    run.finish()

if __name__ == "__main__":
    train()
