import click
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from model import myawesomemodel
from src.data.dataloader import mnist
import os
from omegaconf import OmegaConf
import hydra
import logging

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="config", config_name="config.yaml")
def train(cfg):
    log = logging.getLogger(__name__)
    # Load data
    train_dataset, test_dataset = mnist()
    train_loader = DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)
    log.info("data load complete")
    # Model
    model = myawesomemodel().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    train_losses = []
    test_accuracies = []

    # Training loop
    for epoch in range(cfg.hyperparameters.num_epochs):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        #print(f'Epoch [{epoch+1}/{cfg.hyperparameters.num_epochs}], Average Training Loss: {avg_loss:.4f}')
        log.info(f'Epoch [{epoch+1}/{cfg.hyperparameters.num_epochs}], Average Training Loss: {avg_loss:.4f}')
        
        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        #print(f'Epoch [{epoch+1}/{cfg.hyperparameters.num_epochs}], Test Accuracy: {accuracy:.2f} %')
        log.info(f'Epoch [{epoch+1}/{cfg.hyperparameters.num_epochs}], Test Accuracy: {accuracy:.2f} %')

    torch.save(model.state_dict(), "model.pt")

    # Plotting
    plt.figure(figsize=(10,5))
    plt.title("Training and Test Accuracy over Epochs")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.grid(True)

    # Create a distinct folder for the plot
    plot_folder = r"/Users/helenehjort/Library/Mobile Documents/com~apple~CloudDocs/Human-centered AI/9. semester/02476 Machine Learning Operations/Project/MLOps-Project/src/visualizations"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    plt.savefig(os.path.join(plot_folder, "training_progress.png"))

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='train_model.log', level=logging.INFO, format=log_fmt, filemode='w')
    train()
