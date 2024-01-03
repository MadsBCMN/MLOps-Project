import click
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from model import myawesomemodel
from dataloader import mnist
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--num_epochs", default=20, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    # Load data
    train_dataset, test_dataset = mnist()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = myawesomemodel().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_loss:.4f}')

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f} %')

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
    plot_folder = r"C:\Users\mads.brodthagen\MLOps-Project\reports\figures"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    plt.savefig(os.path.join(plot_folder, "training_progress.png"))

cli.add_command(train)

if __name__ == "__main__":
    cli()
