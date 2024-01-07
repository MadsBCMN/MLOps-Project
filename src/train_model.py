import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from models.model import timm_model
from data.dataloader import dataloader
from sklearn.model_selection import KFold
from omegaconf import OmegaConf
import hydra
import wandb

# Set the working directory to the current directory
os.chdir(sys.path[0])

# setup logging
log = logging.getLogger(__name__)



@hydra.main(config_path="config", config_name="config.yaml")
def train_evaluate(config: OmegaConf) -> None:
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



    # Evaluation
    def evaluate(model: nn.Module, test_loader: DataLoader, device: str) -> float:
        """
        Evaluate the model.

        Args:
            model (nn.Module): Model to evaluate.
            test_loader (DataLoader): Test set data loader.
            device (str): Device to use for evaluation.

        Returns:
            float: Accuracy of the model on the test set.
        """
        log.info(f'Evaluating on the {"validation" if hparams["k_fold"] else "test"} set')
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
        return accuracy

    def train_model(train_loader: DataLoader, test_loader: DataLoader, device: str, params: dict = hparams) -> nn.Module:
        """
        Train the model.

        Args:
            train_loader (DataLoader): Training set data loader.
            test_loader (DataLoader): Test set data loader.
            device (str): Device to use for training.
            params (dict): Hyperparameters.

        Returns:
            nn.Module: Trained model.
        """

        log.info("Start training model...")

        if params["k_fold"]:
            fold_accuracies = []  # List to store accuracy of each fold
            train_dataset = train_loader.dataset  # Get the dataset used to create train_loader

            # Set up K-Fold cross-validation
            k_folds = 5
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=params['seed'])

            for fold, (train_ids, val_ids) in enumerate(kfold.split(train_loader.dataset)):
                log.info(f"Training fold {fold + 1}/{k_folds}")
                # Sample elements for this fold
                train_subsampler = Subset(train_dataset, train_ids)
                val_subsampler = Subset(train_dataset, val_ids)

                # Create data loaders for this fold
                fold_train_loader = DataLoader(train_subsampler, batch_size=params["batch_size"], shuffle=True)
                fold_val_loader = DataLoader(val_subsampler, batch_size=params["batch_size"], shuffle=False)

                # Initialize a new model each fold
                model = timm_model()
                model.to(device)

                # Setup wandb model logging
                wandb.watch(model, log_freq=100)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=params["lr"])

                for epoch in range(params["n_epochs"]):
                    # Training Phase
                    model.train()
                    total_loss = 0
                    for images, labels in fold_train_loader:
                        # Forward pass
                        outputs = model(images.to(device))
                        loss = criterion(outputs, labels.to(device))

                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                    # Log training loss with fold and epoch information
                    run.log({"fold": fold + 1, "epoch": epoch + 1, "train_loss": total_loss / len(fold_train_loader)})
                    log.info(f"Fold {fold + 1} Epoch [{epoch+1}/{params['n_epochs']}], Loss: {total_loss/len(fold_train_loader)}")

                    # Evaluation Phase
                    accuracy = evaluate(model, fold_val_loader, device)


                    # Log validation accuracy with fold and epoch information
                    run.log({"fold": fold + 1, "epoch": epoch + 1, "val_accuracy": accuracy})
                    log.info(f'Fold {fold + 1}, Epoch {epoch + 1}: Validation Accuracy: {accuracy}%')

                fold_accuracies.append(accuracy)  # Store fold accuracy

            mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
            run.log({"mean_val_accuracy": mean_accuracy})
            log.info(f'Mean Validation Accuracy across {k_folds} folds: {mean_accuracy}%')


        else:
            # Standard train-test split
            model = timm_model()
            model.to(device)

            # Setup wandb model logging
            wandb.watch(model, log_freq=100)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=params["lr"])

            for epoch in range(params["n_epochs"]):
                # Training Phase
                model.train()
                total_loss = 0
                for images, labels in train_loader:
                    # Forward pass
                    outputs = model(images.to(device))
                    loss = criterion(outputs, labels.to(device))

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Log training loss with fold and epoch information
                run.log({"epoch": epoch + 1, "train_loss": total_loss / len(train_loader)})
                log.info(f"Epoch [{epoch+1}/{params['n_epochs']}], Loss: {total_loss/len(train_loader)}")
                # Evaluation Phase
                accuracy = evaluate(model, test_loader, device)

                # Log validation accuracy with fold and epoch information
                run.log({"epoch": epoch + 1, "test_accuracy": accuracy})
                log.info(f'Epoch {epoch + 1}: Test Accuracy: {accuracy}%')

        # Save the final model
        torch.save(model.state_dict(), '../models/model.pt')
        run.log_model(path='../models/model.pt', name="resnet18")
        log.info("Model saved")



    # Load datasets
    train_dataset, test_dataset = dataloader()
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    # Train model
    train_model(train_loader, test_loader, device, params=hparams)

    # Finish wandb run
    run.finish()



if __name__ == "__main__":
    train_evaluate()
