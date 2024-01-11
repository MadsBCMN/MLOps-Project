import os
import sys
import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from models.model import timm_model
from data.dataloader import dataloader
from sklearn.model_selection import KFold
from omegaconf import OmegaConf
import hydra
import wandb
from typing import Any, Dict, List, Tuple
from pytorch_lightning.loggers import WandbLogger
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

# Set the working directory to project root
os.chdir(os.path.dirname(sys.path[0]))

# setup logging
log = logging.getLogger(__name__)

# Set the default precision
torch.set_float32_matmul_precision('medium')

# Setting dataloader worker seed
def seed_worker(worker_id: int) -> None:
    """Set the seed for dataloader workers."""
    worker_seed = torch.initial_seed() % 2 ** 32

class LightningModel(pl.LightningModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = timm_model()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_outputs: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total
        self.validation_outputs.append(accuracy)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_accuracy': accuracy}

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        avg_val_accuracy = torch.tensor(self.validation_outputs).mean()
        self.log('avg_val_accuracy', avg_val_accuracy, prog_bar=True)
        self.validation_outputs = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        train_dataset, _ = dataloader()
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, worker_init_fn=seed_worker)

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        print("val_dataloader")
        _, val_dataset = dataloader()
        return DataLoader(val_dataset, batch_size=self.hparams.batch_size, shuffle=False, worker_init_fn=seed_worker)

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        print("test_dataloader")
        _, test_dataset = dataloader()
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, shuffle=False, worker_init_fn=seed_worker)


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def train_evaluate(config: OmegaConf) -> None:
    """Main function for training and evaluation."""
    hparams = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(project="MLOps_project", config=hparams)
    wandb_logger = WandbLogger()

    # Setting the seed
    torch.manual_seed(hparams["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hparams["seed"])


    if config.k_fold:
        # set profiler
        if config.profile:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, on_trace_ready=tensorboard_trace_handler("./profiler/resnet18_lightning_5fold_cv"))
            prof.start()

        fold_accuracies = []  # Store accuracies for each fold
        train_dataset, _ = dataloader()
        kfold = KFold(n_splits=5, shuffle=True, random_state=config.seed)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
            log.info(f"Training fold {fold + 1}/5")

            # Re-initialize the model for each fold
            model = LightningModel(hparams)

            trainer = pl.Trainer(
                logger=wandb_logger,
                max_epochs=config.n_epochs,
                devices=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                log_every_n_steps=10
            )

            train_subsampler = Subset(train_dataset, train_ids)
            val_subsampler = Subset(train_dataset, val_ids)

            train_loader = DataLoader(train_subsampler, batch_size=config.batch_size, shuffle=True, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_subsampler, batch_size=config.batch_size, shuffle=False, worker_init_fn=seed_worker)

            trainer.fit(model, train_loader, val_loader)

            # Store the accuracy for this fold
            fold_accuracy = trainer.callback_metrics.get('avg_val_accuracy', 0)
            fold_accuracies.append(fold_accuracy)

        # Calculate and log the mean accuracy over all folds
        mean_accuracy = torch.tensor(fold_accuracies).mean().item()
        wandb_logger.log_metrics({"mean_val_accuracy": mean_accuracy})
        log.info(f"Mean Validation Accuracy across folds: {mean_accuracy}%")
        if config.profile:
            prof.stop()
    else:
        # Standard train-test split
        model = LightningModel(hparams)
        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=config.n_epochs,
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            log_every_n_steps=10
        )
        trainer.fit(model)

        # Perform test evaluation using the test_dataloader method
        trainer.test(model)

    # Save the final model
    torch.save(model.state_dict(), '../models/model.pt')
    run.log_model(path='../models/model.pt', name="resnet18")
    log.info("Model saved")


    # Finish wandb run
    run.finish()

if __name__ == "__main__":
    train_evaluate()
