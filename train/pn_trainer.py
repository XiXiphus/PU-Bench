"""pn_trainer.py

A simple trainer for Positive-Negative (PN) classification, which is a standard
binary classification task. This trainer inherits from BaseTrainer to reuse
common functionalities like data loading, evaluation, and logging, but implements
its own training loop and loss function suitable for PN learning.
"""

import torch
from tqdm import tqdm

from .base_trainer import BaseTrainer


class PNTrainer(BaseTrainer):
    """Trainer for standard Positive-Negative (PN) classification."""

    def __init__(self, method: str, experiment: str, params: dict):
        """
        Initializes the PNTrainer.

        Args:
            method (str): The name of the method (should be 'pn').
            experiment (str): Name of the experiment.
            params (dict): Dictionary of parameters.
        """
        # The 'method' parameter is passed from the runner script.
        # This ensures consistency with other trainers, though for PNTrainer
        # it will always be 'pn'.
        super().__init__(method=method, experiment=experiment, params=params)

    def create_criterion(self):
        """
        Creates the loss function for PN classification.
        BCEWithLogitsLoss is suitable for binary classification and is numerically
        more stable than using a Sigmoid layer followed by BCELoss.
        """
        return torch.nn.BCEWithLogitsLoss()

    def train_one_epoch(self, epoch_idx: int):
        """
        Executes training for one epoch.

        This method iterates over the training data, performs forward and
        backward passes, and updates the model parameters. It uses the
        'true_labels' from the dataset for standard supervised training.

        Args:
            epoch_idx (int): The current epoch number.
        """
        self.model.train()
        total_loss = 0.0

        num_epochs = self.params.get("num_epochs", "N/A")
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_idx}/{num_epochs} [PN Training]",
            leave=False,
        )

        for batch in progress_bar:
            # PUDataset yields: features, pu_labels, true_labels, ...
            # For PN training, we only need features and true_labels.
            features, _, true_labels, _, _ = batch

            features = features.to(self.device)
            true_labels = true_labels.to(self.device).float()

            # Zero gradients, perform a forward pass, and calculate loss
            self.optimizer.zero_grad()
            outputs = self.model(features).squeeze()
            loss = self.criterion(outputs, true_labels)

            # Perform a backward pass and update weights
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        if avg_loss > 0:
            log_msg = f"Epoch {epoch_idx} - Average Training Loss: {avg_loss:.4f}"

            # Log to both consoles
            self.console.log(log_msg)
            if self.file_console:
                self.file_console.log(log_msg)
