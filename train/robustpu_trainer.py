"""robustpu_trainer.py

RobustPUTrainer inherits from BaseTrainer and implements the Robust-PU method.
This method includes pre-training and self-paced learning stages.
"""

from __future__ import annotations

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from typing import Any, Dict
from tqdm import tqdm

from .base_trainer import BaseTrainer
from loss.loss_nnpu import PULoss
from loss.loss_robustpu import hardness_functions
from loss.spl_utils import calculate_spl_weights
from train.train_utils import evaluate_metrics


class TrainingScheduler:
    """
    Manages the threshold for self-paced learning.
    Adapted from the original implementation in spl_utills.py.
    """

    def __init__(
        self, schedule_type: str, init_ratio: float, max_ratio: float, grow_steps: int
    ):
        self.type = schedule_type
        self.init_ratio = init_ratio
        self.max_ratio = max_ratio
        self.grow_steps = grow_steps
        self.current_step = 0

    def get_next_ratio(self) -> float:
        """Calculates the ratio of samples to select for the current step."""
        if self.current_step >= self.grow_steps:
            self.current_step += 1
            return self.max_ratio

        if self.type == "const":
            step_ratio = 1.0
        elif self.type == "linear":
            step_ratio = self.current_step / self.grow_steps
        elif self.type == "convex":
            step_ratio = (self.current_step / self.grow_steps) ** 2
        elif self.type == "concave":
            step_ratio = (self.current_step / self.grow_steps) ** 0.5
        elif self.type == "exp":
            step_ratio = 2 ** (self.current_step / self.grow_steps) - 1
        else:
            raise ValueError(f"Unknown scheduler type: {self.type}")

        current_ratio = self.init_ratio + step_ratio * (
            self.max_ratio - self.init_ratio
        )
        self.current_step += 1
        return min(current_ratio, self.max_ratio)


class RobustPUTrainer(BaseTrainer):
    """Robust-PU learning trainer"""

    def __init__(self, method: str, experiment: str, params: dict):
        super().__init__(method, experiment, params)
        self.pre_train_params = self.params.get("pre_train", {})
        self.main_train_params = self.params.get("main_train", {})

        # Hardness function
        hardness_name = self.main_train_params.get("hardness", "logistic")
        self.hardness_func = hardness_functions[hardness_name]

        # SPL type
        self.spl_type = self.main_train_params.get("spl_type", "linear")

        # Moving average ratio
        self.moving_ratio = self.main_train_params.get("moving_ratio", 0.9)
        self.moving_weights = None

        # Schedulers
        p_params = self.main_train_params.get("scheduler_p", {})
        self.scheduler_p = TrainingScheduler(
            schedule_type=p_params.get("type", "linear"),
            init_ratio=float(p_params.get("alpha", 0.1)),
            max_ratio=float(p_params.get("max_thresh", 1.0)),
            grow_steps=int(p_params.get("grow_steps", 5)),
        )
        n_params = self.main_train_params.get("scheduler_n", {})
        self.scheduler_n = TrainingScheduler(
            schedule_type=n_params.get("type", "linear"),
            init_ratio=float(n_params.get("alpha", 0.11)),
            max_ratio=float(n_params.get("max_thresh", 1.0)),
            grow_steps=int(n_params.get("grow_steps", 5)),
        )

    def run(self):
        """Override run method to implement Robust-PU's two-stage logic."""
        self.before_training()

        self.console.log(
            "--- [Stage 1/2] Robust-PU: Pre-training ---", style="bold yellow"
        )
        if self.file_console:
            self.file_console.log("--- [Stage 1/2] Robust-PU: Pre-training ---")
        # Disable early stopping during pre-training
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = False
                self.checkpoint_handler.should_stop = False
            except Exception:
                pass
        self._pre_train()

        # Reset early stopping counter before main training
        if self.checkpoint_handler and self.checkpoint_handler.early_stopping_enabled:
            self.console.log(
                "Resetting early stopping counter for main training.", style="blue"
            )
            if self.file_console:
                self.file_console.log(
                    "Resetting early stopping counter for main training."
                )
            self.checkpoint_handler.wait = 0
            self.checkpoint_handler.should_stop = False

        self.console.log(
            "--- [Stage 2/2] Robust-PU: Iterative Refinement ---", style="bold yellow"
        )
        if self.file_console:
            self.file_console.log("--- [Stage 2/2] Robust-PU: Iterative Refinement ---")
        # Re-enable early stopping for the main stage
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = True
            except Exception:
                pass
        final_metrics = self._main_train()

        self.after_training()
        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()
        self._close_file_console()
        return final_metrics

    def _pre_train(self):
        """Stage 1: Pre-training using nnPU."""

        pre_lr = float(self.pre_train_params.get("lr", 1e-3))
        pre_wd = float(self.pre_train_params.get("weight_decay", 1e-4))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=pre_lr, weight_decay=pre_wd
        )

        # Use nnPU or BCE loss based on config
        pre_loss = self.pre_train_params.get("loss", "nnpu")
        if pre_loss == "nnpu":
            self.criterion = PULoss(self.prior, loss="sigmoid", nnpu=True)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        num_epochs = self.pre_train_params.get("epochs", 100)
        for epoch_idx in tqdm(range(1, num_epochs + 1), desc="Pre-training"):
            self.model.train()
            for x, t, _y_true, _idx, _ in self.train_loader:
                x, t = x.to(self.device), t.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x).view(-1)

                if pre_loss == "nnpu":
                    loss = self.criterion(outputs, t)
                else:
                    # For BCE, convert PU labels to binary
                    binary_labels = (t == 1).float()
                    loss = self.criterion(outputs, binary_labels)

                loss.backward()
                self.optimizer.step()

            # Print metrics every epoch (now with validation metrics if available)
            train_metrics = evaluate_metrics(
                self.model, self.train_loader, self.device, self.prior
            )
            test_metrics = evaluate_metrics(
                self.model, self.test_loader, self.device, self.prior
            )
            val_metrics = (
                evaluate_metrics(
                    self.model, self.validation_loader, self.device, self.prior
                )
                if self.validation_loader is not None
                else None
            )
            self._print_metrics(
                epoch_idx,
                num_epochs,
                train_metrics,
                test_metrics,
                "Pre-training",
                val_metrics=val_metrics,
            )

            # Integrate checkpoint improvement logging consistent with BaseTrainer
            if self.checkpoint_handler:
                all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
                if val_metrics is not None:
                    all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                import time as _time

                self.checkpoint_handler(
                    epoch=epoch_idx,
                    all_metrics=all_metrics,
                    model=self.model,
                    elapsed_seconds=(
                        _time.time() - self._run_start_time
                        if self._run_start_time
                        else None
                    ),
                )

    def _main_train(self):
        """Stage 2: Iterative optimization using self-paced learning."""

        main_lr = float(self.main_train_params.get("lr", 1e-2))
        main_wd = float(self.main_train_params.get("weight_decay", 0.0))

        # Number of outer episodes (not inner epochs)
        num_episodes = int(self.main_train_params.get("epochs", 5))
        num_inner_epochs = int(self.main_train_params.get("inner_epochs", 20))

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        for episode in tqdm(range(1, num_episodes + 1), desc="Episodes"):
            # Get current thresholds from schedulers
            thresh_p = self.scheduler_p.get_next_ratio()
            thresh_n = self.scheduler_n.get_next_ratio()

            self.console.log(
                f"Episode {episode}: thresh_p={thresh_p:.3f}, thresh_n={thresh_n:.3f}"
            )

            # Create weighted dataloader for this episode
            weighted_loader = self._create_weighted_dataloader(thresh_p, thresh_n)

            # Optionally restart model parameters
            if self.main_train_params.get("restart", False):
                self.model.apply(self._reset_weights)

            # Create new optimizer for this episode
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=main_lr, weight_decay=main_wd
            )

            # Inner training loop
            should_break_episodes = False
            for inner_epoch in range(1, num_inner_epochs + 1):
                self._train_episode(weighted_loader)

                # Log metrics every epoch
                self.global_epoch += 1
                train_metrics = evaluate_metrics(
                    self.model, self.train_loader, self.device, self.prior
                )
                test_metrics = evaluate_metrics(
                    self.model, self.test_loader, self.device, self.prior
                )
                val_metrics = (
                    evaluate_metrics(
                        self.model, self.validation_loader, self.device, self.prior
                    )
                    if self.validation_loader is not None
                    else None
                )
                self._print_metrics(
                    inner_epoch,
                    num_inner_epochs,
                    train_metrics,
                    test_metrics,
                    f"Episode {episode}",
                    val_metrics=val_metrics,
                )

                if self.checkpoint_handler:
                    all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                    all_metrics.update(
                        {f"test_{k}": v for k, v in test_metrics.items()}
                    )
                    if val_metrics is not None:
                        all_metrics.update(
                            {f"val_{k}": v for k, v in val_metrics.items()}
                        )
                    import time as _time

                    self.checkpoint_handler(
                        epoch=self.global_epoch,
                        all_metrics=all_metrics,
                        model=self.model,
                        elapsed_seconds=(
                            _time.time() - self._run_start_time
                            if self._run_start_time
                            else None
                        ),
                    )

                # Early stopping check
                if self.checkpoint_handler and self.checkpoint_handler.should_stop:
                    self.console.log(
                        f"Early stopping in main training episode {episode}.",
                        style="bold red",
                    )
                    should_break_episodes = True
                    break

            if should_break_episodes:
                break

        return self.checkpoint_handler.best_metrics if self.checkpoint_handler else {}

    def _reset_weights(self, m):
        """Reset model weights for restart option."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.reset_parameters()

    def _create_weighted_dataloader(self, thresh_p: float, thresh_n: float):
        """Calculate weights based on hardness and create weighted data loader."""
        self.model.eval()

        # Process data in batches to avoid OOM
        batch_size = 1000
        all_features, all_pu_labels, all_true_labels = [], [], []
        all_hardness_p, all_hardness_n = [], []

        with torch.no_grad():
            for x, t, y_true, _idx, _ in self.train_loader:
                all_features.append(x)
                all_pu_labels.append(t)
                all_true_labels.append(y_true)

                # Calculate hardness for this batch
                x_device = x.to(self.device)
                logits = self.model(x_device).view(-1)

                # Calculate hardness for positive label
                temper_p = self.main_train_params.get("scheduler_p", {}).get(
                    "temper", 1.0
                )
                hardness_p = self.hardness_func(
                    logits / temper_p, torch.ones_like(logits)
                ).cpu()
                all_hardness_p.append(hardness_p)

                # Calculate hardness for negative label
                temper_n = self.main_train_params.get("scheduler_n", {}).get(
                    "temper", 1.3
                )
                hardness_n = self.hardness_func(
                    logits / temper_n, -torch.ones_like(logits)
                ).cpu()
                all_hardness_n.append(hardness_n)

        all_features = torch.cat(all_features, dim=0)
        all_pu_labels = torch.cat(all_pu_labels, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)
        hardness_p_all = torch.cat(all_hardness_p, dim=0)
        hardness_n_all = torch.cat(all_hardness_n, dim=0)

        # Calculate SPL weights (following original implementation logic)
        # First, calculate weights for all samples using negative hardness
        weights = calculate_spl_weights(hardness_n_all, thresh_n, self.spl_type)

        # Then, override weights for labeled positive samples
        pos_mask = all_pu_labels == 1
        if pos_mask.any():
            pos_hardness = hardness_p_all[pos_mask]
            pos_weights = calculate_spl_weights(pos_hardness, thresh_p, self.spl_type)
            weights[pos_mask] = pos_weights

        # Apply moving average
        if self.moving_weights is None:
            self.moving_weights = weights.clone()
        else:
            self.moving_weights = (
                self.moving_ratio * self.moving_weights
                + (1 - self.moving_ratio) * weights
            )
            weights = self.moving_weights.clone()

        # Log statistics
        unl_mask = all_pu_labels == -1
        num_selected_p = (weights[pos_mask] > 0.1).sum().item() if pos_mask.any() else 0
        num_selected_n = (weights[unl_mask] > 0.1).sum().item() if unl_mask.any() else 0

        # Calculate mean weights for positive and negative unlabeled samples
        unl_weights = weights[unl_mask]
        unl_true_labels = all_true_labels[unl_mask]
        neg_unl_weights = (
            unl_weights[unl_true_labels == 0]
            if (unl_true_labels == 0).any()
            else torch.tensor([0.0])
        )

        self.console.log(
            f"Selected samples - Positive: {num_selected_p}/{pos_mask.sum()}, "
            f"Unlabeled: {num_selected_n}/{unl_mask.sum()}"
        )
        self.console.log(
            f"Mean weight - Labeled: {weights[pos_mask].mean():.3f}, "
            f"Negative-unlabeled: {neg_unl_weights.mean():.3f}"
        )

        # Create weighted dataset with original PU labels
        weighted_dataset = torch.utils.data.TensorDataset(
            all_features, all_pu_labels, weights, all_true_labels
        )

        batch_size = self.main_train_params.get("batch_size", 64)
        return torch.utils.data.DataLoader(
            weighted_dataset, batch_size=batch_size, shuffle=True
        )

    def _train_episode(self, dataloader):
        """Train one epoch on a weighted mini-batch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        # Use PU loss for main training
        loss_type = self.main_train_params.get("loss", "pu")
        if loss_type == "bce":
            criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            criterion = PULoss(self.prior, loss="sigmoid", nnpu=True)

        for x, pu_labels, sample_weights, true_labels in dataloader:
            x = x.to(self.device)
            pu_labels = pu_labels.to(self.device)  # Contains 1 and -1
            sample_weights = sample_weights.to(self.device)

            # Skip batch if all weights are zero
            if sample_weights.sum() < 1e-6:
                continue

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)

            if loss_type == "bce":
                # For BCE, convert PU labels to binary
                binary_labels = (pu_labels == 1).float()
                per_sample_loss = criterion(outputs, binary_labels)
                weighted_loss = per_sample_loss * sample_weights
                loss = weighted_loss.sum() / (sample_weights.sum() + 1e-6)
            else:
                # For PU loss, pass weights directly to the loss function
                # PULoss expects labels to be 1 (positive) or -1 (unlabeled)
                loss = criterion(outputs, pu_labels, sample_weights)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    def create_criterion(self):
        """Placeholder implementation, actually created independently in each stage."""
        return nn.Identity()

    def train_one_epoch(self, epoch_idx: int):
        pass
