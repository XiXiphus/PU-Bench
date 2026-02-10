"""holisticpu_trainer.py

HolisticPUTrainer inherits from BaseTrainer and implements the Holistic-PU method.
This method contains two core stages:
1. Trend analysis stage: In early training, identify reliable negative samples by observing
   the trend of model predictions on unlabeled samples using `jenkspy` natural breaks clustering.
2. Fine-tuning stage: Use high-quality pseudo labels from stage 1 for standard semi-supervised fine-tuning.
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from torch import nn
from typing import Any
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

# EMA
from backbone.ema import ModelEMA
from tqdm import tqdm
from rich.table import Table
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from torch.utils.data import Subset
from .train_utils import evaluate_metrics, seed_worker
from data.holisticpu_dataset import TransformHolisticPU, HolisticPUDatasetWrapper
from data.vector_augment import (
    VectorAugPUDatasetWrapper,
    VectorWeakAugment,
    VectorStrongAugment,
)


# Helper functions


def _three_sigma(x: np.ndarray) -> np.ndarray:
    """Filter extreme values using a hardcoded threshold from the original implementation.

    This is not a standard three-sigma rule but a specific filtering used by the authors.
    The original code had different versions; we use the one from `creditcard.py` and `train.py`.
    """
    # The original implementation contained logic like:
    # idx = np.where(x < 0.2 / 9) # from misc.py, seems dataset-specific
    # return x[idx]
    # To remain general while being faithful, we replicate the standard rule,
    # as it was likely the authors' intention. For exact replication on specific
    # datasets, this function may need to be tailored.
    mean = x.mean()
    std = x.std()
    min_range = mean - 3 * std
    max_range = mean + 3 * std
    idx = np.where((x > min_range) & (x < max_range))
    return x[idx]


def _interleave(x: torch.Tensor, size: int) -> torch.Tensor:
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def _de_interleave(x: torch.Tensor, size: int) -> torch.Tensor:
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 7.0 / 16.0,  # Align with original implementation
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# jenkspy is a core dependency for this method
try:
    import jenkspy
except ImportError:
    print("Error: holisticpu_trainer requires 'jenkspy' library.")
    print("Please run: pip install jenkspy")
    raise

from .base_trainer import BaseTrainer


class HolisticPUTrainer(BaseTrainer):
    """Holistic-PU Learning Trainer.

    HolisticPU is a novel PU learning method that identifies reliable negative samples
    by analyzing the trend of model predictions on unlabeled samples during training.
    The method is based on an important observation: in early training, the model's
    predictions for true positive samples show an upward trend, while predictions
    for negative samples remain relatively stable or decline.

    Core ideas:
    1. Trend analysis: Calculate first-order differences of prediction scores for
       each unlabeled sample across multiple training epochs
    2. Trend scoring: Use log(1 + diff + 0.5 * diff²) transformation to quantify trend strength
    3. Natural clustering: Use Jenks natural breaks algorithm to divide samples into two classes
    4. Pseudo-label generation: Samples with higher trend scores are labeled as positive,
       lower ones as negative

    Two-stage training:
    - Stage 1: Train P vs U classifier while recording prediction trends of U samples
    - Stage 2: Use generated pseudo-labels for standard semi-supervised training
    """

    def __init__(self, method: str, experiment: str, params: dict[str, Any]):
        super().__init__(method, experiment, params)
        # Read HolisticPU-specific parameters from configuration
        self.phase1_epochs = self.params.get("phase1_epochs", 15)
        self.current_phase = 1
        self.pseudo_labels_map = None

        # --- Attributes for both phases ---
        self.model2 = None
        self.optimizer2 = None
        self.scheduler2 = None
        self.labeled_loader = None
        self.unlabeled_loader = None
        self.unlabeled_pred_loader = None  # For prediction, no shuffle, no drop_last

        # Stage-2 hyperparameters
        self.use_ema = self.params.get("use_ema", True)
        self.ema_decay = self.params.get("ema_decay", 0.999)
        self.ema_model = None

    def run(self):
        """Override run method to implement HolisticPU's two-stage logic."""
        self.before_training()
        final_metrics = None

        # --- Stage 1: Trend Analysis & Pseudo-Label Generation ---
        self.console.log(
            "\n--- [Stage 1/2] Holistic-PU: Trend Analysis & Pseudo-Labeling ---",
            style="bold yellow",
        )
        if self.file_console:
            self.file_console.log(
                "\n--- [Stage 1/2] Holistic-PU: Trend Analysis & Pseudo-Labeling ---"
            )

        # Reset early stopping counter before the main analysis/tuning stages
        if self.checkpoint_handler and self.checkpoint_handler.early_stopping_enabled:
            self.console.log(
                "Resetting early stopping counter for main training stages.",
                style="blue",
            )
            if self.file_console:
                self.file_console.log(
                    "Resetting early stopping counter for main training stages."
                )
            self.checkpoint_handler.wait = 0
            self.checkpoint_handler.should_stop = False
        # Disable early stopping during Phase-1 analysis to ensure completion
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = False
            except Exception:
                pass

        pseudo_labels_map = self._run_phase1_analysis()
        self.pseudo_labels_map = pseudo_labels_map

        # --- Stage 2: Fine-tuning with Pseudo-Labels ---
        if pseudo_labels_map:
            self.console.log(
                "\n--- [Stage 2/2] Holistic-PU: Fine-tuning with Pseudo-Labels ---",
                style="bold yellow",
            )
            if self.file_console:
                self.file_console.log(
                    "\n--- [Stage 2/2] Holistic-PU: Fine-tuning with Pseudo-Labels ---"
                )
            # Reinitialize model and optimizer for stage 2 to achieve optimal performance
            # Note: The original author's implementation recreates the model, we follow this approach
            self._build_model()
            self.model2 = self._build_model(return_model=True)
            # Align dataloaders for Phase-2 like the original: L/U same batch, drop_last=True
            self._rebuild_phase2_loaders()

            # After rebuilding the model, dataloaders for phase 2 are already created
            # self._wrap_train_dataset_for_phase2() # This is now done in before_training

            lr = self.params.get("lr", 1e-3)
            wd = self.params.get("weight_decay", 5e-4)
            momentum = self.params.get("momentum", 0.9)
            nesterov = self.params.get("nesterov", True)
            # Allow users to set stage 2 epochs separately, otherwise fall back to global epochs
            num_epochs_phase2 = self.params.get(
                "phase2_epochs", self.params.get("epochs", 100)
            )

            # --- Optimizer and Scheduler for model 1 (self.model) ---
            no_decay = ["bias", "bn"]
            grouped_parameters1 = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": wd,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.SGD(
                grouped_parameters1, lr=lr, momentum=momentum, nesterov=nesterov
            )

            # The original implementation uses steps for scheduler, not epochs.
            # Number of steps per epoch is defined by the number of batches in the unlabeled loader.
            steps_per_epoch_ph2 = len(self.unlabeled_loader)
            num_training_steps = num_epochs_phase2 * steps_per_epoch_ph2
            num_warmup_steps = self.params.get("warmup_steps", 0)

            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

            # --- Optimizer and Scheduler for model 2 (unused, for consistency) ---
            grouped_parameters2 = [
                {
                    "params": [
                        p
                        for n, p in self.model2.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": wd,
                },
                {
                    "params": [
                        p
                        for n, p in self.model2.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer2 = torch.optim.SGD(
                grouped_parameters2, lr=lr, momentum=momentum, nesterov=nesterov
            )
            self.scheduler2 = get_cosine_schedule_with_warmup(
                self.optimizer2,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

            # Create EMA model for the main model
            if self.use_ema:
                self.ema_model = ModelEMA(self, self.model, self.ema_decay)
            self.current_phase = 2

            # Add stage 2 configuration information
            self.console.log(
                f"Stage 2 training config: LR={lr}, Epochs={num_epochs_phase2}, Steps_per_epoch={steps_per_epoch_ph2}"
            )
            if self.file_console:
                self.file_console.log(
                    f"Stage 2 training config: LR={lr}, Epochs={num_epochs_phase2}, Steps_per_epoch={steps_per_epoch_ph2}"
                )
                self.file_console.log(
                    f"Optimizer type: {type(self.optimizer).__name__}, Parameter groups: {len(self.optimizer.param_groups)}"
                )

            # Use a custom training loop that mimics the original implementation, instead of self._run_epochs
            # Re-enable early stopping for Phase-2 before starting fine-tuning
            if self.checkpoint_handler:
                try:
                    self.checkpoint_handler.early_stopping_enabled = True
                except Exception:
                    pass
            final_metrics = self._run_phase2_epochs(
                num_epochs_phase2, steps_per_epoch_ph2
            )
        # else: no pseudo labels, Phase-2 cannot proceed

        # End and record best results
        self.after_training()
        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()

        # Close file console
        self._close_file_console()

        return final_metrics or {}

    def _run_phase1_analysis(self) -> dict[int, int] | None:
        """
        Execute first stage training and analysis.
        In this stage, the model is trained to distinguish P and U, while recording prediction trend changes for U samples.
        """

        # Set optimizer and scheduler for stage 1
        lr = self.params.get("lr", 1e-3)
        wd = self.params.get("weight_decay", 5e-4)
        momentum = self.params.get("momentum", 0.9)
        nesterov = self.params.get("nesterov", True)

        # Parameter grouping, BN layers and biases do not use weight decay
        no_decay = ["bias", "bn"]
        grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": wd,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.SGD(
            grouped_parameters, lr=lr, momentum=momentum, nesterov=nesterov
        )

        # Set learning rate scheduler based on steps, not epochs
        steps_per_epoch = self.params.get("steps_per_epoch", 512)
        num_training_steps = self.phase1_epochs * steps_per_epoch
        num_warmup_steps = self.params.get("warmup_steps", 0)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Create EMA model
        if self.use_ema:
            self.ema_model = ModelEMA(self, self.model, self.ema_decay)

        preds_sequence = []

        # 1. Execute Phase 1 training and prediction
        num_epochs = self.phase1_epochs

        self.console.log(
            f"Stage 1 training config: LR={lr}, Epochs={num_epochs}, Steps_per_epoch={steps_per_epoch}"
        )
        if self.file_console:
            self.file_console.log(
                f"Stage 1 training config: LR={lr}, Epochs={num_epochs}, Steps_per_epoch={steps_per_epoch}"
            )

        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)

        for epoch in tqdm(range(1, num_epochs + 1), desc="Phase 1 Training"):
            self.global_epoch += 1
            self.model.train()

            for batch_idx in range(steps_per_epoch):
                try:
                    (x_l_w, x_l_s), _, y_l_true, _, _ = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.labeled_loader)
                    (x_l_w, x_l_s), _, y_l_true, _, _ = next(labeled_iter)

                try:
                    (x_u_w, x_u_s), _, _, _, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.unlabeled_loader)
                    (x_u_w, x_u_s), _, _, _, _ = next(unlabeled_iter)

                x_l_w, x_l_s, y_l_true = (
                    x_l_w.to(self.device),
                    x_l_s.to(self.device),
                    y_l_true.to(self.device),
                )
                x_u_w, x_u_s = (
                    x_u_w.to(self.device),
                    x_u_s.to(self.device),
                )

                # P samples are class 0, U samples are class 1
                targets_l = torch.zeros_like(y_l_true)
                targets_u = torch.ones(x_u_w.size(0), device=self.device).long()

                self.optimizer.zero_grad()

                # The original implementation uses weak+strong for P, and only weak for U in loss
                # The logits for unlabeled data use weak augmentation input
                logits_l_w = self.model(x_l_w)
                logits_l_s = self.model(x_l_s)
                logits_u_w = self.model(x_u_w)

                # Supervised loss for labeled data (P)
                smoothing = self.params.get("label_smoothing", 0.1)
                loss_l = (
                    F.cross_entropy(logits_l_w, targets_l, label_smoothing=smoothing)
                    + F.cross_entropy(logits_l_s, targets_l, label_smoothing=smoothing)
                ) / 2

                # Unsupervised loss for unlabeled data (U), treated as negative class
                loss_u = F.cross_entropy(logits_u_w, targets_u)

                loss = loss_l + loss_u

                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                if self.use_ema and self.ema_model is not None:
                    self.ema_model.update(self.model)

            # --- Standard evaluation and logging (per epoch) ---
            train_metrics = evaluate_metrics(
                self.model, self.train_loader, self.device, self.prior
            )
            test_metrics = evaluate_metrics(
                self.model, self.test_loader, self.device, self.prior
            )
            val_metrics = (
                evaluate_metrics(
                    self.get_eval_model(),
                    self.validation_loader,
                    self.device,
                    self.prior,
                )
                if self.validation_loader is not None
                else None
            )
            self._print_metrics(
                epoch,
                num_epochs,
                train_metrics,
                test_metrics,
                "Trend_Analysis",
                val_metrics=val_metrics,
            )

            # Call checkpoint to log improvements (like BaseTrainer)
            if self.checkpoint_handler:
                all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
                if val_metrics is not None:
                    all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                eval_model = (
                    self.get_eval_model()
                    if hasattr(self, "get_eval_model")
                    else self.model
                )
                import time as _time

                self.checkpoint_handler(
                    epoch=self.global_epoch,
                    all_metrics=all_metrics,
                    model=eval_model,
                    elapsed_seconds=(
                        _time.time() - self._run_start_time
                        if self._run_start_time
                        else None
                    ),
                )

            # After each epoch, predict and record unlabeled data
            if self.use_ema and self.ema_model is not None:
                eval_model = self.ema_model.ema
            else:
                eval_model = self.model

            eval_model.eval()
            all_scores = []
            unlabeled_count = 0
            with torch.no_grad():
                # Use the prediction loader to ensure all samples are included and in order
                for (x_u_w, _), _, _, _, _ in self.unlabeled_pred_loader:
                    x = x_u_w.to(self.device)
                    unlabeled_count += x.size(0)
                    logits = eval_model(x)

                    # Ensure logits shape is correct
                    if logits.dim() == 0:  # Scalar output for single sample
                        logits = logits.unsqueeze(0)
                    elif logits.dim() == 1 and len(x) == 1:
                        # Single sample, keep 1D
                        pass
                    else:
                        logits = logits.squeeze()

                    # Compatible with multi-class and binary classification models
                    if logits.dim() > 1 and logits.size(-1) > 1:
                        # Multi-class model: use softmax and take 1st output (P class probability)
                        probs = torch.softmax(logits, dim=-1)
                        # We need the probability of being class 0 (Positive)
                        scores = probs[:, 0].cpu().numpy()
                    else:
                        # Binary classification model: use sigmoid, output represents P class probability
                        # A lower logit means higher probability of class 0 (P)
                        scores = 1 - torch.sigmoid(logits).cpu().numpy()
                        # Ensure scores is 1D array
                        if scores.ndim == 0:
                            scores = np.array([scores])

                    all_scores.append(scores)

            if all_scores:
                epoch_preds = np.concatenate(all_scores)
                preds_sequence.append(epoch_preds)
                # Record prediction statistics
                if self.file_console:
                    self.file_console.log(
                        f"Epoch {epoch} - Unlabeled samples: {unlabeled_count}, Pred stats: min={np.min(epoch_preds):.4f}, max={np.max(epoch_preds):.4f}, mean={np.mean(epoch_preds):.4f}"
                    )
            else:
                if self.file_console:
                    self.file_console.log(
                        f"Epoch {epoch} - No unlabeled samples found!"
                    )

        if not preds_sequence:
            self.console.log(
                "No unlabeled samples found in stage 1, unable to generate pseudo-labels.",
                style="bold red",
            )
            if self.file_console:
                self.file_console.log(
                    "No unlabeled samples found in stage 1, unable to generate pseudo-labels."
                )
            return None

        # 2. Analyze trends and generate pseudo-labels
        # (num_epochs, num_unlabeled_samples) -> (num_unlabeled_samples, num_epochs)
        preds_sequence = np.vstack(preds_sequence).T

        trends = np.zeros(len(preds_sequence))
        for i, sequence in enumerate(preds_sequence):
            # Use pandas to calculate first-order differences, consistent with original implementation
            s = pd.Series(sequence)
            diff_1 = s.diff(periods=1).iloc[1:].to_numpy()  # Drop first NaN

            # Check if there are valid difference values
            if len(diff_1) == 0 or np.all(np.isnan(diff_1)):
                trends[i] = 0.0
                continue

            # A log-like transformation from original implementation to handle trends
            # Original implementation does not clip values, we follow that.
            diff_1_transformed = np.log(1 + diff_1 + 0.5 * diff_1**2)

            # Filter out invalid values
            valid_mask = np.isfinite(diff_1_transformed)
            if np.any(valid_mask):
                trends[i] = np.mean(diff_1_transformed[valid_mask])
            else:
                trends[i] = 0.0

        # Record trend statistics
        self.console.log(
            f"Trend statistics: Min={np.min(trends):.4f}, Max={np.max(trends):.4f}, Mean={np.mean(trends):.4f}, Std={np.std(trends):.4f}"
        )
        if self.file_console:
            self.file_console.log(
                f"Trend statistics: Min={np.min(trends):.4f}, Max={np.max(trends):.4f}, Mean={np.mean(trends):.4f}, Std={np.std(trends):.4f}"
            )
            # Add trend distribution information
            positive_trends = np.sum(trends > 0)
            negative_trends = np.sum(trends < 0)
            zero_trends = np.sum(trends == 0)
            self.file_console.log(
                f"Trend distribution: Positive={positive_trends}, Negative={negative_trends}, Zero={zero_trends}"
            )

        # 3. Use Jenks Natural Breaks algorithm for clustering
        breaks = None
        break_point = None
        try:
            # Check validity of trend data
            if len(np.unique(trends)) < 2:
                raise ValueError(
                    "Trend data lacks variability, unable to perform clustering"
                )

            # Initial clustering
            breaks = jenkspy.jenks_breaks(trends, n_classes=2)
            break_point = breaks[1]

            # Optional three-sigma truncation
            if self.params.get("use_three_sigma", False) and break_point > 0:
                trends_std = _three_sigma(trends)
                if len(np.unique(trends_std)) >= 2:
                    breaks = jenkspy.jenks_breaks(trends_std, n_classes=2)
                    break_point = breaks[1]
        except Exception as e:
            self.console.log(f"Jenkspy clustering failed: {e}", style="bold red")
            self.console.log(
                "Will use trend median as fallback split point.", style="yellow"
            )
            if self.file_console:
                self.file_console.log(f"Jenkspy clustering failed: {e}")
                self.file_console.log("Will use trend median as fallback split point.")
            break_point = np.median(trends)

        self.console.log(
            f"Jenks breaks: {breaks if breaks is not None else 'N/A'}, Break point: {break_point:.4f}"
        )
        if self.file_console:
            self.file_console.log(
                f"Jenks breaks: {breaks if breaks is not None else 'N/A'}, Break point: {break_point:.4f}"
            )

        # [Key Fix]: Align with original implementation's pseudo-label direction
        # Trends greater than breakpoint are considered positive (pseudo_label=0), lower trends as negative (pseudo_label=1)
        pseudo_labels_binary = np.where(trends > break_point, 0, 1).astype(int)

        # Align to unlabeled training loader order (sequential, drop_last=True)
        unlabeled_indices = self.unlabeled_loader.dataset.indices
        if len(pseudo_labels_binary) != len(unlabeled_indices):
            raise ValueError(
                f"FATAL: Mismatch between predictions ({len(pseudo_labels_binary)}) and unlabeled indices ({len(unlabeled_indices)})."
            )
        # Save both array (for Phase-2 circular slicing) and map (optional external use)
        self.pseudo_targets_array = pseudo_labels_binary
        self.pseudo_labels_map = dict(zip(unlabeled_indices, pseudo_labels_binary))

        # Count pseudo-label information
        n_pseudo_pos = np.sum(pseudo_labels_binary == 0)  # Positive count (label=0)
        n_pseudo_neg = np.sum(pseudo_labels_binary == 1)  # Negative count (label=1)
        self.console.log(
            f"Generated {n_pseudo_pos} positive pseudo-labels and {n_pseudo_neg} negative pseudo-labels from {len(unlabeled_indices)} unlabeled samples."
        )
        if self.file_console:
            self.file_console.log(
                f"Generated {n_pseudo_pos} positive pseudo-labels and {n_pseudo_neg} negative pseudo-labels from {len(unlabeled_indices)} unlabeled samples."
            )

        # ------------------ Evaluate pseudo-label quality ------------------
        # In many datasets, true labels might not be available for U set in a real scenario.
        # We perform this evaluation only if the dataset wrapper provides true labels.
        wrapped_dataset = self.unlabeled_loader.dataset.dataset
        if hasattr(wrapped_dataset.base_dataset, "true_labels"):
            true_labels_all = wrapped_dataset.base_dataset.true_labels.cpu().numpy()
            # Use the same indices (sequential, drop_last=True) for exact alignment
            true_labels_unlabeled = true_labels_all[unlabeled_indices]

            # [FIX] For evaluation, flip pseudo-labels (0 for P, 1 for N) to match true labels (1 for P, 0 for N)
            pseudo_labels_eval = 1 - pseudo_labels_binary
            acc = accuracy_score(true_labels_unlabeled, pseudo_labels_eval)
            prec = precision_score(
                true_labels_unlabeled, pseudo_labels_eval, zero_division=0
            )
            rec = recall_score(
                true_labels_unlabeled, pseudo_labels_eval, zero_division=0
            )
            f1 = f1_score(true_labels_unlabeled, pseudo_labels_eval, zero_division=0)

            table = Table(title="Phase-1 Pseudo-Label Quality", show_edge=True)
            table.add_column("Metric", justify="left")
            table.add_column("Value", justify="right")
            table.add_row("Accuracy", f"{acc:.4f}")
            table.add_row("Precision", f"{prec:.4f}")
            table.add_row("Recall", f"{rec:.4f}")
            table.add_row("F1-Score", f"{f1:.4f}")
            self.console.print(table)
            if self.file_console:
                self.file_console.print(table)

        # Calculate estimated positive prior probability
        wrapped_dataset_labeled = self.labeled_loader.dataset.dataset
        n_labeled_pos = (
            (wrapped_dataset_labeled.base_dataset.pu_labels == 1).sum().item()
        )
        estimated_prior = (n_pseudo_pos + n_labeled_pos) / (
            len(unlabeled_indices) + n_labeled_pos
        )
        self.console.log(f"Estimated Prior: {estimated_prior:.4f}")
        if self.file_console:
            self.file_console.log(f"Estimated Prior: {estimated_prior:.4f}")

        return self.pseudo_labels_map

    def _run_phase2_epochs(self, num_epochs: int, steps_per_epoch: int):
        """Custom epoch loop for Phase 2 to match original implementation."""
        test_metrics = {}  # Initialize with a default value
        for epoch in range(1, num_epochs + 1):
            self.global_epoch += 1
            self._train_epoch_phase2(epoch, num_epochs, steps_per_epoch)

            # Perform evaluation at the end of each epoch
            train_metrics = evaluate_metrics(
                self.get_eval_model(), self.train_loader, self.device, self.prior
            )
            test_metrics = evaluate_metrics(
                self.get_eval_model(), self.test_loader, self.device, self.prior
            )
            val_metrics = (
                evaluate_metrics(
                    self.get_eval_model(),
                    self.validation_loader,
                    self.device,
                    self.prior,
                )
                if self.validation_loader is not None
                else None
            )
            self._print_metrics(
                epoch,
                num_epochs,
                train_metrics,
                test_metrics,
                "Fine-tuning",
                val_metrics=val_metrics,
            )

            if self.checkpoint_handler:
                all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
                if val_metrics is not None:
                    all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                import time as _time

                self.checkpoint_handler(
                    epoch=self.global_epoch,
                    all_metrics=all_metrics,
                    model=self.get_eval_model(),
                    elapsed_seconds=(
                        _time.time() - self._run_start_time
                        if self._run_start_time
                        else None
                    ),
                )

            # Early stopping check
            if self.checkpoint_handler and self.checkpoint_handler.should_stop:
                self.console.log(
                    "Early stopping in fine-tuning stage.", style="bold red"
                )
                break
        return test_metrics  # Return final test metrics

    def create_criterion(self):
        """Return different loss functions based on current stage."""
        # Both stage 1 and stage 2 use cross-entropy loss
        return nn.CrossEntropyLoss()

    def train_one_epoch(self, epoch_idx: int):
        """
        This method is required by the BaseTrainer, but HolisticPU uses a custom
        run loop that handles training for both phases directly. Therefore, this
        method can be left empty to satisfy the abstract class requirement.
        """
        pass

    def get_eval_model(self):
        """Get model for evaluation"""
        # Original Phase-2 uses the main model for evaluation (not EMA)
        return self.model

    def _train_epoch_phase2(
        self, current_epoch: int, total_epochs: int, steps_per_epoch: int
    ):
        """Stage 2: Train using pseudo-labels, mimicking original implementation."""
        self.model.train()
        if not self.pseudo_labels_map:
            raise RuntimeError(
                "Pseudo-labels not generated when stage 2 training starts."
            )

        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)

        # Use contiguous pseudo targets aligned with unlabeled loader order
        pseudo_targets_arr = self.pseudo_targets_array
        unlabeled_num = len(pseudo_targets_arr)
        unlabeled_idx_ptr = 0

        total_loss = 0.0
        labeled_loss_sum = 0.0
        unlabeled_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Phase 2 Epoch {current_epoch}/{total_epochs}",
        )
        for batch_idx in pbar:
            try:
                (x_l_w, x_l_s), _, y_true_l, _, _ = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_loader)
                (x_l_w, x_l_s), _, y_true_l, _, _ = next(labeled_iter)

            try:
                (x_u_w, x_u_s), _, _, _, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                (x_u_w, x_u_s), _, _, _, _ = next(unlabeled_iter)

            x_l_w = x_l_w.to(self.device)
            x_l_s = x_l_s.to(self.device)
            x_u_w = x_u_w.to(self.device)
            x_u_s = x_u_s.to(self.device)
            batch_size_l = x_l_w.size(0)
            batch_size_u = x_u_w.size(0)

            # CRITICAL: In the original implementation, batch sizes must match for chunk(2) to work
            if batch_size_l != batch_size_u:
                self.console.log(
                    f"[yellow]Warning: Batch size mismatch L={batch_size_l}, U={batch_size_u}, skipping iteration[/yellow]"
                )
                continue

            batch_size = batch_size_l  # Now we know they're equal
            mu = self.params.get("mu", 1)

            # Original Phase-2: two forward passes with size=2 interleave on (P_w,U_w) and (P_s,U_s)
            # IMPORTANT: The original implementation always uses size=2, not 2*mu or 4*mu
            inputs_pw_uw = torch.cat((x_l_w, x_u_w), dim=0)

            # Debug info
            if batch_idx == 0:
                self.console.log(
                    f"[blue]Phase2 Debug: x_l_w.shape={x_l_w.shape}, x_u_w.shape={x_u_w.shape}[/blue]"
                )
                self.console.log(
                    f"[blue]inputs_pw_uw.shape={inputs_pw_uw.shape}, interleave size=2[/blue]"
                )

            inputs_pw_uw = _interleave(inputs_pw_uw, 2).to(self.device)
            logits_pw_uw = self.model(inputs_pw_uw)
            logits_pw_uw = _de_interleave(logits_pw_uw, 2)
            logits_x_w, logits_u = logits_pw_uw.chunk(2)

            inputs_ps_us = torch.cat((x_l_s, x_u_s), dim=0)
            inputs_ps_us = _interleave(inputs_ps_us, 2).to(self.device)
            logits_ps_us = self.model(inputs_ps_us)
            logits_ps_us = _de_interleave(logits_ps_us, 2)
            logits_x_s, logits_u_s = logits_ps_us.chunk(2)

            # 1) Supervised CE on labeled weak (P class == 0), no smoothing
            targets_l = torch.zeros(batch_size, device=self.device, dtype=torch.long)
            loss_labeled = F.cross_entropy(logits_x_w, targets_l)
            labeled_loss_sum += loss_labeled.item()

            # 2) Unsupervised loss (interpolation + consistency)
            # a) Circular slice pseudo targets aligned with loader order
            slice_start = unlabeled_idx_ptr
            slice_end = unlabeled_idx_ptr + batch_size
            if slice_end <= unlabeled_num:
                batch_pseudo_np = pseudo_targets_arr[slice_start:slice_end]
            else:
                # wrap-around
                part1 = pseudo_targets_arr[slice_start:]
                part2 = pseudo_targets_arr[: slice_end - unlabeled_num]
                batch_pseudo_np = np.concatenate([part1, part2], axis=0)
            unlabeled_idx_ptr = (unlabeled_idx_ptr + batch_size) % unlabeled_num
            pseudo_targets = torch.from_numpy(batch_pseudo_np.astype(np.int64)).to(
                self.device
            )

            # b) Interpolation CE on weak
            label_p_onehot = F.one_hot(pseudo_targets, 2).float()
            label_u_onehot = F.one_hot(torch.ones_like(pseudo_targets), 2).float()
            lamda = (current_epoch / total_epochs) ** 0.8
            interpolated_target = lamda * label_p_onehot + (1 - lamda) * label_u_onehot
            loss_interp = F.cross_entropy(logits_u, interpolated_target)

            # c) Consistency CE on strong with confidence mask
            T = float(self.params.get("T", 1.0))
            threshold = float(self.params.get("threshold", 0.9))
            pseudo_label_confidence = torch.softmax(logits_u.detach() / T, dim=-1)
            max_probs, pseudo_targets_conf = torch.max(pseudo_label_confidence, dim=-1)
            mask = max_probs.ge(threshold).float()
            loss_consistency = (
                F.cross_entropy(logits_u_s, pseudo_targets_conf, reduction="none")
                * mask
            ).mean()

            loss_unlabeled = loss_interp + loss_consistency
            unlabeled_loss_sum += loss_unlabeled.item()

            # 3) Total loss
            lambda_u = float(self.params.get("lambda_u", 5.0))
            loss = loss_labeled + lambda_u * loss_unlabeled

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # EMA update
            if self.use_ema and self.ema_model is not None:
                self.ema_model.update(self.model)

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix(
                loss=total_loss / num_batches,
                loss_x=labeled_loss_sum / num_batches,
                loss_u=unlabeled_loss_sum / num_batches,
                lr=self.scheduler.get_last_lr()[0],
            )
        pbar.close()

        # Add training statistics
        if self.file_console and num_batches > 0:
            avg_total = total_loss / num_batches
            avg_labeled = labeled_loss_sum / num_batches
            avg_unlabeled = unlabeled_loss_sum / num_batches
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
            self.file_console.log(
                f"Phase2 - Total: {avg_total:.4f}, Labeled: {avg_labeled:.4f}, Unlabeled: {avg_unlabeled:.4f}, LR: {lr:.6f}, Batches: {num_batches}"
            )

    def _create_ssl_dataloaders(self):
        """Wrap and split training dataset for SSL, creating separate labeled and unlabeled dataloaders."""
        self.console.log(
            "Creating separate P/U dataloaders with weak/strong augmentations..."
        )
        if self.file_console:
            self.file_console.log(
                "Creating separate P/U dataloaders with weak/strong augmentations..."
            )

        base_dataset = self.train_loader.dataset

        # Detect image-like by sample dimensionality
        try:
            sample = base_dataset[0][0]
        except Exception:
            sample = None
        is_image_like = isinstance(sample, torch.Tensor) and sample.dim() >= 3
        self._is_image_like = bool(is_image_like)

        # Create the appropriate augmentation transform
        if is_image_like:
            dataset_mean = getattr(base_dataset, "mean", (0.5,))
            dataset_std = getattr(base_dataset, "std", (0.5,))
            if isinstance(dataset_mean, (int, float)):
                dataset_mean = (dataset_mean,)
            if isinstance(dataset_std, (int, float)):
                dataset_std = (dataset_std,)
            image_size = getattr(base_dataset, "image_size", 32)
            if (
                image_size == 32
                and isinstance(dataset_mean, tuple)
                and len(dataset_mean) == 1
            ):
                image_size = 28  # MNIST/FMNIST case
            transform = TransformHolisticPU(
                mean=dataset_mean, std=dataset_std, image_size=image_size
            )
        else:
            weak = VectorWeakAugment(
                noise_std=float(self.params.get("vec_weak_noise_std", 0.02)),
                dropout_ratio=float(self.params.get("vec_weak_dropout", 0.0)),
            )
            strong = VectorStrongAugment(
                noise_std=float(self.params.get("vec_strong_noise_std", 0.1)),
                dropout_ratio=float(self.params.get("vec_strong_dropout", 0.1)),
                sign_flip_ratio=float(self.params.get("vec_sign_flip_ratio", 0.05)),
            )
            transform = lambda x: (
                (weak(x), strong(x))
                if not isinstance(x, tuple)
                else (weak(x[0]), strong(x[0]))
            )

        # Create wrapped datasets with the transform
        wrapped_dataset = (
            HolisticPUDatasetWrapper(base_dataset, transform)
            if is_image_like
            else VectorAugPUDatasetWrapper(
                base_dataset, weak_aug=weak, strong_aug=strong
            )
        )

        # Split indices for labeled (P) and unlabeled (U) data
        labeled_indices = np.where(base_dataset.pu_labels == 1)[0]
        unlabeled_indices = np.where(base_dataset.pu_labels == -1)[0]

        labeled_subset = Subset(wrapped_dataset, labeled_indices)
        unlabeled_subset = Subset(wrapped_dataset, unlabeled_indices)

        # Create two separate dataloaders
        batch_size = self.params.get("batch_size", 64)
        num_workers = self.params.get("num_workers", 4)

        mu = self.params.get("mu", 1)
        # Phase-1 loaders (robust for very small P): do NOT scale by mu; do NOT drop last
        self.labeled_loader = torch.utils.data.DataLoader(
            labeled_subset,
            batch_size=min(batch_size, max(1, len(labeled_indices))),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
        )

        # Original implementation uses a mu parameter to have larger unlabeled batches
        self.unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_subset,
            batch_size=batch_size * mu,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
        )

        self.unlabeled_pred_loader = torch.utils.data.DataLoader(
            unlabeled_subset,
            batch_size=batch_size * mu,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
        )

        # A combined loader for standard evaluation purposes
        self.train_loader = torch.utils.data.DataLoader(
            wrapped_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

    def _rebuild_phase2_loaders(self):
        """Recreate P/U loaders for Phase-2 so that L/U batch sizes align like the original code.

        Key insight for LP-scarce scenarios:
        - When LP is very small, we need to ensure at least 2 batches for drop_last=True to work
        - Both loaders must produce identical batch sizes for chunk(2) operation
        """
        base_dataset = self.train_loader.dataset.base_dataset
        labeled_indices = np.where(base_dataset.pu_labels == 1)[0]
        unlabeled_indices = np.where(base_dataset.pu_labels == -1)[0]

        wrapped_dataset = self.train_loader.dataset
        labeled_subset = Subset(wrapped_dataset, labeled_indices)
        unlabeled_subset = Subset(wrapped_dataset, unlabeled_indices)

        mu = int(self.params.get("mu", 1))
        batch_base = int(self.params.get("batch_size", 64))
        num_workers = self.params.get("num_workers", 4)

        # Calculate effective batch size ensuring at least 2 batches for labeled data
        # (need 2 batches because drop_last=True will drop the last one)
        num_labeled = len(labeled_indices)
        num_unlabeled = len(unlabeled_indices)

        # For very small LP (like 63), we need special handling
        # Original implementation assumes batch_size < num_labeled, but this may not hold

        # Strategy for extremely small LP:
        # 1. If LP < batch_size, reduce batch_size to at most LP/2 (to get at least 2 batches)
        # 2. If LP is very small (< 10), use batch_size=1 or 2

        if num_labeled < batch_base:
            # LP is smaller than default batch size
            if num_labeled >= 4:
                # Try to get at least 2 batches
                eff_bs = num_labeled // 2
            else:
                # Extremely small LP, use minimum batch size
                eff_bs = min(2, num_labeled)
            use_drop_last = (
                num_labeled >= 2 * eff_bs
            )  # Only drop_last if we have enough for 2 full batches
            self.console.log(
                f"[yellow]Warning: Only {num_labeled} labeled samples, adjusting batch_size to {eff_bs}[/yellow]"
            )
        else:
            # Normal case: LP is larger than batch_size
            max_batch_size = num_labeled // 2  # Ensure at least 2 batches
            eff_bs = min(batch_base, max_batch_size)
            use_drop_last = True

        # Don't scale by mu if it would make batch_size > num_labeled
        if eff_bs * mu > num_labeled:
            final_batch_size = eff_bs
            self.console.log(
                f"[yellow]Not scaling by mu={mu} to avoid batch_size > num_labeled[/yellow]"
            )
        else:
            final_batch_size = eff_bs * mu

        self.console.log(
            f"Phase-2 DataLoader config: batch_size={final_batch_size}, drop_last={use_drop_last}"
        )
        self.console.log(
            f"Labeled samples: {num_labeled}, Unlabeled samples: {num_unlabeled}"
        )

        self.labeled_loader = torch.utils.data.DataLoader(
            labeled_subset,
            batch_size=final_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=use_drop_last,
            worker_init_fn=seed_worker,
        )
        self.unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_subset,
            batch_size=final_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=use_drop_last,
            worker_init_fn=seed_worker,
        )

    # Optional Hook (override as needed by subclasses)
    def before_training(self):
        """Called before training starts, can do additional initialization"""
        # Call parent class's file console initialization
        super().before_training()
        self._create_ssl_dataloaders()

        # Add HolisticPU-specific configuration information to log
        if self.file_console:
            self.file_console.log(f"Stage 1 epochs: {self.phase1_epochs}")
            self.file_console.log(
                f"Stage 2 epochs: {self.params.get('phase2_epochs', self.params.get('epochs'))}"
            )
            self.file_console.log(f"use_ema: {self.use_ema}")
            self.file_console.log(f"ema_decay: {self.ema_decay}")
            self.file_console.log(
                f"use_three_sigma: {self.params.get('use_three_sigma', False)}"
            )
            self.file_console.log(f"=" * 80)

    def after_training(self):
        """Called after training ends"""
        # Call parent class's cleanup logic
        super().after_training()
