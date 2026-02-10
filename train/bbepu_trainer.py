"""bbepu_trainer.py

BBE + nnPU Trainer Implementation
Combines Binomial Bias Estimation with non-negative PU learning.

This trainer implements a two-stage approach:
1. Initial training with standard nnPU using default prior
2. BBE-guided training with adaptive prior estimation

Based on "Mixture Proportion Estimation and PU Learning: A Modern Approach" (Garg et al., 2021)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from .base_trainer import BaseTrainer
from loss.loss_bbepu import BBEPULoss, BBEEstimator


class BBEPUTrainer(BaseTrainer):
    """
    Trainer for BBE + nnPU learning method.

    Features:
    - Two-stage training: warmup + BBE-guided nnPU
    - Dynamic prior estimation using BBE method
    - Theoretical guarantees from both BBE and nnPU
    """

    def before_training(self):
        super().before_training()

        # Training parameters
        self.warmup_epochs = self.params.get("warmup_epochs", 10)
        self.main_epochs = self.params.get("main_epochs", 20)
        self.bbe_update_freq = self.params.get("bbe_update_freq", 10)

        # BBE parameters
        self.delta = self.params.get("delta", 0.1)
        self.gamma_bbe = self.params.get("gamma_bbe", 0.01)

        # nnPU parameters
        self.gamma_nnpu = self.params.get("gamma", 1.0)
        self.beta = self.params.get("beta", 0.0)
        self.loss_type = self.params.get("loss_type", "sigmoid")

        # Initialize BBE estimator for validation-based estimation
        self.bbe_estimator = BBEEstimator(delta=self.delta, gamma=self.gamma_bbe)

        # Track prior estimates
        self.prior_estimates = []
        self.current_stage = 1  # 1: warmup, 2: BBE-guided

        self.console.log(f"BBE + nnPU Training Configuration:", style="bold cyan")
        self.console.log(f"  Warmup epochs: {self.warmup_epochs}")
        self.console.log(f"  Main epochs: {self.main_epochs}")
        self.console.log(f"  BBE update frequency: {self.bbe_update_freq}")
        self.console.log(f"  Initial prior: {self.prior}")

    def create_criterion(self):
        """Create BBE + nnPU loss with initial parameters."""
        loss_type = self.params.get("loss_type", "sigmoid")
        gamma = self.params.get("gamma", 1.0)
        beta = self.params.get("beta", 0.0)
        bbe_update_freq = self.params.get("bbe_update_freq", 10)

        return BBEPULoss(
            initial_prior=self.prior,
            loss_type=loss_type,
            gamma=gamma,
            beta=beta,
            bbe_update_freq=bbe_update_freq,
        )

    def run(self):
        """Custom training workflow with two stages."""
        self.before_training()

        # --- Stage 1: Warmup Training ---
        self.console.log(
            "\n--- [Stage 1/2] BBE+nnPU: Warmup Training ---", style="bold yellow"
        )
        # Disable early stopping during warmup to avoid premature stop
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = False
                self.checkpoint_handler.should_stop = False
            except Exception:
                pass
        if self.warmup_epochs > 0:
            self._run_epochs(self.warmup_epochs, stage_name="Stage 1 (Warmup)")

        # --- Intermediate: Better Prior Estimation ---
        self.console.log("\n--- Computing BBE Prior Estimate ---", style="bold yellow")
        self._estimate_prior_from_validation()

        # --- Stage 2: BBE-Guided Training ---
        self.current_stage = 2
        self.console.log(
            "\n--- [Stage 2/2] BBE+nnPU: BBE-Guided Training ---",
            style="bold yellow",
        )

        # Reset early stopping counter before the main stage
        if self.checkpoint_handler and self.checkpoint_handler.early_stopping_enabled:
            self.console.log(
                "Resetting early stopping counter for BBE-Guided stage.", style="blue"
            )
            if self.file_console:
                self.file_console.log(
                    "Resetting early stopping counter for BBE-Guided stage."
                )
            self.checkpoint_handler.wait = 0
            self.checkpoint_handler.should_stop = False
        # Re-enable early stopping for the final stage
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = True
            except Exception:
                pass

        # Create new loss with better prior estimate
        self.console.log(
            "Creating new loss function with estimated prior.", style="yellow"
        )
        self.criterion = BBEPULoss(
            initial_prior=self.criterion.get_current_prior(),
            loss_type=self.loss_type,
            gamma=self.gamma_nnpu,
            beta=self.beta,
            bbe_update_freq=self.bbe_update_freq,
        )

        if self.main_epochs > 0:
            final_metrics = self._run_epochs(
                self.main_epochs, stage_name="Stage 2 (BBE-Guided)"
            )
        else:
            final_metrics = {}

        # Log final prior estimates
        self._log_prior_estimates()

        self.after_training()
        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()
        self._close_file_console()

        return final_metrics

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch."""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for x, t, _y_true, _idx, _ in self.train_loader:
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)

            # BBE + nnPU loss
            loss = self.criterion(outputs, t)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        # Log current prior estimate
        if epoch_idx % 5 == 0:  # Log every 5 epochs
            current_prior = self.criterion.get_current_prior()
            self.prior_estimates.append(current_prior)

            if hasattr(self, "file_console") and self.file_console:
                self.file_console.log(
                    f"Epoch {epoch_idx}, Stage {self.current_stage}: "
                    f"Prior estimate = {current_prior:.4f}, Loss = {total_loss/batch_count:.4f}"
                )

    def _estimate_prior_from_validation(self):
        """
        Use validation set to get a better prior estimate using BBE.
        This provides a more stable estimate than the online version.
        """
        if not hasattr(self, "validation_loader") or self.validation_loader is None:
            self.console.log(
                "No validation loader available. Skipping BBE validation estimate.",
                style="yellow",
            )
            return

        self.model.eval()

        # Collect predictions on validation set
        all_outputs = []
        all_targets = []
        all_true_labels = []

        with torch.no_grad():
            for x, t, y_true, _idx, _ in self.validation_loader:
                x, t, y_true = (
                    x.to(self.device),
                    t.to(self.device),
                    y_true.to(self.device),
                )
                outputs = self.model(x).view(-1)
                probs = torch.sigmoid(outputs)

                all_outputs.append(probs)
                all_targets.append(t)
                all_true_labels.append(y_true)

        if len(all_outputs) == 0:
            return

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        all_true_labels = torch.cat(all_true_labels)

        # Separate P and U samples
        p_mask = all_targets == 1
        u_mask = all_targets == -1

        if p_mask.sum() > 10 and u_mask.sum() > 10:  # Need sufficient samples
            p_probs = all_outputs[p_mask]
            u_probs = all_outputs[u_mask]
            u_true_labels = all_true_labels[u_mask]

            # Stack u_probs for BBE format
            u_probs_stacked = torch.stack([u_probs, 1 - u_probs], dim=1)

            try:
                # BBE estimation with true labels
                prior_estimate = self.bbe_estimator.estimate_alpha(
                    p_probs, u_probs_stacked, u_true_labels
                )

                self.console.log(
                    f"BBE Validation Estimate: {prior_estimate:.4f} "
                    f"(Original: {self.prior:.4f})",
                    style="green",
                )

                # Update criterion's prior
                self.criterion.current_prior = prior_estimate

                if hasattr(self, "file_console") and self.file_console:
                    self.file_console.log(
                        f"BBE validation-based prior estimate: {prior_estimate:.4f}"
                    )

            except Exception as e:
                self.console.log(f"BBE estimation failed: {str(e)}", style="red")

        self.model.train()

    def _log_prior_estimates(self):
        """Log the evolution of prior estimates during training."""
        if len(self.prior_estimates) > 0:
            self.console.log(
                f"Prior estimate evolution: {[f'{p:.3f}' for p in self.prior_estimates[-5:]]}",
                style="cyan",
            )

        final_prior = self.criterion.get_current_prior()
        self.console.log(
            f"Final prior estimate: {final_prior:.4f} (Original: {self.prior:.4f})",
            style="bold green",
        )

        if hasattr(self, "file_console") and self.file_console:
            self.file_console.log(f"Final BBE prior estimate: {final_prior:.4f}")
            self.file_console.log(f"Prior estimate history: {self.prior_estimates}")
