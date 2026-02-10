"""distpu_trainer.py

DistPUTrainer inherits from BaseTrainer and implements two-stage Dist-PU training:
1. Warm-up stage: Basic LabelDistributionLoss (with optional entropy loss).
2. Mixup stage: Uses pseudo labels + Mixup + composite loss.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Any
from torch import nn

from loss.loss_distpu import LabelDistributionLoss
from loss.loss_entropy import entropy_loss
from .train_utils import (
    PseudoLabeler,
    mixup_data,
    mixup_criterion,
)

from .base_trainer import BaseTrainer


class DistPUTrainer(BaseTrainer):
    """Dist-PU method trainer (with warm-up & mixup two stages)."""

    # Stage is set in before_training() after initialization

    # Stage switching & Criterion creation
    def _create_criterion_for_stage(self, stage_params: dict[str, Any]):
        """Return loss function based on stage parameters."""
        gamma = stage_params.get("gamma", 1.0)

        # Basic distribution loss
        num_bins = stage_params.get("num_bins", 1)
        base_loss = LabelDistributionLoss(
            self.prior, num_bins=num_bins, device=self.device
        )

        # If in warm-up stage, allow adding entropy loss (co_mu)
        if stage_params.get("co_mu", 0) > 0:
            co_mu = stage_params["co_mu"]

            def composite_loss(logits, labels):
                scores = torch.sigmoid(logits)
                unlabeled_scores = scores[labels == 0]
                return base_loss(logits, labels) + co_mu * entropy_loss(
                    unlabeled_scores
                )

            return composite_loss

        # Mixup stage returns base_loss directly, composite logic handled in train_one_epoch
        return base_loss

    # Required interface implementation
    def create_criterion(self):
        """Placeholder implementation, actual criterion created in stage switching."""
        return nn.Identity()

    def train_one_epoch(self, epoch_idx: int):
        if self.current_stage == "warm_up":
            self._train_epoch_warm_up()
        elif self.current_stage == "mixup":
            self._train_epoch_mixup()
        else:
            raise ValueError(f"Unknown stage: {self.current_stage}")

    # Stage training implementation
    def _train_epoch_warm_up(self):
        self.model.train()
        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x).squeeze()
            labels = (t > 0).float()  # {+1,-1} -> {1,0}
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

    def _train_epoch_mixup(self):
        stage_params = self.mixup_cfg
        self.model.train()

        # Dynamic adjustment of co_entropy
        co_entropy_base = stage_params.get("co_entropy", 0.0)
        current_epoch_in_stage = self.global_epoch - self.warm_up_cfg.get("epochs", 0)
        total_mix_epochs = stage_params.get("epochs", 1)
        co_entropy = co_entropy_base * (
            1 - math.cos((current_epoch_in_stage / total_mix_epochs) * (math.pi / 2))
        )

        for x, t, _y_true, idx, _ in self.train_loader:  # type: ignore
            x, t, idx = x.to(self.device), t.to(self.device), idx.to(self.device)

            # 1. Pseudo labels
            pseudo_labels = self.pseudo_labeler.get_pseudo_labels_for_batch(idx)
            pseudo_labels[t == 1] = 1.0  # Fixed labeled positive samples as 1

            # 2. Mixup
            alpha = stage_params.get("alpha", 1.0)
            mixed_x, y_a, y_b, lam = mixup_data(x, pseudo_labels, alpha, self.device)

            # 3. Forward pass
            logits_orig = self.model(x).squeeze()
            scores_orig = torch.sigmoid(logits_orig)

            logits_mix = self.model(mixed_x).squeeze()
            scores_mix = torch.sigmoid(logits_mix)

            # 4. Loss
            self.optimizer.zero_grad()

            labels_dist = (t > 0).float()
            loss_dist = self.criterion(logits_orig, labels_dist)
            loss_ent_orig = entropy_loss(scores_orig[t != 1])
            loss_ent_mix = entropy_loss(scores_mix)
            loss_mix_ce = mixup_criterion(scores_mix, y_a, y_b, lam)

            total_loss = (
                loss_dist
                + co_entropy * loss_ent_orig
                + stage_params.get("co_mix_entropy", 0.0) * loss_ent_mix
                + stage_params.get("co_mixup", 0.0) * loss_mix_ce
            )
            total_loss.backward()
            self.optimizer.step()

            # 5. Update pseudo labels
            self.pseudo_labeler.update_pseudo_labels_for_batch(idx, scores_orig)

    # Multi-stage control
    def before_training(self):
        # First call parent's initialization (including file console)
        super().before_training()

        if "stages" not in self.params:
            raise ValueError("DistPU requires `stages` configuration in params.")
        self.warm_up_cfg = self.params["stages"].get("warm_up", {})
        self.mixup_cfg = self.params["stages"].get("mixup", {})

        # Add DistPU-specific configuration info to logs
        if self.file_console:
            self.file_console.log(
                f"Warm-up epochs: {self.warm_up_cfg.get('epochs', 0)}"
            )
            self.file_console.log(f"Mixup epochs: {self.mixup_cfg.get('epochs', 0)}")
            self.file_console.log(f"=" * 80)

        # Switch to warm-up stage first
        self._set_stage("warm_up", self.warm_up_cfg)

    def after_training(self):
        # Call parent's cleanup logic
        super().after_training()

    def _set_stage(self, stage_name: str, stage_params: dict[str, Any]):
        """Reconfigure optimizer and loss function based on given stage_params."""
        self.current_stage = stage_name

        # Update optimizer & loss using stage parameters
        lr = stage_params.get("lr", self.params.get("lr", 1e-3))
        wd = stage_params.get("weight_decay", self.params.get("weight_decay", 5e-4))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=wd
        )
        self.criterion = self._create_criterion_for_stage(stage_params)

    # Override run() to implement multi-stage training workflow
    def run(self):
        # Initialize stage configuration
        self.before_training()
        final_metrics = None

        # Warm-up
        if self.warm_up_cfg and self.warm_up_cfg.get("epochs", 0) > 0:
            print("\n--- [Stage 1/2] Dist-PU Warm-up ---")
            if self.file_console:
                self.file_console.log("\n--- [Stage 1/2] Dist-PU Warm-up ---")
            # Disable early stopping for the warm-up stage to avoid premature stop
            if self.checkpoint_handler:
                try:
                    self.checkpoint_handler.early_stopping_enabled = False
                    self.checkpoint_handler.should_stop = False
                except Exception:
                    pass
            self._set_stage("warm_up", self.warm_up_cfg)
            final_metrics = self._run_epochs(
                self.warm_up_cfg["epochs"], stage_name="Warm-up"
            )

        # Mixup stage
        if self.mixup_cfg and self.mixup_cfg.get("epochs", 0) > 0:
            print("\n--- [Stage 2/2] Dist-PU Mixup ---")
            if self.file_console:
                self.file_console.log("\n--- [Stage 2/2] Dist-PU Mixup ---")

            # Reset early stopping counter before the main stage
            if (
                self.checkpoint_handler
                and self.checkpoint_handler.early_stopping_enabled
            ):
                self.console.log(
                    "Resetting early stopping counter for Mixup stage.", style="blue"
                )
                if self.file_console:
                    self.file_console.log(
                        "Resetting early stopping counter for Mixup stage."
                    )
                self.checkpoint_handler.wait = 0
                self.checkpoint_handler.should_stop = False
            # Re-enable early stopping for the final stage
            if self.checkpoint_handler:
                try:
                    self.checkpoint_handler.early_stopping_enabled = True
                except Exception:
                    pass

            self._set_stage("mixup", self.mixup_cfg)
            # Initialize pseudo label generator
            self.pseudo_labeler = PseudoLabeler(self.model, self.device)
            self.pseudo_labeler.generate_initial_pseudo_labels(
                self.train_loader, self.device
            )
            final_metrics = self._run_epochs(
                self.mixup_cfg["epochs"], stage_name="Mixup"
            )

        # Persist results/timing/memory into results/{experiment}.json
        self.after_training()

        # Log best checkpoint metrics to file console if available
        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()

        # Close file console
        self._close_file_console()

        return final_metrics or {}
