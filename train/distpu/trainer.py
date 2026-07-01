"""Dist-PU trainer package for PU-Bench.

Reference implementation:
    Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective
    at commit ``cb74be1a87176fd38270873c06374e53905b7354``.

Active source files are ``train.py``, ``utils.py``, ``losses/distributionLoss.py``,
``losses/entropyMinimization.py``, ``customized/mixup.py`` and
``dataTools/mixupDataset.py`` under ``author source file(s): distpu``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..base_trainer import BaseTrainer
from ..utils.reproducibility import seed_worker
from .core import (
    MixupDataset,
    distpu_entropy_loss,
    mixup_bce,
    mixup_two_targets,
)
from .losses import LabelDistributionLoss


class DistPUTrainer(BaseTrainer):
    """Dist-PU method trainer (with warm-up & mixup two stages)."""

    # Stage is set in before_training() after initialization

    # Stage switching & Criterion creation
    def _create_criterion_for_stage(self, stage_params: dict[str, Any]):
        """Return loss function based on stage parameters."""
        # Basic distribution loss
        num_bins = stage_params.get("num_bins", 1)
        base_loss = LabelDistributionLoss(
            self.prior, num_bins=num_bins, device=self.device
        )

        # If in warm-up stage, allow adding entropy loss (co_mu)
        if stage_params.get("co_mu", 0) > 0:
            co_mu = stage_params["co_mu"]

            def composite_loss(logits, labels):
                scores = torch.sigmoid(torch.clamp(logits, min=-10, max=10))
                unlabeled_scores = scores[labels == 0]
                return base_loss(logits, labels) + co_mu * distpu_entropy_loss(
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
            self._train_epoch_mixup(epoch_idx)
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
        self.scheduler.step()

    def _train_epoch_mixup(self, epoch_idx: int):
        stage_params = self.mixup_cfg
        self.model.train()

        co_entropy_base = stage_params.get("co_entropy", 0.0)
        total_mix_epochs = stage_params.get("epochs", 1)
        co_entropy = co_entropy_base * (
            1 - math.cos(((epoch_idx - 1) / total_mix_epochs) * (math.pi / 2))
        )

        for x, t, _y_true, idx, _ in self.train_loader:  # type: ignore
            x, t, idx = x.to(self.device), t.to(self.device), idx.to(self.device)

            idx_cpu = idx.detach().cpu()
            psudos = self.mixup_dataset.psudo_labels[idx_cpu].to(self.device)
            psudos[t == 1] = 1.0

            alpha = stage_params.get("alpha", 1.0)
            mixed_x, y_a, y_b, lam = mixup_two_targets(x, psudos, alpha, self.device)

            logits_orig = torch.clamp(self.model(x).squeeze(), min=-10, max=10)
            scores_orig = torch.sigmoid(logits_orig)

            logits_mix = torch.clamp(self.model(mixed_x).squeeze(), min=-10, max=10)
            scores_mix = torch.sigmoid(logits_mix)

            self.optimizer.zero_grad()

            labels_dist = (t > 0).float()
            loss_dist = self.criterion(logits_orig, labels_dist)
            loss_ent_orig = distpu_entropy_loss(scores_orig[t != 1])
            loss_ent_mix = distpu_entropy_loss(scores_mix)
            loss_mix_ce = mixup_bce(scores_mix, y_a, y_b, lam)

            total_loss = (
                loss_dist
                + co_entropy * loss_ent_orig
                + stage_params.get("co_mix_entropy", 0.0) * loss_ent_mix
                + stage_params.get("co_mixup", 0.0) * loss_mix_ce
            )
            total_loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.mixup_dataset.psudo_labels[idx_cpu] = scores_orig.detach().cpu()
        self.scheduler.step()

    def _loader_worker_kwargs(self) -> dict[str, Any]:
        num_workers = int(self.params.get("num_workers", 0))
        pin_memory = bool(self.params.get("pin_memory", torch.cuda.is_available()))
        kwargs: dict[str, Any] = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "worker_init_fn": seed_worker,
        }
        if num_workers > 0 and "persistent_workers" in self.params:
            kwargs["persistent_workers"] = bool(self.params["persistent_workers"])
        if num_workers > 0 and "prefetch_factor" in self.params:
            kwargs["prefetch_factor"] = int(self.params["prefetch_factor"])
        return kwargs

    def _make_loader(self, dataset, *, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **self._loader_worker_kwargs(),
        )

    def _align_source_loader_contract(self) -> None:
        """Use the Dist-PU source batch contract without changing datasets."""
        train_batch = int(self.params.get("batch_size", 256))
        eval_batch = int(self.params.get("test_batch_size", 128))
        self.train_loader = self._make_loader(
            self.train_loader.dataset,
            batch_size=train_batch,
            shuffle=True,
        )
        if self.validation_loader is not None:
            self.validation_loader = self._make_loader(
                self.validation_loader.dataset,
                batch_size=eval_batch,
                shuffle=False,
            )
        self.test_loader = self._make_loader(
            self.test_loader.dataset,
            batch_size=eval_batch,
            shuffle=False,
        )

    # Multi-stage control
    def before_training(self):
        # First call parent's initialization (including file console)
        super().before_training()

        if "stages" not in self.params:
            raise ValueError("DistPU requires `stages` configuration in params.")
        self.warm_up_cfg = self.params["stages"].get("warm_up", {})
        self.mixup_cfg = self.params["stages"].get("mixup", {})
        self._align_source_loader_contract()
        if (
            self.checkpoint_handler is not None
            and str(getattr(self.checkpoint_handler, "monitor", "")).startswith("val_")
            and self.validation_loader is None
        ):
            raise ValueError(
                "Dist-PU benchmark checkpointing requires a validation loader when monitoring val_* metrics."
            )
        self._early_stopping_initially_enabled = (
            bool(getattr(self.checkpoint_handler, "early_stopping_enabled", False))
            if self.checkpoint_handler
            else False
        )

        # Add DistPU-specific configuration info to logs
        if self.file_console:
            self.file_console.log("Dist-PU source alignment:")
            self.file_console.log(
                "  source=Ray-rui/Dist-PU@cb74be1a87176fd38270873c06374e53905b7354"
            )
            self.file_console.log(
                f"  model={self.model.__class__.__name__}, "
                f"params={sum(p.numel() for p in self.model.parameters()) / 1e6:.3f}M, "
                "backbone=shared_public"
            )
            self.file_console.log(
                f"  batch_size={self.params.get('batch_size', 256)}, "
                f"test_batch_size={self.params.get('test_batch_size', 128)}, "
                f"num_workers={self.params.get('num_workers', 0)}, "
                f"pin_memory={self.params.get('pin_memory', torch.cuda.is_available())}"
            )
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

        lr = stage_params.get("lr", self.params.get("lr", 1e-3))
        wd = stage_params.get("weight_decay", self.params.get("weight_decay", 5e-4))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=wd
        )
        if stage_name == "mixup":
            eta_min = float(stage_params.get("eta_min", 0.7 * lr))
            t_max = int(
                stage_params.get("scheduler_t_max", stage_params.get("epochs", 1))
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, t_max, eta_min=eta_min
            )
        else:
            t_max = int(
                stage_params.get(
                    "scheduler_t_max",
                    self.mixup_cfg.get("epochs", stage_params.get("epochs", 1)),
                )
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, t_max
            )
        self.criterion = self._create_criterion_for_stage(stage_params)

    def _reset_checkpoint_for_mixup(self) -> None:
        self.reset_checkpoint_tracking()

    def _make_sequential_train_loader(self) -> DataLoader:
        return self._make_loader(
            self.train_loader.dataset,
            batch_size=int(self.params.get("test_batch_size", 128)),
            shuffle=False,
        )

    # Override run() to implement multi-stage training workflow
    def run(self):
        # Initialize stage configuration
        self.before_training()
        final_metrics = None
        try:
            # Warm-up
            if self.warm_up_cfg and self.warm_up_cfg.get("epochs", 0) > 0:
                print("\n--- [Stage 1/2] Dist-PU Warm-up ---")
                if self.file_console:
                    self.file_console.log("\n--- [Stage 1/2] Dist-PU Warm-up ---")
                self._set_stage("warm_up", self.warm_up_cfg)
                with self.suspend_checkpointing():
                    final_metrics = self.run_stage(
                        "Warm-up", self.warm_up_cfg["epochs"]
                    )

            # Mixup stage
            if self.mixup_cfg and self.mixup_cfg.get("epochs", 0) > 0:
                print("\n--- [Stage 2/2] Dist-PU Mixup ---")
                if self.file_console:
                    self.file_console.log("\n--- [Stage 2/2] Dist-PU Mixup ---")

                if self.checkpoint_handler and self._early_stopping_initially_enabled:
                    self.console.log(
                        "Resetting early stopping counter for Mixup stage.",
                        style="blue",
                    )
                    if self.file_console:
                        self.file_console.log(
                            "Resetting early stopping counter for Mixup stage."
                        )
                self.set_checkpoint_early_stopping(
                    self._early_stopping_initially_enabled, reset=True
                )

                self._set_stage("mixup", self.mixup_cfg)
                self._reset_checkpoint_for_mixup()
                self.mixup_dataset = MixupDataset()
                self.mixup_dataset.update_psudos(
                    self._make_sequential_train_loader(), self.model, self.device
                )
                final_metrics = self.run_stage("Mixup", self.mixup_cfg["epochs"])
                self.distpu_mixup_final_metrics = final_metrics

            return final_metrics or {}
        finally:
            self.finalize()

    def _compose_result_summary(self) -> dict:
        result = super()._compose_result_summary()
        if hasattr(self, "distpu_mixup_final_metrics"):
            result["distpu_source_notes"] = {
                "best_scope": "mixup_stage_only",
                "benchmark_monitor": (
                    getattr(self.checkpoint_handler, "monitor", None)
                    if self.checkpoint_handler
                    else None
                ),
                "mixup_final_test_metrics": self.distpu_mixup_final_metrics,
            }
        return result
