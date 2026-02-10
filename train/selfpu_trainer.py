"""selfpu_trainer.py

SelfPUTrainer inherits from BaseTrainer and implements a faithful reproduction of the Self-PU method,
including Self-paced pseudo labeling & Mean Teacher consistency.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from typing import Any

from loss.loss_nnpu import PULoss
from .base_trainer import BaseTrainer
from backbone.ema import ModelEMA
from .train_utils import select_model


class SelfPUTrainer(BaseTrainer):
    """Self-PU learning trainer with:
    1) Self-paced pseudo labeling with dataset resampling.
    2) Mean Teacher consistency regularization.
    """

    def __init__(self, method: str, experiment: str, params: dict[str, Any]):
        super().__init__(method, experiment, params)

        # Initialize Self-PU and Mean Teacher specific parameters
        self._init_selfpu_params()
        self._prepare_selfpu_data()

        # Create second model for dual-model architecture
        self.model2 = select_model(
            method=self.method, params=self.params, prior=self.prior
        ).to(self.device)

        # Initialize bias from prior for model2 if it has a single-logit head
        try:
            import math as _math

            def _logit(_p: float) -> float:
                eps = 1e-6
                _p = max(min(float(_p), 1 - eps), eps)
                return _math.log(_p / (1.0 - _p))

            if bool(self.params.get("init_bias_from_prior", True)):
                fc2 = getattr(self.model2, "final_classifier", None)
                if (
                    isinstance(fc2, torch.nn.Linear)
                    and getattr(fc2, "bias", None) is not None
                ):
                    if int(getattr(fc2, "out_features", 0)) == 1:
                        with torch.no_grad():
                            fc2.bias.fill_(_logit(self.prior))
        except Exception:
            pass

        # Ensure dynamic models (e.g., MLPs built on first forward) have parameters before creating optimizer
        try:
            has_params_m2 = any(p.requires_grad for p in self.model2.parameters())
        except Exception:
            has_params_m2 = False
        if not has_params_m2:
            try:
                sample_batch = next(iter(self.train_loader))
                x_sample = sample_batch[0]
                if isinstance(x_sample, (list, tuple)):
                    x_sample = x_sample[0]
                with torch.no_grad():
                    _ = self.model2(x_sample.to(self.device))
            except Exception:
                pass

        # Create optimizer for second model
        lr = self.params.get("lr", 1e-3)
        wd = self.params.get("weight_decay", 5e-4)
        self.optimizer2 = torch.optim.Adam(
            self.model2.parameters(), lr=lr, weight_decay=wd
        )

        # Initialize EMA models for both student models
        self._initialize_ema_models()

        # PU loss for the unlabeled part
        self.criterion_unlabeled = PULoss(self.prior, nnpu=True, loss="sigmoid")

        # Disable early stopping before the final stage begins
        if getattr(self, "checkpoint_handler", None):
            try:
                self.checkpoint_handler.early_stopping_enabled = False
                self.checkpoint_handler.should_stop = False
            except Exception:
                pass

    def _init_selfpu_params(self):
        """Load Self-PU specific hyperparameters from the params dict."""
        # Self-paced learning
        self.sp_enabled = self.params.get("self_paced_enabled", True)
        self.sp_start_epoch = self.params.get("self_paced_start", 10)
        self.sp_update_freq = self.params.get("self_paced_frequency", 10)
        self.sp_rampup_length = self.params.get("self_paced_rampup", 80)

        # Dual model selection ratios
        self.sp_top_p1 = self.params.get("self_paced_top_p1", 0.4)  # Model 1
        self.sp_top_p2 = self.params.get("self_paced_top_p2", 0.6)  # Model 2

        # Dataset management
        self.replacement = self.params.get("replacement", True)
        self.increasing = self.params.get("increasing", True)
        self.flex_ratio = self.params.get("flex_ratio", 0.0)

        # Loss Weights
        self.pu_loss_weight = self.params.get("pu_loss_weight", 1.0)

        # Mean Teacher
        self.mt_enabled = self.params.get("mean_teacher_enabled", True)
        self.mt_start_epoch = self.params.get("mean_teacher_start", 50)
        self.ema_decay = self.params.get("ema_decay", 0.999)
        self.consistency_weight = self.params.get("consistency_weight", 1.0)
        self.consistency_rampup = self.params.get("consistency_rampup", 50)

    def _prepare_selfpu_data(self):
        """Create separate P and U loaders, similar to P3MIX and original Self-PU."""
        full_train_dataset = self.train_loader.dataset

        # Store original indices for later
        self.positive_indices = np.array(
            [i for i, label in enumerate(full_train_dataset.pu_labels) if label == 1]
        )
        self.unlabeled_indices_full = np.array(
            [i for i, label in enumerate(full_train_dataset.pu_labels) if label == -1]
        )

        # Total data size
        self.n_total = len(full_train_dataset)

        # Initialize clean and noisy datasets for both models
        # Initially, clean sets are empty and noisy sets contain all data
        self.clean_indices1 = np.array([], dtype=int)
        self.clean_indices2 = np.array([], dtype=int)
        self.noisy_indices1 = np.arange(self.n_total)
        self.noisy_indices2 = np.arange(self.n_total)

        # Initial loaders use only positive samples
        batch_size = self.params.get("batch_size", 128)

        # Create initial positive loader
        p_dataset = Subset(full_train_dataset, self.positive_indices)
        # Guard: if positive samples are too few, avoid drop_last=True causing empty iterator
        drop_last_p = True
        try:
            if len(p_dataset) < batch_size:
                drop_last_p = False
        except Exception:
            pass
        self.p_loader = DataLoader(
            p_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_p
        )

        # Initialize clean loaders as None (will be created when indices are available)
        self.clean_loader1 = None
        self.clean_loader2 = None

        # Create loaders for noisy datasets (initially all data)
        # Guard: if total data is less than one batch, also do not drop
        drop_last_n1 = True
        try:
            if len(self.noisy_indices1) < batch_size:
                drop_last_n1 = False
        except Exception:
            pass
        self.noisy_loader1 = DataLoader(
            Subset(full_train_dataset, self.noisy_indices1),
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_n1,
        )
        drop_last_n2 = True
        try:
            if len(self.noisy_indices2) < batch_size:
                drop_last_n2 = False
        except Exception:
            pass
        self.noisy_loader2 = DataLoader(
            Subset(full_train_dataset, self.noisy_indices2),
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_n2,
        )

        # Update loader for evaluating all data (no shuffle)
        self.update_loader = DataLoader(
            full_train_dataset, batch_size=1000, shuffle=False
        )

    def _initialize_ema_models(self):
        """Initializes the EMA models for both student models if enabled."""
        if self.mt_enabled:
            self.ema_model1 = ModelEMA(self, self.model, decay=self.ema_decay)
            self.ema_model2 = ModelEMA(self, self.model2, decay=self.ema_decay)
        else:
            self.ema_model1 = None
            self.ema_model2 = None

    def create_criterion(self):
        # Supervised loss for labeled positive samples
        return nn.BCEWithLogitsLoss()

    def train_one_epoch(self, epoch_idx: int):
        # Log current training phase
        if epoch_idx < self.sp_start_epoch:
            phase = "Initial (P only)"
        elif epoch_idx < self.mt_start_epoch:
            phase = "Self-paced"
        else:
            phase = "Self-paced + Mean Teacher"
        self.console.log(f"[Epoch {epoch_idx}] Training phase: {phase}")

        # Re-enable early stopping when entering the final stage
        if epoch_idx >= self.mt_start_epoch and getattr(
            self, "checkpoint_handler", None
        ):
            try:
                if not self.checkpoint_handler.early_stopping_enabled:
                    self.console.log(
                        "Enabling early stopping for final stage.", style="blue"
                    )
                    if getattr(self, "file_console", None):
                        self.file_console.log(
                            "Enabling early stopping for final stage."
                        )
                self.checkpoint_handler.early_stopping_enabled = True
                self.checkpoint_handler.wait = 0
                self.checkpoint_handler.should_stop = False
            except Exception:
                pass

        # Update pseudo-labels and resample based on schedule
        if (
            self.sp_enabled
            and epoch_idx >= self.sp_start_epoch
            and epoch_idx % self.sp_update_freq == 0
        ):
            self.console.log(
                f"Epoch {epoch_idx}: Updating self-paced pseudo-labels...",
                style="yellow",
            )
            self._update_pseudo_labels_dual_models(epoch_idx)

        self.model.train()
        self.model2.train()
        if self.ema_model1:
            self.ema_model1.ema.train()
            self.ema_model2.ema.train()

        # Determine which loaders to use based on training phase
        if epoch_idx < self.sp_start_epoch:
            # Initial phase: use only positive samples for both models
            loader1_iter = iter(self.p_loader)
            loader2_iter = iter(self.p_loader)
            noisy_loader1_iter = iter(self.noisy_loader1)
            noisy_loader2_iter = iter(self.noisy_loader2)
            num_iterations = len(self.p_loader)
        else:
            # Self-paced phase: use positive samples + selected clean samples
            # Always include positive samples in training
            if len(self.clean_indices1) > 0 and len(self.clean_indices2) > 0:
                # Create combined loaders with positive samples + clean samples
                combined_indices1 = np.concatenate(
                    [self.positive_indices, self.clean_indices1]
                )
                combined_indices2 = np.concatenate(
                    [self.positive_indices, self.clean_indices2]
                )

                combined_loader1 = DataLoader(
                    Subset(self.train_loader.dataset, combined_indices1),
                    batch_size=self.params.get("batch_size", 128),
                    shuffle=True,
                    drop_last=True,
                )
                combined_loader2 = DataLoader(
                    Subset(self.train_loader.dataset, combined_indices2),
                    batch_size=self.params.get("batch_size", 128),
                    shuffle=True,
                    drop_last=True,
                )

                loader1_iter = iter(combined_loader1)
                loader2_iter = iter(combined_loader2)
                noisy_loader1_iter = iter(self.noisy_loader1)
                noisy_loader2_iter = iter(self.noisy_loader2)
                num_iterations = min(len(combined_loader1), len(combined_loader2))
            else:
                # Fallback to positive samples if no clean samples selected yet.
                # This ensures we don't mix p_loader and noisy_loader incorrectly.
                loader1_iter = iter(self.p_loader)
                loader2_iter = iter(self.p_loader)
                noisy_loader1_iter = iter(
                    self.p_loader
                )  # Use p_loader to avoid double-counting
                noisy_loader2_iter = iter(
                    self.p_loader
                )  # Use p_loader to avoid double-counting
                num_iterations = len(self.p_loader)

        for i in range(num_iterations):
            # Get clean/positive samples
            x_clean1, t_clean1, _, _, _ = next(loader1_iter)
            x_clean2, t_clean2, _, _, _ = next(loader2_iter)

            # Get noisy samples
            try:
                x_noisy1, t_noisy1, _, _, _ = next(noisy_loader1_iter)
                x_noisy2, t_noisy2, _, _, _ = next(noisy_loader2_iter)
            except StopIteration:
                noisy_loader1_iter = iter(self.noisy_loader1)
                noisy_loader2_iter = iter(self.noisy_loader2)
                x_noisy1, t_noisy1, _, _, _ = next(noisy_loader1_iter)
                x_noisy2, t_noisy2, _, _, _ = next(noisy_loader2_iter)

            # Train Model 1
            loss1 = self._train_step(
                self.model,
                self.optimizer,
                self.ema_model1,
                x_clean1,
                t_clean1,
                x_noisy1,
                t_noisy1,
                epoch_idx,
                model_idx=1,
            )

            # Train Model 2
            loss2 = self._train_step(
                self.model2,
                self.optimizer2,
                self.ema_model2,
                x_clean2,
                t_clean2,
                x_noisy2,
                t_noisy2,
                epoch_idx,
                model_idx=2,
            )

    def _train_step(
        self,
        model,
        optimizer,
        ema_model,
        x_clean,
        t_clean,
        x_noisy,
        t_noisy,
        epoch_idx,
        model_idx,
    ):
        """Single training step for one model."""
        x_clean = x_clean.to(self.device)
        t_clean = t_clean.to(self.device)
        x_noisy = x_noisy.to(self.device)
        t_noisy = t_noisy.to(self.device)

        # Forward pass on clean samples
        logits_clean = model(x_clean).squeeze()

        # Supervised loss on clean samples
        if epoch_idx < self.sp_start_epoch:
            # Initial stage: directly use PU loss (same way as noisy), only utilize positive/unlabeled label information
            loss_clean = self.criterion_unlabeled(logits_clean, t_clean)
        else:
            # Self-paced learning stage: convert clean sample pseudo-labels to 0/1 (binary classification),
            # positive class (t_clean == 1) set to 1, pseudo-negative and unlabeled set to 0, then supervise with BCE
            binary_targets = (t_clean == 1).float()
            loss_clean = self.criterion(logits_clean, binary_targets)

        # Forward pass on noisy samples with PU loss
        logits_noisy = model(x_noisy).squeeze()
        loss_pu = self.criterion_unlabeled(logits_noisy, t_noisy)

        # Mean Teacher consistency loss
        loss_consistency = torch.tensor(0.0, device=self.device)
        if self._check_mean_teacher(epoch_idx) and ema_model:
            with torch.no_grad():
                teacher_logits_clean = ema_model.ema(x_clean).squeeze()
                teacher_logits_noisy = ema_model.ema(x_noisy).squeeze()

            # Consistency on both clean and noisy samples
            student_scores_clean = torch.sigmoid(logits_clean)
            teacher_scores_clean = torch.sigmoid(teacher_logits_clean)
            loss_consistency_clean = F.mse_loss(
                student_scores_clean, teacher_scores_clean
            )

            student_scores_noisy = torch.sigmoid(logits_noisy)
            teacher_scores_noisy = torch.sigmoid(teacher_logits_noisy)
            loss_consistency_noisy = F.mse_loss(
                student_scores_noisy, teacher_scores_noisy
            )

            loss_consistency = (loss_consistency_clean + loss_consistency_noisy) / 2

        # Total loss
        consistency_weight = self._get_consistency_weight()
        loss = (
            loss_clean
            + self.pu_loss_weight * loss_pu
            + consistency_weight * loss_consistency
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA model
        if ema_model and self._check_mean_teacher(epoch_idx):
            ema_model.update(model)

        return loss.item()

    def _check_mean_teacher(self, epoch):
        """Check if Mean Teacher should be enabled at current epoch."""
        return self.mt_enabled and epoch >= self.mt_start_epoch

    def _get_consistency_weight(self):
        """Calculate current consistency weight based on ramp-up schedule."""
        if not self.mt_enabled:
            return 0.0

        # Ramp-up from https://arxiv.org/abs/1610.02242
        current = np.clip(self.global_epoch, 0.0, self.consistency_rampup)
        phase = 1.0 - current / self.consistency_rampup
        rampup_weight = float(np.exp(-5.0 * phase * phase))

        return self.consistency_weight * rampup_weight

    def _update_pseudo_labels_dual_models(self, current_epoch: int):
        """Update pseudo labels based on dual teacher models following Self-PU strategy."""
        if not self.ema_model1 or not self.ema_model2:
            self.console.log(
                "EMA models not available, skipping self-paced update.", style="red"
            )
            return

        # Get predictions from both teacher models
        self.console.log("Getting predictions from both teacher models...")
        scores1 = self._get_all_predictions(self.ema_model1)
        scores2 = self._get_all_predictions(self.ema_model2)

        # Dynamic sampling ratio based on epoch (ramp-up if increasing=True)
        if self.increasing:
            percent = min(
                (current_epoch - self.sp_start_epoch) / self.sp_rampup_length, 1.0
            )
        else:
            percent = 1.0

        # Calculate number of samples to select
        n_unlabeled = len(self.unlabeled_indices_full)
        n_select1 = int(n_unlabeled * self.sp_top_p1 * percent)
        n_select2 = int(n_unlabeled * self.sp_top_p2 * percent)

        # Select confident samples for Model 1
        pseudo_pos_idx1, pseudo_neg_idx1 = self._select_confident_samples(
            scores1, n_select1, self.positive_indices
        )
        selected_indices1 = np.concatenate([pseudo_pos_idx1, pseudo_neg_idx1])

        # Select confident samples for Model 2
        pseudo_pos_idx2, pseudo_neg_idx2 = self._select_confident_samples(
            scores2, n_select2, self.positive_indices
        )
        selected_indices2 = np.concatenate([pseudo_pos_idx2, pseudo_neg_idx2])

        # Update clean and noisy indices
        if self.replacement:
            # Full replacement mode: clean = selected, noisy = rest
            self.clean_indices1 = selected_indices1
            self.clean_indices2 = selected_indices2
            self.noisy_indices1 = np.setdiff1d(
                np.arange(self.n_total), selected_indices1
            )
            self.noisy_indices2 = np.setdiff1d(
                np.arange(self.n_total), selected_indices2
            )
        else:
            # Incremental mode: add to existing clean set
            self.clean_indices1 = np.unique(
                np.concatenate([self.clean_indices1, selected_indices1])
            )
            self.clean_indices2 = np.unique(
                np.concatenate([self.clean_indices2, selected_indices2])
            )
            self.noisy_indices1 = np.setdiff1d(
                np.arange(self.n_total), self.clean_indices1
            )
            self.noisy_indices2 = np.setdiff1d(
                np.arange(self.n_total), self.clean_indices2
            )

        # Log selection statistics
        self._log_selection_stats(
            pseudo_pos_idx1, pseudo_neg_idx1, pseudo_pos_idx2, pseudo_neg_idx2
        )

        # Update data loaders
        self._update_data_loaders()

    def _get_all_predictions(self, ema_model):
        """Get predictions for all samples from a teacher model.
        Ensures a 1D score array aligned to dataset indices, robust for last-batch size 1.
        """
        ema_model.ema.eval()
        # Pre-allocate and fill by sample indices to avoid any ordering issues
        all_scores = np.zeros(self.n_total, dtype=np.float32)

        with torch.no_grad():
            for x, _, _, indices, _ in self.update_loader:
                x = x.to(self.device)
                logits = ema_model.ema(x)
                # Always produce 1D tensor of shape [batch]
                scores = torch.sigmoid(logits).view(-1).cpu().numpy()
                all_scores[indices.cpu().numpy()] = scores

        ema_model.ema.train()
        return all_scores

    def _select_confident_samples(self, scores, n_select, positive_indices):
        """Select confident positive and negative samples following Self-PU strategy."""
        # For Self-PU, we only select from unlabeled samples
        # The positive samples will be handled separately in training

        # Remove known positive samples from consideration
        mask = np.ones(len(scores), dtype=bool)
        mask[positive_indices] = False
        unlabeled_scores = scores[mask]
        unlabeled_indices = np.where(mask)[0]

        # Sort by score
        sorted_idx = np.argsort(unlabeled_scores)

        # Select equal number of pseudo-positives and pseudo-negatives
        n_half = n_select // 2

        # Guard: when n_select == 0, avoid Python slicing with -0 which equals 0
        if n_half <= 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        # Bound n_half to available unlabeled
        n_half = min(n_half, len(unlabeled_indices) // 2)
        if n_half <= 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        # Pseudo-negatives: lowest scores
        pseudo_neg_idx = unlabeled_indices[sorted_idx[:n_half]]

        # Pseudo-positives: highest scores
        pseudo_pos_idx = unlabeled_indices[sorted_idx[-n_half:]]

        # Only return selected pseudo-labeled samples (not including original positive samples)
        return pseudo_pos_idx, pseudo_neg_idx

    def _log_selection_stats(
        self, pseudo_pos_idx1, pseudo_neg_idx1, pseudo_pos_idx2, pseudo_neg_idx2
    ):
        """Log statistics about selected pseudo-labeled samples."""
        dataset = self.train_loader.dataset
        if not hasattr(dataset, "true_labels") or dataset.true_labels is None:
            self.console.log(
                "True labels not available, skipping selection stats.", style="yellow"
            )
            return

        def get_stats(p_idx, n_idx):
            """Calculate precision stats for pseudo-labels."""
            n_p = len(p_idx)
            prec_p = (
                np.mean([dataset.true_labels[i] == 1 for i in p_idx]) if n_p > 0 else 0
            )
            n_n = len(n_idx)
            prec_n = (
                np.mean([dataset.true_labels[i] == 0 for i in n_idx]) if n_n > 0 else 0
            )
            return n_p, prec_p, n_n, prec_n

        n_p1, prec_p1, n_n1, prec_n1 = get_stats(pseudo_pos_idx1, pseudo_neg_idx1)
        n_p2, prec_p2, n_n2, prec_n2 = get_stats(pseudo_pos_idx2, pseudo_neg_idx2)

        self.console.log(
            f"Model 1 Selection: {n_p1} pseudo-P (precision: {prec_p1:.2%}), "
            f"{n_n1} pseudo-N (precision: {prec_n1:.2%})"
        )
        self.console.log(
            f"Model 2 Selection: {n_p2} pseudo-P (precision: {prec_p2:.2%}), "
            f"{n_n2} pseudo-N (precision: {prec_n2:.2%})"
        )

    def _update_data_loaders(self):
        """Update data loaders with new indices."""
        batch_size = self.params.get("batch_size", 128)
        dataset = self.train_loader.dataset

        # Update clean loaders (handle empty clean sets)
        if len(self.clean_indices1) > 0:
            drop_last_c1 = len(self.clean_indices1) >= batch_size
            self.clean_loader1 = DataLoader(
                Subset(dataset, self.clean_indices1),
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last_c1,
            )
        else:
            self.clean_loader1 = None

        if len(self.clean_indices2) > 0:
            drop_last_c2 = len(self.clean_indices2) >= batch_size
            self.clean_loader2 = DataLoader(
                Subset(dataset, self.clean_indices2),
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last_c2,
            )
        else:
            self.clean_loader2 = None

        # Update noisy loaders; if no noisy indices, fallback to positive loader
        if len(self.noisy_indices1) > 0:
            drop_last_n1 = len(self.noisy_indices1) >= batch_size
            self.noisy_loader1 = DataLoader(
                Subset(dataset, self.noisy_indices1),
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last_n1,
            )
        else:
            self.noisy_loader1 = self.p_loader

        if len(self.noisy_indices2) > 0:
            drop_last_n2 = len(self.noisy_indices2) >= batch_size
            self.noisy_loader2 = DataLoader(
                Subset(dataset, self.noisy_indices2),
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last_n2,
            )
        else:
            self.noisy_loader2 = self.p_loader
