from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from .base_trainer import BaseTrainer
from .train_utils import seed_worker
from loss.loss_pulda import (
    PULDALabelDistributionLoss,
    TwoWaySigmoidLoss,
    PULDALabelDistributionLossWithEMA,
    TwoWaySigmoidLossWithEMA,
)


class PULDATrainer(BaseTrainer):
    """PULDA method trainer.

    Stage 1 (warm-up): standard PN training over available PU labels mapped to {0,1}.
    Stage 2 (alignment): Label Distribution Alignment + Mixup on pseudo scores.
    """

    def before_training(self):
        super().before_training()

        # --- Recalculate prior from the actual training set ---
        # This is critical for case-control scenarios where the training prior is
        # intentionally shifted and differs from the validation/test prior.
        try:
            actual_train_prior = (
                self.train_loader.dataset.true_labels.float().mean().item()
            )
            # Log the change if it's significant
            if abs(actual_train_prior - self.prior) > 1e-4:
                self.console.log(
                    f"Overriding initial prior {self.prior:.4f} with actual training set prior {actual_train_prior:.4f}",
                    style="bold yellow",
                )
                self.prior = actual_train_prior
        except Exception as e:
            self.console.log(
                f"Could not determine actual training prior, falling back to config prior. Reason: {e}",
                style="bold red",
            )

        # Build PULDA losses (following original EMA/non-EMA selection)
        self.use_ema = bool(self.params.get("EMA", 1))  # Default to EMA enabled
        temperature = self.params.get("tmpr", 3.5)

        if self.use_ema:
            self.base_loss = PULDALabelDistributionLossWithEMA(
                prior=self.prior,
                temperature=temperature,
                alpha_u=self.params.get("alpha_U", 0.85),
            )
        else:
            self.base_loss = PULDALabelDistributionLoss(
                self.prior, temperature=temperature
            )

        self.use_two_way = bool(self.params.get("two_way", 1))
        self.two_way_loss = None
        if self.use_two_way:
            # For case-control, the prior is high, and the default margin might be too small
            # to prevent collapse. We use a larger default margin in this specific scenario.
            default_margin = (
                0.95 if self.params.get("scenario") == "case-control" else 0.6
            )
            margin = float(self.params.get("margin", default_margin))
            if (
                self.params.get("scenario") == "case-control"
                and self.params.get("margin") is None
            ):
                self.console.log(
                    f"Scenario is 'case-control', using a larger default margin of {margin}",
                    style="yellow",
                )

            if self.use_ema:
                self.two_way_loss = TwoWaySigmoidLossWithEMA(
                    self.prior,
                    margin,
                    float(self.params.get("tmpr", 3.5)),
                    float(self.params.get("alpha_CN", 0.5)),
                )
            else:
                self.two_way_loss = TwoWaySigmoidLoss(
                    self.prior,
                    margin,
                    float(self.params.get("tmpr", 3.5)),
                )

        self.warmup_epochs = int(self.params.get("warm_up_epochs", 60))
        self.align_epochs = int(
            self.params.get(
                "pu_epochs",
                max(1, self.params.get("num_epochs", 60) - self.warmup_epochs),
            )
        )
        self.alpha = float(self.params.get("alpha", 11.0))
        self.co_mixup = float(self.params.get("co_mixup", 4.2))

        # For pseudo-label management
        self.pseudo_labels = None
        self.alignment_stage_initialized = False

        # Stage-specific learning parameters (following original implementation)
        self.warmup_lr = self.params.get("warm_up_lr", 1e-4)
        self.warmup_wd = self.params.get("warm_up_weight_decay", 5e-4)
        self.align_lr = self.params.get("lr", 1e-3)
        self.align_wd = self.params.get("weight_decay", 1e-4)

        # Optional: restore original P/U resampling strategy (fixed P and U per batch)
        # If enabled, rebuild train_loader with a batch sampler that yields
        # batches containing P_batch_size positives and U_batch_size unlabeled
        if int(self.params.get("resample", 1)) == 1:
            p_bs = int(self.params.get("P_batch_size", 16))
            u_bs = int(self.params.get("U_batch_size", 128))

            class _PUSampler(Sampler[list[int]]):
                def __init__(self, p_indices, u_indices, p_batch_size, u_batch_size):
                    self.p_indices = p_indices
                    self.u_indices = u_indices
                    self.p_batch_size = int(p_batch_size)
                    self.u_batch_size = int(u_batch_size)

                def __iter__(self):
                    # Positive iterator repeats forever; unlabeled iterates once per epoch
                    import numpy as _np

                    # fresh shuffles each epoch
                    p_perm = _np.random.permutation(self.p_indices)
                    u_perm = _np.random.permutation(self.u_indices)

                    # cycle positives if depleted
                    p_cursor = 0
                    for u_start in range(0, len(u_perm), self.u_batch_size):
                        u_batch = u_perm[u_start : u_start + self.u_batch_size]
                        # take p_batch_size from p_perm, wrap-around if needed
                        if p_cursor + self.p_batch_size > len(p_perm):
                            # reshuffle and wrap
                            remain = p_perm[p_cursor:]
                            p_perm = _np.random.permutation(self.p_indices)
                            need = self.p_batch_size - len(remain)
                            p_batch = _np.concatenate([remain, p_perm[:need]])
                            p_cursor = need
                        else:
                            p_batch = p_perm[p_cursor : p_cursor + self.p_batch_size]
                            p_cursor += self.p_batch_size

                        yield _np.concatenate([p_batch, u_batch]).tolist()

                def __len__(self) -> int:
                    return max(1, len(self.u_indices) // self.u_batch_size)

            # Extract indices by PU label from the underlying dataset
            base_ds = self.train_loader.dataset
            try:
                pu = base_ds.pu_labels
                p_indices = (pu == 1).nonzero().squeeze().cpu().numpy()
                u_indices = (pu != 1).nonzero().squeeze().cpu().numpy()
            except Exception as _e:
                # Fallback: keep original loader on failure
                p_indices = None
                u_indices = None

            if p_indices is not None and u_indices is not None:
                sampler = _PUSampler(p_indices, u_indices, p_bs, u_bs)
                self.train_loader = DataLoader(
                    base_ds,
                    batch_sampler=sampler,
                    num_workers=self.params.get("num_workers", 4),
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                )

    def create_criterion(self):
        # We compute losses manually
        return torch.nn.Identity()

    def run(self):
        self.before_training()

        # Stage 1: Warm-up (following original setup)
        if self.warmup_epochs > 0:
            # Disable early stopping during warm-up
            if self.checkpoint_handler:
                try:
                    self.checkpoint_handler.early_stopping_enabled = False
                    self.checkpoint_handler.should_stop = False
                except Exception:
                    pass
            self._setup_warmup_stage()
            self._run_epochs(self.warmup_epochs, stage_name="Warm-up")

        # Stage transition: reset optimizer and scheduler for alignment stage
        if self.align_epochs > 0 and not self.alignment_stage_initialized:
            self._setup_alignment_stage()
            self.alignment_stage_initialized = True

        # Stage 2: Alignment + Mixup
        if self.align_epochs > 0:
            # Re-enable early stopping for alignment stage first
            if self.checkpoint_handler:
                try:
                    self.checkpoint_handler.early_stopping_enabled = True
                except Exception:
                    pass
            
            # Reset early stopping counter before the main stage
            if (
                self.checkpoint_handler
                and self.checkpoint_handler.early_stopping_enabled
            ):
                self.console.log(
                    "Resetting early stopping counter for Alignment stage.",
                    style="blue",
                )
                if self.file_console:
                    self.file_console.log(
                        "Resetting early stopping counter for Alignment stage."
                    )
                self.checkpoint_handler.wait = 0
                self.checkpoint_handler.should_stop = False

            self._run_epochs(self.align_epochs, stage_name="Alignment")

        self.after_training()
        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()
        self._close_file_console()

    def _setup_warmup_stage(self):
        """Setup optimizer and scheduler for warm-up stage (following original)."""
        self.console.log("Setting up warm-up stage...", style="yellow")

        # Create optimizer with warm-up parameters
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.warmup_lr, weight_decay=self.warmup_wd
        )

        # Create scheduler for warm-up (CosineAnnealingLR)
        # Original implementation uses T_max=pu_epochs for warm-up scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.align_epochs
        )

    def _setup_alignment_stage(self):
        """Setup optimizer, scheduler and pseudo-labels for alignment stage."""
        self.console.log("Setting up alignment stage...", style="yellow")

        # Initialize pseudo-labels first
        self._initialize_pseudo_labels()

        # Create new optimizer with alignment parameters (following original)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.align_lr, weight_decay=self.align_wd
        )

        # Create scheduler for alignment stage (CosineAnnealingLR with eta_min)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.align_epochs, eta_min=0.7 * self.align_lr
        )

    def train_one_epoch(self, epoch_idx: int):
        # Use global_epoch to determine stage, not stage-local epoch_idx
        # This ensures correct stage detection even when _run_epochs is called multiple times
        # Note: global_epoch is incremented at the start of each epoch in _run_epochs,
        # so after warmup_epochs, global_epoch equals warmup_epochs, and we should switch to alignment
        if self.global_epoch <= self.warmup_epochs:
            self._train_epoch_warmup()
        else:
            self._train_epoch_align()

    # -------------------- Warm-up --------------------
    def _train_epoch_warmup(self):
        self.model.train()
        # Use standard BCE loss for warm-up to avoid gradient vanishing issue
        # The Label Distribution Loss has very small gradients due to expectation
        # over the whole batch, making it hard to train from scratch.
        bce_loss = torch.nn.BCEWithLogitsLoss()
        
        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)
            pu_binary = (t == 1).float()  # P->1, U->0
            self.optimizer.zero_grad()
            # Original warm-up does not clamp logits
            logits = self.model(x).view(-1)

            # Use BCE loss for warm-up instead of PULDA loss
            loss = bce_loss(logits, pu_binary)

            loss.backward()
            self.optimizer.step()

        # Update scheduler at the end of epoch (following original)
        if hasattr(self, "scheduler") and self.scheduler:
            self.scheduler.step()

    # -------------------- Alignment + Mixup --------------------
    def _initialize_pseudo_labels(self):
        """Compute initial pseudo-labels for the entire training set."""
        self.console.log(
            "Initializing pseudo-labels for alignment stage...", style="yellow"
        )
        self.model.eval()
        num_samples = len(self.update_loader.dataset)
        self.pseudo_labels = torch.zeros(
            num_samples, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            for x, _, _, idx, _ in self.update_loader:
                x, idx = x.to(self.device), idx.to(self.device)
                logits = self.model(x).view(-1)
                scores = torch.sigmoid(logits)
                self.pseudo_labels[idx] = scores
        self.model.train()
        self.console.log("Pseudo-labels initialized.", style="green")

    def _train_epoch_align(self):
        self.model.train()
        for x, t, _y_true, idx, _ in self.train_loader:  # type: ignore
            x, t, idx = x.to(self.device), t.to(self.device), idx.to(self.device)

            # --- Logic restored to match original implementation ---
            # 1. Prepare targets for Mixup using OLD pseudo-labels from previous step
            # .clone() is critical to avoid modifying the master tensor for P samples
            pseudo_targets = self.pseudo_labels[idx].clone()
            is_p = t == 1
            pseudo_targets[is_p] = 1.0

            # 2. Perform Mixup
            lam = (
                torch.distributions.beta.Beta(self.alpha, self.alpha)
                .sample()
                .to(self.device)
            )
            perm = torch.randperm(x.size(0), device=self.device)
            x_mix = lam * x + (1.0 - lam) * x[perm]
            y_a = pseudo_targets
            y_b = pseudo_targets[perm]

            # 3. Forward pass on non-mixed data (for alignment loss and pseudo-label update)
            logits_orig = torch.clamp(self.model(x).view(-1), min=-10, max=10)

            # 4. Forward pass on mixed data (for mixup loss)
            logits_mix = torch.clamp(self.model(x_mix).view(-1), min=-10, max=10)

            # 5. Calculate losses
            # Alignment loss uses the original logits
            pu_binary = (t == 1).float()
            loss_align = self.base_loss(logits_orig, pu_binary)
            if self.use_two_way and self.two_way_loss is not None:
                loss_align = loss_align + self.two_way_loss(logits_orig, pu_binary)

            # Mixup loss uses the mixed logits and targets from old pseudo-labels
            scores_mix = torch.sigmoid(logits_mix)
            loss_mix = lam * F.binary_cross_entropy(
                scores_mix, y_a, reduction="none"
            ) + (1.0 - lam) * F.binary_cross_entropy(scores_mix, y_b, reduction="none")
            loss_mix = loss_mix.mean()

            loss = loss_align + self.co_mixup * loss_mix

            # 6. Optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 7. Update pseudo-labels for the current batch for the NEXT iteration
            with torch.no_grad():
                current_scores = torch.sigmoid(logits_orig.detach())
                self.pseudo_labels[idx] = current_scores

        # Update scheduler at the end of epoch (following original)
        if hasattr(self, "scheduler") and self.scheduler:
            self.scheduler.step()
