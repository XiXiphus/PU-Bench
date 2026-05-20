"""PUL-CPBF trainer.

Source note:
- Author snapshot:
  ``author source file(s): pulcpbf_author_provided``.
- Active source files: ``warmup.py`` for phase 1, ``train.py`` for phase 2,
  ``creditcard.py`` for the trend partition, and ``dataset/randaugment.py``
  plus image dataset files for weak/strong augmentation.

Core workflow (two stages):
- Stage 1 (warming-up): P vs U training, using logistic/sigmoid surrogate, negative class terms for unlabeled,
  and supporting co-entropy (probability-based entropy minimization).
- Stage 2 (fine-tuning): Based on pseudo-labels from stage 1 (two strategies: trend/alpha_range),
  perform supervised and consistency training on weak/strong augmented samples, with co-entropy added.

The trainer keeps the active source losses, pseudo-label structure, and
weak/strong image augmentation inside PU-Bench's controlled data split and
public-backbone policy. Vector/tabular datasets use a benchmark-specific
weak/strong extension, so they are not exact reproductions of the source image
recipe.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Dict

from ..base_trainer import BaseTrainer
from ..metrics import evaluate_metrics
from ..reproducibility import seed_worker
from ..augmentations.vector import (
    VectorAugPUDatasetWrapper,
    VectorWeakAugment,
    VectorStrongAugment,
)
from .augment import (
    PULCPBFDatasetWrapper,
    PULCPBFEvalDatasetWrapper,
    TransformPULCPBF,
    TransformPULCPBFEval,
    infer_image_profile,
)


def _select_loss(loss_name: str):
    """Match PULCPBF warmup surrogate losses.

    logistic: softplus(-x)
    sigmoid:  sigmoid(-x)
    """
    losses = {
        "logistic": lambda x: F.softplus(-x),
        "sigmoid": lambda x: torch.sigmoid(-x),
        "CE": lambda x: x,
    }
    if loss_name not in losses:
        raise ValueError(f"Unsupported loss: {loss_name}")
    return losses[loss_name]


def _entropy_minimization(scores: torch.Tensor) -> torch.Tensor:
    """Entropy minimization on probabilities (scores in [0,1])."""
    eps = 1e-12
    scores = torch.clamp(scores, eps, 1 - eps)
    return -(scores * torch.log(scores) + (1 - scores) * torch.log(1 - scores)).mean()


class PULCPBFTrainer(BaseTrainer):
    """PULCPBF method trainer."""

    def __init__(self, method: str, experiment: str, params: dict):
        super().__init__(method, experiment, params)
        # Stage settings
        self.current_phase = 1
        self.phase1_epochs = int(
            self.params.get("phase1_epochs", self.params.get("warming_epochs", 10))
        )
        self.phase2_epochs = int(
            self.params.get("phase2_epochs", self.params.get("epochs", 50))
        )

        # Optimization related
        self.lr = float(self.params.get("lr", 2e-3))
        self.weight_decay = float(self.params.get("weight_decay", 5e-3))
        self.momentum = float(self.params.get("momentum", 0.9))
        self.nesterov = bool(self.params.get("nesterov", True))
        self.warmup_steps = int(self.params.get("warmup_steps", 0))

        # Loss/regularization related
        self.co_entropy = float(self.params.get("co_entropy", 0.0))
        self.lambda_u = float(
            self.params.get("lambda_u", 0.85)
        )  # Consistent with original implementation
        self.mask_threshold = float(self.params.get("mask_threshold", 0.9))
        self.temperature_T = float(self.params.get("T", 0.5))
        self.loss_name = str(self.params.get("loss", "logistic"))

        # Benchmark conveniences are opt-in. The source active warmup has no
        # positive boost, beta floor, or prior-matching regularizer.
        self.warmup_pos_boost = float(self.params.get("warmup_pos_boost", 1.0))
        self.beta_min = float(self.params.get("beta_min", 0.0))
        self.prior_reg_weight = float(self.params.get("prior_reg_weight", 0.0))
        self.phase2_lamda = float(self.params.get("phase2_lamda", 0.5))
        self.mask_threshold_start = float(
            self.params.get("mask_threshold_start", self.mask_threshold)
        )
        self.mask_threshold_end = float(
            self.params.get("mask_threshold_end", self.mask_threshold)
        )
        self.T_start = float(self.params.get("T_start", self.temperature_T))
        self.T_end = float(self.params.get("T_end", self.temperature_T))
        self.co_entropy_ramp_end = int(self.params.get("co_entropy_ramp_end", 0))

        # warmup alpha / beta settings
        self.alpha = float(self.params.get("alpha", 0.5))

        # Pseudo-label strategy
        self.pseudo_label_strategy = str(
            self.params.get("pseudo_label_strategy", "trend")
        ).lower()
        self.alpha_list = list(
            self.params.get("alpha_list", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        )
        self.use_three_sigma = bool(self.params.get("use_three_sigma", False))

        # Stage 2 data augmentation
        self.batch_size = int(self.params.get("batch_size", 64))

        # Pseudo-label cache: index -> {0,1}
        self.pseudo_labels_map: Dict[int, int] | None = None

        # Record U sample prediction probabilities for each epoch in stage 1 (for trend strategy)
        self._phase1_unlabeled_scores: list[np.ndarray] = []
        # Global U indices and position mapping required for trend alignment
        self._trend_u_indices_sorted = None  # type: ignore
        self._trend_u_index_to_pos = None  # type: ignore

        # Model expected input channels (auto-detected and used for input channel alignment)
        self._model_in_channels: int | None = None
        # Whether it is image data flag (detected and cached at Phase-1 startup)
        self._is_image_like: bool | None = None
        # Generic scheduler placeholder (Phase-1 doesn't use, avoid attribute missing)
        self.scheduler = None

    def _ensure_channel_match(self, x: torch.Tensor) -> torch.Tensor:
        """Align input x channel count to model first layer Conv2d in_channels.

        - If model expects 3 channels and x has 1 channel, repeat to 3 channels.
        - If model expects 1 channel and x has 3 channels, take first channel.
        Other cases unchanged.
        """
        if not (isinstance(x, torch.Tensor) and x.dim() == 4):
            return x
        if self._model_in_channels is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    self._model_in_channels = int(m.in_channels)
                    break
        if self._model_in_channels is None:
            return x
        in_c = x.size(1)
        exp_c = self._model_in_channels
        if in_c == exp_c:
            out = x
        elif exp_c == 3 and in_c == 1:
            out = x.repeat(1, 3, 1, 1)
        elif exp_c == 1 and in_c == 3:
            out = x[:, 0:1, ...]
        else:
            out = x
        return out

    # ---------------- BaseTrainer Interface ----------------
    def create_criterion(self):
        # Binary classification BCE with logits (stage 2 main loss)
        return nn.BCEWithLogitsLoss()

    def train_one_epoch(self, epoch_idx: int):
        if self.current_phase == 1:
            self._train_epoch_phase1(epoch_idx)
            # Record unlabeled sample score sequence for this epoch (for trend strategy)
            self._record_unlabeled_scores_for_trend()
        elif self.current_phase == 2:
            self._train_epoch_phase2(epoch_idx)
        else:
            raise ValueError(f"Unknown phase: {self.current_phase}")

    def run(self):
        self._install_image_source_transforms()

        # Stage 1
        self.before_training()
        # Disable early stopping during Phase-1 warming-up
        self.set_checkpoint_early_stopping(False)
        self._run_phase1()

        # Generate stage 2 pseudo-labels
        self._generate_pseudo_labels()

        # Wrap training data (weak/strong augmentation)
        self._ensure_phase2_train_dataset()

        # Reinitialize optimizer/scheduler
        self._init_optimizer_phase2()
        self.current_phase = 2

        # Reset early stopping counter before entering final stage
        if self.checkpoint_handler:
            self.console.log(
                "Resetting early stopping counter for fine-tuning stage.", style="blue"
            )
            if self.file_console:
                self.file_console.log(
                    "Resetting early stopping counter for fine-tuning stage."
                )
        # Re-enable early stopping for Phase-2
        self.set_checkpoint_early_stopping(True, reset=True)

        # Stage 2 training
        final_metrics = self.run_stage("Fine-tuning", self.phase2_epochs)

        # End
        self.finalize()
        return final_metrics

    # ---------------- Stage 1: Warming-up ----------------
    def _run_phase1(self):
        # If non-image (vector/text) data and current model is CNN, switch to generic MLP
        try:
            _batch = next(iter(self.train_loader))
            _x0 = _batch[0]
            if isinstance(_x0, (list, tuple)):
                _x0 = _x0[0]
            is_image_like = isinstance(_x0, torch.Tensor) and _x0.dim() >= 3
        except Exception:
            is_image_like = False
        self._is_image_like = is_image_like
        if not is_image_like:
            has_conv2d = any(isinstance(m, nn.Conv2d) for m in self.model.modules())
            if has_conv2d:
                from backbone.models import MLP_20News

                self.model = MLP_20News(prior=getattr(self, "prior", 0.0))
                self.model.to(self.device)
        if bool(self.params.get("init_bias_from_prior", False)):
            try:
                self._init_bias_from_prior()
            except Exception:
                pass

        # Use generic epoch runner to execute and print Phase-1 (Warming-up) metrics
        self.current_phase = 1
        _ = self._run_epochs(self.phase1_epochs, stage_name="Warming-up")

    def _train_epoch_phase1(self, epoch_idx: int):
        self.model.train()
        loss_fn = _select_loss(self.loss_name)

        # Estimate prior_ (labeled positive ratio)
        ds = self.train_loader.dataset
        pu = ds.pu_labels
        prior_labeled_ratio = (pu == 1).float().mean().item()
        # Align with PULCPBF beta definition
        beta = (
            self.alpha
            * prior_labeled_ratio
            / (
                self.prior
                - self.prior * prior_labeled_ratio
                + self.alpha * prior_labeled_ratio
                + 1e-12
            )
        )
        # Fallback to avoid giving no gradient to P at all
        beta = float(min(0.99, max(self.beta_min, beta)))

        total_loss = 0.0
        num_batches = 0

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            if isinstance(x, (list, tuple)):
                x = x[0]
            # Align input to model expected channels and size (AlzheimerMRI is 1x128x128)
            x = self._ensure_channel_match(x)
            x, t = x.to(self.device), t.to(self.device)

            # Split P / U
            p_mask = t == 1
            u_mask = t == -1
            if not (p_mask.any() or u_mask.any()):
                continue

            self.optimizer.zero_grad()
            logits = self.model(x).view(-1)
            logits_clamped = torch.clamp(logits, -10.0, 10.0)

            # Lx: surrogate for positive examples (P) (per original implementation: loss_fn(logits_x_w).mean())
            Lx = torch.tensor(0.0, device=self.device)
            if p_mask.any():
                Lx = loss_fn(logits_clamped[p_mask]).mean()
                # Original coefficient + extra boost for amplifying supervision signal when LP is very few
                Lx = Lx * (1.0 - beta) * prior_labeled_ratio * self.warmup_pos_boost

            # Ln: negative class surrogate for unlabeled (U): loss_fn(-logits_u_w)
            Ln = torch.tensor(0.0, device=self.device)
            if u_mask.any():
                Ln = loss_fn(-logits_clamped[u_mask]).mean()
                Ln = Ln * beta * (1.0 - prior_labeled_ratio)

            # Entropy minimization: only for unlabeled sample (U) probabilities
            loss_ent = torch.tensor(0.0, device=self.device)
            if self.co_entropy > 0 and u_mask.any():
                probs_u = torch.sigmoid(logits_clamped[u_mask])
                loss_ent = self.co_entropy * _entropy_minimization(probs_u)

            # Prior matching regularization: constrain average prediction probability close to training prior π
            prob_mean = torch.sigmoid(logits_clamped).mean()
            loss_prior = self.prior_reg_weight * (prob_mean - float(self.prior)) ** 2

            loss = Lx + Ln + loss_ent + loss_prior
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        if self.file_console and num_batches > 0:
            self.file_console.log(
                f"Phase1[{epoch_idx}/{self.phase1_epochs}] - Loss: {total_loss/max(1,num_batches):.4f}, beta={beta:.4f}, prior_reg={self.prior_reg_weight:.3f}, pos_boost={self.warmup_pos_boost:.2f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

    def _record_unlabeled_scores_for_trend(self):
        if self.pseudo_label_strategy != "trend":
            return
        self.model.eval()
        # Initialize global U index order and mapping (once only)
        if self._trend_u_indices_sorted is None or self._trend_u_index_to_pos is None:
            ds = self.train_loader.dataset
            try:
                pu = getattr(ds, "pu_labels", None)
                if pu is None:
                    unlabeled_idx = None
                else:
                    if hasattr(pu, "detach"):
                        pu_np = pu.detach().cpu().numpy()
                    else:
                        pu_np = np.array(pu)
                    mask = pu_np == -1
                    idx_arr = getattr(ds, "indices", None)
                    if idx_arr is None:
                        unlabeled_idx = np.arange(len(ds))[mask]
                    else:
                        if hasattr(idx_arr, "detach"):
                            idx_np = idx_arr.detach().cpu().numpy()
                        else:
                            idx_np = np.array(idx_arr)
                        unlabeled_idx = idx_np[mask]
            except Exception:
                unlabeled_idx = None
            if unlabeled_idx is None:
                return
            sorted_idx = np.sort(unlabeled_idx)
            self._trend_u_indices_sorted = sorted_idx
            self._trend_u_index_to_pos = {
                int(i): pos for pos, i in enumerate(sorted_idx)
            }

        epoch_scores = np.full(
            len(self._trend_u_indices_sorted), np.nan, dtype=np.float64
        )
        with torch.no_grad():
            for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = self._ensure_channel_match(x)
                x, t = x.to(self.device), t.to(self.device)
                # Establish GPU and CPU mask sets: x uses GPU mask; _idx (on CPU) uses CPU mask
                u_mask_gpu = t == -1
                if not u_mask_gpu.any():
                    continue
                u_mask_cpu = u_mask_gpu.detach().cpu()
                batch_idx_u = _idx[u_mask_cpu].detach().cpu().numpy().astype(int)
                logits_u = self.model(x[u_mask_gpu]).view(-1)
                scores = torch.sigmoid(logits_u).detach().cpu().numpy()
                for i_val, s_val in zip(batch_idx_u, scores):
                    pos = self._trend_u_index_to_pos.get(int(i_val))  # type: ignore
                    if pos is not None:
                        epoch_scores[pos] = float(s_val)
        # If missing, fill with previous round value or 0.5 to ensure consistent length
        if np.isnan(epoch_scores).any():
            if self._phase1_unlabeled_scores:
                prev = self._phase1_unlabeled_scores[-1]
                mask = np.isnan(epoch_scores)
                if len(prev) == len(epoch_scores):
                    epoch_scores[mask] = prev[mask]
                else:
                    epoch_scores[mask] = 0.5
            else:
                epoch_scores[np.isnan(epoch_scores)] = 0.5
        self._phase1_unlabeled_scores.append(epoch_scores)

    # ---------------- Stage 2: Pseudo-labels + Consistency Training ----------------
    def _init_optimizer_phase2(self):
        phase2_optimizer = str(self.params.get("phase2_optimizer", "adam")).lower()
        if phase2_optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif phase2_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=self.nesterov,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported PULCPBF phase2_optimizer: {phase2_optimizer}")
        self.scheduler = None

        # Main loss (BCE with logits)
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.criterion_u = nn.BCEWithLogitsLoss(reduction="none")

    def _train_epoch_phase2(self, epoch_idx: int):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        labeled_loss_sum, unlabeled_loss_sum = 0.0, 0.0

        lamda = self.phase2_lamda
        prog = float(epoch_idx) / max(1.0, float(self.phase2_epochs))
        cur_mask_th = (
            self.mask_threshold_start
            + (self.mask_threshold_end - self.mask_threshold_start) * prog
        )
        cur_T = self.T_start + (self.T_end - self.T_start) * prog
        if self.co_entropy_ramp_end > 0:
            cur_co_entropy = self.co_entropy * min(
                1.0, float(epoch_idx) / max(1, self.co_entropy_ramp_end)
            )
        else:
            cur_co_entropy = self.co_entropy

        for (x_w, x_s), t, y_true, idx, _ in self.train_loader:  # type: ignore
            x_w = self._ensure_channel_match(x_w)
            x_s = self._ensure_channel_match(x_s)
            x_w, x_s = x_w.to(self.device), x_s.to(self.device)
            t, y_true, idx = (
                t.to(self.device),
                y_true.to(self.device),
                idx.to(self.device),
            )

            labeled_mask = t == 1
            unlabeled_mask = t == -1

            loss_l = torch.tensor(0.0, device=self.device)
            loss_u = torch.tensor(0.0, device=self.device)
            entropy_logits = []

            # Supervised loss (labeled positive samples on weak augmentation; target uses true label)
            if labeled_mask.any():
                logits_x = self.model(x_w[labeled_mask]).view(-1)
                entropy_logits.append(logits_x)
                targets_x = y_true[labeled_mask].float()
                loss_l = self.criterion(logits_x, targets_x)
                labeled_loss_sum += loss_l.item()

            # Unlabeled consistency loss (weak/strong augmentation + pseudo-labels + temperature/mask)
            if unlabeled_mask.any():
                # Pseudo-labels: generated by stage 1 (0/1)
                batch_u_indices = idx[unlabeled_mask].detach().cpu().numpy()
                pseudo_binary = torch.tensor(
                    [self.pseudo_labels_map.get(int(i), 0) for i in batch_u_indices],
                    dtype=torch.long,
                    device=self.device,
                )
                logits_u_w = self.model(x_w[unlabeled_mask]).view(-1)
                logits_u_s = self.model(x_s[unlabeled_mask]).view(-1)
                entropy_logits.extend([logits_u_w, logits_u_s])

                # Temperature scaling + mask for self-training on the strong branch.
                with torch.no_grad():
                    p_u = torch.sigmoid(logits_u_w)
                    p_u2 = torch.stack([1 - p_u, p_u], dim=1)
                    pseudo_soft = p_u2 ** (1.0 / max(cur_T, 1e-6))
                    pseudo_soft = pseudo_soft / pseudo_soft.sum(dim=1, keepdim=True)
                    conf, pseudo_targets_u = torch.max(pseudo_soft, dim=1)
                    mask = (conf >= cur_mask_th).float()

                # Mixed target follows the author phase-2 form, but targets_p
                # is the phase-1 partition label.  Do not use oracle true labels
                # for unlabeled examples.
                targets_p = pseudo_binary.float()
                target_mix = (
                    lamda * pseudo_targets_u.float() + (1.0 - lamda) * targets_p
                )

                # Use the source BCE-with-logits unlabeled weak loss.
                loss_u_w = self.criterion(logits_u_w, target_mix)

                # Strong augmentation consistency with hard pseudo-labels.
                loss_u_s = (
                    self.criterion_u(logits_u_s, pseudo_targets_u.float()) * mask
                ).mean()

                loss_u = loss_u_w + loss_u_s
                unlabeled_loss_sum += loss_u.item()

            # Source train.py applies entropy minimization on the full
            # concatenated batch logits.
            loss_ent = torch.tensor(0.0, device=self.device)
            if cur_co_entropy > 0 and entropy_logits:
                probs = torch.sigmoid(torch.cat(entropy_logits))
                loss_ent = cur_co_entropy * _entropy_minimization(probs)

            loss = loss_l + self.lambda_u * loss_u + loss_ent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        if self.file_console and num_batches > 0:
            avg_total = total_loss / num_batches
            avg_l = labeled_loss_sum / num_batches
            avg_u = unlabeled_loss_sum / num_batches
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
            self.file_console.log(
                f"Phase2[{epoch_idx}/{self.phase2_epochs}] - Total: {avg_total:.4f}, Labeled: {avg_l:.4f}, Unlabeled: {avg_u:.4f}, LR: {lr:.6f}"
            )

    # ---------- Bias init utility ----------
    def _init_bias_from_prior(self):
        """Initialize last layer Linear bias to logit(π) to accelerate convergence and avoid all-negative predictions.
        Only effective when last layer output is 1-dimensional, ignored otherwise.
        """
        pi = float(max(1e-4, min(1 - 1e-4, self.prior)))
        b = float(math.log(pi / (1.0 - pi)))
        last_linear = None
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None and getattr(last_linear, "out_features", None) == 1:
            if getattr(last_linear, "bias", None) is not None:
                with torch.no_grad():
                    last_linear.bias.data.fill_(b)

    # ---------------- Pseudo-label Generation ----------------
    def _generate_pseudo_labels(self):
        # If trend is selected but no U scores recorded yet, try recording one round first
        if (
            str(self.pseudo_label_strategy).lower() == "trend"
            and len(self._phase1_unlabeled_scores) == 0
        ):
            try:
                self._record_unlabeled_scores_for_trend()
            except Exception:
                pass
            if len(self._phase1_unlabeled_scores) == 0:
                # Fallback: automatically fall back to alpha_range to avoid direct interruption
                if hasattr(self, "file_console") and self.file_console:
                    self.file_console.log(
                        "[yellow]Warning:[/yellow] No unlabeled scores captured during Phase-1. "
                        "Falling back to 'alpha_range' pseudo-label strategy."
                    )
                self.pseudo_label_strategy = "alpha_range"

        if self.pseudo_label_strategy == "trend":
            self._generate_pseudo_by_trend()
        elif self.pseudo_label_strategy == "alpha_range":
            self._generate_pseudo_by_alpha_range()
        else:
            raise ValueError(
                f"Unknown pseudo_label_strategy: {self.pseudo_label_strategy}"
            )

    def _generate_pseudo_by_trend(self):
        import pandas as pd

        try:
            import jenkspy
        except ImportError as e:
            raise RuntimeError(
                "Trend strategy requires 'jenkspy'. Please install it or use 'alpha_range'."
            ) from e

        if not self._phase1_unlabeled_scores:
            # If stage 1 recorded no U scores, cannot perform trend strategy
            raise RuntimeError(
                "No unlabeled scores recorded in Phase-1; cannot use 'trend' strategy."
            )

        # Shape: (num_epochs, num_unlabeled) -> (num_unlabeled, num_epochs)
        preds_sequence = np.vstack(self._phase1_unlabeled_scores).T
        trends = np.zeros(len(preds_sequence), dtype=np.float64)
        for i, seq in enumerate(preds_sequence):
            s = pd.Series(seq)
            diff_1 = s.diff(periods=1).iloc[1:].to_numpy()
            if diff_1.size == 0:
                trends[i] = 0.0
                continue
            # Original implementation transformation: log(1 + d + 0.5*d^2)
            eps = 1e-8
            d = np.clip(diff_1, -1 + eps, 1 - eps)
            v = np.log(1 + d + 0.5 * d * d)
            v = v[np.isfinite(v)]
            trends[i] = v.mean() if v.size > 0 else 0.0

        # optional three-sigma
        breaks = None
        try:
            _arr = trends
            if self.use_three_sigma:
                mu, std = _arr.mean(), _arr.std()
                low, high = mu - 3 * std, mu + 3 * std
                _arr = _arr[(_arr > low) & (_arr < high)]
            if len(np.unique(_arr)) < 2:
                raise ValueError("Not enough variability for Jenks breaks")
            breaks = jenkspy.jenks_breaks(_arr, n_classes=2)
            break_point = breaks[1]
        except Exception:
            break_point = float(np.median(trends))

        # Trend greater than split point → pseudo-label=0 (negative class), otherwise 1 (positive class) - consistent with original implementation
        pseudo = (trends <= break_point).astype(int)

        # Extract U global indices from dataset, consistent with ordering used in recording stage
        ds = self.train_loader.dataset
        unlabeled_idx = ds.indices[(ds.pu_labels == -1)].numpy()
        if getattr(self, "_trend_u_indices_sorted", None) is not None:
            order = self._trend_u_indices_sorted
        else:
            order = np.sort(unlabeled_idx)
        self.pseudo_labels_map = dict(
            zip([int(i) for i in order], [int(v) for v in pseudo])
        )

    def _generate_pseudo_by_alpha_range(self):
        # Replicate simplified alpha segment pseudo-label generation in PULCPBF/train.py (based on sign + interval random)
        self.model.eval()

        # Only perform prediction and pseudo-label generation on unlabeled samples
        unlabeled_indices = []
        unlabeled_logits = []

        with torch.no_grad():
            for x, t, _y_true, idx, _ in self.train_loader:  # type: ignore
                if isinstance(x, (list, tuple)):
                    x = x[0]
                # When evaluating pseudo-labels, also align input size and channels
                x = self._ensure_channel_match(x)
                x, t, idx = x.to(self.device), t.to(self.device), idx.to(self.device)
                u_mask = t == -1
                if not u_mask.any():
                    continue

                logits_u = self.model(x[u_mask]).view(-1)
                unlabeled_logits.append(logits_u.detach().cpu())
                unlabeled_indices.append(idx[u_mask].detach().cpu())

        if not unlabeled_logits:
            self.pseudo_labels_map = {}
            return

        # Merge all unlabeled data
        all_unlabeled_logits = torch.cat(unlabeled_logits)
        all_unlabeled_indices = torch.cat(unlabeled_indices)

        # Calculate sign for each alpha
        preds_sign = []
        for _ in self.alpha_list:
            s = torch.sign(all_unlabeled_logits).numpy()  # {-1, 0, 1}
            s[s == 0] = 1  # Rare values treated as positive
            preds_sign.append(s)

        preds_sign = np.array(preds_sign)  # (len(alpha_list), N_unlabeled)
        N_unlabeled = preds_sign.shape[1]
        pseudo_scores = np.zeros(N_unlabeled, dtype=np.float64)

        # First segment: samples predicted as -1, assign ~ U[0, alpha_0]
        alpha0 = float(self.alpha_list[0]) if self.alpha_list else 0.1
        neg_mask = preds_sign[0] == -1
        pseudo_scores[neg_mask] += np.random.uniform(
            0.0, alpha0, size=int(neg_mask.sum())
        )

        # Last segment: samples predicted as +1, assign ~ U[alpha_0, 1]
        pos_mask = preds_sign[-1] == 1
        pseudo_scores[pos_mask] += np.random.uniform(
            alpha0, 1.0, size=int(pos_mask.sum())
        )

        # Middle segment: sign changes (from +1 to -1)
        for i in range(len(self.alpha_list) - 1):
            mask_change = (preds_sign[i] == 1) & (preds_sign[i + 1] == -1)
            if not mask_change.any():
                continue
            pseudo_scores[mask_change] += np.random.uniform(
                float(self.alpha_list[i]),
                float(self.alpha_list[i + 1]),
                size=int(mask_change.sum()),
            )

        # Binarization
        pseudo_binary = (pseudo_scores >= 0.5).astype(int)

        # Directly establish mapping relationship (now indices and pseudo_binary have consistent length)
        self.pseudo_labels_map = {}
        for idx, pseudo_label in zip(all_unlabeled_indices.numpy(), pseudo_binary):
            self.pseudo_labels_map[int(idx)] = int(pseudo_label)

    # ---------------- Phase-2 Data Wrapping ----------------
    def _make_image_train_wrapper(self, base_dataset):
        mean, std, image_size, mode = infer_image_profile(base_dataset, self.params)
        return PULCPBFDatasetWrapper(
            base_dataset=base_dataset,
            transform=TransformPULCPBF(
                mean=mean,
                std=std,
                image_size=image_size,
                mode=mode,
            ),
        )

    def _make_image_eval_wrapper(self, base_dataset):
        mean, std, image_size, _mode = infer_image_profile(base_dataset, self.params)
        return PULCPBFEvalDatasetWrapper(
            base_dataset=base_dataset,
            transform=TransformPULCPBFEval(
                mean=mean,
                std=std,
                image_size=image_size,
            ),
        )

    def _is_image_dataset(self, dataset) -> bool:
        try:
            sample_x = dataset[0][0]
        except Exception:
            return False
        return isinstance(sample_x, torch.Tensor) and sample_x.dim() >= 3

    def _install_image_source_transforms(self):
        """Install PUL-CPBF image transforms for train/val/test loaders."""

        base_dataset = self.train_loader.dataset
        if isinstance(base_dataset, PULCPBFDatasetWrapper):
            return
        if not self._is_image_dataset(base_dataset):
            return

        self.train_loader = torch.utils.data.DataLoader(
            self._make_image_train_wrapper(base_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.params.get("num_workers", 4),
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        if self.validation_loader is not None and self._is_image_dataset(
            self.validation_loader.dataset
        ):
            self.validation_loader = torch.utils.data.DataLoader(
                self._make_image_eval_wrapper(self.validation_loader.dataset),
                batch_size=self.params.get("batch_size", 128),
                shuffle=False,
                num_workers=self.params.get("num_workers", 4),
                pin_memory=True,
                worker_init_fn=seed_worker,
            )
        if self.test_loader is not None and self._is_image_dataset(
            self.test_loader.dataset
        ):
            self.test_loader = torch.utils.data.DataLoader(
                self._make_image_eval_wrapper(self.test_loader.dataset),
                batch_size=self.params.get("batch_size", 128),
                shuffle=False,
                num_workers=self.params.get("num_workers", 4),
                pin_memory=True,
                worker_init_fn=seed_worker,
            )

    def _ensure_phase2_train_dataset(self):
        base_dataset = self.train_loader.dataset

        try:
            sample_x = base_dataset[0][0]
        except Exception:
            sample_x = None

        if isinstance(sample_x, (list, tuple)) and len(sample_x) == 2:
            wrapped = base_dataset
        elif self._is_image_dataset(base_dataset):
            wrapped = self._make_image_train_wrapper(base_dataset)
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
            wrapped = VectorAugPUDatasetWrapper(
                base_dataset=base_dataset, weak_aug=weak, strong_aug=strong
            )

        self.train_loader = torch.utils.data.DataLoader(
            wrapped,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.params.get("num_workers", 4),
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
