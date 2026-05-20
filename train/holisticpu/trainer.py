"""HolisticPU trainer adapted to PU-Bench.

Reference implementation:
    wxr99/HolisticPU at commit 4d4ce7d6ba29722995d308293374c0f475d988d3,
    especially ``main.py``, ``train.py``, ``model/loss.py`` and
    ``utils/misc.py``.

Important source convention: HolisticPU trains a two-logit classifier where
class ``0`` is positive and class ``1`` is negative/unlabeled. PU-Bench keeps
oracle true labels as ``1`` for positive and ``0`` for negative, so metric code
must use the model's ``positive_logit_index = 0`` hook instead of assuming the
second logit is positive. Image augmentation/loss flow tracks the source; the
vector/tabular path is a benchmark-specific extension.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from backbone.ema import ModelEMA
from ..augmentations.vector import (
    VectorAugPUDatasetWrapper,
    VectorStrongAugment,
    VectorWeakAugment,
)
from ..base_trainer import BaseTrainer
from ..metrics import evaluate_metrics, evaluate_proxy_metrics
from ..reproducibility import seed_worker
from .core import (
    as_numpy,
    cosine_schedule_with_warmup,
    de_interleave,
    interleave,
    jenks_breaks,
    soft_cross_entropy,
    source_three_sigma,
)
from .augment import HolisticPUDatasetWrapper, TransformHolisticPU
from .model_selector import select_model


class HolisticPUTrainer(BaseTrainer):
    """Two-stage HolisticPU trainer adapted from the authors' active source."""

    def __init__(self, method: str, experiment: str, params: dict[str, Any]):
        super().__init__(method, experiment, params)

        self.phase1_epochs = int(
            self.params.get(
                "warming_epochs",
                self.params.get("phase1_epochs", self.params.get("epochs", 15)),
            )
        )
        self.phase2_epochs = int(
            self.params.get(
                "ft_epochs",
                self.params.get("phase2_epochs", self.params.get("epochs", 25)),
            )
        )
        self.eval_step = int(
            self.params.get("eval_step", self.params.get("steps_per_epoch", 512))
        )
        self.mu = int(self.params.get("mu", 1))
        self.rho = float(self.params.get("rho", self.params.get("label_smoothing", 0.1)))
        self.temperature = float(self.params.get("T", 1.0))
        self.threshold = float(self.params.get("threshold", 0.9))
        self.use_ema = bool(self.params.get("use_ema", True))
        self.ema_decay = float(self.params.get("ema_decay", 0.999))
        self.use_three_sigma = bool(self.params.get("use_three_sigma", True))
        self.trend_filter_threshold = float(
            self.params.get("trend_filter_threshold", 0.2 / 9.0)
        )

        self.labeled_loader: DataLoader | None = None
        self.unlabeled_loader: DataLoader | None = None
        self.unlabeled_pred_loader: DataLoader | None = None
        self.ema_model: ModelEMA | None = None
        self.pseudo_targets_array: np.ndarray | None = None
        self.pseudo_labels_map: dict[int, int] = {}
        self._unlabeled_base_indices: np.ndarray | None = None
        self._last_phase_losses: dict[str, float] = {}

    def create_criterion(self):
        return nn.CrossEntropyLoss()

    def create_model(self):
        return select_model(params=self.params, prior=self.prior)

    def train_one_epoch(self, epoch_idx: int):
        raise RuntimeError("HolisticPU uses a custom two-stage training loop.")

    def get_eval_model(self) -> nn.Module:
        # Source phase-2 evaluates model1 directly even when EMA is maintained.
        return self.model

    def before_training(self):
        super().before_training()
        self._create_source_loaders()
        if self.file_console:
            self.file_console.log("HolisticPU source alignment:")
            self.file_console.log(
                "  source=wxr99/HolisticPU@4d4ce7d6ba29722995d308293374c0f475d988d3"
            )
            self.file_console.log(
                f"  phase1_epochs={self.phase1_epochs}, phase2_epochs={self.phase2_epochs}, eval_step={self.eval_step}"
            )
            self.file_console.log(
                f"  backbone_policy={self.params.get('backbone_policy', 'controlled')}, "
                f"private_backbone_arch={self.params.get('private_backbone_arch', 'auto')}, "
                f"model={self.model.__class__.__name__}, "
                f"params={sum(p.numel() for p in self.model.parameters()) / 1e6:.3f}M"
            )
            self.file_console.log(
                f"  rho={self.rho}, mu={self.mu}, T={self.temperature}, threshold={self.threshold}"
            )
            self.file_console.log(
                f"  positive_logit_index={getattr(self.model, 'positive_logit_index', 1)}"
            )

    def run(self):
        self.before_training()
        final_metrics: dict[str, float] = {}
        try:
            self.console.log(
                "\n--- [Stage 1/2] HolisticPU trend scoring ---",
                style="bold yellow",
            )
            phase1_early_stop_state = (
                bool(getattr(self.checkpoint_handler, "early_stopping_enabled", False))
                if self.checkpoint_handler is not None
                else False
            )
            self.set_checkpoint_early_stopping(False, reset=True)

            pseudo_targets = self._run_phase1()
            self.pseudo_targets_array = pseudo_targets

            self.set_checkpoint_early_stopping(phase1_early_stop_state)

            self.console.log(
                "\n--- [Stage 2/2] HolisticPU fine tuning ---",
                style="bold yellow",
            )
            self._build_model()
            self._create_source_loaders()
            self.ema_model = ModelEMA(self, self.model, self.ema_decay) if self.use_ema else None
            self.optimizer, self.scheduler = self._make_source_optimizer_and_scheduler()
            final_metrics = self._run_phase2()

            return final_metrics
        finally:
            self.finalize()

    def _make_source_optimizer_and_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer, LambdaLR]:
        no_decay = ["bias", "bn"]
        grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": float(self.params.get("weight_decay", 5e-4)),
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
        optimizer = torch.optim.SGD(
            grouped_parameters,
            lr=float(self.params.get("lr", 0.01)),
            momentum=float(self.params.get("momentum", 0.9)),
            nesterov=bool(self.params.get("nesterov", True)),
        )
        total_steps = int(
            self.params.get(
                "total_steps",
                max(1, self.eval_step) * max(self.phase1_epochs, self.phase2_epochs),
            )
        )
        scheduler = cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.params.get("warmup_steps", self.params.get("warmup", 0))),
            num_training_steps=total_steps,
        )
        return optimizer, scheduler

    def _create_source_loaders(self) -> None:
        base_dataset = self._base_train_dataset()
        pu_labels = as_numpy(base_dataset.pu_labels)
        metadata = getattr(base_dataset, "pu_metadata", {}) or {}
        labeled_value = metadata.get("pu_labeled_label", 1)
        unlabeled_value = metadata.get("pu_unlabeled_label", -1)

        labeled_indices = np.flatnonzero(pu_labels == labeled_value)
        unlabeled_indices = np.flatnonzero(pu_labels == unlabeled_value)
        if len(labeled_indices) == 0 or len(unlabeled_indices) == 0:
            raise ValueError(
                "HolisticPU requires non-empty labeled-positive and unlabeled splits."
            )

        wrapped_dataset = self._wrap_for_holisticpu(base_dataset)
        self._unlabeled_base_indices = unlabeled_indices

        batch_size = int(self.params.get("batch_size", 64)) * self.mu
        if batch_size <= 0:
            raise ValueError(f"Invalid HolisticPU batch size: {batch_size}")
        source_batch_size = batch_size
        if bool(self.params.get("expand_labels", False)) or len(labeled_indices) < batch_size:
            repeats = int(math.ceil(batch_size * self.eval_step / max(1, len(labeled_indices))))
            labeled_indices = np.hstack([labeled_indices for _ in range(max(1, repeats))])
            np.random.shuffle(labeled_indices)
            if self.file_console:
                self.file_console.log(
                    f"expanded labeled positives to {len(labeled_indices)} indices "
                    f"for source batch_size={source_batch_size}, eval_step={self.eval_step}"
                )

        source_drop_last = True
        if min(len(labeled_indices), len(unlabeled_indices)) < batch_size:
            if not bool(self.params.get("allow_small_batch_fallback", True)):
                raise ValueError(
                    "HolisticPU source batch size exceeds available P/U samples. "
                    "Set allow_small_batch_fallback=true to run this benchmark split."
                )
            batch_size = max(1, min(len(labeled_indices), len(unlabeled_indices)))
            source_drop_last = False
            self.console.log(
                f"[yellow]HolisticPU small-split fallback: batch_size={batch_size}, drop_last=False[/yellow]"
            )

        num_workers = int(self.params.get("num_workers", 0))
        pin_memory = torch.cuda.is_available()

        labeled_subset = Subset(wrapped_dataset, labeled_indices.tolist())
        unlabeled_subset = Subset(wrapped_dataset, unlabeled_indices.tolist())

        self.labeled_loader = DataLoader(
            labeled_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=source_drop_last,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
        )
        self.unlabeled_loader = DataLoader(
            unlabeled_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=source_drop_last,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
        )
        self.unlabeled_pred_loader = DataLoader(
            unlabeled_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=source_drop_last,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(
            wrapped_dataset,
            batch_size=int(self.params.get("batch_size", 64)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
        )

    def _base_train_dataset(self):
        dataset = self.train_loader.dataset
        while hasattr(dataset, "base_dataset"):
            dataset = dataset.base_dataset
        return dataset

    def _wrap_for_holisticpu(self, base_dataset):
        sample_x = base_dataset[0][0]
        is_image = isinstance(sample_x, torch.Tensor) and sample_x.ndim >= 3
        if is_image:
            mean, std = self._image_normalization(base_dataset)
            image_size = int(getattr(base_dataset, "image_size", sample_x.shape[-1]))
            aug_mode = self._augment_mode()
            wrapped = HolisticPUDatasetWrapper(
                base_dataset,
                TransformHolisticPU(
                    mean=mean, std=std, image_size=image_size, mode=aug_mode
                ),
            )
        else:
            weak = VectorWeakAugment(
                noise_std=float(self.params.get("vec_weak_noise_std", 0.0)),
                dropout_ratio=float(self.params.get("vec_weak_dropout", 0.0)),
            )
            strong = VectorStrongAugment(
                noise_std=float(self.params.get("vec_strong_noise_std", 0.0)),
                dropout_ratio=float(self.params.get("vec_strong_dropout", 0.0)),
                sign_flip_ratio=float(self.params.get("vec_sign_flip_ratio", 0.0)),
            )
            wrapped = VectorAugPUDatasetWrapper(base_dataset, weak_aug=weak, strong_aug=strong)

        for attr in ("features", "pu_labels", "true_labels", "indices", "pseudo_labels"):
            if hasattr(base_dataset, attr):
                setattr(wrapped, attr, getattr(base_dataset, attr))
        if hasattr(base_dataset, "pu_metadata"):
            wrapped.pu_metadata = base_dataset.pu_metadata
            wrapped.metadata = base_dataset.pu_metadata
        return wrapped

    def _augment_mode(self) -> str:
        dataset_class = str(self.params.get("dataset_class", "")).lower()
        if "fashionmnist" in dataset_class:
            return "fashionmnist"
        if "stl" in dataset_class:
            return "stl"
        if "cifar" in dataset_class:
            return "cifar"
        return "generic"

    def _image_normalization(self, base_dataset) -> tuple[tuple[float, ...], tuple[float, ...]]:
        dataset_class = str(self.params.get("dataset_class", "")).lower()
        if "cifar" in dataset_class or "stl" in dataset_class:
            return (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        if "fashionmnist" in dataset_class:
            return (0.1307,), (0.3081,)
        mean = getattr(base_dataset, "mean", (0.5,))
        std = getattr(base_dataset, "std", (0.5,))
        if isinstance(mean, (int, float)):
            mean = (float(mean),)
        if isinstance(std, (int, float)):
            std = (float(std),)
        return tuple(mean), tuple(std)

    def _next_pu_batch(self, labeled_iter, unlabeled_iter):
        assert self.labeled_loader is not None and self.unlabeled_loader is not None
        try:
            labeled_batch = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(self.labeled_loader)
            labeled_batch = next(labeled_iter)
        try:
            unlabeled_batch = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(self.unlabeled_loader)
            unlabeled_batch = next(unlabeled_iter)
        return labeled_batch, unlabeled_batch, labeled_iter, unlabeled_iter

    def _run_phase1(self) -> np.ndarray:
        assert self.labeled_loader is not None and self.unlabeled_loader is not None
        self.optimizer, self.scheduler = self._make_source_optimizer_and_scheduler()
        self.ema_model = ModelEMA(self, self.model, self.ema_decay) if self.use_ema else None

        score_history: list[np.ndarray] = []
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)

        for epoch in tqdm(range(self.phase1_epochs), desc="HolisticPU phase1"):
            self.global_epoch += 1
            self.model.train()
            losses: list[float] = []

            for _ in range(self.eval_step):
                labeled_batch, unlabeled_batch, labeled_iter, unlabeled_iter = (
                    self._next_pu_batch(labeled_iter, unlabeled_iter)
                )
                (x_l_w, x_l_s), *_ = labeled_batch
                (x_u_w, x_u_s), *_ = unlabeled_batch

                batch_size = min(x_l_w.shape[0], x_u_w.shape[0])
                x_l_w = x_l_w[:batch_size].to(self.device)
                x_l_s = x_l_s[:batch_size].to(self.device)
                x_u_w = x_u_w[:batch_size].to(self.device)
                x_u_s = x_u_s[:batch_size].to(self.device)

                group_size = 4 * max(1, self.mu)
                if (4 * batch_size) % group_size != 0:
                    group_size = 4
                inputs = interleave(torch.cat((x_u_w, x_u_s, x_l_w, x_l_s)), group_size)
                logits = de_interleave(self.model(inputs), group_size)
                logits_u, _logits_u_s = logits[: 2 * batch_size].chunk(2)
                logits_l_w, logits_l_s = logits[2 * batch_size :].chunk(2)

                targets_l = torch.zeros(batch_size, device=self.device, dtype=torch.long)
                targets_u = torch.ones(batch_size, device=self.device, dtype=torch.long)
                loss_l = (
                    F.cross_entropy(logits_l_w, targets_l, label_smoothing=self.rho)
                    + F.cross_entropy(logits_l_s, targets_l, label_smoothing=self.rho)
                ) / 2.0
                loss_u = F.cross_entropy(logits_u, targets_u)
                loss = loss_l + loss_u

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.ema_model is not None:
                    self.ema_model.update(self.model)
                losses.append(float(loss.detach().cpu()))

            scores = self._record_unlabeled_positive_scores()
            score_history.append(scores)
            self._last_phase_losses = {"phase1_loss": float(np.mean(losses)) if losses else float("nan")}
            if self.file_console:
                self.file_console.log(
                    f"phase1 epoch={epoch + 1}/{self.phase1_epochs} loss={self._last_phase_losses['phase1_loss']:.4f} "
                    f"scores=[min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}]"
                )

        if len(score_history) < 2:
            raise RuntimeError("HolisticPU needs at least two phase-1 score snapshots.")

        trends = self._trend_scores(np.vstack(score_history).T)
        pseudo_targets = self._partition_trends(trends)
        self._record_pseudo_label_diagnostics(pseudo_targets, trends)
        return pseudo_targets

    def _record_unlabeled_positive_scores(self) -> np.ndarray:
        assert self.unlabeled_pred_loader is not None
        model = self.ema_model.ema if self.ema_model is not None else self.model
        model.eval()
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for (x_w, _x_s), *_ in self.unlabeled_pred_loader:
                logits = model(x_w.to(self.device))
                probs = torch.softmax(logits, dim=1)
                scores.append(probs[:, 0].detach().cpu().numpy())
        if not scores:
            raise RuntimeError("HolisticPU could not record unlabeled predictions.")
        return np.concatenate(scores, axis=0)

    def _trend_scores(self, score_matrix: np.ndarray) -> np.ndarray:
        diffs = np.diff(score_matrix, axis=1)
        transformed = np.log(1.0 + diffs + 0.5 * diffs**2)
        return np.nanmean(transformed, axis=1)

    def _partition_trends(self, trends: np.ndarray) -> np.ndarray:
        finite_trends = trends[np.isfinite(trends)]
        if len(finite_trends) == 0 or len(np.unique(finite_trends)) < 2:
            break_point = float(np.nanmedian(trends))
            breaks = [float(np.nanmin(trends)), break_point, float(np.nanmax(trends))]
        else:
            breaks = jenks_breaks(finite_trends)
            break_point = float(breaks[1])
            if self.use_three_sigma and break_point > 0:
                filtered = source_three_sigma(finite_trends, self.trend_filter_threshold)
                if len(filtered) > 0 and len(np.unique(filtered)) >= 2:
                    breaks = jenks_breaks(filtered)
                    break_point = float(breaks[1])

        if self.file_console:
            self.file_console.log(f"Jenks breaks: {breaks}; break_point={break_point}")
        self.console.log(f"HolisticPU break point: {break_point:.6f}")
        return np.where(trends > break_point, 0, 1).astype(np.int64)

    def _record_pseudo_label_diagnostics(
        self, pseudo_targets: np.ndarray, trends: np.ndarray
    ) -> None:
        assert self._unlabeled_base_indices is not None
        effective_indices = self._unlabeled_base_indices[: len(pseudo_targets)]
        self.pseudo_labels_map = dict(
            zip(effective_indices.astype(int).tolist(), pseudo_targets.astype(int).tolist())
        )

        pseudo_pos = int((pseudo_targets == 0).sum())
        pseudo_neg = int((pseudo_targets == 1).sum())
        base_dataset = self._base_train_dataset()
        metadata = getattr(base_dataset, "pu_metadata", {}) or {}
        labeled_value = metadata.get("pu_labeled_label", 1)
        n_labeled = int((as_numpy(base_dataset.pu_labels) == labeled_value).sum())
        estimated_prior = (pseudo_pos + n_labeled) / max(1, len(pseudo_targets) + n_labeled)

        self.console.log(
            f"HolisticPU pseudo labels: positive={pseudo_pos}, negative={pseudo_neg}, estimated_prior={estimated_prior:.4f}"
        )
        if self.file_console:
            self.file_console.log(
                f"trend stats: min={np.nanmin(trends):.6f}, max={np.nanmax(trends):.6f}, mean={np.nanmean(trends):.6f}"
            )
            self.file_console.log(
                f"pseudo labels: positive={pseudo_pos}, negative={pseudo_neg}, estimated_prior={estimated_prior:.6f}"
            )

        if hasattr(base_dataset, "true_labels"):
            true_labels = as_numpy(base_dataset.true_labels)[effective_indices]
            pseudo_eval = (pseudo_targets == 0).astype(np.int64)
            acc = float((pseudo_eval == true_labels).mean()) if len(true_labels) else float("nan")
            self._last_phase_losses["pseudo_label_acc"] = acc
            if self.file_console:
                self.file_console.log(f"pseudo_label_oracle_acc={acc:.6f}")

    def _run_phase2(self) -> dict[str, float]:
        assert self.pseudo_targets_array is not None
        assert self.labeled_loader is not None and self.unlabeled_loader is not None
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)
        pseudo_ptr = 0
        final_test_metrics: dict[str, float] = {}

        for epoch in tqdm(range(self.phase2_epochs), desc="HolisticPU phase2"):
            self.global_epoch += 1
            self.model.train()
            losses: list[float] = []
            losses_x: list[float] = []
            losses_u: list[float] = []

            for _ in range(self.eval_step):
                labeled_batch, unlabeled_batch, labeled_iter, unlabeled_iter = (
                    self._next_pu_batch(labeled_iter, unlabeled_iter)
                )
                (x_l_w, x_l_s), *_ = labeled_batch
                (x_u_w, x_u_s), *_ = unlabeled_batch

                batch_size = min(x_l_w.shape[0], x_u_w.shape[0])
                if batch_size == 0:
                    continue
                x_l_w = x_l_w[:batch_size].to(self.device)
                x_l_s = x_l_s[:batch_size].to(self.device)
                x_u_w = x_u_w[:batch_size].to(self.device)
                x_u_s = x_u_s[:batch_size].to(self.device)

                group_size = 4 * max(1, self.mu)
                if (4 * batch_size) % group_size != 0:
                    group_size = 4

                inputs = interleave(torch.cat((x_u_w, x_u_s, x_l_w, x_l_s)), group_size)
                logits = de_interleave(self.model(inputs), group_size)
                logits_u, logits_u_s = logits[: 2 * batch_size].chunk(2)
                logits_l_w, _logits_l_s = logits[2 * batch_size :].chunk(2)

                targets_l = torch.zeros(batch_size, device=self.device, dtype=torch.long)
                loss_x = F.cross_entropy(logits_l_w, targets_l)

                pseudo_batch, pseudo_ptr = self._next_pseudo_targets(batch_size, pseudo_ptr)
                targets_u = torch.ones(batch_size, device=self.device, dtype=torch.long)
                loss_u = self._source_loss_ft(
                    logits_u=logits_u,
                    logits_u_s=logits_u_s,
                    targets_u=targets_u,
                    targets_p=pseudo_batch,
                    epoch=epoch,
                )
                loss = loss_x + loss_u

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.ema_model is not None:
                    self.ema_model.update(self.model)

                losses.append(float(loss.detach().cpu()))
                losses_x.append(float(loss_x.detach().cpu()))
                losses_u.append(float(loss_u.detach().cpu()))

            self._last_phase_losses = {
                "phase2_loss": float(np.mean(losses)) if losses else float("nan"),
                "phase2_loss_x": float(np.mean(losses_x)) if losses_x else float("nan"),
                "phase2_loss_u": float(np.mean(losses_u)) if losses_u else float("nan"),
            }
            final_test_metrics = self._evaluate_and_checkpoint(epoch + 1)
            if self.checkpoint_handler and self.checkpoint_handler.should_stop:
                self.console.log("Early stopping in HolisticPU phase2.", style="yellow")
                break

        return final_test_metrics

    def _next_pseudo_targets(
        self, batch_size: int, start: int
    ) -> tuple[torch.Tensor, int]:
        assert self.pseudo_targets_array is not None
        n = len(self.pseudo_targets_array)
        end = start + batch_size
        if end <= n:
            batch = self.pseudo_targets_array[start:end]
        else:
            batch = np.concatenate(
                [self.pseudo_targets_array[start:], self.pseudo_targets_array[: end - n]]
            )
        next_start = end % n
        return torch.from_numpy(batch.astype(np.int64)).to(self.device), next_start

    def _source_loss_ft(
        self,
        logits_u: torch.Tensor,
        logits_u_s: torch.Tensor,
        targets_u: torch.Tensor,
        targets_p: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        label_u = F.one_hot(targets_u, 2).float().to(self.device)
        label_p = F.one_hot(targets_p, 2).float().to(self.device)
        lamda = (epoch / max(1, self.phase2_epochs)) ** 0.8
        blended_label = lamda * label_p + (1.0 - lamda) * label_u
        loss = soft_cross_entropy(logits_u, blended_label)

        pseudo_label = torch.softmax(logits_u.detach() / self.temperature, dim=-1)
        max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        loss_consistency = (
            F.cross_entropy(logits_u_s, pseudo_targets_u, reduction="none") * mask
        ).mean()
        return loss + loss_consistency

    def _evaluate_and_checkpoint(self, stage_epoch: int) -> dict[str, float]:
        scenario = self.params.get("scenario", "single")
        train_oracle = evaluate_metrics(
            self.get_eval_model(), self.train_loader, self.device, self.prior
        )
        test_metrics = evaluate_metrics(
            self.get_eval_model(), self.test_loader, self.device, self.prior
        )
        train_proxy = evaluate_proxy_metrics(
            self.get_eval_model(), self.train_loader, self.device, self.prior, scenario
        )
        train_metrics = {**train_oracle, **train_proxy, **self._last_phase_losses}

        val_metrics = None
        if self.validation_loader is not None:
            val_oracle = evaluate_metrics(
                self.get_eval_model(), self.validation_loader, self.device, self.prior
            )
            val_proxy = evaluate_proxy_metrics(
                self.get_eval_model(),
                self.validation_loader,
                self.device,
                self.prior,
                scenario,
            )
            val_metrics = {**val_oracle, **val_proxy}

        self._print_metrics(
            stage_epoch,
            self.phase2_epochs,
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
            self.checkpoint_handler(
                epoch=self.global_epoch,
                all_metrics=all_metrics,
                model=self.get_eval_model(),
                elapsed_seconds=(
                    None
                    if self._run_start_time is None
                    else time.time() - self._run_start_time
                ),
            )
        return test_metrics
