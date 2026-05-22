"""Self-PU trainer adapted to PU-Bench.

Primary source:
    VITA-Group/Self-PU at audited snapshot
    a0e332ae4f8110e2490d597876e36bf837e1060f, especially ``train_2s2t.py``,
    ``datasets.py``, ``cifar_datasets.py`` and ``mean_teacher/ramps.py``.

Source facts preserved here:
    - PU labels are +1 for labeled positives and -1 for unlabeled examples.
    - The benchmark entry implements the source's two-student/two-teacher
      Self-PU variant without the expensive self-calibration path from
      ``train_2s2t_mix.py``.
    - Noisy splits are trained with nnPU. Clean self-paced splits are selected
      only from U using high/low model scores, then trained with the source
      entropy-minimization ``--soft-label`` branch by default.
    - Mean-teacher EMA starts only after the configured epoch; teachers are
      copied from students at the switch point before EMA updates begin.

Benchmark boundary:
    This is the controlled-backbone PU-Bench entry. The source paper's dataset
    recipes and private architectures are not reproduced; the estimator is run
    on PU-Bench's own SCAR/SAR splits to measure robustness under a shared
    backbone policy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..base_trainer import BaseTrainer
from ..metrics import evaluate_metrics, evaluate_proxy_metrics
from ..model_factory import select_model
from ..common.pu_risk import PULoss
from ..reproducibility import seed_worker
from .ema import EMATeacher
from .losses import (
    mutual_student_consistency,
    sigmoid_entropy_loss,
    sigmoid_rampup,
    softmax_mse_consistency,
    signed_binary_ce,
)
from .selection import (
    SelectionState,
    SelfPUCleanDataset,
    SelfPUNoisyDataset,
    SelfPUSelector,
)


@dataclass
class SelfPUConfig:
    batch_size: int
    lr: float
    weight_decay: float
    self_paced_enabled: bool
    self_paced_start: int
    self_paced_stop: int
    self_paced_frequency: int
    self_paced_rampup: int
    top1: float
    top2: float
    replacement: bool
    increasing: bool
    flex_ratio: float
    mean_teacher_enabled: bool
    mean_teacher_start: int
    ema_decay: float
    consistency_weight: float
    consistency_rampup: int
    mutual_type: str
    mutual_alpha: float
    soft_label: bool
    pu_loss_weight: float


class SelfPUTrainer(BaseTrainer):
    """Source-aligned Self-PU trainer for PU-Bench datasets."""

    def __init__(self, method: str, experiment: str, params: dict[str, Any]):
        super().__init__(method, experiment, params)
        self.selfpu_cfg = self._parse_selfpu_config()
        self._validate_source_data_contract()
        self._rng = np.random.default_rng(int(self.params.get("seed", 42)))
        self.selector = SelfPUSelector(
            self.train_loader.dataset,
            replacement=self.selfpu_cfg.replacement,
            increasing=self.selfpu_cfg.increasing,
            rampup_length=self.selfpu_cfg.self_paced_rampup,
            flex_ratio=self.selfpu_cfg.flex_ratio,
            rng=self._rng,
        )

        self.model2 = self._build_second_student()
        self.optimizer2 = torch.optim.Adam(
            self.model2.parameters(),
            lr=self.selfpu_cfg.lr,
            weight_decay=self.selfpu_cfg.weight_decay,
        )
        self.scheduler1 = CosineAnnealingLR(
            self.optimizer, T_max=max(1, int(self.params.get("num_epochs", 1)))
        )
        self.scheduler2 = CosineAnnealingLR(
            self.optimizer2, T_max=max(1, int(self.params.get("num_epochs", 1)))
        )
        self.teacher1 = EMATeacher(self.model, self.device)
        self.teacher2 = EMATeacher(self.model2, self.device)
        self._mean_teacher_switched = False
        self._last_selection_epoch: int | None = None
        self._last_completed_epoch: int | None = None
        self._last_report_model_name: str | None = None
        self._last_report_model: torch.nn.Module | None = None

        self.criterion_unlabeled = PULoss(
            self.prior,
            loss=str(self.params.get("loss", "sigmoid")),
            nnpu=True,
            gamma=float(self.params.get("gamma", 1.0)),
            beta=float(self.params.get("beta", 0.0)),
        )
        self.state1 = self._empty_selection_state()
        self.state2 = self._empty_selection_state()
        self.clean_loader1 = None
        self.clean_loader2 = None
        self.noisy_loader1 = self._make_noisy_loader(self.state1.noisy_indices)
        self.noisy_loader2 = self._make_noisy_loader(self.state2.noisy_indices)
        self._epoch_stats: dict[str, float] = {}

        self.set_checkpoint_early_stopping(False)

    def _parse_selfpu_config(self) -> SelfPUConfig:
        return SelfPUConfig(
            batch_size=int(self.params.get("batch_size", 256)),
            lr=float(self.params.get("lr", 5e-4)),
            weight_decay=float(self.params.get("weight_decay", 5e-3)),
            self_paced_enabled=bool(self.params.get("self_paced_enabled", True)),
            self_paced_start=int(self.params.get("self_paced_start", 10)),
            self_paced_stop=int(self.params.get("self_paced_stop", 50)),
            self_paced_frequency=int(self.params.get("self_paced_frequency", 10)),
            self_paced_rampup=int(self.params.get("self_paced_rampup", 100)),
            top1=float(self.params.get("self_paced_top_p1", 0.4)),
            top2=float(self.params.get("self_paced_top_p2", 0.6)),
            replacement=bool(self.params.get("replacement", True)),
            increasing=bool(self.params.get("increasing", True)),
            flex_ratio=float(self.params.get("flex_ratio", 0.0)),
            mean_teacher_enabled=bool(self.params.get("mean_teacher_enabled", True)),
            mean_teacher_start=int(self.params.get("mean_teacher_start", 50)),
            ema_decay=float(self.params.get("ema_decay", 0.999)),
            consistency_weight=float(self.params.get("consistency_weight", 0.3)),
            consistency_rampup=int(self.params.get("consistency_rampup", 400)),
            mutual_type=str(self.params.get("selfpu_type", "mu")),
            mutual_alpha=float(self.params.get("mutual_alpha", 0.1)),
            soft_label=bool(self.params.get("soft_label", True)),
            pu_loss_weight=float(self.params.get("pu_loss_weight", 1.0)),
        )

    def _validate_source_data_contract(self) -> None:
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "pu_labels"):
            raise ValueError("Self-PU requires a PU dataset exposing `pu_labels`.")
        labels = {
            int(value)
            for value in torch.unique(dataset.pu_labels.detach().cpu()).tolist()
        }
        if 1 not in labels or -1 not in labels:
            raise ValueError(
                "Self-PU requires PU labels +1 for positives and -1 for "
                f"unlabeled examples; observed {sorted(labels)}."
            )

    def _build_second_student(self) -> torch.nn.Module:
        model = select_model(
            method=self.method,
            params=self.params,
            prior=self.prior,
        ).to(self.device)
        try:
            has_params = any(p.requires_grad for p in model.parameters())
        except Exception:
            has_params = False
        if not has_params:
            sample_batch = next(iter(self.train_loader))
            x_sample = sample_batch[0]
            if isinstance(x_sample, (list, tuple)):
                x_sample = x_sample[0]
            with torch.no_grad():
                _ = model(x_sample.to(self.device))
        return model

    def _empty_selection_state(self) -> SelectionState:
        all_indices = np.arange(len(self.train_loader.dataset), dtype=int)
        return SelectionState(
            clean_indices=np.array([], dtype=int),
            pseudo_pos_indices=np.array([], dtype=int),
            pseudo_neg_indices=np.array([], dtype=int),
            noisy_indices=all_indices,
        )

    def create_criterion(self):
        return signed_binary_ce

    def _make_loader(self, dataset, *, shuffle: bool, drop_last: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.selfpu_cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last and len(dataset) >= self.selfpu_cfg.batch_size,
            worker_init_fn=seed_worker,
        )

    def _make_clean_loader(self, state: SelectionState):
        if len(state.clean_indices) == 0:
            return None
        clean_dataset = SelfPUCleanDataset(
            self.train_loader.dataset,
            state.pseudo_pos_indices,
            state.pseudo_neg_indices,
        )
        return self._make_loader(clean_dataset, shuffle=True, drop_last=False)

    def _make_noisy_loader(self, indices: np.ndarray):
        noisy_dataset = SelfPUNoisyDataset(
            self.train_loader.dataset,
            np.asarray(indices, dtype=int),
            rng=self._rng,
        )
        return self._make_loader(noisy_dataset, shuffle=False, drop_last=False)

    def train_one_epoch(self, epoch_idx: int):
        self._maybe_start_mean_teacher(epoch_idx)
        self.model.train()
        self.model2.train()
        self._step_schedulers_source_order(epoch_idx)
        self._epoch_stats = {
            "selfpu_clean1": float(len(self.state1.clean_indices)),
            "selfpu_clean2": float(len(self.state2.clean_indices)),
            "selfpu_noisy1": float(len(self.state1.noisy_indices)),
            "selfpu_noisy2": float(len(self.state2.noisy_indices)),
        }

        clean_loss1 = self._run_clean_branch(
            model=self.model,
            teacher=self.teacher1,
            optimizer=self.optimizer,
            loader=self.clean_loader1,
            epoch_idx=epoch_idx,
        )
        clean_loss2 = self._run_clean_branch(
            model=self.model2,
            teacher=self.teacher2,
            optimizer=self.optimizer2,
            loader=self.clean_loader2,
            epoch_idx=epoch_idx,
        )
        if self.clean_loader2 is not None:
            self._update_teachers_once(epoch_idx)
        noisy_loss1 = self._run_noisy_branch(
            model=self.model,
            peer_model=self.model2,
            teacher=self.teacher1,
            optimizer=self.optimizer,
            loader=self.noisy_loader1,
            epoch_idx=epoch_idx,
        )
        noisy_loss2 = self._run_noisy_branch(
            model=self.model2,
            peer_model=self.model,
            teacher=self.teacher2,
            optimizer=self.optimizer2,
            loader=self.noisy_loader2,
            epoch_idx=epoch_idx,
        )
        self._update_teachers_once(epoch_idx)
        self._shuffle_noisy_loaders()
        self._last_completed_epoch = epoch_idx
        self._epoch_stats.update(
            {
                "selfpu_clean_loss1": clean_loss1,
                "selfpu_clean_loss2": clean_loss2,
                "selfpu_noisy_loss1": noisy_loss1,
                "selfpu_noisy_loss2": noisy_loss2,
            }
        )
        if epoch_idx >= self.selfpu_cfg.mean_teacher_start and getattr(
            self, "checkpoint_handler", None
        ):
            self.set_checkpoint_early_stopping(True)

    def _shuffle_noisy_loaders(self) -> None:
        """Match the source's per-epoch reshuffle of the noisy P/U sampler."""

        for loader in (self.noisy_loader1, self.noisy_loader2):
            dataset = getattr(loader, "dataset", None)
            if hasattr(dataset, "shuffle"):
                dataset.shuffle()

    def _step_schedulers_source_order(self, epoch_idx: int) -> None:
        """Apply the source's train-before-loop cosine schedule without warnings."""

        source_epoch = max(0, int(epoch_idx) - 1)
        for scheduler in (self.scheduler1, self.scheduler2):
            t_max = max(1, int(getattr(scheduler, "T_max", 1)))
            eta_min = float(getattr(scheduler, "eta_min", 0.0))
            lrs = []
            for group, base_lr in zip(
                scheduler.optimizer.param_groups,
                scheduler.base_lrs,
            ):
                lr = eta_min + (float(base_lr) - eta_min) * (
                    1.0 + math.cos(math.pi * source_epoch / t_max)
                ) / 2.0
                group["lr"] = lr
                lrs.append(lr)
            scheduler.last_epoch = source_epoch
            scheduler._last_lr = lrs

    def _run_clean_branch(
        self,
        *,
        model: torch.nn.Module,
        teacher: EMATeacher,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader | None,
        epoch_idx: int,
    ) -> float:
        if loader is None:
            return 0.0
        total_loss = 0.0
        total_count = 0
        for x, signed_y, _true_y, _idx, _pseudo in loader:
            x = x.to(self.device)
            signed_y = signed_y.to(self.device)
            logits = model(x).view(-1)
            if self.selfpu_cfg.soft_label:
                loss = sigmoid_entropy_loss(logits)
            else:
                loss = self.criterion(logits, signed_y)
            if self._check_mean_teacher(epoch_idx):
                with torch.no_grad():
                    teacher_logits = teacher.model(x).view(-1)
                loss = (
                    loss
                    + self._consistency_weight(epoch_idx)
                    * softmax_mse_consistency(logits, teacher_logits)
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(x.shape[0])
            total_count += int(x.shape[0])
        return total_loss / max(1, total_count)

    def _run_noisy_branch(
        self,
        *,
        model: torch.nn.Module,
        peer_model: torch.nn.Module,
        teacher: EMATeacher,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        epoch_idx: int,
    ) -> float:
        total_loss = 0.0
        total_count = 0
        mutual_accepts = 0
        mutual_batches = 0
        for x, pu_y, _true_y, _idx, _pseudo in loader:
            x = x.to(self.device)
            pu_y = pu_y.to(self.device)
            logits = model(x).view(-1)
            pu_loss = self.criterion_unlabeled(logits, pu_y)
            loss = self.selfpu_cfg.pu_loss_weight * pu_loss

            if self.selfpu_cfg.mutual_type == "mu" and self._check_mean_teacher(
                epoch_idx
            ):
                with torch.no_grad():
                    peer_logits = peer_model(x).view(-1)
                mutual, accepted = mutual_student_consistency(
                    logits,
                    peer_logits,
                    pu_loss,
                    self.selfpu_cfg.mutual_alpha,
                )
                mutual_batches += 1
                mutual_accepts += int(accepted)
                if accepted:
                    loss = loss + mutual

            if self._check_mean_teacher(epoch_idx):
                with torch.no_grad():
                    teacher_logits = teacher.model(x).view(-1)
                loss = (
                    loss
                    + self._consistency_weight(epoch_idx)
                    * softmax_mse_consistency(logits, teacher_logits)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(x.shape[0])
            total_count += int(x.shape[0])

        if mutual_batches:
            key = "selfpu_mutual_accept_rate"
            previous = self._epoch_stats.get(key)
            value = float(mutual_accepts / mutual_batches)
            self._epoch_stats[key] = value if previous is None else (previous + value) / 2
        return total_loss / max(1, total_count)

    def _maybe_start_mean_teacher(self, epoch_idx: int) -> None:
        if not self._check_mean_teacher(epoch_idx) or self._mean_teacher_switched:
            return
        self.teacher1.copy_from(self.model)
        self.teacher2.copy_from(self.model2)
        self._mean_teacher_switched = True
        self.console.log("Self-PU mean-teacher EMA initialized from students.")

    def _update_teachers_once(self, epoch_idx: int) -> None:
        if self._check_mean_teacher(epoch_idx):
            self.teacher1.update(self.model, self.selfpu_cfg.ema_decay)
            self.teacher2.update(self.model2, self.selfpu_cfg.ema_decay)

    def _check_mean_teacher(self, epoch_idx: int) -> bool:
        cfg = self.selfpu_cfg
        return cfg.mean_teacher_enabled and epoch_idx >= cfg.mean_teacher_start

    def _check_self_paced(self, epoch_idx: int) -> bool:
        cfg = self.selfpu_cfg
        if not cfg.self_paced_enabled:
            return False
        if epoch_idx < cfg.self_paced_start:
            return False
        if epoch_idx >= cfg.self_paced_stop:
            return False
        return True

    def _consistency_weight(self, epoch_idx: int) -> float:
        cfg = self.selfpu_cfg
        if not self._check_mean_teacher(epoch_idx):
            return 0.0
        return cfg.consistency_weight * sigmoid_rampup(
            epoch_idx - 30,
            cfg.consistency_rampup,
        )

    def _maybe_update_self_paced_sets(self, epoch_idx: int) -> None:
        if not self._check_self_paced(epoch_idx):
            return
        cfg = self.selfpu_cfg
        if (epoch_idx - cfg.self_paced_start) % cfg.self_paced_frequency != 0:
            return
        if self._last_selection_epoch == epoch_idx:
            return
        t0 = time.time()
        scores1 = self._predict_scores(
            self.teacher1.model if self._check_mean_teacher(epoch_idx) else self.model
        )
        scores2 = self._predict_scores(
            self.teacher2.model if self._check_mean_teacher(epoch_idx) else self.model2
        )
        self.state1 = self.selector.select(
            scores1,
            epoch=epoch_idx,
            top=cfg.top1,
            previous_pseudo_pos_indices=self.state1.pseudo_pos_indices,
            previous_pseudo_neg_indices=self.state1.pseudo_neg_indices,
            ratio=0.5,
        )
        self.state2 = self.selector.select(
            scores2,
            epoch=epoch_idx,
            top=cfg.top2,
            previous_pseudo_pos_indices=self.state2.pseudo_pos_indices,
            previous_pseudo_neg_indices=self.state2.pseudo_neg_indices,
            ratio=0.5,
        )
        self.clean_loader1 = self._make_clean_loader(self.state1)
        self.clean_loader2 = self._make_clean_loader(self.state2)
        self.noisy_loader1 = self._make_noisy_loader(self.state1.noisy_indices)
        self.noisy_loader2 = self._make_noisy_loader(self.state2.noisy_indices)
        self._last_selection_epoch = epoch_idx
        self._log_selection(epoch_idx, time.time() - t0)

    @torch.no_grad()
    def _predict_scores(self, model: torch.nn.Module) -> np.ndarray:
        model.eval()
        scores = np.zeros(len(self.train_loader.dataset), dtype=np.float32)
        for x, _pu_y, _true_y, indices, _pseudo in self.update_loader:
            x = x.to(self.device)
            logits = model(x).view(-1)
            scores[indices.detach().cpu().numpy()] = (
                torch.sigmoid(logits).detach().cpu().numpy()
            )
        model.train()
        return scores

    def _log_selection(self, epoch_idx: int, elapsed: float) -> None:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.2%}"

        self.console.log(
            "Self-PU selection "
            f"epoch={epoch_idx}: "
            f"model1 clean={len(self.state1.clean_indices)} "
            f"(P={len(self.state1.pseudo_pos_indices)}, "
            f"N={len(self.state1.pseudo_neg_indices)}, "
            f"precP={fmt(self.state1.pos_precision)}, "
            f"precN={fmt(self.state1.neg_precision)}); "
            f"model2 clean={len(self.state2.clean_indices)} "
            f"(P={len(self.state2.pseudo_pos_indices)}, "
            f"N={len(self.state2.pseudo_neg_indices)}, "
            f"precP={fmt(self.state2.pos_precision)}, "
            f"precN={fmt(self.state2.neg_precision)}); "
            f"elapsed={elapsed:.1f}s"
        )

    def _candidate_models(self) -> list[tuple[str, torch.nn.Module]]:
        candidates = [("student1", self.model), ("student2", self.model2)]
        if self._mean_teacher_switched:
            candidates.extend(
                [("teacher1", self.teacher1.model), ("teacher2", self.teacher2.model)]
            )
        return candidates

    def _reporting_metrics(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Report the best Self-PU candidate instead of hard-wiring student1.

        The author source validates ``student1``, ``student2`` and, after the
        mean-teacher switch, both EMA teachers, then keeps the best one.  PU-Bench
        cannot use oracle labels for benchmark model selection, so this
        controlled entry follows the benchmark proxy-accuracy monitor.  If PA
        starts preferring a degenerate candidate, first rule out numerical
        instability in the training loop before changing the proxy metric.
        """

        scenario = self.params.get("scenario", "single")
        selection_loader = self.validation_loader or self.train_loader
        candidate_records = []
        for candidate_id, (name, model) in enumerate(self._candidate_models(), start=1):
            proxy = evaluate_proxy_metrics(
                model,
                selection_loader,
                self.device,
                self.prior,
                scenario,
            )
            score = float(proxy.get("proxy_acc", float("-inf")))
            if np.isnan(score):
                score = float("-inf")
            candidate_records.append((score, candidate_id, name, model, proxy))

        _, candidate_id, name, model, selected_proxy = max(
            candidate_records,
            key=lambda item: item[0],
        )
        if name != self._last_report_model_name:
            self.console.log(f"Self-PU reporting candidate: {name}")
            self._last_report_model_name = name
        self._last_report_model = model
        prior_calibrated_fallback = self._oracle_prior_calibrated_fallback()

        train_metrics = evaluate_metrics(
            model,
            self.train_loader,
            self.device,
            self.prior,
            prior_calibrated_fallback=prior_calibrated_fallback,
        )
        train_metrics.update(
            evaluate_proxy_metrics(
                model,
                self.train_loader,
                self.device,
                self.prior,
                scenario,
            )
        )
        train_metrics["selfpu_report_model_id"] = float(candidate_id)

        val_metrics: dict[str, float] = {}
        if self.validation_loader is not None:
            val_metrics = evaluate_metrics(
                model,
                self.validation_loader,
                self.device,
                self.prior,
                prior_calibrated_fallback=prior_calibrated_fallback,
            )
            val_metrics.update(selected_proxy)
            val_metrics["selfpu_report_model_id"] = float(candidate_id)

        test_metrics = evaluate_metrics(
            model,
            self.test_loader,
            self.device,
            self.prior,
            prior_calibrated_fallback=prior_calibrated_fallback,
        )
        test_metrics["selfpu_report_model_id"] = float(candidate_id)
        return train_metrics, val_metrics, test_metrics

    def get_checkpoint_model(self):
        return self._last_report_model or self.model

    def get_extra_epoch_metrics(self):
        train_metrics, val_metrics, test_metrics = self._reporting_metrics()
        train_metrics.update(dict(self._epoch_stats))

        if self._last_completed_epoch is not None:
            # Source order: train/validate first, then refresh the self-paced
            # split for the next epoch.
            self._maybe_update_self_paced_sets(self._last_completed_epoch)

        return train_metrics, val_metrics, test_metrics
