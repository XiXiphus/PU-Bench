"""Robust-PU trainer adapted to PU-Bench.

Primary sources:
    - Paper: Robust Positive-Unlabeled Learning via Noise Negative Sample
      Self-correction, KDD 2023.
    - Code snapshot:
      author source file(s): robustpu
      woriazzc/Robust-PU at 34d950f2c6e56510855a922acb5f84b6459773ef

Source facts preserved here:
    - PU labels are +1 for labeled positives and -1 for unlabeled examples.
    - Stage 1 pretrains with nnPU by default.
    - Stage 2 alternates hardness measurement, self-paced sample weighting,
      and weighted supervised training with U treated as binary negative.
    - Hardness is computed separately for labeled positives and U using
      temperature-scaled positive/negative pseudo-labels.
    - The default source SPL map is Welsch; benchmark configs may choose other
      source-supported maps explicitly.

Benchmark boundary:
    This is the controlled-backbone PU-Bench entry.  The source paper's dataset
    recipes are not reproduced; the estimator is run on PU-Bench's own
    SCAR/SAR splits to measure robustness.  PU-Bench checkpointing and
    pretraining restore use validation proxy accuracy.  True labels are
    diagnostics only; they are never used for model selection, calibration,
    losses, or sample weights.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..base_trainer import BaseTrainer
from ..metrics import evaluate_metrics, evaluate_proxy_metrics
from ..reproducibility import seed_worker
from .losses import (
    binary_cross_entropy_loss,
    focal_binary_loss,
    hardness_values,
    source_pu_loss,
)
from .model_selector import select_model
from .spl import TrainingScheduler, calculate_spl_weights


@dataclass
class RobustPUStageConfig:
    pre_epochs: int
    pre_lr: float
    pre_weight_decay: float
    pre_batch_size: int
    pre_loss: str
    pre_optimizer: str
    pre_monitor: str | None
    pre_calibration: str | None
    episodes: int
    inner_epochs: int
    main_lr: float
    main_weight_decay: float
    main_batch_size: int
    main_loss: str
    main_optimizer: str
    hardness: str
    spl_type: str
    temper_p: float
    temper_n: float
    focal_gamma: float
    phi: float
    restart: bool


class RobustPUTrainer(BaseTrainer):
    """Source-aligned Robust-PU trainer for PU-Bench datasets."""

    def create_model(self):
        return select_model(params=self.params, prior=self.prior)

    def _prepare_data(self):
        super()._prepare_data()
        self._validate_data_contract()

    def _validate_data_contract(self) -> None:
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "pu_labels"):
            raise ValueError("RobustPU requires a PU dataset exposing `pu_labels`.")

        metadata = getattr(dataset, "pu_metadata", {}) or {}
        expected_labels = {
            "pu_labeled_label": 1,
            "pu_unlabeled_label": -1,
        }
        for key, expected in expected_labels.items():
            observed = metadata.get(key, expected)
            if int(observed) != expected:
                raise ValueError(
                    "RobustPU requires PU labels +1 for labeled positives and "
                    f"-1 for unlabeled samples; got {key}={observed}."
                )

        labels = {
            int(value)
            for value in torch.unique(dataset.pu_labels.detach().cpu()).tolist()
        }
        if 1 not in labels or -1 not in labels:
            raise ValueError(
                "RobustPU requires both labeled-positive (+1) and unlabeled (-1) "
                f"examples; observed PU labels {sorted(labels)}."
            )

        pi_unlabeled = metadata.get("pi_unlabeled")
        if (
            pi_unlabeled is not None
            and abs(float(pi_unlabeled) - float(self.prior)) > 1e-8
        ):
            raise ValueError(
                "RobustPU prior must be the positive fraction inside U "
                f"(pi_unlabeled={pi_unlabeled}), got prior={self.prior}."
            )

    def before_training(self):
        super().before_training()
        self._sync_prior_from_pu_metadata()
        self.robust_cfg = self._parse_stage_config()
        self._validate_proxy_acc_monitor_policy()
        self.scheduler_p = self._make_scheduler("scheduler_p", default_alpha=0.1)
        self.scheduler_n = self._make_scheduler("scheduler_n", default_alpha=0.11)
        self._moving_weights_by_index: dict[int, float] | None = None
        self.last_weight_stats: dict[str, float] = {}

    def create_criterion(self):
        return nn.Identity()

    def train_one_epoch(self, epoch_idx: int):
        raise RuntimeError("RobustPUTrainer uses its source two-stage run loop.")

    def run(self):
        self.before_training()
        original_early_stopping = (
            bool(self.checkpoint_handler.early_stopping_enabled)
            if self.checkpoint_handler
            else False
        )

        if self.robust_cfg.pre_epochs > 0:
            self.set_checkpoint_early_stopping(False)
            self.console.log(
                "--- [Stage 1/2] Robust-PU: nnPU pre-training ---",
                style="bold yellow",
            )
            if self.file_console:
                self.file_console.log("--- [Stage 1/2] Robust-PU: nnPU pre-training ---")
            self._pre_train()

        if self.robust_cfg.episodes > 0:
            self._reset_checkpoint_for_main_stage()
            self.set_checkpoint_early_stopping(original_early_stopping, reset=True)
            self.console.log(
                "--- [Stage 2/2] Robust-PU: self-paced weighted training ---",
                style="bold yellow",
            )
            if self.file_console:
                self.file_console.log(
                    "--- [Stage 2/2] Robust-PU: self-paced weighted training ---"
                )
            self._run_self_paced_training()

        self.after_training()
        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()
        self._close_file_console()
        return self.checkpoint_handler.best_metrics if self.checkpoint_handler else {}

    def get_extra_epoch_metrics(self) -> tuple[dict, dict, dict]:
        return dict(self.last_weight_stats), {}, {}

    def _parse_stage_config(self) -> RobustPUStageConfig:
        pre = self.params.get("pre_train", {}) or {}
        main = self.params.get("main_train", {}) or {}
        scheduler_p = main.get("scheduler_p", {}) or {}
        scheduler_n = main.get("scheduler_n", {}) or {}

        return RobustPUStageConfig(
            pre_epochs=int(pre.get("epochs", self.params.get("pre_epochs", 0))),
            pre_lr=float(pre.get("lr", self.params.get("pre_lr", 1e-3))),
            pre_weight_decay=float(
                pre.get("weight_decay", self.params.get("pre_wd", 0.0))
            ),
            pre_batch_size=int(
                pre.get(
                    "batch_size",
                    self.params.get(
                        "pre_batch_size",
                        self.params.get("batch_size", 128),
                    ),
                )
            ),
            pre_loss=str(pre.get("loss", self.params.get("pre_loss", "nnpu"))).lower(),
            pre_optimizer=str(
                pre.get("optimizer", self.params.get("pre_optimizer", "adam"))
            ).lower(),
            pre_monitor=(
                str(pre.get("monitor", pre.get("selection_monitor")))
                if pre.get("monitor", pre.get("selection_monitor")) is not None
                else None
            ),
            pre_calibration=(
                str(pre.get("calibration", pre.get("logit_calibration")))
                if pre.get("calibration", pre.get("logit_calibration")) is not None
                else None
            ),
            episodes=int(main.get("epochs", self.params.get("epochs", 100))),
            inner_epochs=int(
                main.get("inner_epochs", self.params.get("inner_epochs", 1))
            ),
            main_lr=float(main.get("lr", self.params.get("lr", 1e-4))),
            main_weight_decay=float(main.get("weight_decay", self.params.get("wd", 0.0))),
            main_batch_size=int(
                main.get(
                    "batch_size",
                    self.params.get("batch_size", 64),
                )
            ),
            main_loss=str(main.get("loss", self.params.get("loss", "bce"))).lower(),
            main_optimizer=str(
                main.get("optimizer", self.params.get("optimizer", "adam"))
            ).lower(),
            hardness=str(main.get("hardness", self.params.get("hardness", "logistic"))).lower(),
            spl_type=str(main.get("spl_type", self.params.get("spl_type", "welsch"))).lower(),
            temper_p=float(scheduler_p.get("temper", main.get("temper_p", 1.0))),
            temper_n=float(scheduler_n.get("temper", main.get("temper_n", 1.3))),
            focal_gamma=float(main.get("focal_gamma", self.params.get("focal_gamma", 1.0))),
            phi=float(main.get("moving_ratio", self.params.get("phi", 0.0))),
            restart=bool(main.get("restart", self.params.get("restart", False))),
        )

    def _validate_proxy_acc_monitor_policy(self) -> None:
        monitors = []
        if self.checkpoint_handler:
            monitors.append(("checkpoint.monitor", self.checkpoint_handler.monitor))
        if self.robust_cfg.pre_monitor:
            monitors.append(("pre_train.monitor", self.robust_cfg.pre_monitor))

        for name, monitor in monitors:
            if str(monitor) != "val_proxy_acc":
                raise ValueError(
                    "RobustPU benchmark selection must use val_proxy_acc under "
                    f"PU-Bench policy; got {name}={monitor!r}."
                )

        calibration = self.robust_cfg.pre_calibration
        if calibration and str(calibration).lower() not in {"none", "false", "off"}:
            raise ValueError(
                "RobustPU benchmark entry disables oracle-label pretrain "
                f"calibration; got pre_train.calibration={calibration!r}."
            )

    def _make_scheduler(self, name: str, *, default_alpha: float) -> TrainingScheduler:
        main = self.params.get("main_train", {}) or {}
        cfg = main.get(name, {}) or {}
        return TrainingScheduler(
            schedule_type=str(cfg.get("type", cfg.get("scheduler_type", "linear"))),
            init_ratio=float(cfg.get("alpha", default_alpha)),
            max_thresh=float(cfg.get("max_thresh", 1.0)),
            grow_steps=int(cfg.get("grow_steps", 5)),
            lam=float(cfg.get("lam", cfg.get("eta", 0.5))),
        )

    def _sync_prior_from_pu_metadata(self) -> None:
        metadata = getattr(self.train_loader.dataset, "pu_metadata", {}) or {}
        risk_prior = float(metadata.get("pi_unlabeled", self.prior))
        if abs(risk_prior - float(self.prior)) > 1e-4:
            self.console.log(
                f"Using PU-Bench unlabeled-mixture prior pi_U={risk_prior:.4f} "
                f"instead of initial prior {float(self.prior):.4f}.",
                style="bold yellow",
            )
        self.prior = risk_prior

    def _set_checkpoint_early_stopping(self, enabled: bool, *, reset: bool = False) -> None:
        self.set_checkpoint_early_stopping(enabled, reset=reset)

    def _reset_checkpoint_for_main_stage(self) -> None:
        """Keep pretraining as initialization, not as the selected RobustPU result."""
        self.reset_checkpoint_tracking()

    def _make_source_optimizer(
        self,
        optimizer_name: str,
        *,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        params = self.model.parameters()
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=lr,
                momentum=float(self.params.get("momentum", 0.9)),
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported RobustPU optimizer: {optimizer_name}")

    def _pre_train(self) -> None:
        cfg = self.robust_cfg
        optimizer = self._make_source_optimizer(
            cfg.pre_optimizer,
            lr=cfg.pre_lr,
            weight_decay=cfg.pre_weight_decay,
        )
        criterion = self._make_pu_loss(cfg.pre_loss)
        pretrain_loader = self._make_pretrain_loader()

        best_state: dict[str, Any] | None = None
        best_pretrain_score = float("-inf")
        num_epochs = int(cfg.pre_epochs)

        for epoch_idx in tqdm(
            range(1, num_epochs + 1),
            desc=f"Pre-training ({self.method.upper()})",
        ):
            self.model.train()
            for x, pu_labels, _y_true, _idx, _pseudo in pretrain_loader:
                x = self._source_input(x).to(self.device)
                pu_labels = pu_labels.to(self.device)
                logits = self._positive_logit(self.model(x))
                loss = self._stage_loss(criterion, cfg.pre_loss, logits, pu_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.global_epoch += 1
            train_metrics, val_metrics, test_metrics = self._evaluate_current_model()
            self._print_metrics(
                epoch_idx,
                num_epochs,
                train_metrics,
                test_metrics,
                "Pre-training",
                val_metrics=val_metrics,
            )
            self._checkpoint_epoch(train_metrics, val_metrics, test_metrics)

            pretrain_score = self._internal_selection_score(
                train_metrics,
                val_metrics,
                test_metrics,
                monitor_override=cfg.pre_monitor,
            )
            if pretrain_score > best_pretrain_score:
                best_pretrain_score = pretrain_score
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None and bool(self.params.get("restore_best_pretrain", True)):
            self.model.load_state_dict(best_state)
        if cfg.pre_calibration:
            self._apply_pretrain_calibration(cfg.pre_calibration)

    def _make_pretrain_loader(self) -> DataLoader:
        return DataLoader(
            self.train_loader.dataset,
            batch_size=self.robust_cfg.pre_batch_size,
            shuffle=True,
            num_workers=int(self.params.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
        )

    def _apply_pretrain_calibration(self, calibration: str) -> None:
        calibration = str(calibration).lower()
        if calibration in {"none", "false", "off"}:
            return
        if calibration != "val_oracle_accuracy_threshold":
            raise ValueError(
                "RobustPU pretrain calibration must be "
                "'val_oracle_accuracy_threshold' or 'none'."
            )
        threshold, accuracy = self._calibrate_final_bias_with_validation()
        message = (
            "RobustPU validation logit calibration: "
            f"threshold={threshold:.6g}, bias_delta={-threshold:.6g}, "
            f"val_oracle_accuracy={accuracy:.6g}"
        )
        self.console.log(message)
        if self.file_console:
            self.file_console.log(message)

    def _calibrate_final_bias_with_validation(self) -> tuple[float, float]:
        if self.validation_loader is None:
            raise ValueError(
                "RobustPU validation calibration requires a validation loader. "
                "Set a positive val_ratio or disable pre_train.calibration."
            )
        logits, true_labels = self._collect_logits_and_true_labels(self.validation_loader)
        threshold, accuracy = self._best_binary_accuracy_threshold(logits, true_labels)
        head = self._last_single_logit_linear()
        if head is None:
            raise ValueError(
                "RobustPU validation calibration requires a single-logit linear head."
            )
        with torch.no_grad():
            head.bias.add_(
                torch.as_tensor(
                    -threshold,
                    device=head.bias.device,
                    dtype=head.bias.dtype,
                )
            )
        return threshold, accuracy

    def _collect_logits_and_true_labels(
        self,
        loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        training = self.model.training
        self.model.eval()
        logits_all: list[torch.Tensor] = []
        true_all: list[torch.Tensor] = []
        with torch.no_grad():
            for x, _pu_labels, y_true, _idx, _pseudo in loader:
                x = self._source_input(x).to(self.device)
                logits = self._positive_logit(self.model(x))
                logits_all.append(logits.detach().cpu().view(-1))
                true_all.append(y_true.detach().cpu().view(-1).long())
        self.model.train(training)
        return torch.cat(logits_all), torch.cat(true_all)

    @staticmethod
    def _best_binary_accuracy_threshold(
        logits: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> tuple[float, float]:
        logits = logits.detach().cpu().view(-1).float()
        true_labels = true_labels.detach().cpu().view(-1).long()
        if logits.numel() == 0:
            raise ValueError("Cannot calibrate RobustPU logits on an empty validation set.")
        sorted_logits, _ = torch.sort(logits)
        candidates = [float(sorted_logits[0].item() - 1.0)]
        candidates.extend(
            float(((sorted_logits[idx - 1] + sorted_logits[idx]) * 0.5).item())
            for idx in range(1, sorted_logits.numel())
        )
        candidates.append(float(sorted_logits[-1].item() + 1.0))

        best_threshold = candidates[0]
        best_accuracy = -1.0
        for threshold in candidates:
            predictions = (logits > threshold).long()
            accuracy = float(predictions.eq(true_labels).float().mean().item())
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        return best_threshold, best_accuracy

    def _last_single_logit_linear(self) -> nn.Linear | None:
        found: nn.Linear | None = None
        for module in self.model.modules():
            if (
                isinstance(module, nn.Linear)
                and int(getattr(module, "out_features", 0)) == 1
                and getattr(module, "bias", None) is not None
            ):
                found = module
        return found

    def _run_self_paced_training(self) -> None:
        cfg = self.robust_cfg
        for episode in tqdm(
            range(1, cfg.episodes + 1),
            desc=f"Episodes ({self.method.upper()})",
        ):
            thresh_p = self.scheduler_p.get_next_ratio()
            thresh_n = self.scheduler_n.get_next_ratio()
            self.console.log(
                f"RobustPU episode {episode}: thresh_p={thresh_p:.3f}, "
                f"thresh_n={thresh_n:.3f}"
            )

            weighted_loader = self._create_weighted_dataloader(thresh_p, thresh_n)
            if cfg.restart:
                self._reset_model_parameters()

            optimizer = self._make_source_optimizer(
                cfg.main_optimizer,
                lr=cfg.main_lr,
                weight_decay=cfg.main_weight_decay,
            )
            criterion = self._make_pu_loss(cfg.main_loss)
            last_train_metrics = None
            last_val_metrics = None
            last_test_metrics = None

            for inner_epoch in range(1, cfg.inner_epochs + 1):
                self._train_weighted_epoch(weighted_loader, optimizer, criterion)
                self.global_epoch += 1

                train_metrics, val_metrics, test_metrics = self._evaluate_current_model()
                self._print_metrics(
                    inner_epoch,
                    cfg.inner_epochs,
                    train_metrics,
                    test_metrics,
                    f"Episode {episode}",
                    val_metrics=val_metrics,
                )
                last_train_metrics = train_metrics
                last_val_metrics = val_metrics
                last_test_metrics = test_metrics

            if last_train_metrics is not None and last_test_metrics is not None:
                self._checkpoint_epoch(
                    last_train_metrics,
                    last_val_metrics,
                    last_test_metrics,
                )
            if self.checkpoint_handler and self.checkpoint_handler.should_stop:
                self.console.log(
                    f"Early stopping in RobustPU episode {episode}.",
                    style="bold red",
                )
                return

    def _create_weighted_dataloader(
        self,
        threshold_p: float,
        threshold_n: float,
    ) -> DataLoader:
        cfg = self.robust_cfg
        if cfg.hardness in {"distance", "cos"}:
            raise RuntimeError(
                "RobustPU distance/cos hardness needs source feature hooks and is "
                "not enabled in the controlled PU-Bench entry. Use logistic or sigmoid."
            )

        source_loader = self._make_weight_source_loader()
        self.model.eval()
        data_all: list[torch.Tensor] = []
        pu_all: list[torch.Tensor] = []
        true_all: list[torch.Tensor] = []
        index_all: list[torch.Tensor] = []
        weights_all: list[torch.Tensor] = []

        with torch.no_grad():
            for x, pu_labels, y_true, indices, _pseudo in source_loader:
                data_all.append(x.cpu())
                pu_all.append(pu_labels.cpu())
                true_all.append(y_true.cpu())
                index_all.append(indices.cpu())

                x_dev = self._source_input(x).to(self.device)
                logits = self._positive_logit(self.model(x_dev))
                pu_dev = pu_labels.to(self.device)

                hardness_n = hardness_values(
                    cfg.hardness,
                    logits / cfg.temper_n,
                    -1,
                    gamma=cfg.focal_gamma,
                )
                weights = calculate_spl_weights(
                    hardness_n,
                    threshold_n,
                    spl_type=cfg.spl_type,
                    mix2_gamma=float(self.params.get("mix2_gamma", 1.0)),
                    poly_t=float(self.params.get("poly_t", 3.0)),
                )

                pos_mask = pu_dev == 1
                if pos_mask.any():
                    hardness_p = hardness_values(
                        cfg.hardness,
                        logits[pos_mask] / cfg.temper_p,
                        1,
                        gamma=cfg.focal_gamma,
                    )
                    weights[pos_mask] = calculate_spl_weights(
                        hardness_p,
                        threshold_p,
                        spl_type=cfg.spl_type,
                        mix2_gamma=float(self.params.get("mix2_gamma", 1.0)),
                        poly_t=float(self.params.get("poly_t", 3.0)),
                    )
                weights_all.append(weights.cpu())

        data = torch.cat(data_all, dim=0)
        pu_labels = torch.cat(pu_all, dim=0)
        true_labels = torch.cat(true_all, dim=0)
        indices = torch.cat(index_all, dim=0).long()
        weights = torch.cat(weights_all, dim=0).float()

        if cfg.phi > 0.0:
            weights = self._apply_moving_average(indices, weights, cfg.phi)

        self._record_weight_stats(pu_labels, true_labels, weights)
        weighted_dataset = TensorDataset(data, pu_labels, true_labels, weights)
        return DataLoader(
            weighted_dataset,
            batch_size=cfg.main_batch_size,
            shuffle=True,
            num_workers=int(self.params.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
        )

    def _make_weight_source_loader(self) -> DataLoader:
        dataset = (self.update_loader or self.train_loader).dataset
        return DataLoader(
            dataset,
            batch_size=self.robust_cfg.main_batch_size,
            shuffle=True,
            num_workers=int(self.params.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
        )

    def _apply_moving_average(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        phi: float,
    ) -> torch.Tensor:
        if self._moving_weights_by_index is None:
            self._moving_weights_by_index = {
                int(idx): float(w) for idx, w in zip(indices.tolist(), weights.tolist())
            }
            return weights

        updated = []
        for idx, weight in zip(indices.tolist(), weights.tolist()):
            old = self._moving_weights_by_index.get(int(idx), float(weight))
            new = float(phi) * old + (1.0 - float(phi)) * float(weight)
            self._moving_weights_by_index[int(idx)] = new
            updated.append(new)
        return torch.tensor(updated, dtype=weights.dtype)

    def _record_weight_stats(
        self,
        pu_labels: torch.Tensor,
        true_labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        pos_mask = pu_labels == 1
        unl_mask = pu_labels != 1
        neg_unl_mask = unl_mask & (true_labels == 0)
        pos_unl_mask = unl_mask & (true_labels == 1)

        def _mean(mask: torch.Tensor) -> float:
            return float(weights[mask].mean().item()) if mask.any() else float("nan")

        self.last_weight_stats = {
            "robustpu_weight_labeled_mean": _mean(pos_mask),
            "robustpu_weight_unlabeled_mean": _mean(unl_mask),
            "robustpu_weight_unlabeled_neg_mean": _mean(neg_unl_mask),
            "robustpu_weight_unlabeled_pos_mean": _mean(pos_unl_mask),
            "robustpu_weight_ess": float(
                (weights.sum().item() ** 2)
                / max(float(torch.sum(weights.pow(2)).item()), 1e-12)
            ),
        }

    def _train_weighted_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
    ) -> None:
        self.model.train()
        for x, pu_labels, _true_labels, weights in dataloader:
            x = self._source_input(x).to(self.device)
            pu_labels = pu_labels.to(self.device)
            weights = weights.to(self.device)
            if weights.sum().item() <= 1e-8:
                continue
            logits = self._positive_logit(self.model(x))
            loss = self._stage_loss(
                criterion,
                self.robust_cfg.main_loss,
                logits,
                pu_labels,
                weights,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _make_pu_loss(self, loss_name: str):
        if loss_name == "nnpu":
            return lambda logits, labels, weights=None: source_pu_loss(
                logits,
                labels,
                self.prior,
                weights,
                sur_loss="sigmoid",
                nnpu=True,
            )
        if loss_name == "upu":
            return lambda logits, labels, weights=None: source_pu_loss(
                logits,
                labels,
                self.prior,
                weights,
                sur_loss="sigmoid",
                nnpu=False,
            )
        if loss_name in {"bce", "focal"}:
            return loss_name
        raise ValueError("RobustPU loss must be one of 'bce', 'focal', 'nnpu', 'upu'.")

    def _stage_loss(
        self,
        criterion,
        loss_name: str,
        logits: torch.Tensor,
        pu_labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if loss_name == "bce":
            return binary_cross_entropy_loss(logits, pu_labels, weights)
        if loss_name == "focal":
            return focal_binary_loss(
                logits,
                pu_labels,
                weights,
                gamma=self.robust_cfg.focal_gamma,
            )
        return criterion(logits, pu_labels, weights)

    def _evaluate_current_model(self):
        scenario = self.params.get("scenario", "single")
        prior_calibrated_fallback = self._oracle_prior_calibrated_fallback()
        train_metrics = evaluate_metrics(
            self.model,
            self.train_loader,
            self.device,
            self.prior,
            prior_calibrated_fallback=prior_calibrated_fallback,
        )
        train_metrics.update(
            evaluate_proxy_metrics(
                self.model,
                self.train_loader,
                self.device,
                self.prior,
                scenario,
            )
        )
        val_metrics = (
            evaluate_metrics(
                self.model,
                self.validation_loader,
                self.device,
                self.prior,
                prior_calibrated_fallback=prior_calibrated_fallback,
            )
            if self.validation_loader is not None
            else None
        )
        if val_metrics is not None:
            val_metrics.update(
                evaluate_proxy_metrics(
                    self.model,
                    self.validation_loader,
                    self.device,
                    self.prior,
                    scenario,
                )
            )
        test_metrics = evaluate_metrics(
            self.model,
            self.test_loader,
            self.device,
            self.prior,
            prior_calibrated_fallback=prior_calibrated_fallback,
        )
        if self.last_weight_stats:
            train_metrics.update(self.last_weight_stats)
        return train_metrics, val_metrics, test_metrics

    def _internal_selection_score(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
        test_metrics: dict[str, float],
        monitor_override: str | None = None,
    ) -> float:
        """Select pretrain initialization by PU-Bench validation proxy accuracy."""
        if monitor_override:
            monitor = monitor_override
        elif self.checkpoint_handler:
            monitor = getattr(self.checkpoint_handler, "monitor", None)
        else:
            monitor = None
        monitor = monitor or "val_proxy_acc"
        if "_" in monitor:
            phase, key = monitor.split("_", 1)
        else:
            phase, key = "val", monitor
        if phase == "test":
            raise ValueError(
                "RobustPU does not allow test metrics for model selection. "
                f"Got monitor '{monitor}'."
            )
        if key != "proxy_acc":
            raise ValueError(
                "RobustPU benchmark model selection must use proxy_acc. "
                f"Got monitor '{monitor}'."
            )

        phase_metrics = {
            "train": train_metrics,
            "val": val_metrics,
        }
        metrics = phase_metrics.get(phase)
        if metrics is not None and key in metrics:
            return float(metrics[key])

        for metrics in (val_metrics, train_metrics):
            if metrics is None:
                continue
            if "proxy_acc" in metrics:
                return float(metrics["proxy_acc"])
        return float("-inf")

    def _checkpoint_epoch(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
        test_metrics: dict[str, float],
    ) -> None:
        if not self.checkpoint_handler:
            return
        all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        if val_metrics is not None:
            all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        self.checkpoint_handler(
            epoch=self.global_epoch,
            all_metrics=all_metrics,
            model=self.model,
            elapsed_seconds=(time.time() - self._run_start_time)
            if self._run_start_time
            else None,
        )

    def _positive_logit(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.dim() > 1 and outputs.shape[1] > 1:
            pos_idx = int(getattr(self.model, "positive_logit_index", 1))
            return outputs[:, pos_idx].view(-1)
        return outputs.view(-1)

    def _source_input(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def _reset_model_parameters(self) -> None:
        if hasattr(self.model, "reset_para"):
            self.model.reset_para()
            return
        for module in self.model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
