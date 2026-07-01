"""Packaged VPU trainer.

Source-faithful VPU trainer.

Primary source:
    https://github.com/HC-Feynman/vpu
    - `vpu.py`: variational loss, separate P/X loaders, mixup regularizer,
      Adam betas, and 20-epoch learning-rate decay.
    - `run.py`: source defaults for mix_alpha, lam, val_iterations, and epochs.

Implementation contract:
    - VPU does not use a PU class prior in its objective.
    - Training uses a labeled-positive loader P and an all-training-data loader X.
    - The learned model represents log phi; for single-logit PU-Bench backbones
      we use the equivalent binary log-softmax parameterization.
    - Source-faithful runs should not initialize classifier bias from prior;
      BaseTrainer disables that default for `method == "vpu"`.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..base_trainer import BaseTrainer
from ..utils.reproducibility import seed_worker
from .losses import VPULoss


class VPUTrainer(BaseTrainer):
    """Variational Positive-Unlabeled learning trainer."""

    def _prepare_data(self):
        super()._prepare_data()
        self._validate_source_data_contract()
        self._build_source_loaders()

    def create_criterion(self):
        args = argparse.Namespace(
            mix_alpha=self.params.get("mix_alpha", 0.3),
            lam=self.params.get("lam", 0.03),
        )
        return VPULoss(args)

    def _validate_source_data_contract(self) -> None:
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "pu_labels"):
            raise ValueError("VPU requires a PU dataset exposing `pu_labels`.")

        metadata = getattr(dataset, "pu_metadata", {}) or {}
        labeled_label = int(metadata.get("pu_labeled_label", 1))
        if labeled_label != 1:
            raise ValueError(
                "VPU source reproduction expects labeled positives to use PU label +1; "
                f"got pu_labeled_label={labeled_label}."
            )

        labels = {
            int(value)
            for value in torch.unique(dataset.pu_labels.detach().cpu()).tolist()
        }
        if 1 not in labels:
            raise ValueError("VPU requires at least one labeled-positive (+1) sample.")

    def _build_source_loaders(self) -> None:
        """Build source-style P and X loaders without changing metric loaders."""
        dataset = self.train_loader.dataset
        batch_size = int(self.params.get("batch_size", 500))
        num_workers = int(self.params.get("num_workers", 0))
        pin_memory = bool(torch.cuda.is_available())

        pu_labels = dataset.pu_labels.detach().cpu()
        positive_indices = (
            torch.nonzero(pu_labels == 1, as_tuple=False).view(-1).tolist()
        )
        if not positive_indices:
            raise ValueError("VPU requires a non-empty labeled-positive set P.")

        drop_x = (
            bool(self.params.get("vpu_drop_last", True)) and len(dataset) >= batch_size
        )
        drop_p = (
            bool(self.params.get("vpu_drop_last", True))
            and len(positive_indices) >= batch_size
        )

        self.vpu_x_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_x,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
        )
        self.vpu_p_loader = DataLoader(
            Subset(dataset, positive_indices),
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_p,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
        )
        self._vpu_x_iter = None
        self._vpu_p_iter = None
        self.vpu_loss_trace: list[dict[str, float]] = []

        self.vpu_val_x_loader = None
        self.vpu_val_p_loader = None
        if self.validation_loader is not None:
            val_dataset = self.validation_loader.dataset
            val_labels = val_dataset.pu_labels.detach().cpu()
            val_positive_indices = (
                torch.nonzero(val_labels == 1, as_tuple=False).view(-1).tolist()
            )
            if val_positive_indices:
                self.vpu_val_x_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    worker_init_fn=seed_worker,
                )
                self.vpu_val_p_loader = DataLoader(
                    Subset(val_dataset, val_positive_indices),
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    worker_init_fn=seed_worker,
                )

    def _next_loader_batch(self, attr_name: str, loader: DataLoader):
        iterator = getattr(self, attr_name, None)
        try:
            batch = next(iterator)
        except Exception:
            iterator = iter(loader)
            setattr(self, attr_name, iterator)
            batch = next(iterator)
        return batch

    @staticmethod
    def _features_from_batch(batch):
        x = batch[0]
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def _log_phi_all(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        if outputs.dim() > 1 and outputs.shape[1] > 1:
            return F.log_softmax(outputs, dim=1)

        raw = outputs.view(-1)
        return torch.stack([F.logsigmoid(-raw), F.logsigmoid(raw)], dim=1)

    def _source_var_loss(self, x_loader: DataLoader, p_loader: DataLoader) -> float:
        log_phi_x_parts = []
        log_phi_p_parts = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in x_loader:
                data_x = self._features_from_batch(batch).to(self.device)
                log_phi_x_parts.append(self._log_phi_all(data_x)[:, 1])
            for batch in p_loader:
                data_p = self._features_from_batch(batch).to(self.device)
                log_phi_p_parts.append(self._log_phi_all(data_p)[:, 1])
        if was_training:
            self.model.train()

        if not log_phi_x_parts or not log_phi_p_parts:
            return float("nan")
        log_phi_x = torch.cat(log_phi_x_parts, dim=0)
        log_phi_p = torch.cat(log_phi_p_parts, dim=0)
        var_loss = (
            torch.logsumexp(log_phi_x, dim=0)
            - torch.log(torch.tensor(float(len(log_phi_x)), device=self.device))
            - torch.mean(log_phi_p)
        )
        return float(var_loss.detach().cpu().item())

    def get_extra_epoch_metrics(self) -> tuple[dict, dict, dict]:
        if self.vpu_val_x_loader is None or self.vpu_val_p_loader is None:
            return {}, {}, {}
        return (
            {},
            {
                "vpu_var_loss": self._source_var_loss(
                    self.vpu_val_x_loader,
                    self.vpu_val_p_loader,
                )
            },
            {},
        )

    def _maybe_apply_source_lr_decay(self, epoch_idx: int) -> None:
        decay_every = int(self.params.get("vpu_lr_decay_every", 20))
        if decay_every <= 0 or epoch_idx % decay_every != 0:
            return

        factor = float(self.params.get("vpu_lr_decay_factor", 0.5))
        current_lr = float(self.optimizer.param_groups[0]["lr"]) * factor
        if bool(self.params.get("vpu_reset_optimizer_on_lr_decay", True)):
            model_params = (
                self.model.params()
                if hasattr(self.model, "params")
                else self.model.parameters()
            )
            self.optimizer = self._make_optimizer(
                model_params,
                lr=current_lr,
                weight_decay=float(self.params.get("weight_decay", 0.0)),
            )
            return

        for group in self.optimizer.param_groups:
            group["lr"] = current_lr

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        self._maybe_apply_source_lr_decay(epoch_idx)

        iterations = int(
            self.params.get(
                "val_iterations", self.params.get("vpu_iterations_per_epoch", 30)
            )
        )
        beta_dist = torch.distributions.beta.Beta(
            self.criterion.mix_alpha,
            self.criterion.mix_alpha,
        )

        totals = {"phi_loss": 0.0, "var_loss": 0.0, "reg_loss": 0.0}
        for _ in range(iterations):
            x_batch = self._next_loader_batch("_vpu_x_iter", self.vpu_x_loader)
            p_batch = self._next_loader_batch("_vpu_p_iter", self.vpu_p_loader)

            data_x = self._features_from_batch(x_batch).to(self.device)
            data_p = self._features_from_batch(p_batch).to(self.device)

            if data_p.size(0) != data_x.size(0):
                idx = torch.randint(
                    0,
                    data_p.size(0),
                    (data_x.size(0),),
                    device=self.device,
                )
                data_p = data_p[idx]

            data_all = torch.cat((data_p, data_x), dim=0)
            log_phi_all = self._log_phi_all(data_all)
            log_phi_p = log_phi_all[: data_p.size(0), 1]
            log_phi_x = log_phi_all[data_p.size(0) :, 1]

            target_x = log_phi_x.exp()
            rand_perm = torch.randperm(data_p.size(0), device=self.device)
            data_p_perm = data_p[rand_perm]
            target_p_perm = torch.ones(data_p.size(0), device=self.device)[rand_perm]
            lam = beta_dist.sample().to(self.device)
            sam_data = lam * data_x + (1 - lam) * data_p_perm
            sam_target = lam * target_x + (1 - lam) * target_p_perm
            out_log_phi_mix = self._log_phi_all(sam_data)[:, 1]

            self.optimizer.zero_grad()
            phi_loss, var_loss, reg_loss = self.criterion(
                log_phi_x,
                log_phi_p,
                out_log_phi_mix,
                sam_target,
            )
            phi_loss.backward()
            self.optimizer.step()

            totals["phi_loss"] += float(phi_loss.detach().cpu().item())
            totals["var_loss"] += float(var_loss.detach().cpu().item())
            totals["reg_loss"] += float(reg_loss.detach().cpu().item())

        denom = max(iterations, 1)
        epoch_metrics = {key: value / denom for key, value in totals.items()}
        self.vpu_loss_trace.append(epoch_metrics)
        msg = (
            f"Epoch {epoch_idx} - VPU phi_loss: {epoch_metrics['phi_loss']:.4f}, "
            f"var_loss: {epoch_metrics['var_loss']:.4f}, "
            f"reg_loss: {epoch_metrics['reg_loss']:.4f}"
        )
        self.console.log(msg)
        if self.file_console:
            self.file_console.log(msg)
