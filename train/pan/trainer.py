"""PAN trainer for PU-Bench.

Source note:
    Paper: "Predictive Adversarial Learning from Positive and Unlabeled Data"
    Reference implementation: author source file(s): pan
    Repository: https://github.com/morning-dews/PAN
    Audited source snapshot: 5be703639e4c1eb2c63d869ba150d84ebf0c116e
    Active source file: main.py

This controlled PU-Bench entry keeps public benchmark backbones and datasets,
while implementing PAN-specific P/U batch construction and the source D/R
adversarial objective inside this package.
"""

from __future__ import annotations

import torch
from torch import nn

from ..base_trainer import BaseTrainer
from ..utils.checkpointing import CheckpointBundle
from ..utils.model_factory import select_model
from .core import model_probability, pan_objective


class PANTrainer(BaseTrainer):
    """Predictive Adversarial Network trainer."""

    def _build_model(self):
        self.model = select_model(self.method, self.params, self.prior).to(self.device)
        self.discriminator = select_model(self.method, self.params, self.prior).to(
            self.device
        )

        self._ensure_model_initialized(self.model)
        self._ensure_model_initialized(self.discriminator)
        self._maybe_init_pan_bias_from_prior()

        lr = float(self.params.get("lr", 1e-4))
        lr_r = float(self.params.get("lr_r", lr))
        lr_d = float(self.params.get("lr_d", lr))
        wd = float(self.params.get("weight_decay", 1e-3))
        wd_r = float(self.params.get("weight_decay_r", wd))
        wd_d = float(self.params.get("weight_decay_d", wd))
        betas = self._adam_betas()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr_r, betas=betas, weight_decay=wd_r
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=betas, weight_decay=wd_d
        )

        self.pan_eps = float(self.params.get("pan_eps", 1e-2))
        self.loss2_weight = float(self.params.get("loss2_weight", 1e-4))
        self.pan_labeled_label = 1
        self.pan_unlabeled_label = -1

    def _ensure_model_initialized(self, model: nn.Module) -> None:
        try:
            has_params = any(p.requires_grad for p in model.parameters())
        except Exception:
            has_params = False
        if has_params:
            return

        try:
            sample_batch = next(iter(self.train_loader))
            x_sample = sample_batch[0]
            if isinstance(x_sample, (list, tuple)):
                x_sample = x_sample[0]
            with torch.no_grad():
                _ = model(x_sample.to(self.device))
        except Exception:
            pass

    def _maybe_init_pan_bias_from_prior(self) -> None:
        if not self._should_init_bias_from_prior():
            return
        try:
            import math as _math

            def _logit(_p: float) -> float:
                eps = 1e-6
                _p = max(min(float(_p), 1 - eps), eps)
                return _math.log(_p / (1.0 - _p))

            for model in (self.model, self.discriminator):
                fc = getattr(model, "final_classifier", None)
                if (
                    isinstance(fc, torch.nn.Linear)
                    and getattr(fc, "bias", None) is not None
                ):
                    if int(getattr(fc, "out_features", 0)) == 1:
                        with torch.no_grad():
                            fc.bias.fill_(_logit(self.prior))
        except Exception:
            pass

    def before_training(self):
        super().before_training()
        self._prepare_pan_label_semantics()

    def _prepare_pan_label_semantics(self):
        label_scheme = self.params.get("label_scheme", {}) or {}
        self.pan_labeled_label = int(label_scheme.get("pu_labeled_label", 1))
        self.pan_unlabeled_label = int(label_scheme.get("pu_unlabeled_label", -1))

        full_train_dataset = self.train_loader.dataset
        pu_labels = full_train_dataset.pu_labels

        p_indices = (pu_labels == self.pan_labeled_label).nonzero(as_tuple=True)[0]
        u_indices = (pu_labels == self.pan_unlabeled_label).nonzero(as_tuple=True)[0]
        if p_indices.numel() == 0 or u_indices.numel() == 0:
            raise ValueError(
                "PAN requires both labeled positives and unlabeled samples."
            )
        self.pan_train_dataset = full_train_dataset
        self.pan_p_indices = p_indices.long()
        self.pan_u_indices = u_indices.long()

    def create_criterion(self):
        return nn.Identity()

    def get_checkpoint_model(self):
        return CheckpointBundle(
            modules={
                "recognizer": self.model,
                "discriminator": self.discriminator,
            },
            optimizers={
                "optimizer_r": self.optimizer,
                "optimizer_d": self.optimizer_d,
            },
        )

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        self.discriminator.train()

        p_pool = self.pan_p_indices[torch.randperm(len(self.pan_p_indices))]
        u_pool = self.pan_u_indices[torch.randperm(len(self.pan_u_indices))]
        p_cursor = 0
        u_cursor = 0

        batch_size = int(
            self.params.get("batch_size", self.train_loader.batch_size or 1)
        )
        batch_size = max(2, batch_size)
        total_count = len(self.pan_p_indices) + len(self.pan_u_indices)
        u_ratio = len(self.pan_u_indices) / max(1, total_count)
        steps = max(1, total_count // batch_size)
        used_p = 0
        used_u = 0

        for _ in range(steps):
            next_u = int(round((used_p + used_u + batch_size) * u_ratio - used_u))
            next_u = max(1, min(batch_size - 1, next_u))
            next_p = batch_size - next_u

            p_batch, p_pool, p_cursor = self._next_pan_batch_indices(
                p_pool,
                p_cursor,
                next_p,
            )
            u_batch, u_pool, u_cursor = self._next_pan_batch_indices(
                u_pool,
                u_cursor,
                next_u,
            )
            x_p = self._pan_features(p_batch)
            x_u = self._pan_features(u_batch)
            self._train_pan_step(x_p, x_u)
            used_p += next_p
            used_u += next_u

    def _next_pan_batch_indices(
        self,
        pool: torch.Tensor,
        cursor: int,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        pieces = []
        remaining_count = int(count)
        while remaining_count > 0:
            remaining_pool = len(pool) - cursor
            take = min(remaining_count, remaining_pool)
            pieces.append(pool[cursor : cursor + take])
            cursor += take
            remaining_count -= take
            if cursor >= len(pool) and remaining_count > 0:
                pool = pool[torch.randperm(len(pool))]
                cursor = 0
        return torch.cat(pieces), pool, cursor

    def _pan_features(self, indices: torch.Tensor) -> torch.Tensor:
        features = self.pan_train_dataset.features[indices]
        if isinstance(features, (list, tuple)):
            features = features[0]
        return features.to(self.device)

    def _pan_loss(self, x_p: torch.Tensor, x_u: torch.Tensor):
        recognizer_u = model_probability(self.model(x_u))
        discriminator_p = model_probability(self.discriminator(x_p))
        discriminator_u = model_probability(self.discriminator(x_u))
        return pan_objective(
            recognizer_u,
            discriminator_p,
            discriminator_u,
            eps=self.pan_eps,
            loss2_weight=self.loss2_weight,
        )

    def _train_pan_step(self, x_p: torch.Tensor, x_u: torch.Tensor):
        loss = self._pan_loss(x_p, x_u)

        d_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        r_params = [p for p in self.model.parameters() if p.requires_grad]
        d_grads = torch.autograd.grad(
            -loss, d_params, retain_graph=True, allow_unused=True
        )
        r_grads = torch.autograd.grad(loss, r_params, allow_unused=True)

        self.optimizer_d.zero_grad(set_to_none=True)
        for param, grad in zip(d_params, d_grads):
            param.grad = grad
        self.optimizer_d.step()

        self.optimizer.zero_grad(set_to_none=True)
        for param, grad in zip(r_params, r_grads):
            param.grad = grad
        self.optimizer.step()
