"""p3mixe_trainer.py

P3MIXETrainer inherits from BaseTrainer and implements the P3MIX-E method.

P3MIX-E Core Logic:
1.  **Dual Dataloaders**: Separately samples from positive (P) and unlabeled (U) sets.
2.  **Mean Teacher (EMA)**: Uses an EMA model to generate stable predictions.
3.  **Heuristic Mixup (h-mix)**:
    - Maintains a pool of "hard" positive examples (high prediction entropy).
    - For unlabeled samples with high uncertainty (predictions between p_lower and p_upper),
      it mixes them with samples from the hard positive pool.
    - Other samples undergo random Mixup.
4.  **Early Learning Regularization (ELR)**: Instead of hard pseudo-labeling, it uses
   a consistency loss to align the student model's predictions with the EMA model's
   soft predictions on mixed samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from .base_trainer import BaseTrainer
from .train_utils import sigmoid_rampup
from backbone.ema import ModelEMA


class P3MIXETrainer(BaseTrainer):
    """P3MIX-E method trainer."""

    def __init__(self, method: str, experiment: str, params: dict[str, Any]):
        super().__init__(method, experiment, params)
        self._init_p3mix_params()
        self._prepare_p3mix_data()
        self._build_p3mix_model()
        self.h_inputs_x = torch.Tensor([]).to(self.device)
        self.h_features_x = torch.Tensor([]).to(self.device)
        self.h_entropys_x = torch.Tensor([]).to(self.device)
        self.h_preds_x = torch.Tensor([]).to(self.device)

    def _init_p3mix_params(self):
        """Load P3MIX specific hyperparameters."""
        self.epochs = self.params.get("num_epochs", 100)
        self.val_iterations = self.params.get("val_iterations", 200)
        self.use_ema = self.params.get("mean_teacher", True)
        self.ema_decay = self.params.get("ema_decay", 0.999)
        self.ema_update_start = self.params.get("ema_start", 0)
        self.ema_dynamic_decay = self.params.get("ema_update", False)
        self.ema_rampup_end = self.params.get("ema_end", 100)
        self.start_hmix = self.params.get("start_hmix", 10)
        self.h_positive_pool_size = self.params.get("h_positive", 100)
        self.p_upper_threshold = self.params.get("p_upper", 0.6)
        self.p_lower_threshold = self.params.get("p_lower", 0.4)
        self.mixup_alpha = self.params.get("alpha", 1.0)
        self.mix_layer = self.params.get("mix_layer", -1)
        self.positive_weight = self.params.get("positive_weight", 1.0)
        self.unlabeled_weight = self.params.get("unlabeled_weight", 1.0)
        self.entropy_weight = self.params.get("entropy_weight", 0.1)
        self.elr_weight = self.params.get("elr_weight", 1.0)  # P3MIX-E uses this

    def _prepare_p3mix_data(self):
        full_train_dataset = self.train_loader.dataset
        positive_indices = [
            i for i, label in enumerate(full_train_dataset.pu_labels) if label == 1
        ]
        unlabeled_indices = [
            i for i, label in enumerate(full_train_dataset.pu_labels) if label == -1
        ]
        p_dataset = Subset(full_train_dataset, positive_indices)
        u_dataset = Subset(full_train_dataset, unlabeled_indices)
        batch_size = self.params.get("batch_size", 128)
        self.p_loader = DataLoader(
            p_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.u_loader = DataLoader(
            u_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.p_update_loader = DataLoader(p_dataset, batch_size=1000, shuffle=False)

    def _build_p3mix_model(self):
        model_map = {
            "CIFAR10": "MixCNN_CIFAR10",
            "FashionMNIST": "MixCNN_FashionMNIST",
            "MNIST": "MixCNN_MNIST",
            "AlzheimerMRI": "MixCNN_AlzheimerMRI",
            "20News": "MixMLP_20News",
            "IMDB": "MixMLP_20News",
            # Tabular datasets reuse the 20News MixMLP
            "Mushrooms": "MixMLP_20News",
            "Spambase": "MixMLP_20News",
            "Connect4": "MixMLP_20News",
        }
        dataset_class = self.params.get("dataset_class")
        mix_model_name = model_map.get(dataset_class)
        if not mix_model_name:
            raise ValueError(f"No Mix model defined for {dataset_class}")
        from backbone import mix_models

        selected_model_cls = getattr(mix_models, mix_model_name)
        self.model = selected_model_cls(prior=self.prior).to(self.device)

        # Ensure dynamic model is built before creating optimizer
        try:
            has_params = any(p.requires_grad for p in self.model.parameters())
        except Exception:
            has_params = False
        if not has_params:
            # Prefer using positive loader; fallback to unlabeled
            bootstrap_batch = None
            try:
                bootstrap_batch, *_ = next(iter(self.p_loader))
            except Exception:
                pass
            if bootstrap_batch is None:
                try:
                    bootstrap_batch, *_ = next(iter(self.u_loader))
                except Exception:
                    pass
            if bootstrap_batch is not None:
                with torch.no_grad():
                    _ = self.model(bootstrap_batch.to(self.device))

        if self.use_ema:
            self.ema_model = ModelEMA(self, self.model, decay=self.ema_decay)
        else:
            self.ema_model = None
        # Initialize bias from prior for fairness (single-logit head)
        try:
            import math as _math

            def _logit(_p: float) -> float:
                eps = 1e-6
                _p = max(min(float(_p), 1 - eps), eps)
                return _math.log(_p / (1.0 - _p))

            if bool(self.params.get("init_bias_from_prior", True)):
                fc = getattr(self.model, "final_classifier", None)
                if (
                    isinstance(fc, torch.nn.Linear)
                    and getattr(fc, "bias", None) is not None
                ):
                    if int(getattr(fc, "out_features", 0)) == 1:
                        with torch.no_grad():
                            fc.bias.fill_(_logit(self.prior))
        except Exception:
            pass
        lr = self.params.get("lr", 1e-3)
        wd = self.params.get("weight_decay", 5e-4)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=wd
        )
        milestones = self.params.get("milestones", [50, 100])
        scheduler_gamma = self.params.get("scheduler_gamma", 0.1)
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=milestones, gamma=scheduler_gamma
        )

    def _update_ema_variables(self, epoch_idx: int):
        if not self.use_ema or not self.ema_model:
            return
        alpha = self.ema_decay
        if self.ema_dynamic_decay:
            alpha = sigmoid_rampup(epoch_idx + 1, self.ema_rampup_end) * alpha
        else:
            alpha = min(1 - 1 / (epoch_idx - self.ema_update_start + 1), alpha)
        for ema_param, param in zip(
            self.ema_model.ema.parameters(), self.model.parameters()
        ):
            ema_param.data.mul_(alpha).add_((param.data), alpha=1.0 - alpha)

    def _check_mean_teacher(self, epoch_idx: int) -> bool:
        return self.use_ema and epoch_idx >= self.ema_update_start

    def get_hinput(self, features_ori: torch.Tensor) -> torch.Tensor:
        n_unlabeled = features_ori.size(0)
        pool_size = int(self.h_inputs_x.size(0))
        if pool_size <= 0:
            # Empty pool; return an empty tensor to signal fallback
            return torch.empty(0, dtype=torch.int64, device=self.device)
        # Sample within actual pool size to avoid OOB
        return torch.randint(
            low=0,
            high=pool_size,
            size=(n_unlabeled,),
            device=self.device,
            dtype=torch.int64,
        )

    def update_hinput(self):
        net = self.model
        net.eval()
        with torch.no_grad():
            h_inputs_x_, h_features_x_, h_entropys_x_, h_preds_x_ = [], [], [], []
            for data, _, _, _, _ in self.p_update_loader:
                data = data.to(self.device, non_blocking=True)
                outputs_x, features_x = net(data, flag_feature=True)
                preds_x = torch.sigmoid(outputs_x)
                entropys_x = -(
                    preds_x * F.logsigmoid(outputs_x)
                    + (1.0 - preds_x) * F.logsigmoid(-outputs_x)
                )
                h_inputs_x_.extend(list(data.cpu().numpy()))
                h_features_x_.extend(list(features_x.cpu().numpy()))
                h_entropys_x_.extend(list(entropys_x.cpu().numpy().flatten()))
                h_preds_x_.extend(list(preds_x.cpu().numpy().flatten()))
        h_group_x = list(zip(h_inputs_x_, h_features_x_, h_entropys_x_, h_preds_x_))
        h_group_x.sort(key=lambda x: x[2], reverse=True)
        top_h_samples = h_group_x[: self.h_positive_pool_size]
        self.h_inputs_x = torch.tensor(
            np.array([x[0] for x in top_h_samples]), device=self.device
        )
        self.h_features_x = torch.tensor(
            np.array([x[1] for x in top_h_samples]), device=self.device
        )
        self.h_entropys_x = torch.tensor(
            np.array([x[2] for x in top_h_samples]), device=self.device
        )
        self.h_preds_x = torch.tensor(
            np.array([x[3] for x in top_h_samples]), device=self.device
        )
        net.train()

    def create_criterion(self):
        return torch.nn.Identity()

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        if self.ema_model:
            self.ema_model.ema.train()
        u_loader_iter, p_loader_iter = iter(self.u_loader), iter(self.p_loader)

        for i in range(self.val_iterations):
            try:
                data_p, _, _, _, _ = next(p_loader_iter)
            except StopIteration:
                p_loader_iter = iter(self.p_loader)
                data_p, _, _, _, _ = next(p_loader_iter)
            try:
                data_u, _, _, _, _ = next(u_loader_iter)
            except StopIteration:
                u_loader_iter = iter(self.u_loader)
                data_u, _, _, _, _ = next(u_loader_iter)

            data_p, data_u = data_p.to(self.device, non_blocking=True), data_u.to(
                self.device, non_blocking=True
            )
            target_p = torch.ones(
                data_p.shape[0], device=self.device, dtype=torch.float32
            )[:, None]
            target_u = torch.zeros(
                data_u.shape[0], device=self.device, dtype=torch.float32
            )[:, None]
            target_p_, target_u_ = torch.cat(
                (1.0 - target_p, target_p), dim=1
            ), torch.cat((1.0 - target_u, target_u), dim=1)
            data = torch.cat((data_p, data_u), dim=0)
            targets_ = torch.cat((target_p_, target_u_), dim=0)
            idx_p, idx_u = slice(0, len(data_p)), slice(len(data_p), len(data))

            with torch.no_grad():
                ema_net = self.ema_model.ema if self.use_ema else self.model
                outputs = ema_net(data)
                p = torch.sigmoid(outputs)
                targets_elr = torch.cat([1.0 - p, p], dim=1)
                if epoch_idx >= self.start_hmix:
                    outputs_tuple = self.model(data_u, flag_feature=True)
                    outputs_ori, features_ori = outputs_tuple
                    p_indicator_u = torch.sigmoid(outputs_ori).detach()
                    p_indicator_p = torch.ones(
                        len(data_p), dtype=torch.float32, device=self.device
                    )[:, None]
                    p_indicator = torch.cat((p_indicator_p, p_indicator_u), dim=0).view(
                        -1
                    )
                    h_p = torch.ones(
                        len(self.h_inputs_x), dtype=torch.float32, device=self.device
                    )[:, None]
                    h_targets_elr = torch.cat([1.0 - h_p, h_p], dim=1)

            if epoch_idx >= self.start_hmix:
                h_input_b_idx = self.get_hinput(features_ori)
                if h_input_b_idx.numel() == 0:
                    # Fallback to standard random mixup when pool is empty
                    idx = torch.randperm(data.size(0), device=self.device)
                    data_b, targets_b, targets_elr_b = (
                        data[idx],
                        targets_[idx],
                        targets_elr[idx],
                    )
                else:
                    h_target_b = torch.ones(
                        len(h_input_b_idx), dtype=torch.float32, device=self.device
                    )[:, None]
                    h_target_b_ = torch.cat([1.0 - h_target_b, h_target_b], dim=1)
                    idx1 = torch.randint(
                        low=0,
                        high=data.size(0),
                        size=(data_p.size(0),),
                        device=self.device,
                    )
                    data_b1 = torch.cat(
                        [data[idx1], self.h_inputs_x[h_input_b_idx]], dim=0
                    )
                    targets_b1 = torch.cat([targets_[idx1], h_target_b_], dim=0)
                    targets_elr_b1 = torch.cat(
                        [targets_elr[idx1], h_targets_elr[h_input_b_idx]], dim=0
                    )
                    idx2 = torch.randint(
                        low=0,
                        high=data.size(0),
                        size=(data_u.size(0),),
                        device=self.device,
                    )
                    idx = torch.cat([idx1, idx2])
                    data_b, targets_b, targets_elr_b = (
                        data[idx],
                        targets_[idx],
                        targets_elr[idx],
                    )
                    p_indicator[p_indicator >= self.p_upper_threshold] = 1.0
                    p_indicator[p_indicator <= self.p_lower_threshold] = 1.0
                    p_indicator[idx_p] = 1.0
                    p_indicator[p_indicator != 1.0] = 0.0
                    if data_b.dim() == 4:
                        data_b = (
                            p_indicator[:, None, None, None] * data_b
                            + (1.0 - p_indicator[:, None, None, None]) * data_b1
                        )
                    else:
                        data_b = (
                            p_indicator[:, None] * data_b
                            + (1.0 - p_indicator[:, None]) * data_b1
                        )
                    targets_b = (
                        p_indicator[:, None] * targets_b
                        + (1.0 - p_indicator[:, None]) * targets_b1
                    )
                    targets_elr_b = (
                        p_indicator[:, None] * targets_elr_b
                        + (1.0 - p_indicator[:, None]) * targets_elr_b1
                    )
            else:
                idx = torch.randperm(data.size(0))
                data_b, targets_b, targets_elr_b = (
                    data[idx],
                    targets_[idx],
                    targets_elr[idx],
                )

            data_a, targets_a, targets_elr_a = data, targets_, targets_elr
            l = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            l = max(l, 1.0 - l)

            outputs = self.model(data_a, data_b, l, self.mix_layer)
            logits = torch.sigmoid(outputs)
            logits_ = torch.cat([1.0 - logits, logits], dim=1)
            logits_ = torch.clamp(logits_, 1e-4, 1.0 - 1e-4)

            mix_targets = l * targets_a + (1.0 - l) * targets_b
            mix_targets_elr = l * targets_elr_a + (1.0 - l) * targets_elr_b

            loss_p = -(mix_targets[idx_p] * (logits_[idx_p]).log()).sum(1).mean()
            loss_u = -(mix_targets[idx_u] * (logits_[idx_u]).log()).sum(1).mean()
            loss_ent = -(logits_ * logits_.log()).sum(1).mean()
            loss_elr = (1.0 - (mix_targets_elr * logits_).sum(dim=1)).log().mean()

            loss = (
                self.positive_weight * loss_p
                + self.unlabeled_weight * loss_u
                + self.entropy_weight * loss_ent
                + self.elr_weight * loss_elr
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self._check_mean_teacher(epoch_idx):
            self._update_ema_variables(epoch_idx)
        if epoch_idx >= self.start_hmix - 1:
            self.update_hinput()
        self.scheduler.step()
