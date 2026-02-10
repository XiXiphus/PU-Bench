from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from .base_trainer import BaseTrainer
from .train_utils import select_model


class PANTrainer(BaseTrainer):
    """
    Trainer for Predictive Adversarial Networks (PAN).
    Manages the adversarial training loop between a Recognizer and a Discriminator.
    """

    def _build_model(self):
        # The recognizer is the main model for evaluation.
        # The BaseTrainer's self.model will be the Recognizer.
        self.model = select_model(self.method, self.params, self.prior).to(self.device)

        # The Discriminator will use the same base architecture.
        self.discriminator = select_model(self.method, self.params, self.prior).to(
            self.device
        )

        # Initialize bias from prior for single-logit heads (fairness)
        try:
            import math as _math

            def _logit(_p: float) -> float:
                eps = 1e-6
                _p = max(min(float(_p), 1 - eps), eps)
                return _math.log(_p / (1.0 - _p))

            if bool(self.params.get("init_bias_from_prior", True)):
                for m in [self.model, self.discriminator]:
                    fc = getattr(m, "final_classifier", None)
                    if (
                        isinstance(fc, torch.nn.Linear)
                        and getattr(fc, "bias", None) is not None
                    ):
                        if int(getattr(fc, "out_features", 0)) == 1:
                            with torch.no_grad():
                                fc.bias.fill_(_logit(self.prior))
        except Exception:
            pass

        lr_r = self.params.get("lr_r", 0.001)
        lr_d = self.params.get("lr_d", 0.001)

        # Ensure dynamic models (e.g., MLPs built on first forward) have parameters before creating optimizers
        for m in [self.model, self.discriminator]:
            try:
                has_params = any(p.requires_grad for p in m.parameters())
            except Exception:
                has_params = False
            if not has_params:
                try:
                    sample_batch = next(iter(self.train_loader))
                    x_sample = sample_batch[0]
                    if isinstance(x_sample, (list, tuple)):
                        x_sample = x_sample[0]
                    with torch.no_grad():
                        _ = m(x_sample.to(self.device))
                except Exception:
                    pass

        # BaseTrainer expects self.optimizer for the main model (Recognizer)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_r)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)

    def before_training(self):
        super().before_training()
        # Create separate data loaders for P and U sets for the adversarial loop
        self._prepare_pan_data_loaders()

    def _prepare_pan_data_loaders(self):
        full_train_dataset = self.train_loader.dataset

        p_indices = (full_train_dataset.pu_labels == 1).nonzero(as_tuple=True)[0]
        u_indices = (full_train_dataset.pu_labels == -1).nonzero(as_tuple=True)[0]

        p_dataset = Subset(full_train_dataset, p_indices)
        u_dataset = Subset(full_train_dataset, u_indices)

        batch_size = self.params.get("batch_size", 128)
        self.p_loader = DataLoader(
            p_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.u_loader = DataLoader(
            u_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    def create_criterion(self):
        # Losses are computed manually within the training steps
        return nn.Identity()

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        self.discriminator.train()

        # Use the shorter loader to determine the number of iterations
        num_iter = min(len(self.p_loader), len(self.u_loader))
        p_iter = iter(self.p_loader)
        u_iter = iter(self.u_loader)

        for i in range(num_iter):
            x_p, _, _, _, _ = next(p_iter)
            x_u, _, _, _, _ = next(u_iter)
            x_p, x_u = x_p.to(self.device), x_u.to(self.device)

            # D-Step: Train Discriminator
            self._train_d_step(x_u)

            # R-Step: Train Recognizer
            self._train_r_step(x_p, x_u, epoch_idx)

    def _train_d_step(self, x_u):
        self.optimizer_d.zero_grad()

        with torch.no_grad():
            r_on_u = self.model(x_u)
            # Find samples that the recognizer classifies as positive
            fake_positives = x_u[r_on_u.view(-1) > 0]

        if fake_positives.size(0) == 0:
            # No samples to train on, skip this step
            return

        # U samples are "real" (target 1), fake positives are "fake" (target 0)
        d_on_u = self.discriminator(x_u)
        d_on_fake = self.discriminator(fake_positives)

        loss_d_real = F.binary_cross_entropy_with_logits(
            d_on_u, torch.ones_like(d_on_u)
        )
        loss_d_fake = F.binary_cross_entropy_with_logits(
            d_on_fake, torch.zeros_like(d_on_fake)
        )

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_d.step()

    def _train_r_step(self, x_p, x_u, epoch):
        self.optimizer.zero_grad()

        # 1. Supervised loss on Positive samples
        r_on_p = self.model(x_p)
        loss_p = F.binary_cross_entropy_with_logits(r_on_p, torch.ones_like(r_on_p))

        # 2. Adversarial loss on Unlabeled samples (to fool the discriminator)
        r_on_u = self.model(x_u)
        d_on_u = self.discriminator(x_u)
        loss_u_fake = F.binary_cross_entropy_with_logits(
            d_on_u, torch.ones_like(d_on_u)
        )

        # 3. Reliability loss on Unlabeled samples
        # Reliability weight rl is based on discriminator's confidence
        rl = torch.sigmoid(d_on_u).detach()
        loss_u_real = rl * F.binary_cross_entropy_with_logits(
            r_on_u, torch.zeros_like(r_on_u), reduction="none"
        )
        loss_u_real = loss_u_real.mean()

        co_pu = self.params.get("co_pu", 1.0)
        co_rr = self.params.get("co_rr", 0.5)

        loss_r = loss_p + co_pu * loss_u_fake + co_rr * loss_u_real
        loss_r.backward()
        self.optimizer.step()
