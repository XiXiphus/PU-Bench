"""vpu_trainer.py

VPUTrainer inherits from BaseTrainer and implements the VPU method's Mixup training loop.
"""

from __future__ import annotations

import argparse
import torch
from loss.loss_vpu import VPULoss

from .base_trainer import BaseTrainer


class VPUTrainer(BaseTrainer):
    """VPU learning trainer"""

    def create_criterion(self):
        mix_alpha = self.params.get("mix_alpha", 0.3)
        args = argparse.Namespace(mix_alpha=mix_alpha)
        return VPULoss(args)

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            p_mask = t == 1
            p_features = x[p_mask]
            if len(p_features) == 0:
                continue

            # log σ(ϕ(x))
            log_phi_all = torch.sigmoid(self.model(x)).log()
            targets = p_mask.float()

            # Mixup: sample p_mix from same batch
            if p_features.size(0) >= x.size(0):
                rand_perm = torch.randperm(p_features.size(0), device=self.device)[
                    : x.size(0)
                ]
                p_mix = p_features[rand_perm]
            else:
                idx = torch.randint(
                    0, p_features.size(0), (x.size(0),), device=self.device
                )
                p_mix = p_features[idx]

            m = torch.distributions.beta.Beta(
                self.criterion.mix_alpha, self.criterion.mix_alpha
            )
            lam = m.sample().to(self.device)
            lam_float = float(lam)
            sam_data = lam * x + (1 - lam) * p_mix

            pos_prob = torch.sigmoid(self.model(x)).detach()
            sam_target = lam_float * pos_prob + (1 - lam_float) * torch.ones_like(
                pos_prob
            )
            out_log_phi_mix = torch.sigmoid(self.model(sam_data)).log()

            self.optimizer.zero_grad()
            loss = self.criterion(
                log_phi_all, targets, out_log_phi_mix, sam_target, lam_float
            )
            loss.backward()
            self.optimizer.step()
