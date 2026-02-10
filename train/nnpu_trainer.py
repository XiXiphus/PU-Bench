"""nnpu_trainer.py

NNPUTrainer inherits from BaseTrainer and implements the nnPU method's loss and training loop.
"""

from __future__ import annotations

import torch
from loss.loss_nnpu import PULoss

from .base_trainer import BaseTrainer


class NNPUTrainer(BaseTrainer):
    """nnPU learning method trainer"""

    # Required interfaces
    def create_criterion(self):
        gamma = self.params.get("gamma", 1.0)
        beta = self.params.get("beta", 0.0)
        return PULoss(self.prior, loss="sigmoid", nnpu=True, gamma=gamma, beta=beta)

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (nnPU)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)

            # nnPU loss directly uses ±1 labels t
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
