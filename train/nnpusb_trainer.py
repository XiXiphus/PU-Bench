"""nnpusb_trainer.py

NNPUSBTrainer inherits from BaseTrainer and implements the nnPUSB method training loop.
"""

from __future__ import annotations

from loss.loss_nnpusb import nnPUSBloss
from .base_trainer import BaseTrainer


class NNPUSBTrainer(BaseTrainer):
    """nnPUSB learning trainer"""

    def create_criterion(self):
        gamma = self.params.get("gamma", 1.0)
        beta = self.params.get("beta", 0.0)
        weight = self.params.get("weight", 1.0)
        return nnPUSBloss(self.prior, weight=weight, nnPU=True, gamma=gamma, beta=beta)

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
