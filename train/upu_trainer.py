"""upu_trainer.py

UPUTrainer inherits from BaseTrainer and implements the UPU method's loss and training loop.
"""

from __future__ import annotations

from loss.loss_nnpu import PULoss
from .base_trainer import BaseTrainer


class UPUTrainer(BaseTrainer):
    """UPU learning trainer (without nnPU correction term)"""

    def create_criterion(self):
        return PULoss(self.prior, loss="sigmoid", nnpu=False)

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
