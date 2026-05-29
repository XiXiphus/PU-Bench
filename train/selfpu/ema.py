"""EMA teacher utilities for Self-PU."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMATeacher:
    """Source-style exponential moving average teacher.

    Self-PU copies the student into the teacher when the mean-teacher phase
    starts, then updates teacher parameters with
    ``alpha = min(1 - 1 / (step + 1), ema_decay)``.  Source code updates
    parameters only; buffers follow the teacher model's own forward passes.
    """

    def __init__(self, model: nn.Module, device: torch.device | str):
        self.model = deepcopy(model).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.step = 0

    @torch.no_grad()
    def copy_from(self, student: nn.Module) -> None:
        self.model.load_state_dict(student.state_dict())
        self.step = 0

    @torch.no_grad()
    def update(self, student: nn.Module, decay: float) -> None:
        alpha = min(1.0 - 1.0 / float(self.step + 1), float(decay))
        for teacher_param, student_param in zip(
            self.model.parameters(),
            student.parameters(),
        ):
            teacher_param.data.mul_(alpha).add_(
                student_param.data,
                alpha=1.0 - alpha,
            )
        self.step += 1
