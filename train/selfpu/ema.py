"""EMA teacher utilities for Self-PU."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMATeacher:
    """Source-style exponential moving average teacher.

    Self-PU copies the student into the teacher when the mean-teacher phase
    starts, then updates teacher parameters with
    ``alpha = min(1 - 1 / (step + 1), ema_decay)``.
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
        teacher_state = self.model.state_dict()
        student_state = student.state_dict()
        for name, teacher_value in teacher_state.items():
            student_value = student_state[name].detach()
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(alpha).add_(student_value, alpha=1.0 - alpha)
            else:
                teacher_value.copy_(student_value)
        self.step += 1
