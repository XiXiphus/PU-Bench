"""Shared generic Mixup helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    if device:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    scores: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    """Compute Mixup loss as a convex combination of two BCE losses."""
    loss_a = F.binary_cross_entropy(scores, y_a, reduction="mean")
    loss_b = F.binary_cross_entropy(scores, y_b, reduction="mean")
    return lam * loss_a + (1 - lam) * loss_b
