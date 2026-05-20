"""Mixup helpers for PULDA.

Source snapshot:
    author source file(s): pulda/customized/mixup.py
    jiangyangby/PULDA at 7b3dcad95bd7caa0a9477af37a05764fbe6e27bc
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mixup_two_targets(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Source-aligned binary mixup with two soft targets."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a = y
    y_b = y[index]
    return mixed_x, y_a, y_b, lam


def mixup_bce(
    scores: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Binary cross-entropy mixup loss from the PULDA source."""
    loss_a = F.binary_cross_entropy(scores, targets_a)
    loss_b = F.binary_cross_entropy(scores, targets_b)
    return lam * loss_a + (1.0 - lam) * loss_b
