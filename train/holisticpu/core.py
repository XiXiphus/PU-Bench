"""HolisticPU source-aligned primitives.

These helpers mirror the active source implementation in ``wxr99/HolisticPU``
without mixing method-specific details into PU-Bench's shared trainer modules.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

try:
    import jenkspy
except ImportError as exc:  # pragma: no cover - import-time dependency guard
    raise ImportError("HolisticPU requires `jenkspy`. Install it before running.") from exc


def interleave(x: torch.Tensor, size: int) -> torch.Tensor:
    shape = list(x.shape)
    return x.reshape([-1, size] + shape[1:]).transpose(0, 1).reshape([-1] + shape[1:])


def de_interleave(x: torch.Tensor, size: int) -> torch.Tensor:
    shape = list(x.shape)
    return x.reshape([size, -1] + shape[1:]).transpose(0, 1).reshape([-1] + shape[1:])


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 7.0 / 16.0,
) -> LambdaLR:
    """Match ``utils/misc.py:get_cosine_schedule_with_warmup``."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, lr_lambda)


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def jenks_breaks(values: np.ndarray) -> list[float]:
    try:
        return list(jenkspy.jenks_breaks(values, n_classes=2))
    except TypeError:
        return list(jenkspy.jenks_breaks(values, 2))


def source_three_sigma(values: np.ndarray, threshold: float) -> np.ndarray:
    """Replicate the source helper name, not the statistical three-sigma rule.

    The authors' ``utils/misc.py`` filters ``x < 0.2 / 9`` before rerunning
    Jenks if the first break point is positive.
    """

    return values[np.where(values < threshold)]
