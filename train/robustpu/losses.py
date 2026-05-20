"""Robust-PU losses and hardness functions.

Primary source:
    author source file(s): robustpu/lossFunc.py
    woriazzc/Robust-PU at 34d950f2c6e56510855a922acb5f84b6459773ef

The source uses +1 for labeled positives and -1 for unlabeled examples during
PU and hardness computations.  For weighted supervised training, unlabeled
examples are treated as binary negatives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted BCE with Robust-PU's PU label convention."""

    logits = logits.view(-1)
    labels = labels.view(-1)
    targets = (labels == 1).to(dtype=logits.dtype)
    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights.view(-1).to(logits.device)).mean()


def focal_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    gamma: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = logits.view(-1)
    labels = labels.view(-1)
    targets = (labels == 1).to(dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    focal_weights = torch.where(
        targets == 1,
        torch.pow(1.0 - probs, gamma),
        torch.pow(probs, gamma),
    ).detach()
    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    if weights is not None:
        focal_weights = focal_weights * weights.view(-1).to(logits.device)
    out = per_sample * focal_weights
    if reduction == "none":
        return out
    return out.mean()


def logistic_hardness(logits: torch.Tensor, labels: torch.Tensor | float | int):
    return F.softplus(-logits.view(-1) * torch.as_tensor(labels, device=logits.device))


def sigmoid_hardness(logits: torch.Tensor, labels: torch.Tensor | float | int):
    return torch.sigmoid(-logits.view(-1) * torch.as_tensor(labels, device=logits.device))


def crps_hardness(logits: torch.Tensor, labels: torch.Tensor | float | int):
    return sigmoid_hardness(logits, labels).pow(2)


def brier_hardness(logits: torch.Tensor, labels: torch.Tensor | float | int):
    return 2.0 * crps_hardness(logits, labels)


def focal_hardness(
    logits: torch.Tensor,
    labels: torch.Tensor | float | int,
    gamma: float = 1.0,
):
    labels_tensor = torch.full_like(logits.view(-1), float(labels))
    return focal_binary_loss(logits, labels_tensor, gamma=gamma, reduction="none")


def hardness_values(
    name: str,
    logits: torch.Tensor,
    labels: torch.Tensor | float | int,
    *,
    gamma: float = 1.0,
) -> torch.Tensor:
    name = str(name).lower()
    if name == "logistic":
        return logistic_hardness(logits, labels)
    if name == "sigmoid":
        return sigmoid_hardness(logits, labels)
    if name == "crps":
        return crps_hardness(logits, labels)
    if name == "brier":
        return brier_hardness(logits, labels)
    if name == "focal":
        return focal_hardness(logits, labels, gamma=gamma)
    raise ValueError(
        "RobustPU hardness must be one of "
        "'logistic', 'sigmoid', 'crps', 'brier', or 'focal'."
    )
