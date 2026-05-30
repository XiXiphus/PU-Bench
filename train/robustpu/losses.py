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


def _source_surrogate_loss(name: str):
    if name == "sigmoid":
        return lambda x: torch.sigmoid(-x)
    if name == "logistic":
        return lambda x: F.softplus(-x)
    raise ValueError("RobustPU PU loss surrogate must be 'sigmoid' or 'logistic'.")


def source_pu_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prior: float,
    weights: torch.Tensor | None = None,
    *,
    sur_loss: str = "sigmoid",
    nnpu: bool = True,
    gamma: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """Robust-PU source PU risk scalar from ``lossFunc.py``."""

    logits = logits.view(-1)
    labels = labels.view(-1)
    if weights is None:
        weights = torch.ones_like(logits)
    else:
        weights = weights.view(-1).to(device=logits.device, dtype=logits.dtype)

    loss = _source_surrogate_loss(str(sur_loss).lower())
    positive = (labels == 1).to(dtype=logits.dtype)
    unlabeled = (labels == -1).to(dtype=logits.dtype)
    n_positive = max(1.0, float(positive.sum().item()))
    n_unlabeled = max(1.0, float(unlabeled.sum().item()))

    y_positive = loss(logits) * weights
    y_unlabeled = loss(-logits) * weights
    positive_risk = torch.sum(float(prior) * positive * y_positive / n_positive)
    negative_risk = torch.sum(
        (unlabeled / n_unlabeled - float(prior) * positive / n_positive)
        * y_unlabeled
    )
    if nnpu and negative_risk.item() < -float(beta):
        return -float(gamma) * negative_risk
    return positive_risk + negative_risk


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
