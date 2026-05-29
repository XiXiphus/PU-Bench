"""Loss helpers for Self-PU's clean/noisy branches."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sigmoid_loss(logits: torch.Tensor) -> torch.Tensor:
    """Source Self-PU sigmoid loss: 1 / (1 + exp(logit))."""

    return torch.sigmoid(-logits)


def source_nnpu_loss(
    logits: torch.Tensor,
    pu_labels: torch.Tensor,
    prior: float,
    *,
    beta: float = 0.0,
    gamma: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Noisy-branch nnPU scalar used by the Self-PU source.

    The source ``PULoss`` returns a tuple and the training loop optimizes the
    second value.  In the negative-risk branch that value is ``-gamma * R_n``;
    this scalar is also used by Self-PU's mutual-consistency gate.  The
    ``sample_weight`` path is the source ``sigmoid_eps`` variant used by
    ``train_2s2t_mix.py`` self-calibration.
    """

    logits = logits.view(-1)
    labels = pu_labels.view(-1)
    if sample_weight is None:
        weights = torch.ones_like(logits)
    else:
        weights = sample_weight.to(device=logits.device, dtype=logits.dtype).view(-1)
        if weights.shape[0] != logits.shape[0]:
            raise ValueError(
                "Self-PU sample weights must match logits; "
                f"got {tuple(weights.shape)} and {tuple(logits.shape)}."
            )
    positive = labels == 1
    unlabeled = labels == -1
    n_positive = max(1, int(positive.sum().item()))
    n_unlabeled = max(1, int(unlabeled.sum().item()))

    y_positive = sigmoid_loss(logits) * weights
    y_unlabeled = sigmoid_loss(-logits) * weights
    positive_risk = (
        float(prior) * y_positive[positive].sum() / float(n_positive)
    )
    negative_risk = (
        y_unlabeled[unlabeled].sum() / float(n_unlabeled)
        - float(prior) * y_unlabeled[positive].sum() / float(n_positive)
    )
    if negative_risk < -float(beta):
        return -float(gamma) * negative_risk
    return positive_risk + negative_risk


def sigmoid_entropy_values(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample binary entropy used by Self-PU clean/meta branches."""

    probs_pos = torch.sigmoid(logits.view(-1))
    probs = torch.stack((1.0 - probs_pos, probs_pos), dim=1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=1)


def sigmoid_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Entropy minimization used by the source ``--soft-label`` clean branch."""

    return sigmoid_entropy_values(logits).mean()


def signed_binary_ce(logits: torch.Tensor, signed_labels: torch.Tensor) -> torch.Tensor:
    targets = (signed_labels.view(-1) == 1).float()
    return F.binary_cross_entropy_with_logits(logits.view(-1), targets)


def _as_consistency_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() <= 1:
        logits = logits.view(-1, 1)
    else:
        logits = logits.view(logits.shape[0], -1)
    if logits.shape[1] == 1:
        return torch.cat((torch.zeros_like(logits), logits), dim=1)
    return logits


def softmax_mse_consistency(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Mean-teacher consistency from the source dependency.

    The source ``softmax_mse_loss`` sums MSE over probabilities, divides by the
    number of classes, then the training loop divides by batch size.
    """

    student = _as_consistency_logits(student_logits)
    teacher = _as_consistency_logits(teacher_logits)
    if student.shape != teacher.shape:
        raise ValueError(
            "Self-PU consistency requires matching student/teacher shapes; "
            f"got {tuple(student.shape)} and {tuple(teacher.shape)}."
        )
    loss = F.mse_loss(
        F.softmax(student, dim=1),
        F.softmax(teacher, dim=1),
        reduction="sum",
    )
    num_classes = max(1, int(student.shape[1]))
    batch_size = max(1, int(student.shape[0]))
    return loss / float(num_classes * batch_size)


def mutual_student_consistency(
    logits: torch.Tensor,
    peer_logits: torch.Tensor,
    pu_loss: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, bool]:
    """Source ``type == 'mu'`` mutual consistency gate for noisy batches."""

    aux = F.mse_loss(
        torch.sigmoid(logits.view(-1)),
        torch.sigmoid(peer_logits.view(-1)).detach(),
    )
    threshold = pu_loss.detach() * float(alpha)
    return aux, bool(aux.detach() < threshold)


def sigmoid_rampup(epoch: int, rampup_length: int) -> float:
    """Mean-teacher sigmoid ramp-up from the source dependency."""

    if rampup_length <= 0:
        return 1.0
    current = max(0.0, min(float(epoch), float(rampup_length)))
    phase = 1.0 - current / float(rampup_length)
    return float(math.exp(-5.0 * phase * phase))
