"""Loss helpers for Self-PU's clean/noisy branches."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sigmoid_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Entropy minimization used by the source ``--soft-label`` clean branch."""

    probs_pos = torch.sigmoid(logits.view(-1))
    probs = torch.stack((1.0 - probs_pos, probs_pos), dim=1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()


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
