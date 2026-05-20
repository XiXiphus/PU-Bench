"""PAN source-objective helpers."""

from __future__ import annotations

import torch


def model_probability(outputs: torch.Tensor) -> torch.Tensor:
    """Return the positive probability from a one-logit/two-logit model output."""
    if outputs.dim() > 1 and outputs.shape[-1] > 1:
        positive_index = 1
        positive_index = max(0, min(positive_index, outputs.shape[-1] - 1))
        return torch.softmax(outputs, dim=1)[:, positive_index].view(-1)

    raw = outputs.view(-1)
    if torch.all(raw >= 0) and torch.all(raw <= 1):
        return raw
    return torch.sigmoid(raw)


def pan_objective(
    recognizer_u: torch.Tensor,
    discriminator_p: torch.Tensor,
    discriminator_u: torch.Tensor,
    *,
    eps: float,
    loss2_weight: float,
) -> torch.Tensor:
    """Source-style PAN objective from the author ``main.py`` PAN branch.

    The discriminator maximizes this objective; the recognizer minimizes it.
    """
    pair_count = min(discriminator_p.numel(), discriminator_u.numel())
    if pair_count <= 0 or recognizer_u.numel() == 0:
        raise ValueError("PAN objective needs both labeled-positive and unlabeled samples.")

    score_d_p = discriminator_p.view(-1)
    score_d_u = discriminator_u.view(-1)
    score_d_u_h = score_d_u[:pair_count]
    d_rate_u = score_d_u_h.detach()

    d_pos = torch.log(torch.clamp(score_d_p + eps, min=eps))
    d_unlabeled_margin = torch.log(torch.clamp(1.0 - score_d_u_h + eps, min=eps))
    d_reference = torch.log(torch.clamp(1.0 - torch.mean(d_rate_u), min=eps))
    loss1 = d_pos.sum() + torch.maximum(
        d_unlabeled_margin - d_reference,
        torch.zeros_like(score_d_u_h),
    ).sum()

    score_r_u = recognizer_u.view(-1)
    loss2 = loss2_weight * (
        (
            torch.log(torch.clamp(1.0 - score_r_u + eps, min=eps))
            - torch.log(torch.clamp(score_r_u + eps, min=eps))
        )
        * (2.0 * score_d_u - 1.0)
    ).sum()

    return loss1 + loss2
