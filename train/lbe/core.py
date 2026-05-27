"""Core EM formulas for LBE-PU.

These helpers mirror the author-provided implementation under
``author source file(s): lbe_author_provided/LBE_MLP``:

- ``em.py::EStep`` for posterior inference;
- ``train.py::pretrain`` for observed-label pretraining;
- ``train.py::em_train`` for the small-loss classifier M-step and eta M-step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def observed_label_pretrain_loss(
    logits: torch.Tensor,
    q: torch.Tensor,
    proportion_labeled: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Source pretrain loss for observed label indicator ``q``.

    ``q`` is 1 for labeled positives and 0 for unlabeled samples. The source
    weights the labeled-positive term by ``1 - mean(q)`` and the unlabeled term
    by ``mean(q)``.
    """

    q = q.view(-1).float()
    if proportion_labeled is None:
        proportion_labeled = q.mean()
    elif not torch.is_tensor(proportion_labeled):
        proportion_labeled = torch.tensor(
            float(proportion_labeled), device=q.device, dtype=q.dtype
        )
    else:
        proportion_labeled = proportion_labeled.to(device=q.device, dtype=q.dtype)
    weights = q * (1 - proportion_labeled) + (1 - q) * proportion_labeled
    return F.binary_cross_entropy_with_logits(
        logits.view(-1),
        q,
        weight=weights,
        reduction="mean",
    )


def posterior_y1(
    p_y1_x: torch.Tensor,
    eta_x: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Return ``P(y=1 | x, q)`` from source ``EStep``."""

    p_y1_x = p_y1_x.view(-1)
    eta_x = eta_x.view(-1)
    q = q.view(-1).float()

    p_q_given_y1_x = ((1 - eta_x) ** (1 - q)) * (eta_x**q)
    p_q_given_y0_x = 1 - q
    p_y1_q_x = p_y1_x * p_q_given_y1_x
    p_y0_q_x = (1 - p_y1_x) * p_q_given_y0_x
    denom = p_y1_q_x + p_y0_q_x
    posterior = torch.where(
        denom > 0,
        p_y1_q_x / denom,
        torch.zeros_like(denom),
    )
    return torch.where(q == 1, torch.ones_like(posterior), posterior)


def m_step_classifier_per_sample_loss(
    logits: torch.Tensor,
    pst_y1: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Source per-sample classifier M-step loss before global top-k filtering."""

    p_y1_x = torch.sigmoid(logits.view(-1))
    pst_y1 = pst_y1.view(-1).float()
    pst = torch.stack([1 - pst_y1, pst_y1], dim=1)
    probs = torch.stack([1 - p_y1_x, p_y1_x], dim=1)
    return -(pst * torch.log(probs + eps)).sum(dim=1)


def m_step_classifier_loss(
    logits: torch.Tensor,
    pst_y1: torch.Tensor,
    topk_keep: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Source classifier M-step loss with 80% small-loss retention."""

    per_sample = m_step_classifier_per_sample_loss(logits, pst_y1, eps=eps)
    keep = max(1, int(per_sample.numel() * float(topk_keep)))
    return per_sample.topk(largest=False, k=keep)[0].mean()


def m_step_eta_loss(
    eta_logits: torch.Tensor,
    q: torch.Tensor,
    pst_y1: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Source eta-model M-step loss."""

    eta = torch.sigmoid(eta_logits.view(-1))
    q = q.view(-1).float()
    pst_y1 = pst_y1.view(-1).float()
    likelihood = ((1 - eta) ** (1 - q)) * (eta**q)
    return -(pst_y1 * torch.log(likelihood + eps)).mean()


__all__ = [
    "observed_label_pretrain_loss",
    "posterior_y1",
    "m_step_classifier_per_sample_loss",
    "m_step_classifier_loss",
    "m_step_eta_loss",
]
