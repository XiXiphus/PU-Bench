"""Shared nnPU/uPU risk estimators.

This module owns PU risk code used by more than one method package. Method
packages may re-export these symbols locally, but cross-method imports should
depend on ``train.common`` instead of borrowing from another method package.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def choose_loss(loss_name):
    losses = {
        "zero-one": lambda x: (torch.sign(-x) + 1) / 2,
        "sigmoid": lambda x: torch.sigmoid(-x),
        "logistic": lambda x: F.softplus(-x),
        "squared": lambda x: torch.square(x - 1) / 2,
        "savage": lambda x: 4 / torch.square(1 + torch.exp(x)),
        "LSIF": (lambda x: torch.square(x - 1) / 2, lambda x: x - 1),
        "log": (lambda x: -torch.log(x)),
    }
    return losses[loss_name]


class PULoss(nn.Module):
    """PyTorch port of Kiryo et al.'s nnPU/uPU risk estimator.

    This estimator preserves the source label convention: ``+1`` for labeled
    positives and ``-1`` for unlabeled samples.
    """

    def __init__(self, prior, loss="zero-one", gamma=1, beta=0, nnpu=True):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise ValueError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = choose_loss(loss)
        self.nnpu = nnpu
        self.positive = 1
        self.unlabeled = -1

    def forward(self, x, t, weights=None):
        x = x.view(-1)
        t = t.view(-1)
        positive_mask = t == self.positive
        unlabeled_mask = t == self.unlabeled

        n_positive = max(1, positive_mask.sum().item())
        n_unlabeled = max(1, unlabeled_mask.sum().item())

        if weights is None:
            weights = torch.ones_like(t, dtype=x.dtype)

        positive_risk = (
            self.prior
            * torch.sum(self.loss_func(x[positive_mask]) * weights[positive_mask])
            / n_positive
        )

        negative_risk = (
            torch.sum(self.loss_func(-x[unlabeled_mask]) * weights[unlabeled_mask])
            / n_unlabeled
            - self.prior
            * torch.sum(self.loss_func(-x[positive_mask]) * weights[positive_mask])
            / n_positive
        )

        if self.nnpu:
            if negative_risk.item() < -self.beta:
                objective_nnpu = positive_risk - self.beta
                grad_source = -self.gamma * negative_risk
                loss = grad_source + (objective_nnpu - grad_source).detach()
            else:
                loss = positive_risk + negative_risk
        else:
            loss = positive_risk + negative_risk

        return loss


def pu_loss(x, t, prior, loss=None, nnpu=True):
    """Functional wrapper for non-negative/unbiased PU learning."""

    return PULoss(prior=prior, loss=loss, nnpu=nnpu)(x, t)


__all__ = ["PULoss", "choose_loss", "pu_loss"]
