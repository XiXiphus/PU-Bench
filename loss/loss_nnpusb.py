import torch
from torch import nn
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


class nnPUSBloss(nn.Module):
    """PUSB (PU learning with Selected Bias) loss function

    Based on the original implementation from:
    "Positive-Unlabeled Learning with Selected Bias"
    """

    def __init__(self, prior, weight=1.0, gamma=1, beta=0, nnPU=True):
        super(nnPUSBloss, self).__init__()

        if not 0 < prior < 1:
            raise ValueError("The class prior should be in (0, 1)")

        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.nnPU = nnPU
        self.weight = weight
        self.name = "nnpusb" if nnPU else "pusb"

    def forward(self, logits, target, test=False):
        """
        Args:
            logits: Model outputs (before sigmoid), shape (N,) or (N, 1)
            target: Labels where 1 = positive, -1 = unlabeled
        """
        # Ensure logits is 1D
        logits = logits.view(-1)
        target = target.view(-1)

        # Create masks for positive and unlabeled samples
        positive = (target == 1).float()
        unlabeled = (target == -1).float()

        # Count samples (with minimum to avoid division by zero)
        n_positive = torch.clamp(positive.sum(), min=1)
        n_unlabeled = torch.clamp(unlabeled.sum(), min=1)

        # Compute losses using the correct formulation
        # For positive samples: log(1 + exp(-g))
        loss_positive = F.softplus(-logits)  # This is log(1 + exp(-g))
        # For negative risk calculation: log(1 + exp(g))
        loss_negative = F.softplus(logits)  # This is log(1 + exp(g))

        # Apply weight to positive samples (for biased selection)
        weighted_loss_positive = loss_positive * positive * self.weight

        # Calculate positive risk
        positive_risk = self.prior * weighted_loss_positive.sum() / n_positive

        # Calculate negative risk
        # R_u^- - π R_p^-
        negative_risk = (unlabeled * loss_negative).sum() / n_unlabeled - self.prior * (
            positive * loss_negative
        ).sum() / n_positive

        # Apply non-negative correction if enabled
        if self.nnPU and negative_risk < -self.beta:
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk
