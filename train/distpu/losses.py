"""Source-aligned Dist-PU label distribution loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class LabelDistributionLoss(torch.nn.Module):
    """Histogram-based Label Distribution Loss for PU learning.

    This loss function encourages the distribution of predicted scores (probabilities)
    for positive and unlabeled sets to match target distributions.

    - For positive samples, the target distribution is a sharp peak at 1.0.
    - For unlabeled samples, the target is a mix of the positive distribution
      and a negative distribution (a sharp peak at 0.0), weighted by the class prior.

    The distance between the predicted and target distributions is measured by L1 loss.

    Reference:
    Zhao, Y., Xu, Q., Jiang, Y., Wen, P., & Huang, Q. (2022).
    Dist-PU: Positive-Unlabeled Learning From a Label Distribution Perspective.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition (CVPR).
    """

    def __init__(
        self,
        prior: float,
        num_bins: int = 1,
        device: torch.device | None = None,
    ):
        """
        Args:
            prior (float): The class prior π, i.e., the prevalence of the
                           positive class in the training data P(y=+1).
            num_bins (int): The number of bins to use for building the
                            histograms of score distributions. Defaults to 1,
                            following the original implementation.
            device (torch.device, optional): The device to move tensors to.
        """
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError("The class prior must be in the range (0, 1).")

        self.prior = prior
        # Weight for the unlabeled loss component, from the original paper's code
        self.frac_prior = 1.0 / (2 * self.prior)
        self.num_bins = num_bins
        self.device = device or torch.device("cpu")

        # Bin boundaries for the histogram, from 0 to 1.
        self.step = 1.0 / self.num_bins
        self.t_size = self.num_bins + 1
        self.t = torch.arange(0, 1 + self.step, self.step).view(1, -1).to(self.device)

        # Define target distributions (proxies)
        proxy_p = np.zeros(self.t_size, dtype=float)
        proxy_n = np.zeros_like(proxy_p)
        proxy_p[-1] = 1.0
        proxy_n[0] = 1.0

        # The unlabeled set is a mixture of P and N
        proxy_mix = self.prior * proxy_p + (1 - self.prior) * proxy_n
        self.proxy_positive = torch.from_numpy(proxy_p).requires_grad_(False).float().to(
            self.device
        )
        self.proxy_unlabeled = (
            torch.from_numpy(proxy_mix).requires_grad_(False).float().to(self.device)
        )

    def _create_histogram(self, scores: torch.Tensor) -> torch.Tensor:
        """Creates a soft histogram of scores.

        Instead of hard assignments, this uses a triangular kernel for binning,
        making the process differentiable.

        Args:
            scores (torch.Tensor): A tensor of prediction scores (probabilities).

        Returns:
            torch.Tensor: A normalized histogram representing the distribution.
        """
        if scores.numel() == 0:
            return torch.zeros_like(self.t).squeeze(0)

        scores = scores.view(-1, 1)
        scores_rep = scores.repeat(1, self.t_size)
        hist = torch.abs(scores_rep - self.t)
        inds = hist > self.step
        hist = self.step - hist
        hist[inds] = 0
        return hist.sum(dim=0) / (len(scores) * self.step)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): The raw output (logits) from the model.
            labels (torch.Tensor): Labels for the samples. Must use:
                                   - `1` for labeled positive samples.
                                   - `0` for unlabeled samples.

        Returns:
            torch.Tensor: The calculated distribution loss.
        """
        # Convert logits to probabilities
        scores = torch.sigmoid(logits)

        # Separate scores for positive and unlabeled samples
        positive_scores = scores[labels == 1]
        unlabeled_scores = scores[labels == 0]

        loss_p = 0
        loss_u = 0
        if positive_scores.numel() > 0:
            hist_positive = self._create_histogram(positive_scores)
            loss_p = F.l1_loss(hist_positive, self.proxy_positive, reduction="mean")
        if unlabeled_scores.numel() > 0:
            hist_unlabeled = self._create_histogram(unlabeled_scores)
            loss_u = F.l1_loss(hist_unlabeled, self.proxy_unlabeled, reduction="mean")

        return loss_p + self.frac_prior * loss_u
