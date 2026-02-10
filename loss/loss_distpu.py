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
        self.bin_width = 1.0 / self.num_bins
        self.bin_centers = (
            torch.arange(0, 1 + self.bin_width, self.bin_width)
            .view(1, -1)
            .to(self.device)
        )

        # Define target distributions (proxies)
        proxy_p = torch.zeros(self.num_bins + 1, device=self.device)
        proxy_n = torch.zeros(self.num_bins + 1, device=self.device)
        proxy_p[-1] = 1.0  # Positives should have scores of 1
        proxy_n[0] = 1.0  # Negatives should have scores of 0

        # The unlabeled set is a mixture of P and N
        self.proxy_unlabeled = self.prior * proxy_p + (1 - self.prior) * proxy_n
        self.proxy_positive = proxy_p

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
            return torch.zeros_like(self.bin_centers).squeeze(0)

        scores_reshaped = scores.view(-1, 1)
        # Calculate distance to each bin center and apply triangular kernel
        # Equivalent to: max(0, 1 - |score - bin_center| / bin_width)
        distances = torch.abs(scores_reshaped - self.bin_centers)
        in_range_mask = distances <= self.bin_width
        weights = (self.bin_width - distances) * in_range_mask

        # Normalize to form a valid distribution
        histogram = weights.sum(dim=0)
        return histogram / (histogram.sum() + 1e-8)

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

        # Create histograms from the scores
        hist_positive = self._create_histogram(positive_scores)
        hist_unlabeled = self._create_histogram(unlabeled_scores)

        # Calculate L1 loss between predicted and target histograms
        loss_p = F.l1_loss(hist_positive, self.proxy_positive, reduction="mean")
        loss_u = F.l1_loss(hist_unlabeled, self.proxy_unlabeled, reduction="mean")

        return loss_p + self.frac_prior * loss_u
