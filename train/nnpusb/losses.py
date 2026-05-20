from torch import nn
import torch.nn.functional as F


class nnPUSBloss(nn.Module):
    """PyTorch port of MasaKat0's neural PUSB/nnPUSB PU risk.

    Source implementation:
        https://github.com/MasaKat0/PUlearning/blob/master/BiasedPUlearning/nnPUSB/train_nnPU.py

    The source constructs selected-biased labeled positives in the data script,
    then trains a logistic nnPU risk with labels ``1`` for labeled positives and
    ``0`` for unlabeled examples. PU-Bench keeps its canonical unlabeled label
    as ``-1``; this loss is the direct label-mapped analogue. It intentionally
    does not apply any extra positive-risk weight. Selection bias belongs in the
    data construction, not in the estimator.

    Source risk:
        positive_risk = pi * E_P[softplus(-g)]
        negative_risk = E_U[softplus(g)] - pi * E_P[softplus(g)]
        loss = -negative_risk if negative_risk < 0 else positive_risk + negative_risk
    """

    def __init__(self, prior, gamma=1, beta=0, nnPU=True, unlabeled_label=-1):
        super(nnPUSBloss, self).__init__()

        if not 0 < prior < 1:
            raise ValueError("The class prior should be in (0, 1)")

        self.prior = float(prior)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.nnPU = nnPU
        self.positive_label = 1
        self.unlabeled_label = int(unlabeled_label)
        self.name = "nnpusb" if nnPU else "pusb"

    def forward(self, logits, target, test=False):
        """
        Args:
            logits: Model outputs (before sigmoid), shape (N,) or (N, 1)
            target: PU-Bench labels where 1 = positive, -1 = unlabeled
        """
        logits = logits.view(-1)
        target = target.view(-1)

        positive = target == self.positive_label
        unlabeled = target == self.unlabeled_label

        n_positive = max(1, positive.sum().item())
        n_unlabeled = max(1, unlabeled.sum().item())

        loss_positive = F.softplus(-logits)
        loss_negative = F.softplus(logits)

        positive_risk = self.prior * loss_positive[positive].sum() / n_positive

        negative_risk = (unlabeled * loss_negative).sum() / n_unlabeled - self.prior * (
            loss_negative[positive]
        ).sum() / n_positive

        if self.nnPU and negative_risk.item() < -self.beta:
            return -self.gamma * negative_risk
        return positive_risk + negative_risk


__all__ = ["nnPUSBloss"]
