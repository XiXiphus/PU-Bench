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
    """Wrapper of loss function for PU learning in PyTorch."""

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
    """Wrapper of loss function for non-negative/unbiased PU learning.
        .. math::
            \\begin{array}{lc}
            L = \\pi R_p^+(f) + \\max(R_u^-(f) - \\pi R_p^-(f), \\beta) & {\\rm if nnPU learning}\\\\
            L = \\pi R_p^+(f) + R_u^-(f) - \\pi R_p^-(f) & {\\rm otherwise}
            \\end{array}
    Args:
        x (torch.Tensor): Input tensor.
            The shape of ``x`` should be (:math:`N`, 1).
        t (torch.Tensor): Target tensor.
            The shape of ``t`` should be (:math:`N`,).
        prior (float): Constant variable for class prior.
        loss (function): loss function.
            The loss function should be non-increasing.
        nnpu (bool): Whether use non-negative PU learning or unbiased PU learning.
            In default setting, non-negative PU learning will be used.
    Returns:
        torch.Tensor: A tensor object holding a scalar of the PU loss.
    See:
        Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama.
        "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
        Advances in neural information processing systems. 2017.
        du Plessis, Marthinus Christoffel, Gang Niu, and Masashi Sugiyama.
        "Convex formulation for learning from positive and unlabeled data."
        Proceedings of The 32nd International Conference on Machine Learning. 2015.
    """
    return PULoss(prior=prior, loss=loss, nnpu=nnpu)(x, t)
