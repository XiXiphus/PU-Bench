import torch
import torch.nn as nn
import torch.nn.functional as F


class PULDALabelDistributionLoss(nn.Module):
    """
    Label Distribution Alignment (PULDA) loss.
    - Encourages E[sigmoid(logits)|P] -> 1
    - Encourages E[sigmoid(logits)|U] -> prior
    Uses a softplus-based distance measure for the unlabeled term.
    """

    def __init__(self, prior: float, temperature: float = 1.0):
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError("prior must be in (0,1)")
        self.prior = float(prior)
        self.temperature = float(temperature)
        # Precompute targets
        self.target_p = 1.0
        self.target_u = self.prior

    def _softplus_distance(self, x1: torch.Tensor, x2: float) -> torch.Tensor:
        # Symmetric softplus distance on difference
        xdiff = x1 - x2
        return F.softplus(xdiff, beta=self.temperature) + F.softplus(
            -xdiff, beta=self.temperature
        )

    def forward(
        self, logits: torch.Tensor, pu_labels_binary: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: raw logits, shape [N]
            pu_labels_binary: binary labels where 1 indicates labeled positive (P), 0 indicates unlabeled (U)
        """
        scores = torch.sigmoid(logits.view(-1))
        labels = pu_labels_binary.view(-1)
        mask_p = labels == 1
        mask_u = labels == 0

        loss_p = torch.tensor(0.0, device=logits.device)
        loss_u = torch.tensor(0.0, device=logits.device)

        if mask_p.any():
            exp_p = scores[mask_p].mean()
            loss_p = self.target_p - exp_p
        if mask_u.any():
            exp_u = scores[mask_u].mean()
            loss_u = self._softplus_distance(exp_u, self.target_u)
        # Weighting same as original: 2*prior for labeled term
        return 2.0 * self.prior * loss_p + loss_u


class TwoWaySigmoidLoss(nn.Module):
    """
    Optional two-way sigmoid margin loss from PULDA to avoid trivial solutions.
    """

    def __init__(self, prior: float, margin: float = 0.6, temperature: float = 1.0):
        super().__init__()
        self.prior = float(prior)
        self.margin = float(margin)
        # Following original: force slope and distance temperature = 1.0
        self.slope = 1.0
        self.dist_temperature = 1.0

    def _pos_term(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.slope * z) * torch.sigmoid(
            -self.slope * (z - self.margin)
        )

    def _neg_term(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.slope * (z + self.margin)) * torch.sigmoid(
            -self.slope * z
        )

    def _softplus_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        xdiff = x1 - x2
        return F.softplus(xdiff, beta=self.dist_temperature) + F.softplus(
            -xdiff, beta=self.dist_temperature
        )

    def forward(
        self, logits: torch.Tensor, pu_labels_binary: torch.Tensor
    ) -> torch.Tensor:
        labels = pu_labels_binary.view(-1)
        z = logits.view(-1)
        mask_p = labels == 1
        mask_u = labels == 0

        c_p_plus = torch.tensor(0.0, device=logits.device)
        c_p_minus = torch.tensor(0.0, device=logits.device)
        c_u_minus = torch.tensor(0.0, device=logits.device)

        if mask_p.any():
            zp = z[mask_p]
            c_p_plus = self._pos_term(zp).mean()
            c_p_minus = self._neg_term(zp).mean()
        if mask_u.any():
            zu = z[mask_u]
            c_u_minus = self._neg_term(zu).mean()

        return self.prior * c_p_plus + self._softplus_distance(
            c_u_minus, self.prior * c_p_minus
        )


class PULDALabelDistributionLossWithEMA(PULDALabelDistributionLoss):
    """
    EMA version of PULDA Label Distribution Loss.
    Uses exponential moving average to smooth the expectation estimation.
    """

    def __init__(self, prior: float, temperature: float = 1.0, alpha_u: float = 0.85):
        super().__init__(prior, temperature)
        self.alpha_u = float(alpha_u)
        self.one_minus_alpha_u = 1.0 - self.alpha_u
        self.exp_y_hat_u_ema = None

    def forward(
        self, logits: torch.Tensor, pu_labels_binary: torch.Tensor
    ) -> torch.Tensor:
        """First call: initialize EMA, then switch to second_forward."""
        scores = torch.sigmoid(logits.view(-1))
        labels = pu_labels_binary.view(-1)
        mask_p = labels == 1
        mask_u = labels == 0

        loss_p = torch.tensor(0.0, device=logits.device)
        loss_u = torch.tensor(0.0, device=logits.device)

        if mask_p.any():
            exp_p = scores[mask_p].mean()
            loss_p = self.target_p - exp_p
        if mask_u.any():
            exp_u = scores[mask_u].mean()
            loss_u = self._softplus_distance(exp_u, self.target_u)
            # Initialize EMA
            self.exp_y_hat_u_ema = exp_u.detach()

        # Switch to EMA version after first unlabeled batch
        if mask_u.any():
            self.forward = self._second_forward

        return 2.0 * self.prior * loss_p + loss_u

    def _second_forward(
        self, logits: torch.Tensor, pu_labels_binary: torch.Tensor
    ) -> torch.Tensor:
        """EMA-enabled forward pass."""
        scores = torch.sigmoid(logits.view(-1))
        labels = pu_labels_binary.view(-1)
        mask_p = labels == 1
        mask_u = labels == 0

        loss_p = torch.tensor(0.0, device=logits.device)
        loss_u = torch.tensor(0.0, device=logits.device)

        if mask_p.any():
            exp_p = scores[mask_p].mean()
            loss_p = self.target_p - exp_p
        if mask_u.any():
            current_exp_u = scores[mask_u].mean()
            # EMA update
            exp_u_ema = (
                self.alpha_u * self.exp_y_hat_u_ema
                + self.one_minus_alpha_u * current_exp_u
            )
            loss_u = self._softplus_distance(exp_u_ema, self.target_u)
            self.exp_y_hat_u_ema = exp_u_ema.detach()

        # Scale unlabeled loss by EMA factor (following original)
        return 2.0 * self.prior * loss_p + loss_u / self.one_minus_alpha_u


class TwoWaySigmoidLossWithEMA(TwoWaySigmoidLoss):
    """
    EMA version of Two-Way Sigmoid Loss.
    Uses exponential moving average for the margin loss terms.
    """

    def __init__(
        self,
        prior: float,
        margin: float = 0.6,
        temperature: float = 1.0,
        alpha_cn: float = 0.5,
    ):
        super().__init__(prior, margin, temperature)
        self.alpha_cn = float(alpha_cn)
        self.one_minus_alpha_cn = 1.0 - self.alpha_cn
        self.c_p_minus_ema = None
        self.c_u_minus_ema = None

    def forward(
        self, logits: torch.Tensor, pu_labels_binary: torch.Tensor
    ) -> torch.Tensor:
        """First call: initialize EMA values, then switch to second_forward."""
        labels = pu_labels_binary.view(-1)
        z = logits.view(-1)
        mask_p = labels == 1
        mask_u = labels == 0

        c_p_plus = torch.tensor(0.0, device=logits.device)
        c_p_minus = torch.tensor(0.0, device=logits.device)
        c_u_minus = torch.tensor(0.0, device=logits.device)

        if mask_p.any():
            zp = z[mask_p]
            c_p_plus = self._pos_term(zp).mean()
            c_p_minus = self._neg_term(zp).mean()
            # Initialize EMA
            self.c_p_minus_ema = c_p_minus.detach()
        if mask_u.any():
            zu = z[mask_u]
            c_u_minus = self._neg_term(zu).mean()
            # Initialize EMA
            self.c_u_minus_ema = c_u_minus.detach()

        # Switch to EMA version after initialization
        if self.c_p_minus_ema is not None and self.c_u_minus_ema is not None:
            self.forward = self._second_forward

        return self.prior * c_p_plus + self._softplus_distance(
            c_u_minus, self.prior * c_p_minus
        )

    def _second_forward(
        self, logits: torch.Tensor, pu_labels_binary: torch.Tensor
    ) -> torch.Tensor:
        """EMA-enabled forward pass."""
        labels = pu_labels_binary.view(-1)
        z = logits.view(-1)
        mask_p = labels == 1
        mask_u = labels == 0

        c_p_plus = torch.tensor(0.0, device=logits.device)
        c_p_minus = self.c_p_minus_ema  # Use EMA value as default
        c_u_minus = self.c_u_minus_ema  # Use EMA value as default

        if mask_p.any():
            zp = z[mask_p]
            c_p_plus = self._pos_term(zp).mean()
            # EMA update
            current_c_p_minus = self._neg_term(zp).mean()
            c_p_minus = (
                self.alpha_cn * self.c_p_minus_ema
                + self.one_minus_alpha_cn * current_c_p_minus
            )
            self.c_p_minus_ema = c_p_minus.detach()
        if mask_u.any():
            zu = z[mask_u]
            # EMA update
            current_c_u_minus = self._neg_term(zu).mean()
            c_u_minus = (
                self.alpha_cn * self.c_u_minus_ema
                + self.one_minus_alpha_cn * current_c_u_minus
            )
            self.c_u_minus_ema = c_u_minus.detach()

        # Scale by EMA factor (following original)
        return (
            self.prior * c_p_plus
            + self._softplus_distance(c_u_minus, self.prior * c_p_minus)
            / self.one_minus_alpha_cn
        )
