"""loss_bbepu.py

BBE (Binomial Bias Estimation) + nnPU Loss Implementation
Combines mixture proportion estimation with non-negative PU learning.

Based on:
- "Mixture Proportion Estimation and PU Learning: A Modern Approach" (Garg et al., 2021)
- "Positive-Unlabeled Learning with Non-Negative Risk Estimator" (Kiryo et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats


class BBEEstimator:
    """
    BBE (Binomial Bias Estimation) for mixture proportion estimation.

    This implementation is adapted from the original paper's code:
    "Mixture Proportion Estimation and PU Learning: A Modern Approach"
    """

    def __init__(self, delta=0.1, gamma=0.01):
        self.delta = delta  # Confidence parameter for DKW bound
        self.gamma = gamma  # Relaxation parameter

    def dkw_bound(self, x, y, t, m, n):
        """
        Compute DKW (Dvoretzky-Kiefer-Wolfowitz) bound for confidence interval.

        Args:
            x, y, t: Current values for bound computation
            m, n: Sample sizes
        """
        temp = np.sqrt(np.log(4 / self.delta) / 2 / n) + np.sqrt(
            np.log(4 / self.delta) / 2 / m
        )
        bound = temp * (1 + self.gamma) / (y / n)

        estimate = t
        return estimate, t - bound, t + bound

    def estimate_alpha(self, p_probs, u_probs, u_targets):
        """
        Estimate mixture proportion using BBE method.

        Args:
            p_probs: Probabilities for positive samples [N_p]
            u_probs: Probabilities for unlabeled samples [N_u, 2] (class probabilities)
            u_targets: True targets for unlabeled samples [N_u]

        Returns:
            float: Estimated mixture proportion alpha
        """
        # Convert to numpy if needed
        if torch.is_tensor(p_probs):
            p_probs = p_probs.detach().cpu().numpy()
        if torch.is_tensor(u_probs):
            u_probs = u_probs.detach().cpu().numpy()
        if torch.is_tensor(u_targets):
            u_targets = u_targets.detach().cpu().numpy()

        # Handle different input formats
        if u_probs.ndim == 1:
            u_probs_pos = u_probs
        else:
            u_probs_pos = u_probs[:, 0] if u_probs.shape[1] == 2 else u_probs

        # Sort probabilities
        p_indices = np.argsort(p_probs)
        sorted_p_probs = p_probs[p_indices]

        u_indices = np.argsort(u_probs_pos)
        sorted_u_probs = u_probs_pos[u_indices]
        sorted_u_targets = u_targets[u_indices]

        # Reverse sort for algorithm
        sorted_p_probs = sorted_p_probs[::-1]
        sorted_u_probs = sorted_u_probs[::-1]
        sorted_u_targets = sorted_u_targets[::-1]

        num = len(sorted_u_probs)

        estimate_arr = []
        upper_cfb = []
        lower_cfb = []

        i = 0
        j = 0

        while i < num:
            start_interval = sorted_u_probs[i]

            if i < num - 1 and start_interval > sorted_u_probs[i + 1]:
                pass
            else:
                i += 1
                continue

            # Count P samples above threshold
            while j < len(sorted_p_probs) and sorted_p_probs[j] >= start_interval:
                j += 1

            if j > 1 and i > 1:
                t = (i) * 1.0 * len(sorted_p_probs) / j / len(sorted_u_probs)
                estimate, lower, upper = self.dkw_bound(
                    i, j, t, len(sorted_u_probs), len(sorted_p_probs)
                )
                estimate_arr.append(estimate)
                upper_cfb.append(upper)
                lower_cfb.append(lower)
            i += 1

        if len(upper_cfb) != 0:
            idx = np.argmin(upper_cfb)
            mpe_estimate = estimate_arr[idx]
            return float(np.clip(mpe_estimate, 0.01, 0.99))  # Clip to valid range
        else:
            return 0.5  # Default fallback


class BBEPULoss(nn.Module):
    """
    Combined BBE estimation and nnPU loss for end-to-end training.
    """

    def __init__(
        self,
        initial_prior=0.5,
        loss_type="sigmoid",
        gamma=1.0,
        beta=0.0,
        bbe_update_freq=10,
        min_prior=0.01,
        max_prior=0.99,
    ):
        super(BBEPULoss, self).__init__()

        self.current_prior = initial_prior
        self.loss_type = loss_type
        self.gamma = gamma
        self.beta = beta
        self.bbe_estimator = BBEEstimator()
        # Allow disabling online prior update by setting freq<=0
        self.bbe_update_freq = (
            int(bbe_update_freq) if bbe_update_freq is not None else 10
        )
        self.min_prior = min_prior
        self.max_prior = max_prior

        # Tracking
        self.step_count = 0
        self.prior_history = []

        # Loss function mapping
        self.loss_funcs = {
            "sigmoid": lambda x: torch.sigmoid(-x),
            "logistic": lambda x: F.softplus(-x),
            "squared": lambda x: torch.square(x - 1) / 2,
        }
        self.loss_func = self.loss_funcs[loss_type]

    def update_prior_estimate(self, outputs, targets):
        """
        Update prior estimate using BBE method.

        Args:
            outputs: Model outputs [batch_size]
            targets: Target labels (1 for P, -1 for U) [batch_size]
        """
        with torch.no_grad():
            # Get probabilities
            probs = torch.sigmoid(outputs)

            # Separate P and U samples
            p_mask = targets == 1
            u_mask = targets == -1

            if p_mask.sum() > 0 and u_mask.sum() > 0:
                p_probs = probs[p_mask]
                u_probs = probs[u_mask]

                # For BBE, we need true labels for U samples
                # Since we don't have them in practice, we use a proxy based on probabilities
                # This is a limitation - in real scenarios you'd need validation data
                u_targets_proxy = (u_probs > 0.5).float()

                # Stack u_probs for BBE format
                u_probs_stacked = torch.stack([u_probs, 1 - u_probs], dim=1)

                try:
                    new_prior = self.bbe_estimator.estimate_alpha(
                        p_probs, u_probs_stacked, u_targets_proxy
                    )

                    # Smooth update
                    momentum = 0.9
                    self.current_prior = (
                        momentum * self.current_prior + (1 - momentum) * new_prior
                    )
                    self.current_prior = np.clip(
                        self.current_prior, self.min_prior, self.max_prior
                    )
                    self.prior_history.append(self.current_prior)

                except Exception:
                    # If BBE fails, keep current prior
                    pass

    def forward(self, outputs, targets, weights=None):
        """
        Forward pass with BBE-estimated prior and nnPU loss.

        Args:
            outputs: Model outputs [batch_size]
            targets: Target labels (1 for P, -1 for U) [batch_size]
            weights: Sample weights (optional)
        """
        self.step_count += 1

        # Update prior estimate periodically (only if enabled)
        if self.bbe_update_freq > 0 and (self.step_count % self.bbe_update_freq == 0):
            self.update_prior_estimate(outputs, targets)

        # nnPU loss computation with current prior
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        positive_mask = targets == 1
        unlabeled_mask = targets == -1

        n_positive = max(1, positive_mask.sum().item())
        n_unlabeled = max(1, unlabeled_mask.sum().item())

        if weights is None:
            weights = torch.ones_like(targets, dtype=outputs.dtype)

        # Positive risk
        positive_risk = (
            self.current_prior
            * torch.sum(self.loss_func(outputs[positive_mask]) * weights[positive_mask])
            / n_positive
        )

        # Negative risk
        negative_risk = (
            torch.sum(
                self.loss_func(-outputs[unlabeled_mask]) * weights[unlabeled_mask]
            )
            / n_unlabeled
            - self.current_prior
            * torch.sum(
                self.loss_func(-outputs[positive_mask]) * weights[positive_mask]
            )
            / n_positive
        )

        # Non-negative PU loss
        if negative_risk.item() < -self.beta:
            objective = positive_risk - self.beta
            grad_source = -self.gamma * negative_risk
            loss = grad_source + (objective - grad_source).detach()
        else:
            loss = positive_risk + negative_risk

        return loss

    def get_current_prior(self):
        """Get current estimated prior."""
        return self.current_prior

    def get_prior_history(self):
        """Get history of prior estimates."""
        return self.prior_history.copy()
