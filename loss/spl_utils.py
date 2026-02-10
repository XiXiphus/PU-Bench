"""SPL (Self-Paced Learning) utilities for Robust-PU.

This module contains weight calculation functions for self-paced learning.
"""

import torch
import numpy as np


def calculate_spl_weights(x, thresh, spl_type="linear", eps=1e-7):
    """
    Calculate SPL weights for given hardness values.

    Args:
        x: Hardness values (loss or distance)
        thresh: Threshold for weight calculation
        spl_type: Type of SPL weight function
        eps: Small value for numerical stability

    Returns:
        Weights for each sample
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # Ensure threshold is positive
    thresh = max(thresh, eps)

    if spl_type == "hard":
        # Binary weights: 1 if x < thresh, 0 otherwise
        weights = (x < thresh).float()

    elif spl_type == "linear":
        # Linear decay from 1 to 0
        weights = torch.clamp(1.0 - x / thresh, min=0.0, max=1.0)

    elif spl_type == "logistic":
        # Logistic function for smooth transition
        weights = (1.0 + torch.exp(torch.tensor(-thresh))) / (
            1.0 + torch.exp(x - thresh)
        )

    elif spl_type == "welsch":
        # Welsch robust loss function
        weights = torch.exp(-x / (thresh * thresh))

    elif spl_type == "cauchy":
        # Cauchy robust loss function
        weights = 1.0 / (1.0 + (x / thresh) ** 2)

    elif spl_type == "poly":
        # Polynomial function (t=2 for quadratic)
        t = 2.0
        weights = torch.clamp(torch.pow(1.0 - x / thresh, 1.0 / (t - 1)), min=0.0)

    elif spl_type == "log":
        # Logarithmic function
        thresh = min(thresh, 1.0 - eps)
        weights = torch.log(x + 1.0 - thresh) / torch.log(torch.tensor(1.0 - thresh))
        weights = torch.clamp(weights, min=0.0, max=1.0)
        weights = 1.0 - weights  # Invert so smaller x gets higher weight

    else:
        raise ValueError(f"Unknown SPL type: {spl_type}")

    # Ensure weights are in [0, 1]
    weights = torch.clamp(weights, min=0.0, max=1.0)

    return weights
