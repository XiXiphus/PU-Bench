"""Shared training schedule helpers."""

from __future__ import annotations

import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential ramp-up from https://arxiv.org/abs/1610.02242."""
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear ramp-up utility."""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    return current / rampup_length
