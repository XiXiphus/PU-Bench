"""Self-paced weighting utilities for Robust-PU.

Primary source:
    author source file(s): robustpu/spl_utills.py
    woriazzc/Robust-PU at 34d950f2c6e56510855a922acb5f84b6459773ef
"""

from __future__ import annotations

import math

import torch


class TrainingScheduler:
    """Robust-PU pacing function for SPL thresholds."""

    def __init__(
        self,
        schedule_type: str,
        init_ratio: float,
        max_thresh: float,
        grow_steps: int,
        *,
        lam: float = 0.5,
    ) -> None:
        self.type = str(schedule_type).lower()
        self.init_ratio = float(init_ratio)
        self.max_thresh = float(max_thresh)
        self.grow_steps = max(1, int(grow_steps))
        self.lam = float(lam)
        self.step = 0

    def get_next_ratio(self) -> float:
        if self.type == "const":
            ratio = self.init_ratio
        elif self.type == "linear":
            ratio = self.init_ratio + (
                (self.max_thresh - self.init_ratio) / self.grow_steps
            ) * self.step
        elif self.type == "convex":
            ratio = self.init_ratio + (self.max_thresh - self.init_ratio) * math.sin(
                self.step / self.grow_steps * math.pi * 0.5
            )
        elif self.type == "concave":
            if self.step > self.grow_steps:
                ratio = self.max_thresh
            else:
                ratio = self.init_ratio + (
                    self.max_thresh - self.init_ratio
                ) * (1.0 - math.cos(self.step / self.grow_steps * math.pi * 0.5))
        elif self.type == "exp":
            if not 0.0 <= self.lam <= 1.0:
                raise ValueError("RobustPU exp scheduler requires lam in [0, 1].")
            ratio = self.init_ratio + (self.max_thresh - self.init_ratio) * (
                1.0 - self.lam**self.step
            )
        else:
            raise ValueError(f"Invalid RobustPU scheduler type: {self.type}")

        if self.init_ratio < self.max_thresh:
            ratio = min(ratio, self.max_thresh)
        else:
            ratio = max(ratio, self.max_thresh)
        self.step += 1
        return float(ratio)


def calculate_spl_weights(
    hardness: torch.Tensor,
    threshold: float,
    *,
    spl_type: str,
    mix2_gamma: float = 1.0,
    poly_t: float = 3.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Map hardness values to Robust-PU self-paced sample weights."""

    x = hardness.detach()
    threshold = max(float(threshold), eps)
    spl_type = str(spl_type).lower()

    if spl_type == "hard":
        weights = (x < threshold).float()
    elif spl_type == "linear":
        weights = 1.0 - x / threshold
        weights = torch.where(x >= threshold, torch.zeros_like(weights), weights)
    elif spl_type == "log":
        threshold = min(threshold, 1.0 - 1e-1)
        if not 0.0 < threshold < 1.0:
            raise ValueError("RobustPU log SPL requires threshold in (0, 1).")
        weights = torch.log(x + 1.0 - threshold) / torch.log(
            torch.tensor(1.0 - threshold, device=x.device, dtype=x.dtype)
        )
        weights = torch.where(x >= threshold, torch.zeros_like(weights), weights)
    elif spl_type == "mix2":
        gamma = float(mix2_gamma)
        weights = gamma * (1.0 / torch.sqrt(torch.clamp(x, min=eps)) - 1.0 / threshold)
        weights = torch.where(
            x <= (threshold * gamma / (threshold + gamma)) ** 2,
            torch.ones_like(weights),
            weights,
        )
        weights = torch.where(x >= threshold**2, torch.zeros_like(weights), weights)
    elif spl_type == "logistic":
        weights = (1.0 + torch.exp(torch.tensor(-threshold, device=x.device))) / (
            1.0 + torch.exp(x - threshold)
        )
    elif spl_type == "poly":
        t = float(poly_t)
        if t <= 1:
            raise ValueError("RobustPU poly SPL requires poly_t > 1.")
        weights = torch.pow(torch.clamp(1.0 - x / threshold, min=0.0), 1.0 / (t - 1.0))
        weights = torch.where(x >= threshold, torch.zeros_like(weights), weights)
    elif spl_type == "welsch":
        weights = torch.exp(-x / (threshold * threshold))
    elif spl_type == "cauchy":
        weights = 1.0 / (1.0 + x / (threshold * threshold))
    elif spl_type == "huber":
        sqrt_x = torch.sqrt(torch.clamp(x, min=eps))
        weights = threshold / sqrt_x
        weights = torch.where(sqrt_x <= threshold, torch.ones_like(weights), weights)
    elif spl_type == "l1l2":
        weights = 1.0 / torch.sqrt(threshold + x)
    else:
        raise ValueError(f"Invalid RobustPU SPL type: {spl_type}")

    return torch.clamp(weights, min=0.0, max=1.0)
