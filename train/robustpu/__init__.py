"""Robust-PU method package."""

from __future__ import annotations

from .losses import (
    binary_cross_entropy_loss,
    focal_binary_loss,
    hardness_values,
)
from .models import SourceCNN, SourceNormalNN, select_private_model
from .spl import TrainingScheduler, calculate_spl_weights

__all__ = [
    "RobustPUTrainer",
    "SourceCNN",
    "SourceNormalNN",
    "TrainingScheduler",
    "binary_cross_entropy_loss",
    "calculate_spl_weights",
    "focal_binary_loss",
    "hardness_values",
    "select_private_model",
]


def __getattr__(name: str):
    if name == "RobustPUTrainer":
        from .trainer import RobustPUTrainer

        return RobustPUTrainer
    raise AttributeError(name)
