"""VPU method package."""

from .losses import VPULoss
from .trainer import VPUTrainer

__all__ = ["VPUTrainer", "VPULoss"]
