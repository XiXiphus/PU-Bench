"""nnPUSB method package."""

from .losses import nnPUSBloss
from .trainer import NNPUSBTrainer

__all__ = ["NNPUSBTrainer", "nnPUSBloss"]
