"""PUL-CPBF method package."""

from __future__ import annotations

__all__ = ["PULCPBFTrainer"]


def __getattr__(name: str):
    if name == "PULCPBFTrainer":
        from .trainer import PULCPBFTrainer

        return PULCPBFTrainer
    raise AttributeError(name)
