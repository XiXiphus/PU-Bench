"""PULDA method package."""

from __future__ import annotations

__all__ = ["PULDATrainer"]


def __getattr__(name: str):
    if name == "PULDATrainer":
        from .trainer import PULDATrainer

        return PULDATrainer
    raise AttributeError(name)
