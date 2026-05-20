"""Self-PU method package."""

from __future__ import annotations

__all__ = ["SelfPUTrainer"]


def __getattr__(name: str):
    if name == "SelfPUTrainer":
        from .trainer import SelfPUTrainer

        return SelfPUTrainer
    raise AttributeError(name)
