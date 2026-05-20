"""VAE-PU method package."""

from __future__ import annotations

__all__ = ["VAEPUTrainer"]


def __getattr__(name: str):
    if name == "VAEPUTrainer":
        from .trainer import VAEPUTrainer

        return VAEPUTrainer
    raise AttributeError(name)
