"""LaGAM method package.

The trainer import is lazy so method-private modules such as models, losses, and
augmentations can be imported without pulling in the full trainer stack.
"""

__all__ = ["LaGAMTrainer"]


def __getattr__(name: str):
    if name == "LaGAMTrainer":
        from .trainer import LaGAMTrainer

        return LaGAMTrainer
    raise AttributeError(name)
