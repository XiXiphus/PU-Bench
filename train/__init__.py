"""PU-Bench training package.

Method trainers live in method-local packages. ``run_train`` resolves them
through ``train.registry`` instead of importing legacy ``*_trainer.py`` shims.
"""

from __future__ import annotations

__all__ = ["BaseTrainer", "import_trainer_class", "list_registered_methods"]


def __getattr__(name: str):
    if name == "BaseTrainer":
        from train.base_trainer import BaseTrainer

        return BaseTrainer
    if name in {"import_trainer_class", "list_registered_methods"}:
        from train.registry import import_trainer_class, list_registered_methods

        return {
            "import_trainer_class": import_trainer_class,
            "list_registered_methods": list_registered_methods,
        }[name]
    raise AttributeError(f"module 'train' has no attribute {name!r}")
