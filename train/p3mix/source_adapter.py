"""P3Mix source input adapters.

The source P3Mix image loaders normalize CIFAR-10 inside ``hmix_image_dataset.py``.
PU-Bench keeps the benchmark split machinery, so this wrapper applies only the
method-local input scaling needed by source-hparam runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.reproducibility import seed_worker


class P3MixSourceInputDataset(Dataset):
    """Delegate dataset that transforms only the feature tensor."""

    def __init__(self, base_dataset: Dataset, mode: str):
        self.base_dataset = base_dataset
        self.mode = str(mode or "none").lower()
        for attr in (
            "features",
            "pu_labels",
            "true_labels",
            "indices",
            "pseudo_labels",
            "pu_metadata",
            "metadata",
            "source_indices",
            "source_roles",
        ):
            if hasattr(base_dataset, attr):
                setattr(self, attr, getattr(base_dataset, attr))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        x, *rest = item
        if isinstance(x, torch.Tensor) and self.mode == "cifar10_hmix":
            # Source hmix_image_dataset.normalise(..., mean=std=0.5) maps
            # uint8 CIFAR pixels to [-1, 1]. PU-Bench stores CIFAR as [0, 1].
            x = x.float().mul(2.0).sub(1.0)
        return (x, *rest)


def _loader_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_workers": int(params.get("num_workers", 0)),
        "pin_memory": bool(params.get("pin_memory", torch.cuda.is_available())),
        "worker_init_fn": seed_worker,
    }


def _wrap_dataset(dataset: Dataset, mode: str) -> Dataset:
    if mode in {"", "none", "false", "bench"}:
        return dataset
    if isinstance(dataset, P3MixSourceInputDataset):
        return dataset
    return P3MixSourceInputDataset(dataset, mode)


def install_p3mix_source_input_adapters(trainer: Any) -> None:
    """Rebuild P3Mix loaders with source image input scaling when configured."""

    mode = str(trainer.params.get("source_input_normalization", "none")).lower()
    if mode in {"", "none", "false", "bench"}:
        return

    batch_size = int(trainer.params.get("batch_size", 128))
    eval_batch_size = int(trainer.params.get("eval_batch_size", batch_size))
    kwargs = _loader_kwargs(trainer.params)

    trainer.train_loader = DataLoader(
        _wrap_dataset(trainer.train_loader.dataset, mode),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **kwargs,
    )
    if trainer.validation_loader is not None:
        trainer.validation_loader = DataLoader(
            _wrap_dataset(trainer.validation_loader.dataset, mode),
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            **kwargs,
        )
    trainer.test_loader = DataLoader(
        _wrap_dataset(trainer.test_loader.dataset, mode),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )


def p3mix_loader_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    return _loader_kwargs(params)


def create_p3mix_source_ema(
    model_factory: Any, device: torch.device
) -> SimpleNamespace:
    """Create the source-style EMA teacher as an independent detached model."""

    ema_model = model_factory().to(device)
    for param in ema_model.parameters():
        param.detach_()
        param.requires_grad_(False)
    ema_model.eval()
    return SimpleNamespace(ema=ema_model)
