"""LaGAM meta-model selection kept inside the method package."""

from __future__ import annotations

from ..model_factory import infer_model_name
from .models import (
    MetaCNN_AlzheimerMRI,
    MetaCNN_CIFAR10,
    MetaCNN_FashionMNIST,
    MetaCNN_MNIST,
    MetaDynamicMLP,
)


def select_model(params: dict, prior: float):
    model_name = infer_model_name(params)
    if model_name == "cnn_cifar10":
        return MetaCNN_CIFAR10(prior)
    if model_name == "cnn_fashionmnist":
        return MetaCNN_FashionMNIST(prior)
    if model_name == "cnn_mnist":
        return MetaCNN_MNIST(prior)
    if model_name == "cnn_alzheimermri":
        return MetaCNN_AlzheimerMRI(prior)
    if model_name in {"mlp_20News", "mlp_IMDB", "mlp_mushrooms", "mlp_spambase"}:
        return MetaDynamicMLP(prior)
    raise ValueError(f"No LaGAM meta model defined for {model_name!r}")


__all__ = ["select_model"]
