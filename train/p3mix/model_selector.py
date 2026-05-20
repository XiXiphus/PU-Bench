"""P3Mix mixup-compatible model selection kept inside the method package."""

from __future__ import annotations

from . import models as mix_models


_MODEL_BY_DATASET = {
    "CIFAR10": "MixCNN_CIFAR10",
    "FashionMNIST": "MixCNN_FashionMNIST",
    "MNIST": "MixCNN_MNIST",
    "AlzheimerMRI": "MixCNN_AlzheimerMRI",
    "20News": "MixMLP_20News",
    "IMDB": "MixMLP_20News",
    "Mushrooms": "MixMLP_20News",
    "Spambase": "MixMLP_20News",
    "Connect4": "MixMLP_20News",
}


def select_model(params: dict, prior: float):
    dataset_class = params.get("dataset_class")
    mix_model_name = _MODEL_BY_DATASET.get(dataset_class)
    if not mix_model_name:
        raise ValueError(f"No P3Mix model defined for {dataset_class}")
    selected_model_cls = getattr(mix_models, mix_model_name)
    return selected_model_cls(prior=prior)


__all__ = ["select_model"]
