"""HolisticPU model selection kept inside the method package."""

from __future__ import annotations

from backbone.models import (
    HolisticPU_CNN_AlzheimerMRI,
    HolisticPU_CNN_CIFAR10,
    HolisticPU_CNN_FashionMNIST,
    HolisticPU_CNN_MNIST,
    HolisticPU_MLP_20News,
    HolisticPU_MLP_IMDB,
)

from ..utils.model_factory import infer_model_name
from .models import select_private_model


def select_model(params: dict, prior: float):
    dataset_class = params.get("dataset_class")
    if not dataset_class:
        raise ValueError("Parameter 'dataset_class' not found in the configuration.")

    model_name = infer_model_name(params)
    backbone_policy = str(params.get("backbone_policy", "controlled")).lower()
    if backbone_policy == "private":
        return select_private_model(
            dataset_class=str(dataset_class),
            model_name=model_name,
            prior=prior,
            arch=str(params.get("private_backbone_arch", "auto")),
        )
    if backbone_policy != "controlled":
        raise ValueError(
            "HolisticPU backbone_policy must be either 'controlled' or 'private', "
            f"got '{backbone_policy}'."
        )

    if model_name == "cnn_cifar10":
        return HolisticPU_CNN_CIFAR10(prior)
    if model_name == "cnn_fashionmnist":
        return HolisticPU_CNN_FashionMNIST(prior)
    if model_name == "cnn_mnist":
        return HolisticPU_CNN_MNIST(prior)
    if model_name == "cnn_alzheimermri":
        return HolisticPU_CNN_AlzheimerMRI(prior)
    if model_name == "mlp_20News":
        return HolisticPU_MLP_20News(prior)
    if model_name == "mlp_IMDB":
        return HolisticPU_MLP_IMDB(prior)
    if model_name == "mlp_mushrooms":
        return HolisticPU_MLP_20News(prior)
    if model_name == "mlp_spambase":
        return HolisticPU_MLP_20News(prior)

    raise ValueError(f"No controlled HolisticPU model defined for {model_name!r}")


__all__ = ["select_model"]
