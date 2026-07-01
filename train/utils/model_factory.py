"""Public benchmark backbone factory for PU-Bench trainers."""

from __future__ import annotations

from backbone.models import (
    CNN_CIFAR10,
    CNN_MNIST,
    MLP_IMDB,
    CNN_AlzheimerMRI,
    CNN_FashionMNIST,
    MLP_20News,
)


def infer_model_name(params: dict) -> str:
    """Infer or normalize the public benchmark backbone name."""

    dataset_class = params.get("dataset_class")
    if not dataset_class:
        raise ValueError("Parameter 'dataset_class' not found in the configuration.")

    model_name = params.get("model")
    low_cls = str(dataset_class).lower()
    if not model_name:
        if "cifar10" in low_cls:
            model_name = "cnn_cifar10"
        elif "fashionmnist" in low_cls:
            model_name = "cnn_fashionmnist"
        elif "mnist" in low_cls:
            model_name = "cnn_mnist"
        elif "alzheimer" in low_cls or "mri" in low_cls:
            model_name = "cnn_alzheimermri"
        elif "20news" in low_cls or "newsgroup" in low_cls:
            model_name = "mlp_20News"
        elif "imdb" in low_cls:
            model_name = "mlp_IMDB"
        elif "mushroom" in low_cls or "mushrooms" in low_cls:
            model_name = "mlp_mushrooms"
        elif "spambase" in low_cls:
            model_name = "mlp_spambase"
        elif "connect" in low_cls:
            model_name = "mlp_spambase"
        else:
            raise ValueError(
                f"Could not infer model for dataset_class '{dataset_class}'"
            )
    elif ("alzheimer" in low_cls or "mri" in low_cls) and model_name in (
        "cnn_cifar10",
        "cnn_mnist",
        "cnn_fashionmnist",
    ):
        model_name = "cnn_alzheimermri"

    return str(model_name)


def select_public_model(params: dict, prior: float):
    """Construct a public benchmark backbone.

    Method-private model families belong in ``train/<method>/`` selectors and
    should not be added to this shared factory.
    """

    model_name = infer_model_name(params)
    if model_name == "cnn_cifar10":
        return CNN_CIFAR10(prior)
    if model_name == "cnn_fashionmnist":
        return CNN_FashionMNIST(prior)
    if model_name == "cnn_mnist":
        return CNN_MNIST(prior)
    if model_name == "cnn_alzheimermri":
        return CNN_AlzheimerMRI(prior)
    if model_name == "mlp_20News":
        return MLP_20News(prior)
    if model_name == "mlp_IMDB":
        return MLP_IMDB(prior)
    if model_name == "mlp_mushrooms":
        return MLP_20News(prior)
    if model_name == "mlp_spambase":
        return MLP_20News(prior)

    raise ValueError(f"Could not find a public model for model_name '{model_name}'")


def select_model(method: str, params: dict, prior: float):
    """Backward-compatible public model selector.

    The ``method`` argument is retained for callers that have not yet moved to
    ``create_model()``. It no longer dispatches to method-private families.
    """

    return select_public_model(params=params, prior=prior)


__all__ = ["infer_model_name", "select_model", "select_public_model"]
