"""PUL-CPBF model selection kept inside the method package."""

from __future__ import annotations

from ..model_factory import infer_model_name, select_public_model
from .models import select_private_model


def select_model(params: dict, prior: float):
    dataset_class = params.get("dataset_class")
    if not dataset_class:
        raise ValueError("Parameter 'dataset_class' not found in the configuration.")

    model_name = infer_model_name(params)
    backbone_policy = str(params.get("backbone_policy", "controlled")).lower()

    if backbone_policy == "controlled":
        return select_public_model(params=params, prior=prior)

    if backbone_policy in {"private", "source_available"}:
        try:
            return select_private_model(
                dataset_class=str(dataset_class),
                model_name=model_name,
                prior=prior,
                arch=str(params.get("private_backbone_arch", "auto")),
            )
        except ValueError:
            if backbone_policy == "source_available":
                return select_public_model(params=params, prior=prior)
            raise

    raise ValueError(
        "PULCPBF backbone_policy must be 'controlled', 'private', or "
        f"'source_available', got '{backbone_policy}'."
    )


__all__ = ["select_model"]
