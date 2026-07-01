"""Data loader construction for PU-Bench training runs."""

from __future__ import annotations

from typing import Tuple

from data.AlzheimerMRI_PU import load_alzheimer_mri_pu
from data.CIFAR10_PU import load_cifar10_pu
from data.data_utils import (
    PU_SAMPLING_REFERENCES,
    PUDataloader,
    canonical_case_control_mode,
    get_pu_risk_prior,
)
from data.FashionMNIST_PU import load_fashionmnist_pu
from data.IMDB_PU import load_imdb_pu
from data.MNIST_PU import load_mnist_pu
from data.News20_PU import load_20news_pu

from .reproducibility import seed_worker


def prepare_loaders(
    dataset_name: str,
    data_config: dict,
    batch_size: int = 128,
    data_dir: str = "data",
    shuffle_train: bool = True,
    method: str = "default",
) -> Tuple[PUDataloader, PUDataloader | None, PUDataloader, float, PUDataloader | None]:
    """Create PU datasets and wrap them in PUDataloader instances."""
    dataset_class = data_config.get("dataset_class", "")
    if "cifar" in dataset_class.lower():
        loader_func = load_cifar10_pu
    elif "fashionmnist" in dataset_class.lower():
        loader_func = load_fashionmnist_pu
    elif "mnist" in dataset_class.lower():
        loader_func = load_mnist_pu
    elif "alzheimer" in dataset_class.lower() or "mri" in dataset_class.lower():
        loader_func = load_alzheimer_mri_pu
    elif "20news" in dataset_class.lower() or "newsgroup" in dataset_class.lower():
        loader_func = load_20news_pu
    elif "imdb" in dataset_class.lower():
        loader_func = load_imdb_pu
    elif "mushroom" in dataset_class.lower():
        from data.Mushrooms_PU import load_mushrooms_pu

        loader_func = load_mushrooms_pu
    elif "spambase" in dataset_class.lower():
        from data.Spambase_PU import load_spambase_pu

        loader_func = load_spambase_pu
    elif "connect" in dataset_class.lower():
        from data.Connect4_PU import load_connect4_pu

        loader_func = load_connect4_pu
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name} / {dataset_class}")

    import inspect

    sig = inspect.signature(loader_func)
    loader_params = {
        p.name: data_config[p.name]
        for p in sig.parameters.values()
        if p.name in data_config and p.name != "data_dir"
    }

    if "label_scheme" in data_config:
        scheme = data_config["label_scheme"]
        if isinstance(scheme, dict):
            loader_params.update(scheme)

    train_dataset, val_dataset, test_dataset = loader_func(
        data_dir=data_dir, **loader_params
    )

    if "cifar" in dataset_class.lower():
        train_dataset.mean = (0.4914, 0.4822, 0.4465)
        train_dataset.std = (0.2023, 0.1994, 0.2010)
    elif (
        "mnist" in dataset_class.lower()
        or "fashionmnist" in dataset_class.lower()
        or "alzheimer" in dataset_class.lower()
    ):
        train_dataset.mean = (0.5,)
        train_dataset.std = (0.5,)
        if "alzheimer" in dataset_class.lower():
            try:
                train_dataset.image_size = 128
            except Exception:
                pass

    scenario = data_config.get("scenario")
    semantic_cc_mode = (
        canonical_case_control_mode(data_config.get("case_control_mode"))
        if scenario == "case-control"
        else None
    )
    reference_key = semantic_cc_mode if semantic_cc_mode is not None else scenario

    context_metadata = {
        "dataset_class": dataset_class,
        "scenario": scenario,
        "selection_strategy": data_config.get("selection_strategy"),
        "case_control_mode": data_config.get("case_control_mode"),
        "case_control_semantics": semantic_cc_mode,
        "c_requested": data_config.get("labeled_ratio"),
        "prior_source": "pi_unlabeled",
        "sampling_reference": PU_SAMPLING_REFERENCES.get(reference_key),
    }
    for ds in (train_dataset, val_dataset):
        if ds is not None and hasattr(ds, "pu_metadata"):
            ds.pu_metadata.update(
                {k: v for k, v in context_metadata.items() if v is not None}
            )
            ds.metadata = ds.pu_metadata

    prior = get_pu_risk_prior(train_dataset)
    num_workers = data_config.get("num_workers", 0)

    train_loader = PUDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    validation_loader = None
    if val_dataset and len(val_dataset) > 0:
        validation_loader = PUDataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

    test_loader = PUDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    update_loader = None
    if method in [
        "selfpu",
        "holisticpu",
        "robustpu",
        "pulda",
        "vaepu",
        "lbe",
        "bbepu",
    ]:
        update_loader = PUDataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

    return train_loader, validation_loader, test_loader, prior, update_loader
