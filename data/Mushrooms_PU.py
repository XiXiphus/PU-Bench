import numpy as np
from typing import Tuple

from .data_utils import (
    PUDataset,
    build_pu_datasets_from_binary_arrays,
)


def load_mushrooms_pu(
    data_dir: str = "./datasets",
    n_labeled: int | None = None,
    labeled_ratio: float = 0.2,
    val_ratio: float = 0.2,
    target_prevalence: float | None = None,
    selection_strategy: str = "random",
    scenario: str = "single",
    case_control_mode: str = "naive_mode",
    random_seed: int = 42,
    true_positive_label: int = 1,
    true_negative_label: int = 0,
    pu_labeled_label: int = 1,
    pu_unlabeled_label: int = -1,
    with_replacement: bool = True,
    print_stats: bool = False,
    dataset_log_file: str | None = None,
) -> Tuple[PUDataset, PUDataset, PUDataset]:
    """
    Load UCI Mushrooms dataset (via OpenML), transform into PU format.

    Labels: 'p' (poisonous) vs 'e' (edible). We map 'p' -> 1 (positive), 'e' -> 0 (negative) by default.
    Features: all nominal -> one-hot encoded dense float32.

    The shared PU builder performs source-level train/validation splitting
    before PU sampling, then constructs PU labels and audit metadata.
    """
    # 1) Load from OpenML
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    ds = fetch_openml(
        name="mushroom",
        version=1,
        as_frame=False,
        data_home=data_dir,
        parser="liac-arff",
    )
    X_raw = np.asarray(ds.data)
    y_raw = np.asarray(ds.target)

    # 2) Map labels: 'p'->1, 'e'->0
    y_bin = (y_raw == "p").astype(int)

    # 3) Split before fitting preprocessing transforms to avoid test leakage.
    test_ratio = 0.2
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw.astype(str),
        y_bin,
        test_size=test_ratio,
        stratify=y_bin,
        random_state=random_seed,
    )

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train = enc.fit_transform(X_train_raw).astype(np.float32)
    X_test = enc.transform(X_test_raw).astype(np.float32)

    return build_pu_datasets_from_binary_arrays(
        X_train,
        y_train,
        X_test,
        y_test,
        positive_classes=["p"],
        negative_classes=["e"],
        n_labeled=n_labeled,
        labeled_ratio=labeled_ratio,
        val_ratio=val_ratio,
        target_prevalence=target_prevalence,
        selection_strategy=selection_strategy,
        scenario=scenario,
        case_control_mode=case_control_mode,
        random_seed=random_seed,
        true_positive_label=true_positive_label,
        true_negative_label=true_negative_label,
        pu_labeled_label=pu_labeled_label,
        pu_unlabeled_label=pu_unlabeled_label,
        with_replacement=with_replacement,
        print_stats=print_stats,
        dataset_log_file=dataset_log_file,
        dataset_name="Mushrooms",
    )
