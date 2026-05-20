import numpy as np
from typing import Tuple

from .data_utils import (
    PUDataset,
    build_pu_datasets_from_binary_arrays,
)


def load_spambase_pu(
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
    Load UCI Spambase (via OpenML), produce PU-ready datasets.
    Labels: 1 for spam (positive), 0 for non-spam (negative).
    Features: numeric -> standard scaled float32.

    The shared PU builder performs source-level train/validation splitting
    before PU sampling, then constructs PU labels and audit metadata.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    ds = fetch_openml(
        name="spambase",
        version=1,
        as_frame=False,
        data_home=data_dir,
        parser="liac-arff",
    )
    X_raw = np.asarray(ds.data)
    y = np.asarray(ds.target).astype(int)

    # Split before fitting preprocessing transforms to avoid test leakage.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return build_pu_datasets_from_binary_arrays(
        X_train,
        y_train,
        X_test,
        y_test,
        positive_classes=[1],
        negative_classes=[0],
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
        dataset_name="Spambase",
    )
