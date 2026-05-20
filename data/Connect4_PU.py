import numpy as np
from typing import Tuple

from .data_utils import (
    PUDataset,
    build_pu_datasets_from_binary_arrays,
)


def load_connect4_pu(
    data_dir: str = "./datasets",
    positive_classes: list = ["win"],
    negative_classes: list = ["loss", "draw"],
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
    Load UCI Connect-4 (via OpenML), produce PU-ready datasets.

    Dataset notes:
      - Features: 42 categorical cells (values typically in {'x','o','b'}) + optional turn-related attrs.
      - Target: {'win','loss','draw'} → map to binary with 'win'→1 (positive), others→0 (negative).

    The shared PU builder performs source-level train/validation splitting
    before PU sampling, then constructs PU labels and audit metadata.
    """
    # 1) Load from OpenML
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    ds = fetch_openml(name="connect-4", version=1, as_frame=False, data_home=data_dir)
    X_raw = ds.data
    y_raw = ds.target

    # 2) Map labels to binary robustly: positive if label indicates a win
    y_arr = np.asarray(y_raw)
    y_str = np.char.lower(np.char.strip(y_arr.astype(str)))
    y_bin = np.isin(y_str, positive_classes).astype(int)

    # 3) Ensure dense float32 features. The OpenML connect-4 is a sparse ARFF; when
    #    loaded with as_frame=False, data may be a scipy sparse matrix already one-hot encoded.
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception:
        sp = None  # graceful fallback

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_bin, test_size=0.2, stratify=y_bin, random_state=random_seed
    )

    if sp is not None and sp.issparse(X_train_raw):
        X_train = X_train_raw.toarray().astype(np.float32)
        X_test = X_test_raw.toarray().astype(np.float32)
    else:
        X_train_arr = np.asarray(X_train_raw)
        X_test_arr = np.asarray(X_test_raw)
        if X_train_arr.dtype.kind in ("U", "S", "O"):
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_train = enc.fit_transform(X_train_arr.astype(str)).astype(np.float32)
            X_test = enc.transform(X_test_arr.astype(str)).astype(np.float32)
        else:
            X_train = X_train_arr.astype(np.float32)
            X_test = X_test_arr.astype(np.float32)

    return build_pu_datasets_from_binary_arrays(
        X_train,
        y_train,
        X_test,
        y_test,
        positive_classes=positive_classes,
        negative_classes=negative_classes,
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
        dataset_name="Connect4",
    )
