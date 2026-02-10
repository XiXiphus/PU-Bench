import numpy as np
from typing import Tuple

from .data_utils import (
    PUDataset,
    split_train_val,
    create_pu_training_set,
    print_dataset_statistics,
    resample_by_prevalence,
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
    print_stats: bool = True,
    dataset_log_file: str | None = None,
) -> Tuple[PUDataset, PUDataset, PUDataset]:
    """
    Load UCI Connect-4 (via OpenML), produce PU-ready datasets.

    Dataset notes:
      - Features: 42 categorical cells (values typically in {'x','o','b'}) + optional turn-related attrs.
      - Target: {'win','loss','draw'} → map to binary with 'win'→1 (positive), others→0 (negative).
    """
    rng = np.random.RandomState(random_seed)

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

    if sp is not None and sp.issparse(X_raw):
        X_dense = X_raw.toarray().astype(np.float32)
    else:
        X_arr = np.asarray(X_raw)
        if X_arr.dtype.kind in ("U", "S", "O"):
            # String/categorical → one-hot encode
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_dense = enc.fit_transform(X_arr.astype(str)).astype(np.float32)
        else:
            X_dense = X_arr.astype(np.float32)

    # 4) Train/Val/Test split: first split test from full, then val from train
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_dense, y_bin, test_size=0.2, stratify=y_bin, random_state=random_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=random_seed,
    )

    # 5) Optional test prevalence adjustment
    if target_prevalence is not None and target_prevalence > 0:
        X_test, y_test = resample_by_prevalence(
            X_test, y_test, target_prevalence, random_seed
        )

    # 6) Create PU training set
    pu_train_features, pu_train_true_labels_01, train_labeled_mask = (
        create_pu_training_set(
            X_train,
            y_train,
            n_labeled=n_labeled,
            labeled_ratio=labeled_ratio,
            selection_strategy=selection_strategy,
            scenario=scenario,
            with_replacement=with_replacement,
            case_control_mode=case_control_mode,
        )
    )

    # 7) Map to final label coding
    final_pu_train_true_labels = np.full_like(
        pu_train_true_labels_01, true_negative_label
    )
    final_pu_train_true_labels[pu_train_true_labels_01 == 1] = true_positive_label

    final_val_labels = np.full_like(y_val, true_negative_label)
    final_val_labels[y_val == 1] = true_positive_label

    final_test_labels = np.full_like(y_test, true_negative_label)
    final_test_labels[y_test == 1] = true_positive_label

    final_pu_train_labels = np.full(
        len(pu_train_true_labels_01), pu_unlabeled_label, dtype=int
    )
    final_pu_train_labels[train_labeled_mask == 1] = pu_labeled_label

    # 8) Build datasets
    train_dataset = PUDataset(
        features=pu_train_features,
        pu_labels=final_pu_train_labels,
        true_labels=final_pu_train_true_labels,
    )
    val_dataset = PUDataset(
        features=X_val,
        pu_labels=final_val_labels,
        true_labels=final_val_labels,
    )
    test_dataset = PUDataset(
        features=X_test,
        pu_labels=final_test_labels,
        true_labels=final_test_labels,
    )

    # 9) Stats
    print_dataset_statistics(
        train_dataset,
        val_dataset,
        test_dataset,
        train_labeled_mask,
        positive_classes=positive_classes,
        negative_classes=negative_classes,
        true_positive_label=true_positive_label,
        true_negative_label=true_negative_label,
        pu_labeled_label=pu_labeled_label,
        pu_unlabeled_label=pu_unlabeled_label,
        val_ratio=val_ratio,
        log_file=dataset_log_file,
        also_print=print_stats,
    )

    return train_dataset, val_dataset, test_dataset
