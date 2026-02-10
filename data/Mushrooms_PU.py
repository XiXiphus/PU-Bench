import numpy as np
from typing import Tuple

from .data_utils import (
    PUDataset,
    split_train_val,
    create_pu_training_set,
    print_dataset_statistics,
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
    print_stats: bool = True,
    dataset_log_file: str | None = None,
) -> Tuple[PUDataset, PUDataset, PUDataset]:
    """
    Load UCI Mushrooms dataset (via OpenML), transform into PU format.

    Labels: 'p' (poisonous) vs 'e' (edible). We map 'p' -> 1 (positive), 'e' -> 0 (negative) by default.
    Features: all nominal -> one-hot encoded dense float32.
    """
    rng = np.random.RandomState(random_seed)

    # 1) Load from OpenML
    # Avoid hard network requirement by trying sklearn's fetch_openml
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OneHotEncoder

    ds = fetch_openml(name="mushroom", version=1, as_frame=True, data_home=data_dir)
    X_df = ds.data
    y_raw = ds.target.to_numpy()

    # 2) Map labels: 'p'->1, 'e'->0
    y_bin = (y_raw == "p").astype(int)

    # 3) One-hot encode categorical features
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_onehot = enc.fit_transform(X_df.astype(str))
    X_onehot = X_onehot.astype(np.float32)

    # 4) Train/Val split
    X_train, y_train, X_val, y_val = split_train_val(
        X_onehot, y_bin, val_ratio, random_state=random_seed
    )

    # 5) Optional test prevalence adjustment: For Mushrooms we keep natural test split by random split from remaining
    # Here we split test from remaining data not used in train/val
    # Using a simple random split from remaining pool (simulate standard benchmark pattern)
    # Build test as remaining from original after removing train indices already handled by split_train_val
    # Since split_train_val returns only train/val, we will create test from the full set by another split
    # Strategy: split full into (train+val) vs test with ratio 0.2 (complement of (1 - val_ratio) may overfit)
    # To keep consistency with other loaders that use original dataset's test split, we simply reuse X_val as validation
    # and create a separate test via another random split from the leftover indices.
    # Here, a simple approach: split X_onehot/y_bin into train_and_val vs test (20%), then within train_and_val we already did val split above.

    # If val_ratio < 1.0, create an independent test split from full data with 20%
    test_ratio = 0.2
    from sklearn.model_selection import train_test_split

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_onehot, y_bin, test_size=test_ratio, stratify=y_bin, random_state=random_seed
    )

    # Recompute train vs val from trainval to align indices
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=random_seed,
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

    # 7) Map labels to requested coding
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
        positive_classes=[],
        negative_classes=[],
        true_positive_label=true_positive_label,
        true_negative_label=true_negative_label,
        pu_labeled_label=pu_labeled_label,
        pu_unlabeled_label=pu_unlabeled_label,
        val_ratio=val_ratio,
        log_file=dataset_log_file,
        also_print=print_stats,
    )

    return train_dataset, val_dataset, test_dataset
