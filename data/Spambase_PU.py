import numpy as np
from typing import Tuple

from .data_utils import (
    PUDataset,
    split_train_val,
    create_pu_training_set,
    print_dataset_statistics,
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
    print_stats: bool = True,
    dataset_log_file: str | None = None,
) -> Tuple[PUDataset, PUDataset, PUDataset]:
    """
    Load UCI Spambase (via OpenML), produce PU-ready datasets.
    Labels: 1 for spam (positive), 0 for non-spam (negative).
    Features: numeric -> standard scaled float32.
    """
    rng = np.random.RandomState(random_seed)

    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    ds = fetch_openml(name="spambase", version=1, as_frame=True, data_home=data_dir)
    X_df = ds.data
    y = ds.target.astype(int).to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values).astype(np.float32)

    # Train/Val/Test split: first split test from full, then val from train
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=random_seed,
    )

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
