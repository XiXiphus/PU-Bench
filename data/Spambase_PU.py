import numpy as np
from typing import Tuple

from .data_utils import (
    PUDataset,
    split_pu_val,
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

    Validation set is split AFTER PU labeling so that it preserves the PU
    structure (labeled positive vs. unlabeled), enabling realistic proxy-metric
    based model selection.
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

    # Train/Test split: split test from full data (keep trainval for PU creation)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_seed
    )

    # Create PU training set from ALL training data (before val split)
    pu_features, pu_true_labels_01, labeled_mask = create_pu_training_set(
        X_trainval,
        y_trainval,
        n_labeled=n_labeled,
        labeled_ratio=labeled_ratio,
        selection_strategy=selection_strategy,
        scenario=scenario,
        with_replacement=with_replacement,
        case_control_mode=case_control_mode,
    )

    # Split validation from PU data (AFTER PU labeling) to preserve PU structure
    (
        pu_train_features, pu_train_true_labels_01, train_labeled_mask,
        pu_val_features, pu_val_true_labels_01, val_labeled_mask,
    ) = split_pu_val(pu_features, pu_true_labels_01, labeled_mask, val_ratio, random_state=random_seed)

    # --- Label formatting ---

    # Train true_labels
    final_train_true_labels = np.full_like(pu_train_true_labels_01, true_negative_label)
    final_train_true_labels[pu_train_true_labels_01 == 1] = true_positive_label
    # Train pu_labels
    final_train_pu_labels = np.full(len(pu_train_true_labels_01), pu_unlabeled_label, dtype=int)
    final_train_pu_labels[train_labeled_mask == 1] = pu_labeled_label

    # Val true_labels (for Oracle metrics)
    final_val_true_labels = np.full_like(pu_val_true_labels_01, true_negative_label)
    final_val_true_labels[pu_val_true_labels_01 == 1] = true_positive_label
    # Val pu_labels (PU structure preserved!)
    final_val_pu_labels = np.full(len(pu_val_true_labels_01), pu_unlabeled_label, dtype=int)
    final_val_pu_labels[val_labeled_mask == 1] = pu_labeled_label

    # Test true_labels
    final_test_labels = np.full_like(y_test, true_negative_label)
    final_test_labels[y_test == 1] = true_positive_label

    train_dataset = PUDataset(
        features=pu_train_features,
        pu_labels=final_train_pu_labels,
        true_labels=final_train_true_labels,
    )
    val_dataset = PUDataset(
        features=pu_val_features,
        pu_labels=final_val_pu_labels,
        true_labels=final_val_true_labels,
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
