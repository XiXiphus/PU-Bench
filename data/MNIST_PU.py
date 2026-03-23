import torchvision
import numpy as np
from typing import Tuple
from .data_utils import (
    PUDataset,
    split_pu_val,
    create_pu_training_set,
    print_dataset_statistics,
    resample_by_prevalence,
)


def load_mnist_pu(
    data_dir: str = "datasets/",
    positive_classes: list = [0, 2, 4, 6, 8],
    negative_classes: list = [1, 3, 5, 7, 9],
    n_labeled: int = None,
    labeled_ratio: float = 0.2,
    val_ratio: float = 0.0,
    target_prevalence: float = None,
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
    Load, preprocess and return MNIST dataset for PU learning.

    Validation set is split AFTER PU labeling so that it preserves the PU
    structure (labeled positive vs. unlabeled), enabling realistic proxy-metric
    based model selection.
    """
    np.random.seed(random_seed)

    # Load original MNIST data
    train_set_raw = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    test_set_raw = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)

    def _extract_and_process(dataset):
        features = dataset.data.numpy()
        labels = dataset.targets.numpy()

        binary_labels = np.full_like(labels, -1, dtype=int)
        binary_labels[np.isin(labels, positive_classes)] = 1
        binary_labels[np.isin(labels, negative_classes)] = 0
        valid_mask = binary_labels != -1

        proc_features = features[valid_mask]
        proc_labels = binary_labels[valid_mask]

        # Normalize and adjust dimensions: (N, H, W) -> (N, 1, H, W)
        proc_features = proc_features.astype(np.float32) / 255.0
        proc_features = (proc_features - 0.5) / 0.5
        proc_features = proc_features[:, np.newaxis, :, :]

        return proc_features, proc_labels

    train_features, train_labels = _extract_and_process(train_set_raw)
    test_features, test_labels = _extract_and_process(test_set_raw)

    # Adjust test set prevalence if needed
    if target_prevalence is not None and target_prevalence > 0:
        test_features, test_labels = resample_by_prevalence(
            test_features, test_labels, target_prevalence, random_seed
        )

    # Create PU training set from ALL training data (before val split)
    pu_features, pu_true_labels_01, labeled_mask = create_pu_training_set(
        train_features,
        train_labels,
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

    # Val true_labels
    final_val_true_labels = np.full_like(pu_val_true_labels_01, true_negative_label)
    final_val_true_labels[pu_val_true_labels_01 == 1] = true_positive_label
    # Val pu_labels (PU structure preserved!)
    final_val_pu_labels = np.full(len(pu_val_true_labels_01), pu_unlabeled_label, dtype=int)
    final_val_pu_labels[val_labeled_mask == 1] = pu_labeled_label

    # Test true_labels
    final_test_labels = np.full_like(test_labels, true_negative_label)
    final_test_labels[test_labels == 1] = true_positive_label

    # Create PUDataset instances
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
        features=test_features,
        pu_labels=final_test_labels,
        true_labels=final_test_labels,
    )

    print_dataset_statistics(
        train_dataset,
        val_dataset,
        test_dataset,
        train_labeled_mask,
        positive_classes,
        negative_classes,
        true_positive_label,
        true_negative_label,
        pu_labeled_label,
        pu_unlabeled_label,
        val_ratio,
        log_file=dataset_log_file,
        also_print=print_stats,
    )

    return train_dataset, val_dataset, test_dataset
