import torchvision
import numpy as np
from typing import Tuple
from .data_utils import (
    PUDataset,
    split_train_val,
    create_pu_training_set,
    print_dataset_statistics,
    resample_by_prevalence,
)


def load_cifar10_pu(
    data_dir: str = "datasets/",
    positive_classes: list = [0, 1, 8, 9],  # e.g., vehicles
    negative_classes: list = [2, 3, 4, 5, 6, 7],  # e.g., animals
    n_labeled: int = None,
    labeled_ratio: float = 0.2,
    val_ratio: float = 0.2,
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
    Load, preprocess and return CIFAR-10 dataset for PU learning.

    Args:
        data_dir (str): Data storage directory.
        positive_classes (list): CIFAR-10 class indices defined as positive.
        negative_classes (list): CIFAR-10 class indices defined as negative.
        n_labeled (int): Number of labeled positive examples. Overrides labeled_ratio if provided.
        labeled_ratio (float): Ratio of positive examples to sample as labeled.
        val_ratio (float): Validation set ratio from original training data.
        target_prevalence (float): If specified, resamples the test set to match this positive prevalence.
        selection_strategy (str): Strategy for selecting labeled positive examples ('random', 'all', 'sar_pusb'|'pusb').
        scenario (str): PU learning scenario ('single', 'case-control').
        random_seed (int): Random seed for reproducibility.
        true_positive_label (int): True label value for positive examples (for evaluation).
        true_negative_label (int): True label value for negative examples (for evaluation).
        pu_labeled_label (int): PU training label value for labeled positive examples.
        pu_unlabeled_label (int): PU training label value for unlabeled samples.
        with_replacement (bool): Whether to sample with replacement when needed (case-control scenario, etc.).

    Returns:
        A tuple containing (train_dataset, validation_dataset, test_dataset) as PUDataset objects.
    """
    np.random.seed(random_seed)

    # Load original CIFAR-10 data
    train_set_raw = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True
    )
    test_set_raw = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True
    )

    def _extract_and_process(dataset):
        features = dataset.data
        labels = np.array(dataset.targets)

        binary_labels = np.full_like(labels, -1, dtype=int)
        binary_labels[np.isin(labels, positive_classes)] = 1
        binary_labels[np.isin(labels, negative_classes)] = 0
        valid_mask = binary_labels != -1

        proc_features = features[valid_mask]
        proc_labels = binary_labels[valid_mask]

        # Normalize and adjust dimensions: (N, H, W, C) uint8 -> (N, C, H, W) float32 [0,1]
        proc_features = proc_features.astype(np.float32) / 255.0
        proc_features = np.transpose(proc_features, (0, 3, 1, 2))

        return proc_features, proc_labels

    train_features, train_labels = _extract_and_process(train_set_raw)
    test_features, test_labels = _extract_and_process(test_set_raw)

    # Split validation set
    train_features, train_labels, val_features, val_labels = split_train_val(
        train_features, train_labels, val_ratio, random_state=random_seed
    )

    if target_prevalence is not None and target_prevalence > 0:
        print(f"Target test prevalence: {target_prevalence:.4f}")

        test_features, test_labels = resample_by_prevalence(
            test_features, test_labels, target_prevalence, random_seed
        )

    pu_train_features, pu_train_true_labels_01, train_labeled_mask = (
        create_pu_training_set(
            train_features,
            train_labels,
            n_labeled=n_labeled,
            labeled_ratio=labeled_ratio,
            selection_strategy=selection_strategy,
            scenario=scenario,
            with_replacement=with_replacement,
            case_control_mode=case_control_mode,
        )
    )

    # Label formatting
    final_pu_train_true_labels = np.full_like(
        pu_train_true_labels_01, true_negative_label
    )
    final_pu_train_true_labels[pu_train_true_labels_01 == 1] = true_positive_label
    final_val_labels = np.full_like(val_labels, true_negative_label)
    final_val_labels[val_labels == 1] = true_positive_label
    final_test_labels = np.full_like(test_labels, true_negative_label)
    final_test_labels[test_labels == 1] = true_positive_label
    final_pu_train_labels = np.full(
        len(pu_train_true_labels_01), pu_unlabeled_label, dtype=int
    )
    final_pu_train_labels[train_labeled_mask == 1] = pu_labeled_label

    # Create PUDataset instances
    train_dataset = PUDataset(
        features=pu_train_features,
        pu_labels=final_pu_train_labels,
        true_labels=final_pu_train_true_labels,
    )
    val_dataset = PUDataset(
        features=val_features,
        pu_labels=final_val_labels,
        true_labels=final_val_labels,
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
