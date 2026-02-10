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


def load_fashionmnist_pu(
    data_dir: str = "datasets/",
    positive_classes: list = [0, 2, 3, 4, 6],  # e.g., clothing
    negative_classes: list = [1, 5, 7, 8, 9],  # e.g., footwear/accessories
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
    Load, preprocess and return Fashion-MNIST dataset for PU learning.

    Args:
        data_dir (str): Dataset root directory.
        positive_classes (list): Positive class labels.
        negative_classes (list): Negative class labels.
        n_labeled (int, optional): Number of labeled samples. If None, calculated from labeled_ratio.
        labeled_ratio (float, optional): Ratio of labeled samples. Ignored if n_labeled is not None.
        val_ratio (float): Validation set ratio.
        selection_strategy (str): Strategy for selecting labeled samples ("random", "all", "sar_pusb"|"pusb").
        scenario (str): Scenario ("single", "case-control").
        random_seed (int): Random seed.
        true_positive_label (int): True label value for positive examples (for evaluation).
        true_negative_label (int): True label value for negative examples (for evaluation).
        pu_labeled_label (int): PU training label value for labeled positive examples.
        pu_unlabeled_label (int): PU training label value for unlabeled samples.
        with_replacement (bool): Whether to sample with replacement when needed (case-control scenario, etc.).
    """
    np.random.seed(random_seed)

    train_set_raw = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True
    )
    test_set_raw = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True
    )

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

    train_features, train_labels, val_features, val_labels = split_train_val(
        train_features, train_labels, val_ratio, random_state=random_seed
    )

    if target_prevalence is not None and target_prevalence > 0:
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
