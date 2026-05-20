import torchvision
import numpy as np
from typing import Tuple
from .data_utils import (
    PUDataset,
    build_pu_datasets_from_binary_arrays,
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
    print_stats: bool = False,
    dataset_log_file: str | None = None,
) -> Tuple[PUDataset, PUDataset, PUDataset]:
    """
    Load, preprocess and return MNIST dataset for PU learning.

    The shared PU builder performs source-level train/validation splitting
    before PU sampling, then constructs PU labels and audit metadata.
    """
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

    return build_pu_datasets_from_binary_arrays(
        train_features,
        train_labels,
        test_features,
        test_labels,
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
        dataset_name="MNIST",
    )
