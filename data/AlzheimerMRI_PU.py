import os
import numpy as np
from PIL import Image
from typing import Tuple, List
from .data_utils import (
    PUDataset,
    split_train_val,
    create_pu_training_set,
    print_dataset_statistics,
    resample_by_prevalence,
)


def load_alzheimer_mri_pu(
    data_dir: str = "datasets/Alzheimer_MRI_4_classes_dataset/",
    positive_classes: list = ["ModerateDemented", "MildDemented"],  # Dementia patients
    negative_classes: list = ["NonDemented", "VeryMildDemented"],  # Normal or mild
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
    image_size: tuple = (128, 128),  # Resize images
) -> Tuple[PUDataset, PUDataset, PUDataset]:
    """
    Load, preprocess and return Alzheimer MRI dataset for PU learning.

    Args:
        data_dir (str): Data storage directory.
        positive_classes (list): Class names defined as positive (dementia).
        negative_classes (list): Class names defined as negative (normal/mild).
        n_labeled (int): Number of labeled positive examples. Overrides labeled_ratio if provided.
        labeled_ratio (float): Ratio of positive examples to sample as labeled.
        val_ratio (float): Validation set ratio from original training data.
        target_prevalence (float): If specified, resamples the test set to match this positive prevalence.
        selection_strategy (str): Strategy for selecting labeled positive examples.
        scenario (str): PU learning scenario ('single', 'case-control').
        random_seed (int): Random seed for reproducibility.
        true_positive_label (int): True label value for positive examples.
        true_negative_label (int): True label value for negative examples.
        pu_labeled_label (int): PU training label value for labeled positive examples.
        pu_unlabeled_label (int): PU training label value for unlabeled samples.
        with_replacement (bool): Whether to sample with replacement.
        print_stats (bool): Whether to print dataset statistics.
        dataset_log_file (str): Log file path for dataset statistics.
        image_size (tuple): Target size for resizing images (height, width).

    Returns:
        A tuple containing (train_dataset, validation_dataset, test_dataset) as PUDataset objects.
    """
    np.random.seed(random_seed)

    def load_images_from_folders(
        base_dir: str, class_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load images from class folders."""
        images = []
        labels = []
        filenames = []

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found. Skipping...")
                continue

            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, filename)
                    try:
                        # Open image and convert to grayscale
                        img = Image.open(img_path).convert("L")
                        # Resize
                        img = img.resize(image_size, Image.LANCZOS)
                        # Convert to numpy array
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(class_idx)
                        filenames.append(f"{class_name}/{filename}")
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue

        return np.array(images), np.array(labels), filenames

    # Combine all classes
    all_classes = positive_classes + negative_classes

    # Load all images
    all_images, all_labels, all_filenames = load_images_from_folders(
        data_dir, all_classes
    )

    if len(all_images) == 0:
        raise ValueError(f"No images found in {data_dir}")

    print(f"Loaded {len(all_images)} images from {len(all_classes)} classes")

    # Convert multi-class labels to binary labels
    binary_labels = np.zeros_like(all_labels)
    for i, class_name in enumerate(all_classes):
        if class_name in positive_classes:
            binary_labels[all_labels == i] = 1
        else:
            binary_labels[all_labels == i] = 0

    # Normalize and adjust dimensions: (N, H, W) -> (N, 1, H, W)
    features = all_images.astype(np.float32) / 255.0
    features = (features - 0.5) / 0.5  # Standardize to [-1, 1]
    features = features[:, np.newaxis, :, :]  # Add channel dimension

    # Split training and test sets (80/20)
    indices = np.arange(len(features))
    np.random.shuffle(indices)

    split_idx = int(0.8 * len(features))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_features = features[train_indices]
    train_labels = binary_labels[train_indices]
    test_features = features[test_indices]
    test_labels = binary_labels[test_indices]

    # Split validation set from training set
    train_features, train_labels, val_features, val_labels = split_train_val(
        train_features, train_labels, val_ratio, random_state=random_seed
    )

    # Adjust test set class ratio (if needed)
    if target_prevalence is not None and target_prevalence > 0:
        test_features, test_labels = resample_by_prevalence(
            test_features, test_labels, target_prevalence, random_seed
        )

    # Create PU training set
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

    # PU training labels
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
