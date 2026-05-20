import os
import numpy as np
from PIL import Image
from typing import Tuple, List
from .data_utils import (
    PUDataset,
    build_pu_datasets_from_binary_arrays,
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
    print_stats: bool = False,
    dataset_log_file: str | None = None,
    image_size: tuple = (128, 128),  # Resize images
) -> Tuple[PUDataset, PUDataset, PUDataset]:
    """
    Load, preprocess and return Alzheimer MRI dataset for PU learning.

    The shared PU builder performs source-level train/validation splitting
    before PU sampling, then constructs PU labels and audit metadata.

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

    from sklearn.model_selection import train_test_split

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        binary_labels,
        test_size=0.2,
        stratify=binary_labels,
        random_state=random_seed,
    )

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
        dataset_name="AlzheimerMRI",
    )
