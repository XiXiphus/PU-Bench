"""train_utils.py

Provides common utility functions for training workflows, including data loading,
model selection, metric evaluation, seeding, and helper
utilities used by various PU-learning methods.

Main functions:
    - prepare_loaders:  Return train/val/test DataLoaders with class prior π,
                        plus an optional non-shuffled update_loader.
    - select_model:     Instantiate the model that matches the method/dataset.
    - evaluate_metrics: Evaluate on a DataLoader and return a metrics dict
                        (PU risk, error, accuracy, precision, recall, F1).
    - set_global_seed:  Set global random seeds for reproducibility.

"""

from __future__ import annotations

import os
import json
import copy
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data.data_utils import PUDataloader, PUDataset
from backbone.models import (
    CNN_CIFAR10,
    CNN_FashionMNIST,
    CNN_MNIST,
    CNN_AlzheimerMRI,
    MLP_20News,
    HolisticPU_CNN_CIFAR10,
    HolisticPU_CNN_FashionMNIST,
    HolisticPU_CNN_MNIST,
    HolisticPU_CNN_AlzheimerMRI,
    HolisticPU_MLP_20News,
    MLP_IMDB,
    HolisticPU_MLP_IMDB,
)
from backbone.meta_models import (
    MetaCNN_CIFAR10,
    MetaCNN_FashionMNIST,
    MetaCNN_MNIST,
    MetaCNN_AlzheimerMRI,
)
from backbone.mix_models import (
    MixCNN_CIFAR10,
    MixCNN_FashionMNIST,
    MixCNN_MNIST,
    MixCNN_AlzheimerMRI,
    MixMLP_20News,
    MixMLP_IMDB,
)

from data.CIFAR10_PU import load_cifar10_pu
from data.FashionMNIST_PU import load_fashionmnist_pu
from data.MNIST_PU import load_mnist_pu
from data.AlzheimerMRI_PU import load_alzheimer_mri_pu
from data.News20_PU import load_20news_pu
from data.IMDB_PU import load_imdb_pu
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------


def prepare_loaders(
    dataset_name: str,
    data_config: dict,
    batch_size: int = 128,
    data_dir: str = "data",
    shuffle_train: bool = True,
    method: str = "default",
) -> Tuple[PUDataloader, PUDataloader | None, PUDataloader, float, PUDataloader | None]:
    """Create PU datasets and wrap them in PUDataloader instances.

    Returns:
        train_loader:      Training loader.
        validation_loader: Optional validation loader (may be None).
        test_loader:       Test loader.
        prior:             Class prior π (positive proportion in training set).
        update_loader:     Optional non-shuffled train loader used by certain
                           methods for updates/analysis (may be None).
    """
    dataset_class = data_config.get("dataset_class", "")
    if "cifar" in dataset_class.lower():
        loader_func = load_cifar10_pu
    elif "fashionmnist" in dataset_class.lower():
        loader_func = load_fashionmnist_pu
    elif "mnist" in dataset_class.lower():
        loader_func = load_mnist_pu
    elif "alzheimer" in dataset_class.lower() or "mri" in dataset_class.lower():
        loader_func = load_alzheimer_mri_pu
    elif "20news" in dataset_class.lower() or "newsgroup" in dataset_class.lower():
        loader_func = load_20news_pu
    elif "imdb" in dataset_class.lower():
        loader_func = load_imdb_pu
    elif "mushroom" in dataset_class.lower():
        from data.Mushrooms_PU import load_mushrooms_pu

        loader_func = load_mushrooms_pu
    elif "spambase" in dataset_class.lower():
        from data.Spambase_PU import load_spambase_pu

        loader_func = load_spambase_pu
    elif "connect" in dataset_class.lower():
        from data.Connect4_PU import load_connect4_pu

        loader_func = load_connect4_pu
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name} / {dataset_class}")

    # Pass only parameters that appear in the selected loader's signature
    import inspect

    sig = inspect.signature(loader_func)
    loader_params = {
        p.name: data_config[p.name]
        for p in sig.parameters.values()
        if p.name in data_config and p.name != "data_dir"
    }

    # NOTE: Previously, dataset statistics were written to a plain-text file via
    # loader parameters (dataset_log_file/print_stats). We now centralize structured
    # result logging in BaseTrainer → result.json, so we stop passing those params.

    # Merge label_scheme fields, if provided, into loader parameters
    if "label_scheme" in data_config:
        scheme = data_config["label_scheme"]
        if isinstance(scheme, dict):
            loader_params.update(scheme)

    train_dataset, val_dataset, test_dataset = loader_func(
        data_dir=data_dir, **loader_params
    )

    # LaGAM-specific: build a validation split if none exists
    if method.lower() == "lagam" and (val_dataset is None or len(val_dataset) == 0):
        console = Console()
        console.log(
            "LaGAM method detected with an empty validation set. Creating one from the training set.",
            style="yellow",
        )
        lagam_val_ratio = data_config.get("lagam_val_ratio", 0.1)
        if lagam_val_ratio > 0 and len(train_dataset) > 0:
            train_indices = np.arange(len(train_dataset))

            # Stratified split using true labels
            new_train_indices, val_indices = train_test_split(
                train_indices,
                test_size=lagam_val_ratio,
                stratify=train_dataset.true_labels.numpy(),
                random_state=data_config.get("seed", 42),
            )

            # Create a validation dataset (use clean labels for convenience)
            # Use local, contiguous indices for the split datasets to ensure
            # downstream modules (e.g., LaGAM feature writing and clustering)
            # can safely index tensors sized to the split length.
            _val_len = len(val_indices)
            val_dataset = PUDataset(
                features=train_dataset.features[val_indices],
                pu_labels=train_dataset.true_labels[
                    val_indices
                ],  # reuse true labels for PU labels in val
                true_labels=train_dataset.true_labels[val_indices],
                indices=torch.arange(_val_len),
                pseudo_labels=train_dataset.pseudo_labels[val_indices],
            )

            # Shrink the training dataset accordingly
            _tr_len = len(new_train_indices)
            train_dataset = PUDataset(
                features=train_dataset.features[new_train_indices],
                pu_labels=train_dataset.pu_labels[new_train_indices],
                true_labels=train_dataset.true_labels[new_train_indices],
                indices=torch.arange(_tr_len),
                pseudo_labels=train_dataset.pseudo_labels[new_train_indices],
            )
            console.log(
                f"Split training set: {len(train_dataset)} for training, {len(val_dataset)} for LaGAM validation.",
                style="green",
            )

    # Attach dataset normalization stats for later augmentations
    if "cifar" in dataset_class.lower():
        train_dataset.mean = (0.4914, 0.4822, 0.4465)
        train_dataset.std = (0.2023, 0.1994, 0.2010)
    elif (
        "mnist" in dataset_class.lower()
        or "fashionmnist" in dataset_class.lower()
        or "alzheimer" in dataset_class.lower()
    ):
        train_dataset.mean = (0.5,)
        train_dataset.std = (0.5,)
        # Provide expected size hints for image augmentation and evaluation adaptation (Alzheimer MRI uses 128x128 grayscale)
        if "alzheimer" in dataset_class.lower():
            try:
                train_dataset.image_size = 128
            except Exception:
                pass

    # Class prior π is the positive fraction in the training set (using true labels)
    prior = (train_dataset.true_labels == 1).float().mean().item()

    num_workers = data_config.get("num_workers", 0)

    train_loader = PUDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    validation_loader = None
    if val_dataset and len(val_dataset) > 0:
        validation_loader = PUDataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

    test_loader = PUDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    # Optional non-shuffled train loader for methods that need it
    update_loader = None
    if method in [
        "selfpu",
        "holisticpu",
        "robustpu",
        "pulda",
        "vaepu",
        "lbe",
        "bbepu",
    ]:
        update_loader = PUDataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

    return train_loader, validation_loader, test_loader, prior, update_loader


def select_model(method: str, params: dict, prior: float):
    """Select an appropriate model instance based on the method and dataset configuration."""
    dataset_class = params.get("dataset_class")
    if not dataset_class:
        raise ValueError("Parameter 'dataset_class' not found in the configuration.")

    # Infer a model name if one is not explicitly provided
    model_name = params.get("model")
    if not model_name:
        low_cls = dataset_class.lower()
        if "cifar10" in low_cls:
            model_name = "cnn_cifar10"
        elif "fashionmnist" in low_cls:
            model_name = "cnn_fashionmnist"
        elif "mnist" in low_cls:
            model_name = "cnn_mnist"
        elif "alzheimer" in low_cls or "mri" in low_cls:
            model_name = "cnn_alzheimermri"
        elif "20news" in low_cls or "imdb" in low_cls:
            model_name = "mlp_" + dataset_class
        elif "mushroom" in low_cls or "mushrooms" in low_cls:
            model_name = "mlp_mushrooms"
        elif "spambase" in low_cls:
            model_name = "mlp_spambase"
        elif "connect" in low_cls:
            # Reuse tabular MLP backbone
            model_name = "mlp_spambase"
        else:
            raise ValueError(
                f"Could not infer model for dataset_class '{dataset_class}'"
            )
    else:
        # Force switch to corresponding CNN backbone for AlzheimerMRI (even if method YAML specifies other CNN)
        low_cls = dataset_class.lower()
        if ("alzheimer" in low_cls or "mri" in low_cls) and model_name in (
            "cnn_cifar10",
            "cnn_mnist",
            "cnn_fashionmnist",
        ):
            model_name = "cnn_alzheimermri"

    method_lower = method.lower()

    # Method-specific variants
    if method_lower == "holisticpu":
        if model_name == "cnn_cifar10":
            return HolisticPU_CNN_CIFAR10(prior)
        if model_name == "cnn_fashionmnist":
            return HolisticPU_CNN_FashionMNIST(prior)
        if model_name == "cnn_mnist":
            return HolisticPU_CNN_MNIST(prior)
        if model_name == "cnn_alzheimermri":
            return HolisticPU_CNN_AlzheimerMRI(prior)
        if model_name == "mlp_20News":
            return HolisticPU_MLP_20News(prior)
        if model_name == "mlp_IMDB":
            return HolisticPU_MLP_IMDB(prior)
        # Tabular/text MLP variants should also use 2-class outputs under HolisticPU
        if model_name == "mlp_mushrooms":
            return HolisticPU_MLP_20News(prior)
        if model_name == "mlp_spambase":
            return HolisticPU_MLP_20News(prior)

    elif method_lower == "lagam":
        if model_name == "cnn_cifar10":
            return MetaCNN_CIFAR10(prior)
        if model_name == "cnn_fashionmnist":
            return MetaCNN_FashionMNIST(prior)
        if model_name == "cnn_mnist":
            return MetaCNN_MNIST(prior)
        if model_name == "cnn_alzheimermri":
            return MetaCNN_AlzheimerMRI(prior)

    elif method_lower in ["p3mixc", "p3mixe"]:
        if model_name == "cnn_cifar10":
            return MixCNN_CIFAR10(prior)
        if model_name == "cnn_fashionmnist":
            return MixCNN_FashionMNIST(prior)
        if model_name == "cnn_mnist":
            return MixCNN_MNIST(prior)
        if model_name == "cnn_alzheimermri":
            return MixCNN_AlzheimerMRI(prior)
        if model_name == "mlp_20News":
            return MixMLP_20News(prior)
        if model_name == "mlp_IMDB":
            return MixMLP_IMDB(prior)
        if model_name == "mlp_mushrooms":
            return MixMLP_20News(prior)
        if model_name == "mlp_spambase":
            return MixMLP_20News(prior)

    # Default (baseline) models
    if model_name == "cnn_cifar10":
        return CNN_CIFAR10(prior)
    if model_name == "cnn_fashionmnist":
        return CNN_FashionMNIST(prior)
    if model_name == "cnn_mnist":
        return CNN_MNIST(prior)
    if model_name == "cnn_alzheimermri":
        return CNN_AlzheimerMRI(prior)
    if model_name == "mlp_20News":
        return MLP_20News(prior)
    if model_name == "mlp_IMDB":
        return MLP_IMDB(prior)
    if model_name == "mlp_mushrooms":
        # Reuse 20News MLP (dense tabular/text) for Mushrooms tabular
        return MLP_20News(prior)
    if model_name == "mlp_spambase":
        # Reuse 20News MLP for Spambase tabular features
        return MLP_20News(prior)

    raise ValueError(
        f"Could not find a matching model for method '{method}' and model_name '{model_name}'"
    )


_zero_one_loss = lambda x: (torch.sign(-x) + 1) / 2


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prior: float,
) -> dict[str, float]:
    """Evaluate a model on a PU-formatted DataLoader.

    The evaluation computes:
        - "risk":  Unbiased PU risk estimate based on a zero-one loss surrogate.
                   Uses logits (log-odds) to keep the decision centered at 0.
        - "error": 1 - accuracy.
        - "accuracy", "precision", "recall", "f1": Standard classification metrics
                   computed against true labels.

    Notes:
        * If the model outputs probabilities (in [0, 1]), we threshold at 0.5 for
          binary predictions and convert to logits via logit(p) for risk.
        * If the model outputs logits, we threshold at 0 and use raw logits for risk.
        * For multi-class outputs (C>1), we reduce to argmax for predictions and
          use the score of class 0 for the risk terms to maintain a single score.
    """
    y_true_all, y_pred_all = [], []
    y_scores_all = []
    total_risk_sum = 0.0

    # Helper: adapt input to model's expected channels/size (first Conv2d)
    def _adapt_input_for_model(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not (isinstance(x, torch.Tensor) and x.dim() == 4):
            return x
        exp_c = None
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                exp_c = int(mod.in_channels)
                break
        if exp_c is None:
            return x
        in_c = x.size(1)
        out = x
        if exp_c == 3 and in_c == 1:
            out = out.repeat(1, 3, 1, 1)
        elif exp_c == 1 and in_c == 3:
            out = out[:, 0:1, ...]
        h, w = out.size(2), out.size(3)
        # For different models, resample according to their explicitly declared expected input size;
        # if not declared, use empirical size based on channel count (3->32x32, 1->28x28)
        target_size = None
        # Prioritize reading model declared expected_image_size attribute (including wrapped models like Mix models/method-specific models)
        try:
            model_for_query = m
            # Some methods may attach actual backbone to submodules, can expand as needed here
            # Currently directly read top-level expected_image_size
        except Exception:
            model_for_query = m
        if hasattr(model_for_query, "expected_image_size"):
            try:
                sz = getattr(model_for_query, "expected_image_size")
                if isinstance(sz, (tuple, list)) and len(sz) == 2:
                    target_size = (int(sz[0]), int(sz[1]))
            except Exception:
                target_size = None
        if target_size is None:
            if exp_c == 3:
                target_size = (32, 32)
            elif exp_c == 1:
                target_size = (28, 28)
        if target_size is not None and (h != target_size[0] or w != target_size[1]):
            out = F.interpolate(
                out, size=target_size, mode="bilinear", align_corners=False
            )
        return out

    model.eval()
    with torch.no_grad():
        for x, t, y_true, _, _ in loader:
            # For methods providing multiple augmented views (e.g., weak/strong),
            # use the first one during evaluation to keep consistency.
            if isinstance(x, (list, tuple)):
                x = x[0]

            x = x.to(device)
            x = _adapt_input_for_model(model, x)
            t, y_true = t.to(device), y_true.to(device)
            outputs = model(x)

            # (1) Convert model outputs to binary predictions; handle both
            #     probability and logit forms in a unified way.
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                # Multi-class: argmax for predictions; pick class-0 score for risk
                preds_binary = torch.argmax(outputs, dim=1).long()
                eval_scores = outputs[:, 0]
            else:
                raw = outputs.view(-1)
                # If it looks like probabilities (bounded in [0, 1]), threshold at 0.5
                # and convert to logits for risk computation.
                if torch.all(raw >= 0) and torch.all(raw <= 1):
                    preds_binary = (raw >= 0.5).long()
                    eval_scores = torch.logit(torch.clamp(raw, 1e-6, 1 - 1e-6))
                else:
                    # Treat as logits (log-odds)
                    preds_binary = (raw > 0).long()
                    eval_scores = raw

            y_pred_all.extend(preds_binary.cpu().numpy())
            y_true_all.extend(y_true.cpu().numpy())
            y_scores_all.extend(eval_scores.detach().cpu().numpy())

            # (2) Unbiased PU risk (zero-one surrogate) using labeled-positive (t=+1)
            #     and unlabeled (t=-1) partitions.
            pos_mask, unl_mask = (t == 1), (t == -1)
            risk_pos_term = _zero_one_loss(eval_scores[pos_mask]).sum()
            risk_neg_term = _zero_one_loss(-eval_scores[pos_mask]).sum()
            risk_unl_term = _zero_one_loss(-eval_scores[unl_mask]).sum()

            batch_risk = prior * (risk_pos_term - risk_neg_term) + risk_unl_term
            total_risk_sum += batch_risk.item()

    num_samples = len(y_true_all)
    risk = total_risk_sum / max(1, num_samples)

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    y_score_arr = np.array(y_scores_all)

    # Prior-calibrated fallback: if predictions collapse to a single class
    # (e.g., all-positive due to biased logits), recalibrate the decision
    # threshold so that the predicted positive fraction matches the training prior π.
    try:
        unique_preds = np.unique(y_pred_arr)
        if unique_preds.size == 1:
            n = len(y_score_arr)
            k = int(round(float(prior) * float(n)))
            if 0 < k < n:
                sorted_scores = np.sort(y_score_arr)
                # Threshold between the k-th and (k+1)-th from the top (tie-safe)
                thr = (sorted_scores[n - k] + sorted_scores[n - k - 1]) / 2.0
                y_pred_arr = (y_score_arr >= thr).astype(int)
    except Exception:
        pass

    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)

    # AUC (robust to single-class batches)
    try:
        if len(np.unique(y_true_arr)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except Exception:
        auc = float("nan")

    return {
        "error": 1 - acc,
        "risk": risk,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


# ---------------------------------------------------------------------
# Global seeding
# ---------------------------------------------------------------------


def set_global_seed(seed: int):
    """Set global RNG seeds for Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """Worker initialization function for DataLoader to ensure reproducibility.
    
    This function should be passed as `worker_init_fn` to DataLoader when
    num_workers > 0. It sets the random seed for each worker based on the
    base seed and worker_id to ensure different workers have different but
    deterministic random states.
    
    Args:
        worker_id: The ID of the worker process (provided by DataLoader)
    """
    # Get the base seed from torch's initial seed (set by set_global_seed)
    base_seed = torch.initial_seed()
    # Create a unique seed for each worker
    worker_seed = base_seed + worker_id
    # Set seeds for all random number generators in the worker
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))  # numpy seed must be within 32-bit
    torch.manual_seed(worker_seed)


# ---------------------------------------------------------------------
# Dist-PU Mixup utilities
# ---------------------------------------------------------------------


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup to a batch.

    Returns:
        mixed_x:  Mixed inputs.
        y_a:      Original targets (first partner).
        y_b:      Original targets (second partner).
        lam:      Mixing coefficient λ ~ Beta(alpha, alpha).

    Reference:
        H. Zhang et al., "mixup: Beyond empirical risk minimization," ICLR 2018.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    if device:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    scores: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    """Compute Mixup loss as a convex combination of two BCE losses."""
    loss_a = F.binary_cross_entropy(scores, y_a, reduction="mean")
    loss_b = F.binary_cross_entropy(scores, y_b, reduction="mean")
    return lam * loss_a + (1 - lam) * loss_b


# ---------------------------------------------------------------------
# Dist-PU pseudo-labeling utilities
# ---------------------------------------------------------------------


class PseudoLabeler:
    """Generate and maintain pseudo-labels for a dataset indexed by sample ids."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.pseudo_labels = None
        self.sample_indices = None

    def generate_initial_pseudo_labels(self, loader: DataLoader, device: torch.device):
        """Generate initial pseudo-labels for all samples provided by `loader`."""
        print("--- Generating initial pseudo-labels ---")
        self.model.eval()
        all_indices = []
        all_pseudo_labels = []

        with torch.no_grad():
            for x, _, _, indices, _ in tqdm(loader, desc="Pseudo-Labeling"):
                x = x.to(device)
                outputs = self.model(x)
                pseudo_labels = torch.sigmoid(outputs).squeeze().cpu()

                all_indices.append(indices.cpu())
                all_pseudo_labels.append(pseudo_labels)

        all_indices_tensor = torch.cat(all_indices)
        all_pseudo_labels_tensor = torch.cat(all_pseudo_labels)

        sort_indices = torch.argsort(all_indices_tensor)
        self.sample_indices = all_indices_tensor[sort_indices]
        self.pseudo_labels = all_pseudo_labels_tensor[sort_indices]

        assert len(torch.unique(self.sample_indices)) == len(
            self.pseudo_labels
        ), "Mismatch in pseudo-label and sample index count."
        print(f"✓ Generated {len(self.pseudo_labels)} pseudo-labels.")

    def get_pseudo_labels_for_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieve pseudo-labels for a given batch of sample indices."""
        # Ensure CPU indexing, then move back to the model device if needed
        cpu_indices = indices.to("cpu")
        return self.pseudo_labels[cpu_indices].to(self.device)

    def update_pseudo_labels_for_batch(
        self, indices: torch.Tensor, new_scores: torch.Tensor
    ):
        """Update stored pseudo-labels for a subset of indices using new model scores."""
        self.pseudo_labels[indices] = new_scores.detach().cpu()


console = Console()


class ModelCheckpoint:
    """Save the best model during training according to a monitored metric."""

    def __init__(
        self,
        save_dir: str,
        filename: str,
        monitor: str,
        mode: str = "max",
        save_model: bool = True,
        verbose: bool = True,
        file_console: Console | None = None,
        early_stopping_params: dict | None = None,
    ):
        """
        Args:
            save_dir (str): Directory to save the model.
            filename (str): Model filename.
            monitor (str): Metric to monitor, formatted as "phase_metric"
                           (e.g., "test_f1", "train_accuracy").
            mode (str):     "max" or "min".
            save_model (bool): Whether to persist model weights.
            verbose (bool):   Whether to log improvements.
            file_console (Console | None): Rich console to also write logs to a file.
            early_stopping_params (dict | None): Parameters for early stopping.
        """
        self.save_dir = save_dir
        self.filename = filename
        self.save_path = os.path.join(self.save_dir, self.filename)
        self.monitor = monitor
        self.mode = mode
        self.save_model = save_model
        self.verbose = verbose
        self.file_console = file_console

        if self.mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', but got '{mode}'")

        self.best_score = -np.inf if self.mode == "max" else np.inf
        self.best_epoch = -1
        self.best_metrics = None
        self.best_elapsed_seconds: float | None = None

        # Early stopping attributes
        self.early_stopping_enabled = False
        self.patience = float("inf")
        self.min_delta = 0.0
        self.wait = 0
        self.should_stop = False

        if early_stopping_params and early_stopping_params.get("enabled", False):
            self.early_stopping_enabled = True
            self.patience = early_stopping_params.get("patience", 10)
            self.min_delta = early_stopping_params.get("min_delta", 0)
            if self.verbose:
                self._log(
                    f"Early stopping enabled: patience={self.patience}, min_delta={self.min_delta}",
                    "bold blue",
                )

        if self.save_model:
            os.makedirs(self.save_dir, exist_ok=True)

    def _log(self, message: str, style: str = None):
        """Log to stdout and, if provided, to a file-backed Rich Console."""
        if style:
            message = f"[{style}]{message}[/{style}]"
        console.log(message)
        if self.file_console:
            self.file_console.log(message)

    def __call__(
        self,
        epoch: int,
        all_metrics: dict[str, float],
        model: torch.nn.Module,
        elapsed_seconds: float | None = None,
    ):
        """Check after each epoch whether to update 'best' and save the model."""
        current_score = all_metrics.get(self.monitor)
        if current_score is None:
            # Fallback: try test_* or train_* for the same metric suffix
            try:
                key_suffix = (
                    self.monitor.split("_", 1)[1]
                    if "_" in self.monitor
                    else self.monitor
                )
                alt_keys = [f"test_{key_suffix}", f"train_{key_suffix}"]
                for alt in alt_keys:
                    if alt in all_metrics:
                        current_score = all_metrics[alt]
                        break
            except Exception:
                current_score = None
        if current_score is None:
            if not hasattr(self, "_warned"):
                warning_msg = (
                    f"Warning: monitored metric '{self.monitor}' not found in evaluation results. "
                    f"Skipping checkpoint logic. Available keys: {list(all_metrics.keys())}"
                )
                self._log(warning_msg, "bold yellow")
                self._warned = True
            return

        improved = False
        if self.mode == "max":
            if current_score > self.best_score + self.min_delta:
                improved = True
        else:
            if current_score < self.best_score - self.min_delta:
                improved = True

        if improved:
            old_best = self.best_score
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_metrics = all_metrics
            # Track time-to-best if provided
            try:
                self.best_elapsed_seconds = (
                    float(elapsed_seconds) if elapsed_seconds is not None else None
                )
            except Exception:
                self.best_elapsed_seconds = None

            if self.verbose:
                old_best_str = f"{old_best:.4f}" if np.isfinite(old_best) else "N/A"
                message = f"Epoch {epoch}: {self.monitor} improved from {old_best_str} to {current_score:.4f}."
                if self.save_model:
                    message += f" Saving model to {self.save_path}"
                self._log(message, "bold green")

            if self.save_model:
                torch.save(model.state_dict(), self.save_path)

            # Reset wait counter on improvement
            self.wait = 0
        elif self.early_stopping_enabled:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                if self.verbose:
                    self._log(
                        f"Epoch {epoch}: Early stopping triggered after {self.patience} epochs of no improvement on '{self.monitor}'.",
                        "bold red",
                    )

    def log_best_metrics(self):
        """Render a Rich table with the best metrics recorded so far."""
        if self.best_metrics is None:
            warning_msg = "No best metrics recorded. Perhaps the score never improved from initialization."
            self._log(warning_msg, "bold yellow")
            return

        extra = (
            f", time_to_best={self.best_elapsed_seconds:.2f}s"
            if hasattr(self, "best_elapsed_seconds")
            and self.best_elapsed_seconds is not None
            else ""
        )
        table = Table(
            title=f"Best Metrics ({self.monitor} @ Epoch {self.best_epoch}{extra})"
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Train", style="magenta")
        table.add_column("Test", style="green")

        train_metrics = {
            k.replace("train_", ""): v
            for k, v in self.best_metrics.items()
            if k.startswith("train_")
        }
        test_metrics = {
            k.replace("test_", ""): v
            for k, v in self.best_metrics.items()
            if k.startswith("test_")
        }

        metric_keys = sorted(list(set(train_metrics.keys()) | set(test_metrics.keys())))

        for key in metric_keys:
            train_val = train_metrics.get(key)
            test_val = test_metrics.get(key)
            table.add_row(
                key,
                f"{train_val:.4f}" if train_val is not None else "N/A",
                f"{test_val:.4f}" if test_val is not None else "N/A",
            )

        console.print(table)
        if self.file_console:
            self.file_console.print(table)


# ---------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------


def sigmoid_rampup(current, rampup_length):
    """Exponential ramp-up from https://arxiv.org/abs/1610.02242."""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear ramp-up utility."""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length
