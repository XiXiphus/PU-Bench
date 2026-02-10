import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from typing import List


class PUDataset(Dataset):
    """
    Positive-Unlabeled (PU) Dataset wrapper.

    Each sample contains five components:
    1. features:      Input features.
    2. pu_labels:     PU labels (labeled positive as +1, unlabeled as -1).
    3. true_labels:   True binary labels (positive as 1, negative as 0) for evaluation only.
    4. indices:       Original indices of samples.
    5. pseudo_labels: Pseudo labels or confidence scores.
    """

    def __init__(
        self,
        features: Union[np.ndarray, torch.Tensor],
        pu_labels: Union[np.ndarray, torch.Tensor],
        true_labels: Union[np.ndarray, torch.Tensor],
        indices: Union[np.ndarray, torch.Tensor] = None,
        pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    ):
        # Preserve dtype of features (float for images, long for token ids, etc.)
        self.features = (
            torch.from_numpy(features) if isinstance(features, np.ndarray) else features
        )
        self.pu_labels = (
            torch.from_numpy(pu_labels).long()
            if isinstance(pu_labels, np.ndarray)
            else pu_labels
        )
        self.true_labels = (
            torch.from_numpy(true_labels).long()
            if isinstance(true_labels, np.ndarray)
            else true_labels
        )

        if indices is None:
            self.indices = torch.arange(len(self.features))
        else:
            self.indices = (
                torch.from_numpy(indices).long()
                if isinstance(indices, np.ndarray)
                else indices
            )

        if pseudo_labels is None:
            self.pseudo_labels = torch.zeros(len(self.features))
        else:
            self.pseudo_labels = (
                torch.from_numpy(pseudo_labels).float()
                if isinstance(pseudo_labels, np.ndarray)
                else pseudo_labels
            )

        assert (
            len(self.features)
            == len(self.pu_labels)
            == len(self.true_labels)
            == len(self.indices)
            == len(self.pseudo_labels)
        ), "All data tensors must have the same length"

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.pu_labels[idx],
            self.true_labels[idx],
            self.indices[idx],
            self.pseudo_labels[idx],
        )

    def get_subset(self, subset_indices):
        mask = np.isin(self.indices.numpy(), subset_indices)
        return PUDataset(
            self.features[mask],
            self.pu_labels[mask],
            self.true_labels[mask],
            indices=self.indices[mask],
            pseudo_labels=self.pseudo_labels[mask],
        )


class PUDataloader(DataLoader):
    """
    DataLoader class for Positive-Unlabeled Learning.
    Extends PyTorch's DataLoader with specific functionality for PU learning.
    """

    def __init__(
        self,
        dataset: PUDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(
            dataset, batch_size, shuffle, num_workers=num_workers, **kwargs
        )


def compute_pn_scores(
    features: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
) -> np.ndarray:
    """Train a logistic regression classifier and return P(y=1|x) scores."""
    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", random_state=42)
    clf.fit(features, labels)
    probs = clf.predict_proba(features)[:, 1]
    return probs


def split_train_val(
    features: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split original training set into training and validation sets."""
    if val_ratio == 0.0:
        return (
            features,
            labels,
            np.empty((0, *features.shape[1:]), dtype=features.dtype),
            np.empty(0, dtype=labels.dtype),
        )

    train_f, val_f, train_y, val_y = train_test_split(
        features,
        labels,
        test_size=val_ratio,
        stratify=labels,
        random_state=random_state,
    )
    return train_f, train_y, val_f, val_y


def create_pu_training_set(
    features: np.ndarray,
    labels: np.ndarray,
    n_labeled: int = None,
    labeled_ratio: float = None,
    selection_strategy: str = "random",
    scenario: str = "single",
    with_replacement: bool = True,
    case_control_mode: str = "naive_mode",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate PU training data based on given scenario and strategy."""
    assert scenario in [
        "single",
        "case-control",
    ], "Scenario must be 'single' or 'case-control'"

    pos_indices = np.where(labels == 1)[0]
    n_pos = len(pos_indices)

    if n_labeled is None and labeled_ratio is None:
        raise ValueError("Must provide either n_labeled or labeled_ratio")

    # Deprecated alias 'pusb' is no longer supported; please use 'sar_pusb' explicitly in configs

    # For SAR strategies, pre-compute PN scores once
    pn_probs = None
    if selection_strategy in ["sar_pusb", "sar_lbeA", "sar_lbeB"]:
        flat_features = features.reshape(features.shape[0], -1)
        pn_probs = compute_pn_scores(flat_features, labels)

    if n_labeled is None:
        n_labeled = int(n_pos * labeled_ratio)
    n_labeled = min(n_labeled, n_pos)

    if selection_strategy == "random":
        labeled_pos_idx = (
            np.random.choice(pos_indices, size=n_labeled, replace=False)
            if n_labeled > 0
            else np.array([], dtype=int)
        )
    elif selection_strategy == "all":
        n_labeled = n_pos
        labeled_pos_idx = pos_indices
    elif selection_strategy == "sar_pusb":
        scores = pn_probs[pos_indices]
        scores = scores / scores.mean()
        scores = scores**20
        scores = scores / scores.max()
        ranked = pos_indices[np.argsort(scores)[::-1]]
        labeled_pos_idx = ranked[:n_labeled]
    elif selection_strategy in ["sar_lbeA", "sar_lbeB"]:
        k = 10  # As specified in the LBE paper
        # ShrinkCoef from the original implementation's syn function
        shrink_coef = 1.0
        scores = pn_probs[pos_indices]

        if selection_strategy == "sar_lbeA":
            # Favors high-posterior positives, original formula: p = (scores)^k
            weights = scores**k
        else:  # sar_lbeB
            # Favors boundary/ambiguous positives, original formula: p = (1.5 + ShrinkCoef - scores)^k
            weights = (1.5 + shrink_coef - scores) ** k
            # Ensure weights are non-negative, as scores can be close to 1
            weights = np.maximum(weights, 0)

        # Normalize weights to form a probability distribution
        sum_weights = weights.sum()
        if sum_weights > 0:
            p = weights / sum_weights
        else:
            # Fallback to uniform if all weights are zero
            p = np.full(len(pos_indices), 1.0 / len(pos_indices))

        # Apply smoothing for strategy 1, as in the original `mySampling` function
        if selection_strategy == "sar_lbeA":
            uniform_p = np.full(len(pos_indices), 1.0 / len(pos_indices))
            p = 0.9 * p + 0.1 * uniform_p
            # Re-normalize just in case of floating point inaccuracies
            p /= p.sum()

        labeled_pos_idx = np.random.choice(
            pos_indices, size=n_labeled, replace=False, p=p
        )
    else:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

    labeled_mask = np.zeros(labels.shape, dtype=int)
    labeled_mask[labeled_pos_idx] = 1

    if scenario == "single":
        # SS: Keep original training set size, only select labeled L from positive class, rest as U
        return features, labels, labeled_mask
    else:  # case-control
        # Two modes:
        # - naive_mode: Following nnPUlearning approach: U sampled from overall, positive class count n_up,
        #               prioritizing unselected positive samples for L, reusing L positive samples when insufficient;
        #               default |U| = |X| (also supports |P|+|U|=|X| semantics).
        # - story_mode: Maintain current implementation (calculate size by c and π, sample U from overall).

        mode = str(case_control_mode or "naive_mode").lower()
        if mode not in ["naive_mode", "story_mode"]:
            raise ValueError(
                f"Unknown case_control_mode: {case_control_mode}. Use 'naive_mode' or 'story_mode'."
            )

        if mode == "story_mode":
            # Original implementation: use c and π to control size, sample U from overall
            if labeled_ratio is None:
                raise ValueError(
                    "For 'case-control' scenario, 'labeled_ratio' (label frequency c) must be provided."
                )

            c = float(labeled_ratio)
            if c >= 1.0:
                fallback_mask = np.zeros(labels.shape, dtype=int)
                fallback_mask[pos_indices] = 1
                print(
                    "[PU] case-control with c=1 detected → fallback to single+all (keep original set; all positives labeled)."
                )
                return features, labels, fallback_mask

            n = len(labels)
            pi = labels.mean() if n > 0 else 0.0
            A = 1.0 / (1.0 - c + c * pi)
            P_num = int(np.ceil(A * c * (pi * n)))
            U_num = int(np.ceil(A * (1.0 - c) * n))

            if selection_strategy == "random":
                labeled_pos_idx_cc = (
                    np.random.choice(pos_indices, size=P_num, replace=with_replacement)
                    if P_num > 0 and len(pos_indices) > 0
                    else np.array([], dtype=int)
                )
            elif selection_strategy == "sar_pusb":
                flat_features = features.reshape(features.shape[0], -1)
                pn_probs = compute_pn_scores(flat_features, labels)
                scores = pn_probs[pos_indices]
                scores = scores / scores.mean()
                scores = scores**20
                scores = scores / scores.max()
                ranked = pos_indices[np.argsort(scores)[::-1]]
                if len(ranked) >= P_num:
                    labeled_pos_idx_cc = ranked[:P_num]
                else:
                    extra = (
                        np.random.choice(
                            pos_indices, size=P_num - len(ranked), replace=True
                        )
                        if len(pos_indices) > 0 and P_num - len(ranked) > 0
                        else np.array([], dtype=int)
                    )
                    labeled_pos_idx_cc = np.concatenate([ranked, extra])
            elif selection_strategy in ["sar_lbeA", "sar_lbeB"]:
                k = 10
                shrink_coef = 1.0
                scores = pn_probs[pos_indices]

                if selection_strategy == "sar_lbeA":
                    weights = scores**k
                else:  # sar_lbeB
                    weights = (1.5 + shrink_coef - scores) ** k
                    weights = np.maximum(weights, 0)

                sum_weights = weights.sum()
                if sum_weights > 0:
                    p = weights / sum_weights
                else:
                    p = np.full(len(pos_indices), 1.0 / len(pos_indices))

                if selection_strategy == "sar_lbeA":
                    uniform_p = np.full(len(pos_indices), 1.0 / len(pos_indices))
                    p = 0.9 * p + 0.1 * uniform_p
                    p /= p.sum()

                labeled_pos_idx_cc = np.random.choice(
                    pos_indices, size=P_num, replace=with_replacement, p=p
                )
            elif selection_strategy == "all":
                if len(pos_indices) >= P_num:
                    labeled_pos_idx_cc = pos_indices[:P_num]
                else:
                    extra = (
                        np.random.choice(
                            pos_indices, size=P_num - len(pos_indices), replace=True
                        )
                        if len(pos_indices) > 0 and P_num - len(pos_indices) > 0
                        else np.array([], dtype=int)
                    )
                    labeled_pos_idx_cc = np.concatenate([pos_indices, extra])
            else:
                raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

            all_idx = np.arange(n)
            unlabeled_idx_cc = (
                np.random.choice(all_idx, size=U_num, replace=with_replacement)
                if U_num > 0 and n > 0
                else np.array([], dtype=int)
            )

            new_features = np.concatenate(
                (features[labeled_pos_idx_cc], features[unlabeled_idx_cc]), axis=0
            )
            new_labels = np.concatenate(
                (labels[labeled_pos_idx_cc], labels[unlabeled_idx_cc]), axis=0
            )
            new_labeled_mask = np.concatenate(
                (np.ones(P_num, dtype=int), np.zeros(U_num, dtype=int)), axis=0
            )

            return new_features, new_labels, new_labeled_mask

        # naive_mode
        # Use previously selected labeled_pos_idx from positive class as L;
        # Construct U: take n_up from positive class (prioritize unselected positive samples for L, reuse L when insufficient),
        # then concatenate with all negative samples; default |U| = |X| (corresponding to n_up = n_p).
        n = len(labels)
        pos_mask = labels == 1
        neg_mask = labels == 0
        n_p = int(pos_mask.sum())
        n_lp = int(len(labeled_pos_idx))

        # Adopt common setting |U| = |X| → n_up = n_p
        U_num = n
        n_up = n_p

        # To support |P|+|U|=|X|, can be achieved by setting U_num to n - n_lp:
        # Reserve interface extension: when user explicitly passes labeled_ratio==1 and scenario=case-control, fallback to single+all
        if labeled_ratio is not None and float(labeled_ratio) >= 1.0:
            fallback_mask = np.zeros(labels.shape, dtype=int)
            fallback_mask[pos_indices] = 1
            print(
                "[PU] case-control naive_mode with c=1 detected → fallback to single+all (keep original set; all positives labeled)."
            )
            return features, labels, fallback_mask

        # Calculate positive sample indices for U: first take unlabeled positive samples, reuse labeled positive samples when insufficient
        pos_rest_idx = np.setdiff1d(pos_indices, labeled_pos_idx, assume_unique=False)
        # Concatenate L once, sufficient to cover upper bound n_up ≤ n_p
        pos_for_unlabeled = np.concatenate([pos_rest_idx, labeled_pos_idx], axis=0)[
            :n_up
        ]

        neg_indices = np.where(neg_mask)[0]
        # In |U|=|X| case, all negative samples enter U
        unlabeled_idx_naive = np.concatenate([pos_for_unlabeled, neg_indices], axis=0)

        new_features = np.concatenate(
            (features[labeled_pos_idx], features[unlabeled_idx_naive]), axis=0
        )
        new_labels = np.concatenate(
            (labels[labeled_pos_idx], labels[unlabeled_idx_naive]), axis=0
        )
        new_labeled_mask = np.concatenate(
            (np.ones(n_lp, dtype=int), np.zeros(len(unlabeled_idx_naive), dtype=int)),
            axis=0,
        )

        return new_features, new_labels, new_labeled_mask


def print_dataset_statistics(
    train_dataset: PUDataset,
    val_dataset: PUDataset,
    test_dataset: PUDataset,
    train_labeled_mask: np.ndarray,
    positive_classes: List[int],
    negative_classes: List[int],
    true_positive_label: int,
    true_negative_label: int,
    pu_labeled_label: int,
    pu_unlabeled_label: int,
    val_ratio: float,
    log_file: str | None = None,
    also_print: bool = False,
):
    """Write PU dataset statistics to a log file (and optionally print).

    If log_file is provided, statistics will be appended to that file.
    Set also_print=True to echo the same content to stdout.
    """
    lines: list[str] = []
    lines.append("--- PU Dataset Statistics ---")

    lines.append("Class to Binary Label Mapping:")
    lines.append(
        f"  - Positive Classes {positive_classes} -> {true_positive_label} (Positive)"
    )
    lines.append(
        f"  - Negative Classes {negative_classes} -> {true_negative_label} (Negative)"
    )
    lines.append("")

    lines.append("PU Label Mapping:")
    lines.append(f"  - Labeled (L) -> {pu_labeled_label}")
    lines.append(f"  - Unlabeled (U) -> {pu_unlabeled_label}")
    lines.append("")

    # Training set statistics
    train_stats = [
        [
            len(train_dataset),
            int(sum(train_labeled_mask == 1)),
            int(sum(train_labeled_mask == 0)),
            int(sum(train_dataset.true_labels[train_labeled_mask == 0] == 1)),
            int(sum(train_dataset.true_labels[train_labeled_mask == 0] == 0)),
        ]
    ]
    train_headers = [
        "Total",
        "Labeled (L)",
        "Unlabeled (U)",
        "Positives in U",
        "Negatives in U",
    ]
    lines.append("Training Set Statistics:")
    lines.append(tabulate(train_stats, headers=train_headers, tablefmt="grid"))
    lines.append("")

    # Training set ratio statistics
    num_labeled = int(sum(train_labeled_mask == 1))
    pos_in_unlabeled = int(sum(train_dataset.true_labels[train_labeled_mask == 0] == 1))
    total_pos_in_train = num_labeled + pos_in_unlabeled
    total_train = len(train_dataset)
    ratio_p_in_train = total_pos_in_train / total_train if total_train > 0 else 0
    ratio_lp_in_p_train = (
        num_labeled / total_pos_in_train if total_pos_in_train > 0 else 0
    )

    train_ratios_stats = [
        [
            f"{total_pos_in_train}",
            f"{ratio_p_in_train:.2%}",
            f"{ratio_lp_in_p_train:.2%}",
        ]
    ]
    train_ratios_headers = ["Total Positives", "Prior (P/Total)", "LP/P Ratio"]
    lines.append("Training Set Ratios:")
    lines.append(
        tabulate(train_ratios_stats, headers=train_ratios_headers, tablefmt="grid")
    )
    lines.append("")

    # Validation and Test set statistics
    other_sets_stats = []
    if val_ratio > 0 and len(val_dataset) > 0:
        total_val = len(val_dataset)
        pos_val = int(sum(val_dataset.true_labels == 1))
        ratio_p_in_val = pos_val / total_val if total_val > 0 else 0
        other_sets_stats.append(
            [
                "Validation",
                total_val,
                pos_val,
                int(sum(val_dataset.true_labels == 0)),
                f"{ratio_p_in_val:.2%}",
            ]
        )

    total_test = len(test_dataset)
    pos_test = int(sum(test_dataset.true_labels == 1))
    ratio_p_in_test = pos_test / total_test if total_test > 0 else 0
    other_sets_stats.append(
        [
            "Test",
            total_test,
            pos_test,
            int(sum(test_dataset.true_labels == 0)),
            f"{ratio_p_in_test:.2%}",
        ]
    )
    other_headers = ["Set", "Total", "Positives", "Negatives", "Positive Ratio"]
    lines.append("Validation & Test Set Statistics:")
    lines.append(tabulate(other_sets_stats, headers=other_headers, tablefmt="grid"))

    lines.append("------------------------------------")

    text = "\n".join(lines) + "\n"
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        except Exception:
            pass
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text)
    if also_print or not log_file:
        print(text, end="")


def resample_by_prevalence(features, labels, target_prevalence, random_seed=42):
    """
    Resample dataset by downsampling the majority class to meet target positive prevalence.

    Args:
        features (np.ndarray): Feature data.
        labels (np.ndarray): Label data (0 for negative, 1 for positive).
        target_prevalence (float): Target positive prevalence.
        random_seed (int): Random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled features and labels.
    """
    if not (0 < target_prevalence < 1):
        raise ValueError(
            f"Target prevalence must be in (0, 1), got {target_prevalence}"
        )

    np.random.seed(random_seed)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_features, pos_labels = features[pos_mask], labels[pos_mask]
    neg_features, neg_labels = features[neg_mask], labels[neg_mask]

    n_pos_avail, n_neg_avail = len(pos_features), len(neg_features)

    if n_pos_avail == 0 or n_neg_avail == 0:
        raise ValueError("Cannot resample when one class has no samples.")

    original_prevalence = n_pos_avail / (n_pos_avail + n_neg_avail)

    if target_prevalence > original_prevalence:
        n_pos_new = n_pos_avail
        n_neg_new = int(
            np.floor(n_pos_new * (1 - target_prevalence) / target_prevalence)
        )
        if n_neg_new < 0:
            raise ValueError("Calculated negative sample count cannot be negative.")

        indices = np.random.choice(n_neg_avail, n_neg_new, replace=False)
        pos_features_new, pos_labels_new = pos_features, pos_labels
        neg_features_new, neg_labels_new = neg_features[indices], neg_labels[indices]

    elif target_prevalence < original_prevalence:
        n_neg_new = n_neg_avail
        n_pos_new = int(
            np.floor(n_neg_new * target_prevalence / (1 - target_prevalence))
        )
        if n_pos_new < 0:
            raise ValueError("Calculated positive sample count cannot be negative.")

        indices = np.random.choice(n_pos_avail, n_pos_new, replace=False)
        neg_features_new, neg_labels_new = neg_features, neg_labels
        pos_features_new, pos_labels_new = pos_features[indices], pos_labels[indices]
    else:
        return features, labels

    new_features = np.concatenate([pos_features_new, neg_features_new], axis=0)
    new_labels = np.concatenate([pos_labels_new, neg_labels_new], axis=0)

    shuffle_indices = np.random.permutation(len(new_features))
    return new_features[shuffle_indices], new_labels[shuffle_indices]
