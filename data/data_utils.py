import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tabulate import tabulate


CASE_CONTROL_NNPU_ALIASES = {"naive_mode", "nnpu_full_u", "nnpu_full_u_mode"}
CASE_CONTROL_STORY_ALIASES = {"story_mode", "story_equal_n", "story_equal_total_n"}


# Sampling semantics used here are intentionally source-anchored:
# - nnPU original implementation:
#   https://github.com/kiryor/nnPUlearning/blob/master/dataset.py
# - Mielniczuk and Wawrzenczyk (2024), single-sample vs case-control:
#   https://arxiv.org/abs/2312.02095
PU_SAMPLING_REFERENCES = {
    "single": (
        "single-sample/one-sample PU: one iid sample of (X,S); under SCAR, "
        "c=P(S=1|Y=1) and U is the S=0 remainder."
    ),
    "nnpu_full_u": (
        "nnPU full-U case-control convention: L is sampled from positives and "
        "U is the full population mixture; L and U may overlap."
    ),
    "story_equal_n": (
        "Mielniczuk-Wawrzenczyk 2024 story-mode case-control construction: "
        "sample sizes are scaled by A=1/(1-c+c*pi) to preserve total n."
    ),
}


def canonical_case_control_mode(case_control_mode: str | None) -> str:
    """Normalize legacy case-control mode names into semantic names."""
    mode = str(case_control_mode or "nnpu_full_u").lower()
    if mode in CASE_CONTROL_NNPU_ALIASES:
        return "nnpu_full_u"
    if mode in CASE_CONTROL_STORY_ALIASES:
        return "story_equal_n"
    raise ValueError(
        "Unknown case_control_mode: "
        f"{case_control_mode}. Use 'nnpu_full_u'/'naive_mode' or "
        "'story_equal_n'/'story_mode'."
    )


def _safe_ratio(num: int | float, den: int | float) -> float | None:
    return (float(num) / float(den)) if den else None


def _as_numpy(values: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def make_rng(
    random_seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.random.Generator:
    """Create or reuse a local RNG.

    Dataset construction should not mutate NumPy's global RNG state. Keeping all
    draws behind a local Generator makes PU splits reproducible and auditable.
    """
    return rng if rng is not None else np.random.default_rng(random_seed)


def _choice(
    rng: np.random.Generator,
    values: np.ndarray,
    size: int,
    replace: bool = False,
    p: np.ndarray | None = None,
) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=int)
    if len(values) == 0:
        return np.array([], dtype=int)
    return np.asarray(rng.choice(values, size=size, replace=replace, p=p), dtype=int)


def _select_sar_pusb_positives(
    rng: np.random.Generator,
    pos_indices: np.ndarray,
    pn_probs: np.ndarray,
    n_selected: int,
) -> np.ndarray:
    """Select positives with MasaKat0/PUlearning's nnPUSB accept-reject rule.

    Source path:
        `BiasedPUlearning/nnPUSB/main_nnPUSB_mnist.py`

    The source computes classifier probabilities on positive examples, divides
    by their mean, divides by their maximum, accepts each positive when
    `prob > Uniform(0, 1)`, then shuffles and truncates to the requested count.
    It does not rank top-k positives and does not apply the `**20` exponent used
    in the older linear PUSB experiment script.
    """
    if n_selected <= 0 or len(pos_indices) == 0:
        return np.array([], dtype=int)

    scores = np.asarray(pn_probs[pos_indices], dtype=float)
    mean_score = float(scores.mean()) if len(scores) else 0.0
    if not np.isfinite(mean_score) or mean_score <= 0.0:
        scores = np.ones_like(scores, dtype=float)
    else:
        scores = scores / mean_score

    max_score = float(scores.max()) if len(scores) else 0.0
    if not np.isfinite(max_score) or max_score <= 0.0:
        accept_prob = np.ones_like(scores, dtype=float)
    else:
        accept_prob = scores / max_score
    accept_prob = np.clip(accept_prob, 0.0, 1.0)

    accepted = pos_indices[accept_prob > rng.random(len(pos_indices))]
    accepted = rng.permutation(accepted)
    return np.asarray(accepted[:n_selected], dtype=int)


def summarize_pu_split(
    true_labels: Union[np.ndarray, torch.Tensor],
    pu_labels: Union[np.ndarray, torch.Tensor],
    positive_label: int = 1,
    negative_label: int = 0,
    labeled_label: int = 1,
    unlabeled_label: int = -1,
) -> Dict[str, Any]:
    """Compute the PU split statistics that define the empirical risk prior.

    In nnPU-style risk decompositions, ``prior`` means the positive fraction in
    the unlabeled mixture U, not the positive fraction after concatenating L and
    U.  This follows the original nnPU implementation, which returns
    ``n_up / n_u`` as the class prior.
    """
    y = _as_numpy(true_labels)
    s = _as_numpy(pu_labels)

    labeled = s == labeled_label
    unlabeled = s == unlabeled_label
    positive = y == positive_label
    negative = y == negative_label

    n_total = int(len(y))
    n_labeled = int(labeled.sum())
    n_unlabeled = int(unlabeled.sum())
    n_pos_total = int(positive.sum())
    n_neg_total = int(negative.sum())
    n_pos_unlabeled = int((positive & unlabeled).sum())
    n_neg_unlabeled = int((negative & unlabeled).sum())
    n_pos_labeled = int((positive & labeled).sum())

    return {
        "n_total": n_total,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "n_pos_total": n_pos_total,
        "n_neg_total": n_neg_total,
        "n_pos_labeled": n_pos_labeled,
        "n_pos_unlabeled": n_pos_unlabeled,
        "n_neg_unlabeled": n_neg_unlabeled,
        "pi_constructed_train": _safe_ratio(n_pos_total, n_total),
        "pi_unlabeled": _safe_ratio(n_pos_unlabeled, n_unlabeled),
        "c_realized": _safe_ratio(n_pos_labeled, n_pos_total),
    }


def get_pu_risk_prior(dataset: Dataset) -> float:
    """Return the empirical PU risk prior pi_U from a dataset.

    The preferred value is ``dataset.pu_metadata['pi_unlabeled']``.  Falling
    back to constructed positive prevalence is intentionally last-resort because
    it is wrong for nnPU full-U case-control data, where L positives are also
    present inside U.
    """
    metadata = getattr(dataset, "pu_metadata", {}) or {}
    pi_u = metadata.get("pi_unlabeled")
    if pi_u is not None:
        return float(pi_u)

    if hasattr(dataset, "true_labels") and hasattr(dataset, "pu_labels"):
        observed = summarize_pu_split(dataset.true_labels, dataset.pu_labels)
        pi_u = observed.get("pi_unlabeled")
        if pi_u is not None:
            return float(pi_u)

    true_labels = getattr(dataset, "true_labels", None)
    if true_labels is None:
        raise ValueError("Cannot infer PU risk prior: dataset has no true_labels.")
    y = _as_numpy(true_labels)
    return float((y == 1).mean())


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
        metadata: Dict[str, Any] | None = None,
        source_indices: Union[np.ndarray, torch.Tensor] = None,
        source_roles: Union[np.ndarray, List[str]] = None,
    ):
        metadata_dict = dict(metadata or {})
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

        if source_indices is None:
            self.source_indices = self.indices.clone()
        else:
            self.source_indices = (
                torch.from_numpy(source_indices).long()
                if isinstance(source_indices, np.ndarray)
                else (
                    source_indices.long()
                    if isinstance(source_indices, torch.Tensor)
                    else torch.as_tensor(source_indices).long()
                )
            )

        if source_roles is None:
            self.source_roles = np.where(
                _as_numpy(self.pu_labels)
                == metadata_dict.get("pu_labeled_label", 1),
                "L",
                "U",
            ).astype("<U1")
        else:
            self.source_roles = np.asarray(source_roles).astype("<U1")

        assert len(self.source_indices) == len(self.source_roles) == len(
            self.features
        ), "Source audit fields must have the same length as features"

        self.pu_metadata = metadata_dict
        self.pu_metadata.update(
            summarize_pu_split(
                self.true_labels,
                self.pu_labels,
                positive_label=self.pu_metadata.get("true_positive_label", 1),
                negative_label=self.pu_metadata.get("true_negative_label", 0),
                labeled_label=self.pu_metadata.get("pu_labeled_label", 1),
                unlabeled_label=self.pu_metadata.get("pu_unlabeled_label", -1),
            )
        )
        self.metadata = self.pu_metadata

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
            metadata=self.pu_metadata,
            source_indices=self.source_indices[mask],
            source_roles=self.source_roles[mask],
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
    random_state: int | None = 42,
) -> np.ndarray:
    """Train a logistic regression classifier and return P(y=1|x) scores."""
    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", random_state=random_state)
    clf.fit(features, labels)
    probs = clf.predict_proba(features)[:, 1]
    return probs


def create_pu_training_set(
    features: np.ndarray,
    labels: np.ndarray,
    n_labeled: int = None,
    labeled_ratio: float = None,
    selection_strategy: str = "random",
    scenario: str = "single",
    with_replacement: bool = True,
    case_control_mode: str = "naive_mode",
    random_seed: int | None = None,
    rng: np.random.Generator | None = None,
    source_indices: np.ndarray | None = None,
    return_indices: bool = False,
    return_roles: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a PU training sample with explicit sampling semantics.

    Terminology:
        - ``single`` is the one-sample setting: a single PN sample is observed,
          positives are labeled with frequency ``c``, and U is the unlabeled
          remainder ``S=0``.
        - case-control ``nnpu_full_u`` (legacy ``naive_mode``) follows the
          original nnPU code path where U is the full population mixture and may
          overlap L.
        - case-control ``story_equal_n`` (legacy ``story_mode``) follows the
          sample-size convention in Mielniczuk-Wawrzenczyk (2024): L and U are
          scaled by ``A=1/(1-c+c*pi)`` so the generated PU sample has roughly the
          original total size.

    The class prior used by PU risks is not returned here. It is derived later
    as the positive fraction inside U via ``PUDataset.pu_metadata['pi_unlabeled']``.
    """
    assert scenario in [
        "single",
        "case-control",
    ], "Scenario must be 'single' or 'case-control'"

    rng = make_rng(random_seed=random_seed, rng=rng)
    source_indices = (
        np.arange(len(labels), dtype=int)
        if source_indices is None
        else np.asarray(source_indices, dtype=int)
    )
    if len(source_indices) != len(labels):
        raise ValueError("source_indices must have the same length as labels.")

    pos_indices = np.where(labels == 1)[0]
    n_pos = len(pos_indices)

    if n_labeled is None and labeled_ratio is None:
        raise ValueError("Must provide either n_labeled or labeled_ratio")

    # Deprecated alias 'pusb' is no longer supported; please use 'sar_pusb' explicitly in configs

    # For SAR strategies, pre-compute PN scores once
    pn_probs = None
    if selection_strategy in ["sar_pusb", "sar_lbeA", "sar_lbeB"]:
        flat_features = features.reshape(features.shape[0], -1)
        pn_probs = compute_pn_scores(flat_features, labels, random_state=random_seed)

    if n_labeled is None:
        n_labeled = int(n_pos * labeled_ratio)
    n_labeled = min(n_labeled, n_pos)

    if selection_strategy == "random":
        labeled_pos_idx = (
            _choice(rng, pos_indices, size=n_labeled, replace=False)
            if n_labeled > 0
            else np.array([], dtype=int)
        )
    elif selection_strategy == "all":
        n_labeled = n_pos
        labeled_pos_idx = pos_indices
    elif selection_strategy == "sar_pusb":
        labeled_pos_idx = _select_sar_pusb_positives(
            rng,
            pos_indices,
            pn_probs,
            n_labeled,
        )
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

        labeled_pos_idx = _choice(rng, pos_indices, size=n_labeled, replace=False, p=p)
    else:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

    labeled_mask = np.zeros(labels.shape, dtype=int)
    labeled_mask[labeled_pos_idx] = 1

    def _pack_result(
        out_features: np.ndarray,
        out_labels: np.ndarray,
        out_labeled_mask: np.ndarray,
        out_source_indices: np.ndarray,
        out_roles: np.ndarray,
    ):
        result: tuple[Any, ...] = (out_features, out_labels, out_labeled_mask)
        if return_indices:
            result = (*result, out_source_indices)
        if return_roles:
            result = (*result, out_roles)
        return result

    def _pack_permuted_result(
        out_features: np.ndarray,
        out_labels: np.ndarray,
        out_labeled_mask: np.ndarray,
        out_source_indices: np.ndarray,
        out_roles: np.ndarray,
    ):
        perm = rng.permutation(len(out_labels))
        return _pack_result(
            out_features[perm],
            out_labels[perm],
            out_labeled_mask[perm],
            out_source_indices[perm],
            out_roles[perm],
        )

    if scenario == "single":
        # One-sample PU: keep the original sample; selected positives are L and
        # every other point is U. Under SCAR, c=P(S=1|Y=1); under SAR, e(x)
        # varies through selection_strategy.
        roles = np.where(labeled_mask == 1, "L", "U").astype("<U1")
        return _pack_result(features, labels, labeled_mask, source_indices, roles)
    else:  # case-control
        mode = canonical_case_control_mode(case_control_mode)

        if mode == "story_equal_n":
            # Story-mode case-control: use c and population pi to choose L/U
            # sizes; U is sampled from the overall population mixture.
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
                roles = np.where(fallback_mask == 1, "L", "U").astype("<U1")
                return _pack_result(
                    features, labels, fallback_mask, source_indices, roles
                )

            n = len(labels)
            pi = labels.mean() if n > 0 else 0.0
            A = 1.0 / (1.0 - c + c * pi)
            P_num = int(np.ceil(A * c * (pi * n)))
            U_num = int(np.ceil(A * (1.0 - c) * n))

            if selection_strategy == "random":
                labeled_pos_idx_cc = (
                    _choice(rng, pos_indices, size=P_num, replace=with_replacement)
                    if P_num > 0 and len(pos_indices) > 0
                    else np.array([], dtype=int)
                )
            elif selection_strategy == "sar_pusb":
                flat_features = features.reshape(features.shape[0], -1)
                pn_probs = compute_pn_scores(flat_features, labels, random_state=random_seed)
                labeled_pos_idx_cc = _select_sar_pusb_positives(
                    rng,
                    pos_indices,
                    pn_probs,
                    P_num,
                )
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

                labeled_pos_idx_cc = _choice(
                    rng, pos_indices, size=P_num, replace=with_replacement, p=p
                )
            elif selection_strategy == "all":
                if len(pos_indices) >= P_num:
                    labeled_pos_idx_cc = pos_indices[:P_num]
                else:
                    extra = (
                        _choice(
                            rng,
                            pos_indices,
                            size=P_num - len(pos_indices),
                            replace=True,
                        )
                        if len(pos_indices) > 0 and P_num - len(pos_indices) > 0
                        else np.array([], dtype=int)
                    )
                    labeled_pos_idx_cc = np.concatenate([pos_indices, extra])
            else:
                raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

            all_idx = np.arange(n)
            unlabeled_idx_cc = (
                _choice(rng, all_idx, size=U_num, replace=with_replacement)
                if U_num > 0 and n > 0
                else np.array([], dtype=int)
            )

            new_features = np.concatenate(
                (features[labeled_pos_idx_cc], features[unlabeled_idx_cc]), axis=0
            )
            new_labels = np.concatenate(
                (labels[labeled_pos_idx_cc], labels[unlabeled_idx_cc]), axis=0
            )
            P_actual = len(labeled_pos_idx_cc)
            new_labeled_mask = np.concatenate(
                (np.ones(P_actual, dtype=int), np.zeros(U_num, dtype=int)), axis=0
            )
            new_source_indices = np.concatenate(
                (source_indices[labeled_pos_idx_cc], source_indices[unlabeled_idx_cc]),
                axis=0,
            )
            new_roles = np.concatenate(
                (
                    np.full(P_actual, "L", dtype="<U1"),
                    np.full(U_num, "U", dtype="<U1"),
                ),
                axis=0,
            )

            return _pack_permuted_result(
                new_features,
                new_labels,
                new_labeled_mask,
                new_source_indices,
                new_roles,
            )

        # nnPU full-U case-control: use previously selected positives as L; U is
        # the full population mixture, so selected positives may also appear in
        # U. This matches the original nnPU branch with n_unlabeled == len(x).
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
            roles = np.where(fallback_mask == 1, "L", "U").astype("<U1")
            return _pack_result(features, labels, fallback_mask, source_indices, roles)

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
        new_source_indices = np.concatenate(
            (source_indices[labeled_pos_idx], source_indices[unlabeled_idx_naive]),
            axis=0,
        )
        new_roles = np.concatenate(
            (
                np.full(n_lp, "L", dtype="<U1"),
                np.full(len(unlabeled_idx_naive), "U", dtype="<U1"),
            ),
            axis=0,
        )

        return _pack_permuted_result(
            new_features, new_labels, new_labeled_mask, new_source_indices, new_roles
        )


def split_source_train_val(
    labels: np.ndarray,
    val_ratio: float,
    random_seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split source rows before PU sampling.

    Splitting before PU construction prevents repeated case-control draws of the
    same source row from crossing train/validation boundaries.
    """
    n = len(labels)
    all_indices = np.arange(n, dtype=int)
    if val_ratio <= 0.0 or n == 0:
        return all_indices, np.array([], dtype=int)

    n_val = int(round(n * float(val_ratio)))
    if n_val <= 0:
        return all_indices, np.array([], dtype=int)
    if n_val >= n:
        raise ValueError("val_ratio leaves no source rows for training.")

    labels_arr = np.asarray(labels)
    unique, counts = np.unique(labels_arr, return_counts=True)
    can_stratify = len(unique) > 1 and np.all(counts >= 2) and n_val >= len(unique)
    stratify = labels_arr if can_stratify else None

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=n_val,
        stratify=stratify,
        random_state=random_seed,
    )
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _format_true_labels(
    labels_01: np.ndarray,
    true_positive_label: int,
    true_negative_label: int,
) -> np.ndarray:
    out = np.full(len(labels_01), true_negative_label, dtype=int)
    out[np.asarray(labels_01) == 1] = true_positive_label
    return out


def _format_pu_labels(
    labeled_mask: np.ndarray,
    pu_labeled_label: int,
    pu_unlabeled_label: int,
) -> np.ndarray:
    out = np.full(len(labeled_mask), pu_unlabeled_label, dtype=int)
    out[np.asarray(labeled_mask) == 1] = pu_labeled_label
    return out


def _empty_feature_block(features: np.ndarray) -> np.ndarray:
    return np.empty((0, *features.shape[1:]), dtype=features.dtype)


def _build_pu_dataset_from_source_arrays(
    features: np.ndarray,
    labels_01: np.ndarray,
    source_indices: np.ndarray,
    split_name: str,
    n_labeled: int | None,
    labeled_ratio: float | None,
    selection_strategy: str,
    scenario: str,
    case_control_mode: str,
    with_replacement: bool,
    true_positive_label: int,
    true_negative_label: int,
    pu_labeled_label: int,
    pu_unlabeled_label: int,
    rng: np.random.Generator,
    random_seed: int | None,
    metadata: dict[str, Any],
) -> PUDataset:
    (
        pu_features,
        pu_labels_01,
        labeled_mask,
        pu_source_indices,
        pu_roles,
    ) = create_pu_training_set(
        features,
        labels_01,
        n_labeled=n_labeled,
        labeled_ratio=labeled_ratio,
        selection_strategy=selection_strategy,
        scenario=scenario,
        with_replacement=with_replacement,
        case_control_mode=case_control_mode,
        random_seed=random_seed,
        rng=rng,
        source_indices=source_indices,
        return_indices=True,
        return_roles=True,
    )

    split_metadata = dict(metadata)
    split_metadata.update(
        {
            "split": split_name,
            "source_rows": int(len(features)),
            "source_positive_ratio": _safe_ratio(int((labels_01 == 1).sum()), len(labels_01)),
        }
    )

    return PUDataset(
        features=pu_features,
        pu_labels=_format_pu_labels(labeled_mask, pu_labeled_label, pu_unlabeled_label),
        true_labels=_format_true_labels(
            pu_labels_01, true_positive_label, true_negative_label
        ),
        metadata=split_metadata,
        source_indices=pu_source_indices,
        source_roles=pu_roles,
    )


def build_pu_datasets_from_binary_arrays(
    train_features: np.ndarray,
    train_labels_01: np.ndarray,
    test_features: np.ndarray,
    test_labels_01: np.ndarray,
    *,
    positive_classes: list | None = None,
    negative_classes: list | None = None,
    n_labeled: int | None = None,
    labeled_ratio: float | None = 0.2,
    val_ratio: float = 0.0,
    target_prevalence: float | None = None,
    selection_strategy: str = "random",
    scenario: str = "single",
    case_control_mode: str = "naive_mode",
    random_seed: int | None = 42,
    true_positive_label: int = 1,
    true_negative_label: int = 0,
    pu_labeled_label: int = 1,
    pu_unlabeled_label: int = -1,
    with_replacement: bool = True,
    print_stats: bool = False,
    dataset_log_file: str | None = None,
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[PUDataset, PUDataset, PUDataset]:
    """Build train/validation/test PUDatasets from binary PN arrays.

    Dataset loaders should stop here: their job is only to produce binary
    ``(X_train, y_train, X_test, y_test)`` arrays. This function owns source-level
    validation splitting, PU sampling, label remapping, source audit fields, and
    structured PU metadata.
    """
    rng = make_rng(random_seed=random_seed)
    train_labels_01 = np.asarray(train_labels_01, dtype=int)
    test_labels_01 = np.asarray(test_labels_01, dtype=int)

    if labeled_ratio is None and n_labeled is not None:
        n_pos = int((train_labels_01 == 1).sum())
        labeled_ratio = _safe_ratio(min(int(n_labeled), n_pos), n_pos)

    train_source_idx, val_source_idx = split_source_train_val(
        train_labels_01, val_ratio, random_seed=random_seed
    )

    train_x = train_features[train_source_idx]
    train_y = train_labels_01[train_source_idx]
    val_x = train_features[val_source_idx]
    val_y = train_labels_01[val_source_idx]

    semantic_cc_mode = (
        canonical_case_control_mode(case_control_mode)
        if scenario == "case-control"
        else None
    )
    reference_key = semantic_cc_mode if semantic_cc_mode is not None else scenario

    common_metadata = dict(metadata or {})
    common_metadata.update(
        {
            "dataset_name": dataset_name,
            "scenario": scenario,
            "selection_strategy": selection_strategy,
            "case_control_mode": case_control_mode,
            "case_control_semantics": semantic_cc_mode,
            "c_requested": labeled_ratio,
            "source_split_policy": "split_source_before_pu_sampling",
            "prior_source": "pi_unlabeled",
            "sampling_reference": PU_SAMPLING_REFERENCES.get(reference_key),
            "sar_pusb_selection_rule": (
                "MasaKat0_accept_reject_mean_max"
                if selection_strategy == "sar_pusb"
                else None
            ),
            "random_seed": random_seed,
            "true_positive_label": true_positive_label,
            "true_negative_label": true_negative_label,
            "pu_labeled_label": pu_labeled_label,
            "pu_unlabeled_label": pu_unlabeled_label,
        }
    )

    train_dataset = _build_pu_dataset_from_source_arrays(
        train_x,
        train_y,
        train_source_idx,
        "train",
        n_labeled=n_labeled,
        labeled_ratio=labeled_ratio,
        selection_strategy=selection_strategy,
        scenario=scenario,
        case_control_mode=case_control_mode,
        with_replacement=with_replacement,
        true_positive_label=true_positive_label,
        true_negative_label=true_negative_label,
        pu_labeled_label=pu_labeled_label,
        pu_unlabeled_label=pu_unlabeled_label,
        rng=rng,
        random_seed=random_seed,
        metadata=common_metadata,
    )

    val_n_labeled = None
    if val_ratio > 0 and len(val_x) > 0:
        val_dataset = _build_pu_dataset_from_source_arrays(
            val_x,
            val_y,
            val_source_idx,
            "validation",
            n_labeled=val_n_labeled,
            labeled_ratio=labeled_ratio,
            selection_strategy=selection_strategy,
            scenario=scenario,
            case_control_mode=case_control_mode,
            with_replacement=with_replacement,
            true_positive_label=true_positive_label,
            true_negative_label=true_negative_label,
            pu_labeled_label=pu_labeled_label,
            pu_unlabeled_label=pu_unlabeled_label,
            rng=rng,
            random_seed=random_seed,
            metadata=common_metadata,
        )
    else:
        val_dataset = PUDataset(
            features=_empty_feature_block(train_features),
            pu_labels=np.empty(0, dtype=int),
            true_labels=np.empty(0, dtype=int),
            metadata={**common_metadata, "split": "validation", "source_rows": 0},
            source_indices=np.empty(0, dtype=int),
            source_roles=np.empty(0, dtype="<U1"),
        )

    test_source_indices = np.arange(len(test_labels_01), dtype=int)
    if target_prevalence is not None and target_prevalence > 0:
        test_features, test_labels_01, test_source_indices = resample_by_prevalence(
            test_features,
            test_labels_01,
            target_prevalence,
            rng=rng,
            source_indices=test_source_indices,
        )

    test_true_labels = _format_true_labels(
        test_labels_01, true_positive_label, true_negative_label
    )
    test_dataset = PUDataset(
        features=test_features,
        pu_labels=test_true_labels,
        true_labels=test_true_labels,
        metadata={
            **common_metadata,
            "split": "test",
            "source_rows": int(len(test_features)),
            "target_prevalence": target_prevalence,
        },
        source_indices=test_source_indices,
        source_roles=np.full(len(test_features), "T", dtype="<U1"),
    )

    print_dataset_statistics(
        train_dataset,
        val_dataset,
        test_dataset,
        train_labeled_mask=None,
        positive_classes=positive_classes or [],
        negative_classes=negative_classes or [],
        true_positive_label=true_positive_label,
        true_negative_label=true_negative_label,
        pu_labeled_label=pu_labeled_label,
        pu_unlabeled_label=pu_unlabeled_label,
        val_ratio=val_ratio,
        log_file=dataset_log_file,
        also_print=print_stats,
    )

    return train_dataset, val_dataset, test_dataset


def print_dataset_statistics(
    train_dataset: PUDataset,
    val_dataset: PUDataset,
    test_dataset: PUDataset,
    train_labeled_mask: np.ndarray | None,
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

    train_pu = _as_numpy(train_dataset.pu_labels)
    train_true = _as_numpy(train_dataset.true_labels)
    train_labeled = train_pu == pu_labeled_label
    train_unlabeled = train_pu == pu_unlabeled_label
    if train_labeled_mask is not None:
        mask = np.asarray(train_labeled_mask) == 1
        if len(mask) == len(train_labeled):
            train_labeled = mask
            train_unlabeled = ~mask

    # Training set statistics
    train_stats = [
        [
            len(train_dataset),
            int(train_labeled.sum()),
            int(train_unlabeled.sum()),
            int(((train_true == true_positive_label) & train_unlabeled).sum()),
            int(((train_true == true_negative_label) & train_unlabeled).sum()),
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
    num_labeled = int(train_labeled.sum())
    pos_in_unlabeled = int(
        ((train_true == true_positive_label) & train_unlabeled).sum()
    )
    total_pos_in_train = num_labeled + pos_in_unlabeled
    total_train = len(train_dataset)
    ratio_p_in_train = total_pos_in_train / total_train if total_train > 0 else 0
    pi_unlabeled = (
        pos_in_unlabeled / int(train_unlabeled.sum()) if train_unlabeled.sum() else 0
    )
    ratio_lp_in_p_train = (
        num_labeled / total_pos_in_train if total_pos_in_train > 0 else 0
    )

    train_ratios_stats = [
        [
            f"{total_pos_in_train}",
            f"{ratio_p_in_train:.2%}",
            f"{pi_unlabeled:.2%}",
            f"{ratio_lp_in_p_train:.2%}",
        ]
    ]
    train_ratios_headers = [
        "Total Positives",
        "P/Constructed",
        "Risk Prior pi_U",
        "LP/P Ratio",
    ]
    lines.append("Training Set Ratios:")
    lines.append(
        tabulate(train_ratios_stats, headers=train_ratios_headers, tablefmt="grid")
    )
    lines.append("")

    # Validation and Test set statistics
    other_sets_stats = []
    if val_ratio > 0 and len(val_dataset) > 0:
        total_val = len(val_dataset)
        val_true = _as_numpy(val_dataset.true_labels)
        pos_val = int((val_true == true_positive_label).sum())
        ratio_p_in_val = pos_val / total_val if total_val > 0 else 0
        other_sets_stats.append(
            [
                "Validation",
                total_val,
                pos_val,
                int((val_true == true_negative_label).sum()),
                f"{ratio_p_in_val:.2%}",
            ]
        )

    total_test = len(test_dataset)
    test_true = _as_numpy(test_dataset.true_labels)
    pos_test = int((test_true == true_positive_label).sum())
    ratio_p_in_test = pos_test / total_test if total_test > 0 else 0
    other_sets_stats.append(
        [
            "Test",
            total_test,
            pos_test,
            int((test_true == true_negative_label).sum()),
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
    if also_print:
        print(text, end="")


def resample_by_prevalence(
    features,
    labels,
    target_prevalence,
    random_seed=42,
    rng: np.random.Generator | None = None,
    source_indices: np.ndarray | None = None,
):
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

    rng = make_rng(random_seed=random_seed, rng=rng)
    return_sources = source_indices is not None
    source_indices = (
        np.arange(len(labels), dtype=int)
        if source_indices is None
        else np.asarray(source_indices, dtype=int)
    )

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_features, pos_labels = features[pos_mask], labels[pos_mask]
    neg_features, neg_labels = features[neg_mask], labels[neg_mask]
    pos_sources = source_indices[pos_mask]
    neg_sources = source_indices[neg_mask]

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

        indices = _choice(rng, np.arange(n_neg_avail), n_neg_new, replace=False)
        pos_features_new, pos_labels_new = pos_features, pos_labels
        neg_features_new, neg_labels_new = neg_features[indices], neg_labels[indices]
        pos_sources_new, neg_sources_new = pos_sources, neg_sources[indices]

    elif target_prevalence < original_prevalence:
        n_neg_new = n_neg_avail
        n_pos_new = int(
            np.floor(n_neg_new * target_prevalence / (1 - target_prevalence))
        )
        if n_pos_new < 0:
            raise ValueError("Calculated positive sample count cannot be negative.")

        indices = _choice(rng, np.arange(n_pos_avail), n_pos_new, replace=False)
        neg_features_new, neg_labels_new = neg_features, neg_labels
        pos_features_new, pos_labels_new = pos_features[indices], pos_labels[indices]
        neg_sources_new, pos_sources_new = neg_sources, pos_sources[indices]
    else:
        if return_sources:
            return features, labels, source_indices
        return features, labels

    new_features = np.concatenate([pos_features_new, neg_features_new], axis=0)
    new_labels = np.concatenate([pos_labels_new, neg_labels_new], axis=0)
    new_sources = np.concatenate([pos_sources_new, neg_sources_new], axis=0)

    shuffle_indices = rng.permutation(len(new_features))
    if not return_sources:
        return new_features[shuffle_indices], new_labels[shuffle_indices]
    return (
        new_features[shuffle_indices],
        new_labels[shuffle_indices],
        new_sources[shuffle_indices],
    )
