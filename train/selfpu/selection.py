"""Self-paced dataset partitioning for Self-PU."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SelectionState:
    clean_indices: np.ndarray
    pseudo_pos_indices: np.ndarray
    pseudo_neg_indices: np.ndarray
    noisy_indices: np.ndarray
    pos_precision: float | None = None
    neg_precision: float | None = None


class SelfPUCleanDataset(Dataset):
    """Clean self-paced subset with method-private signed pseudo labels."""

    def __init__(
        self,
        base_dataset: Dataset,
        pseudo_pos_indices: np.ndarray,
        pseudo_neg_indices: np.ndarray,
    ):
        self.base_dataset = base_dataset
        self.pseudo_pos_indices = np.asarray(pseudo_pos_indices, dtype=int)
        self.pseudo_neg_indices = np.asarray(pseudo_neg_indices, dtype=int)
        self.indices = np.concatenate(
            (self.pseudo_pos_indices, self.pseudo_neg_indices)
        ).astype(int)
        self.labels = {
            int(index): 1 for index in self.pseudo_pos_indices.tolist()
        } | {int(index): -1 for index in self.pseudo_neg_indices.tolist()}

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, row: int):
        base_index = int(self.indices[row])
        x, _pu_label, true_label, _index, pseudo = self.base_dataset[base_index]
        signed_label = torch.tensor(self.labels[base_index], dtype=torch.long)
        return x, signed_label, true_label, torch.tensor(base_index), pseudo


class SelfPUNoisyDataset(Dataset):
    """Source-style noisy sampler over labeled positives and remaining U.

    The Self-PU source keeps clean selected examples out of the noisy split and
    then alternates one labeled positive with roughly ``sample_ratio - 1``
    unlabeled examples. This wrapper reproduces that method-local sampling
    convention without changing the shared PU-Bench dataset layer.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: np.ndarray,
        rng: np.random.Generator,
    ):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=int)
        pu_labels = base_dataset.pu_labels.detach().cpu().numpy()
        self.p_indices = self.indices[pu_labels[self.indices] == 1]
        self.u_indices = self.indices[pu_labels[self.indices] == -1]
        self.rng = rng
        self.shuffle()

    def shuffle(self) -> None:
        self.p_indices = self.rng.permutation(self.p_indices)
        self.u_indices = self.rng.permutation(self.u_indices)
        if len(self.p_indices) == 0:
            self.sample_ratio = 1
        else:
            self.sample_ratio = int(len(self.u_indices) / len(self.p_indices)) + 1

    def __len__(self) -> int:
        if len(self.p_indices) == 0:
            return int(len(self.u_indices))
        return int(len(self.p_indices) * max(1, self.sample_ratio))

    def __getitem__(self, row: int):
        if len(self.p_indices) == 0:
            base_index = int(self.u_indices[row % max(1, len(self.u_indices))])
        elif len(self.u_indices) == 0 or row % self.sample_ratio == 0:
            p_row = (row // self.sample_ratio) % len(self.p_indices)
            base_index = int(self.p_indices[p_row])
        else:
            u_row = row - (row // self.sample_ratio + 1)
            base_index = int(self.u_indices[u_row % len(self.u_indices)])
        return self.base_dataset[base_index]


class SelfPUSelector:
    """Self-paced confident U selector aligned to ``train_2s2t.py``."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        replacement: bool,
        increasing: bool,
        rampup_length: int,
        flex_ratio: float,
        rng: np.random.Generator,
    ):
        self.base_dataset = base_dataset
        pu_labels = base_dataset.pu_labels.detach().cpu().numpy()
        self.all_indices = np.arange(len(base_dataset), dtype=int)
        self.positive_indices = self.all_indices[pu_labels == 1]
        self.unlabeled_indices = self.all_indices[pu_labels == -1]
        self.replacement = bool(replacement)
        self.increasing = bool(increasing)
        self.rampup_length = max(1, int(rampup_length))
        self.flex_ratio = float(flex_ratio)
        self.rng = rng

    def select(
        self,
        scores: np.ndarray,
        *,
        epoch: int,
        top: float,
        previous_pseudo_pos_indices: np.ndarray,
        previous_pseudo_neg_indices: np.ndarray,
        ratio: float = 0.5,
    ) -> SelectionState:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.shape[0] != len(self.base_dataset):
            raise ValueError(
                f"SelfPU selector expected {len(self.base_dataset)} scores, "
                f"got {scores.shape[0]}."
            )

        if self.increasing:
            percent = min(max(float(epoch), 0.0) / float(self.rampup_length), 1.0)
        else:
            percent = 1.0

        # Source mode "A": n_all is the number selected from each tail.
        # With the source call-site default ratio=0.5 and flex=0, the clean
        # set contains 2 * n_all examples.
        n_all = int(
            len(self.unlabeled_indices)
            * (1.0 - float(ratio))
            * percent
            * float(top)
        )
        n_each_target = max(0, int(n_all * (1.0 - self.flex_ratio)))

        previous_pos = np.asarray(previous_pseudo_pos_indices, dtype=int)
        previous_neg = np.asarray(previous_pseudo_neg_indices, dtype=int)
        previous_clean = np.concatenate((previous_pos, previous_neg)).astype(int)

        if n_each_target == 0:
            noisy = np.setdiff1d(self.all_indices, previous_clean, assume_unique=False)
            return SelectionState(
                clean_indices=previous_clean,
                pseudo_pos_indices=previous_pos,
                pseudo_neg_indices=previous_neg,
                noisy_indices=noisy,
            )

        if self.replacement:
            candidates = self.unlabeled_indices
        else:
            candidates = np.setdiff1d(self.unlabeled_indices, previous_clean)

        ordered = candidates[np.argsort(scores[candidates])]
        if self.replacement:
            n_each = n_each_target
        else:
            n_each = max(0, n_each_target - len(previous_clean) // 2)
        n_each = min(n_each, len(ordered) // 2)
        pseudo_neg = ordered[:n_each].astype(int)
        pseudo_pos = ordered[-n_each:].astype(int)

        if self.replacement:
            clean = np.concatenate((pseudo_pos, pseudo_neg)).astype(int)
            clean_pos = pseudo_pos
            clean_neg = pseudo_neg
        else:
            clean_pos = np.unique(np.concatenate((previous_pos, pseudo_pos))).astype(int)
            clean_neg = np.unique(np.concatenate((previous_neg, pseudo_neg))).astype(int)
            clean = np.concatenate((clean_pos, clean_neg)).astype(int)

        noisy = np.setdiff1d(self.all_indices, clean, assume_unique=False)

        pos_precision, neg_precision = self._precision(pseudo_pos, pseudo_neg)
        return SelectionState(
            clean_indices=clean,
            pseudo_pos_indices=clean_pos,
            pseudo_neg_indices=clean_neg,
            noisy_indices=noisy,
            pos_precision=pos_precision,
            neg_precision=neg_precision,
        )

    def _precision(
        self,
        pseudo_pos: np.ndarray,
        pseudo_neg: np.ndarray,
    ) -> tuple[float | None, float | None]:
        true_labels = getattr(self.base_dataset, "true_labels", None)
        if true_labels is None:
            return None, None
        y = true_labels.detach().cpu().numpy()
        pos_precision = (
            float((y[pseudo_pos] == 1).mean()) if len(pseudo_pos) else None
        )
        neg_precision = (
            float((y[pseudo_neg] == 0).mean()) if len(pseudo_neg) else None
        )
        return pos_precision, neg_precision
