"""P/U batch sampler for PULDA.

Source snapshot:
    author source file(s): pulda/dataTools/PUSampler.py
    jiangyangby/PULDA at 7b3dcad95bd7caa0a9477af37a05764fbe6e27bc

The source repeats labeled positives indefinitely and iterates unlabeled
examples once per epoch, yielding fixed-size P and U blocks.  PU-Bench keeps the
same estimator-side batching rule while sourcing P/U membership from its own
canonical PU dataset construction.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

import numpy as np
from torch.utils.data import Sampler


class PULDAResamplingBatchSampler(Sampler[list[int]]):
    """Source-aligned PULDA sampler with fixed P and U counts per batch."""

    def __init__(
        self,
        p_indices: Iterable[int],
        u_indices: Iterable[int],
        p_batch_size: int,
        u_batch_size: int,
    ) -> None:
        self.p_indices = np.asarray(list(p_indices), dtype=int)
        self.u_indices = np.asarray(list(u_indices), dtype=int)
        self.p_batch_size = int(p_batch_size)
        self.u_batch_size = int(u_batch_size)

        if self.p_batch_size <= 0 or self.u_batch_size <= 0:
            raise ValueError("PULDA P/U batch sizes must be positive.")
        if len(self.p_indices) == 0:
            raise ValueError("PULDA resampling requires at least one labeled positive.")
        if len(self.u_indices) < self.u_batch_size:
            raise ValueError(
                "PULDA source-style resampling drops incomplete U batches; "
                f"got {len(self.u_indices)} unlabeled samples with U_batch_size={self.u_batch_size}. "
                "Lower U_batch_size or set resample=0."
            )

    def __iter__(self) -> Iterator[list[int]]:
        p_iter = _iterate_eternally(self.p_indices)
        u_iter = _iterate_once(self.u_indices)
        for p_batch, u_batch in zip(
            _grouper(p_iter, self.p_batch_size),
            _grouper(u_iter, self.u_batch_size),
        ):
            yield list(p_batch) + list(u_batch)

    def __len__(self) -> int:
        return len(self.u_indices) // self.u_batch_size


def _iterate_once(indices: np.ndarray) -> np.ndarray:
    return np.random.permutation(indices)


def _iterate_eternally(indices: np.ndarray) -> Iterator[int]:
    def infinite_shuffles() -> Iterator[np.ndarray]:
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def _grouper(iterable: Iterable[int], n: int) -> Iterator[tuple[int, ...]]:
    args = [iter(iterable)] * n
    return zip(*args)
