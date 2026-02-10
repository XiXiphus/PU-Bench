from __future__ import annotations

import torch
from torch.utils.data import Dataset
from typing import Tuple


class VectorWeakAugment:
    """
    Weak augmentations for vector (tabular/text embedding) inputs.
    - Gaussian noise
    - Feature dropout (mask a small ratio of features to zero)
    """

    def __init__(self, noise_std: float = 0.02, dropout_ratio: float = 0.0):
        self.noise_std = float(noise_std)
        self.dropout_ratio = float(dropout_ratio)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure float tensor
        if not torch.is_floating_point(x):
            x = x.float()
        out = x
        if self.noise_std > 0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise
        if self.dropout_ratio > 0:
            mask = (torch.rand_like(out) > self.dropout_ratio).float()
            out = out * mask
        return out


class VectorStrongAugment:
    """
    Strong augmentations for vector inputs.
    - Stronger Gaussian noise
    - Higher feature dropout ratio
    - Random sign flip on a small subset of features
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        dropout_ratio: float = 0.1,
        sign_flip_ratio: float = 0.05,
    ):
        self.noise_std = float(noise_std)
        self.dropout_ratio = float(dropout_ratio)
        self.sign_flip_ratio = float(sign_flip_ratio)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.float()
        out = x
        if self.noise_std > 0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise
        if self.dropout_ratio > 0:
            mask = (torch.rand_like(out) > self.dropout_ratio).float()
            out = out * mask
        if self.sign_flip_ratio > 0:
            flip_mask = torch.rand_like(out) < self.sign_flip_ratio
            out = torch.where(flip_mask, -out, out)
        return out


class VectorAugPUDatasetWrapper(Dataset):
    """
    Wrap a PU dataset that yields (features, pu_labels, true_labels, indices, pseudo_labels)
    into a dataset that yields ((x_w, x_s), pu_labels, true_labels, indices, pseudo_labels),
    where x_w uses weak augmentations and x_s uses strong augmentations for vector inputs.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        weak_aug: VectorWeakAugment | None = None,
        strong_aug: VectorStrongAugment | None = None,
    ):
        self.base_dataset = base_dataset
        self.weak_aug = weak_aug or VectorWeakAugment()
        self.strong_aug = strong_aug or VectorStrongAugment()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x, pu_label, true_label, idx, pseudo_label = self.base_dataset[index]
        # Ensure tensor type
        if isinstance(x, torch.Tensor):
            x_tensor = x
        else:
            x_tensor = torch.as_tensor(x)

        x_w = self.weak_aug(x_tensor)
        x_s = self.strong_aug(x_tensor)
        return (x_w, x_s), pu_label, true_label, idx, pseudo_label
