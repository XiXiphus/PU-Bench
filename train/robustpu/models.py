"""Private Robust-PU backbones.

Primary source:
    author source file(s): robustpu/models.py

These stay inside ``train/robustpu`` so the PU-Bench controlled public
backbones remain separate from the authors' method-specific capacity choices.
The selector targets PU-Bench dataset classes; it does not reproduce every
source dataset recipe.
"""

from __future__ import annotations

import torch
from torch import nn


class _RobustPUPrivateMixin:
    backbone_policy = "private"


class SourceNormalNN(_RobustPUPrivateMixin, nn.Module):
    """Source normalNN with dynamic input dimension for PU-Bench tensors."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.fc1: nn.Linear | None = None
        self.fc2: nn.Linear | None = None
        self.af = nn.ReLU(inplace=True)

    def _build(self, in_features: int, device: torch.device) -> None:
        self.fc1 = nn.Linear(int(in_features), 100).to(device)
        self.fc2 = nn.Linear(100, 1).to(device)
        self.reset_para()

    def forward(self, x: torch.Tensor, return_fea: bool = False):
        h = torch.flatten(x, start_dim=1)
        if self.fc1 is None or self.fc2 is None:
            self._build(h.shape[1], h.device)
        h = self.fc1(h)
        h = self.af(h)
        fea = h.detach().clone()
        h = self.fc2(h).squeeze(-1)
        if return_fea:
            return h, fea
        return h

    def reset_para(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class SourceCNN(_RobustPUPrivateMixin, nn.Module):
    """Source Robust-PU CNN for PU-Bench CIFAR-10."""

    expected_image_size = (32, 32)

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.b1 = nn.BatchNorm2d(96)
        self.b2 = nn.BatchNorm2d(96)
        self.b3 = nn.BatchNorm2d(96)
        self.b4 = nn.BatchNorm2d(192)
        self.b5 = nn.BatchNorm2d(192)
        self.b6 = nn.BatchNorm2d(192)
        self.b7 = nn.BatchNorm2d(192)
        self.b8 = nn.BatchNorm2d(192)
        self.b9 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(10 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)
        self.af = nn.ReLU(inplace=True)
        self.reset_para()

    def forward(self, x: torch.Tensor, return_fea: bool = False):
        h = self.af(self.b1(self.conv1(x)))
        h = self.af(self.b2(self.conv2(h)))
        h = self.af(self.b3(self.conv3(h)))
        h = self.af(self.b4(self.conv4(h)))
        h = self.af(self.b5(self.conv5(h)))
        h = self.af(self.b6(self.conv6(h)))
        h = self.af(self.b7(self.conv7(h)))
        h = self.af(self.b8(self.conv8(h)))
        h = self.af(self.b9(self.conv9(h)))
        h = h.reshape(h.shape[0], -1)
        h = self.af(self.fc1(h))
        h = self.af(self.fc2(h))
        fea = h.detach().clone()
        h = self.fc3(h).squeeze(-1)
        if return_fea:
            return h, fea
        return h

    def reset_para(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def select_private_model(
    dataset_class: str,
    model_name: str,
    prior: float,
    arch: str = "auto",
) -> nn.Module:
    arch = str(arch).lower()
    low_cls = dataset_class.lower()

    if arch == "auto":
        arch = "cnn" if "cifar" in low_cls else "normalnn"

    if arch == "cnn":
        if "cifar" not in low_cls:
            raise ValueError(
                "RobustPU private CNN is source-shaped for 3x32 CIFAR inputs; "
                f"got dataset_class='{dataset_class}'. Use private_backbone_arch='normalnn'."
            )
        return SourceCNN(prior)

    if arch in {"normalnn", "mlp"}:
        return SourceNormalNN(prior)

    raise ValueError(
        f"Unsupported RobustPU private_backbone_arch='{arch}' for "
        f"dataset_class='{dataset_class}', model_name='{model_name}'."
    )


__all__ = ["SourceNormalNN", "SourceCNN", "select_private_model"]
