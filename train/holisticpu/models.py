"""Private HolisticPU backbones for PU-Bench datasets.

They stay inside ``train/holisticpu`` so the benchmark's controlled public
backbones remain separate from HolisticPU's method-specific capacity choices.
The selector below targets PU-Bench dataset classes, not every dataset recipe
that appears in the original HolisticPU repository.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _HolisticPUPrivateMixin:
    positive_logit_index = 0
    backbone_policy = "private"


class PrivateCNNCIFAR(_HolisticPUPrivateMixin, nn.Module):
    """HolisticPU private CNN for PU-Bench CIFAR-10."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=2)
        self.conv3 = nn.Conv2d(96, 192, 1)
        self.conv4 = nn.Conv2d(192, 10, 1)
        self.fc1 = nn.Linear(1960, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1, 1960)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class PrivateLeNet(_HolisticPUPrivateMixin, nn.Module):
    """HolisticPU private LeNet-style CNN for PU-Bench MNIST/FashionMNIST."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(6)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.bn_fc1 = nn.BatchNorm1d(84)
        self.layer1 = nn.Sequential(self.conv1, self.mp, self.relu)
        self.layer2 = nn.Sequential(self.conv2, self.mp, self.relu)
        self.layer3 = nn.Sequential(self.conv3, self.relu)
        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3])
        self.layer4 = nn.Sequential(self.fc1, self.bn_fc1, self.relu)
        self.classifier = nn.Linear(84, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        h = h.view(h.size(0), -1)
        h = self.layer4(h)
        return self.classifier(h)


class PrivateAlzheimerCNN(_HolisticPUPrivateMixin, nn.Module):
    """HolisticPU private CNN for PU-Bench 1-channel 128x128 AlzheimerMRI."""

    expected_image_size = (128, 128)

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout(0.2),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout(0.3),
                ),
            ]
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.final_classifier = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        h = self.gap(h)
        h = torch.flatten(h, 1)
        h = self.classifier_head(h)
        return self.final_classifier(h)


class PrivateDynamicMLP(_HolisticPUPrivateMixin, nn.Module):
    """HolisticPU private MLP for PU-Bench SBERT/tabular vector datasets."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.built = False
        self.layers = nn.ModuleList()
        self.classifier_head: nn.Module | None = None
        self.final_classifier: nn.Module | None = None

    def _build(self, in_features: int) -> None:
        hidden1, hidden2, hidden3 = 512, 256, 128
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, hidden1), nn.ReLU(), nn.Dropout(0.3)
                ),
                nn.Sequential(nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(0.3)),
                nn.Sequential(nn.Linear(hidden2, hidden3), nn.ReLU(), nn.Dropout(0.2)),
            ]
        )
        self.classifier_head = nn.Sequential(nn.Linear(hidden3, 64), nn.ReLU())
        self.final_classifier = nn.Linear(64, 2)
        self.built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.built:
            self._build(int(x.shape[1]))
            self.to(x.device)
        h = x
        for layer in self.layers:
            h = layer(h)
        assert self.classifier_head is not None
        assert self.final_classifier is not None
        h = self.classifier_head(h)
        return self.final_classifier(h)


def select_private_model(dataset_class: str, model_name: str, prior: float, arch: str = "auto"):
    """Return a HolisticPU private backbone for PU-Bench dataset classes."""

    low_cls = dataset_class.lower()
    low_arch = (arch or "auto").lower()

    if low_arch in {"cnn_cifar", "private_cnn_cifar"}:
        return PrivateCNNCIFAR(prior)
    if low_arch in {"lenet", "private_lenet"}:
        return PrivateLeNet(prior)
    if low_arch in {"alzheimercnn", "private_alzheimer_cnn"}:
        return PrivateAlzheimerCNN(prior)
    if low_arch in {"dynamic_mlp", "private_dynamic_mlp"}:
        return PrivateDynamicMLP(prior)
    if low_arch != "auto":
        raise ValueError(f"Unsupported HolisticPU private_backbone_arch='{arch}'")

    if "cifar10" in low_cls or model_name == "cnn_cifar10":
        return PrivateCNNCIFAR(prior)
    if "fashionmnist" in low_cls or "mnist" in low_cls or model_name in {
        "cnn_fashionmnist",
        "cnn_mnist",
    }:
        return PrivateLeNet(prior)
    if "alzheimer" in low_cls or "mri" in low_cls or model_name == "cnn_alzheimermri":
        return PrivateAlzheimerCNN(prior)
    if any(
        key in low_cls
        for key in ("20news", "newsgroup", "imdb", "mushroom", "spambase", "connect")
    ) or model_name in {"mlp_20News", "mlp_IMDB", "mlp_mushrooms", "mlp_spambase"}:
        return PrivateDynamicMLP(prior)
    raise ValueError(
        "HolisticPU private backbone selection has no PU-Bench mapping for "
        f"dataset_class='{dataset_class}', model_name='{model_name}'."
    )


__all__ = [
    "PrivateCNNCIFAR",
    "PrivateLeNet",
    "PrivateAlzheimerCNN",
    "PrivateDynamicMLP",
    "select_private_model",
]
