"""PUL-CPBF private source-style backbones."""

from __future__ import annotations

import torch
from torch import nn


class PULCPBFSourceLeNet(nn.Module):
    """One-logit LeNet used by the author PUL-CPBF FashionMNIST path."""

    backbone_policy = "private"
    num_classifier = 1

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.bn_fc1 = nn.BatchNorm1d(84)

        self.layer1 = nn.Sequential(self.conv1, self.mp, self.relu)
        self.layer2 = nn.Sequential(self.conv2, self.mp, self.relu)
        self.layer3 = nn.Sequential(self.conv3, self.relu)
        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3])
        self.classifier_head = nn.Sequential(self.fc1, self.bn_fc1, self.relu)
        self.classifier = nn.Linear(84, self.num_classifier)
        self.final_classifier = self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer_module in self.layers:
            h = layer_module(h)
        h = h.view(h.size(0), -1)
        h = self.classifier_head(h)
        return self.classifier(h).view(-1)


def select_private_model(
    *, dataset_class: str, model_name: str, prior: float, arch: str = "auto"
) -> nn.Module:
    arch = str(arch).lower()
    if arch not in {"auto", "lenet"}:
        raise ValueError(f"Unsupported PULCPBF private_backbone_arch='{arch}'.")

    dataset_lower = dataset_class.lower()
    if model_name in {"cnn_mnist", "cnn_fashionmnist"} or "mnist" in dataset_lower:
        return PULCPBFSourceLeNet(prior)

    raise ValueError(
        "PULCPBF has no package-local private backbone for "
        f"dataset_class='{dataset_class}', model_name='{model_name}'."
    )


__all__ = ["PULCPBFSourceLeNet", "select_private_model"]
