"""mix_models.py

This module provides Mixup-compatible CNN models.
These models' forward passes are adapted to support feature mixing at intermediate layers,
as required by methods like P3MIX.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class MixCNN_CIFAR10(nn.Module):
    """Mixup-compatible CNN for CIFAR-10 (3x32x32)."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior

        # Decompose the original Sequential 'features' into a ModuleList
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=3, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96, 96, kernel_size=3, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Conv2d(96, 192, kernel_size=3, padding=1),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=3, padding=1),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                ),
            ]
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(192 * 8 * 8, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
        )
        self.final_classifier = nn.Linear(1000, 1)

    def forward(
        self,
        x: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: float | None = None,
        mix_layer: int = 1000,
        flag_feature: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h, h2 = x, x2
        # Perform mix at the input layer
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        # Pass through intermediate layers
        for i, layer_module in enumerate(self.layers):
            if i <= mix_layer:
                h = layer_module(h)
                if h2 is not None:
                    h2 = layer_module(h2)

            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2

            if i > mix_layer:
                h = layer_module(h)

        # Classifier
        h_flat = torch.flatten(h, 1)
        features = self.classifier_head(h_flat)
        logits = self.final_classifier(features)

        if flag_feature:
            return logits, features
        else:
            return logits


class MixLeNet(nn.Module):
    """Mixup-compatible LeNet for MNIST/FashionMNIST (1x28x28)."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

        self.layer1 = nn.Sequential(self.conv1, self.pool1, self.relu1)
        self.layer2 = nn.Sequential(self.conv2, self.pool2, self.relu2)
        self.layers = nn.ModuleList([self.layer1, self.layer2])

        self.classifier_head = nn.Sequential(self.fc1, self.relu3)
        self.final_classifier = self.fc2

    def forward(
        self,
        x: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: float | None = None,
        mix_layer: int = 1000,
        flag_feature: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h, h2 = x, x2
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        for i, layer_module in enumerate(self.layers):
            if i <= mix_layer:
                h = layer_module(h)
                if h2 is not None:
                    h2 = layer_module(h2)

            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2

            if i > mix_layer:
                h = layer_module(h)

        h_flat = torch.flatten(h, 1)
        features = self.classifier_head(h_flat)
        logits = self.final_classifier(features)

        if flag_feature:
            return logits, features
        else:
            return logits


# Aliases for consistency with select_model
MixCNN_MNIST = MixLeNet
MixCNN_FashionMNIST = MixLeNet


class MixCNN_AlzheimerMRI(MixLeNet):
    """Mixup-compatible deeper CNN for Alzheimer MRI (1x128x128)."""

    def __init__(self, prior: float = 0.0):
        # Initialize parent class placeholder, then completely replace with deeper structure
        super().__init__(prior)
        self.expected_image_size = (128, 128)

        # Redefine as deeper multi-layer convolutional structure, maintaining compatibility with P3MIX mixing interface
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # 128 -> 64
                    nn.Dropout(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # 64 -> 32
                    nn.Dropout(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # 32 -> 16
                    nn.Dropout(0.2),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # 16 -> 8
                    nn.Dropout(0.3),
                ),
            ]
        )

        # GAP + 64-dim head for alignment with other methods
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.classifier_head = nn.Sequential(self.fc1, self.relu3, nn.Dropout(0.3))
        self.final_classifier = self.fc2

    def forward(
        self,
        x: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: float | None = None,
        mix_layer: int = 1000,
        flag_feature: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h, h2 = x, x2
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        for i, layer_module in enumerate(self.layers):
            if i <= mix_layer:
                h = layer_module(h)
                if h2 is not None:
                    h2 = layer_module(h2)

            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2

            if i > mix_layer:
                h = layer_module(h)

        # GAP + FC
        h = self.gap(h)
        h_flat = torch.flatten(h, 1)
        features = self.classifier_head(h_flat)
        logits = self.final_classifier(features)

        if flag_feature:
            return logits, features
        else:
            return logits


__all__ = [
    "MixCNN_CIFAR10",
    "MixCNN_FashionMNIST",
    "MixCNN_MNIST",
    "MixCNN_AlzheimerMRI",
    "MixLeNet",
]


# ==============================================================================
# 20News Mix-compatible MLP (feature mixing between linear layers)
# ==============================================================================


class MixMLP20News(nn.Module):
    """Mixup-compatible MLP for 20News dense features.

    Allows mixing at input (-1) or at hidden layer indices (0, 1, 2).
    """

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.built = False
        self.layers = nn.ModuleList()
        self.classifier_head = None
        self.final_classifier = None

    def _build(self, in_features: int):
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
        self.final_classifier = nn.Linear(64, 1)
        self.built = True

    def forward(
        self,
        x: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: float | None = None,
        mix_layer: int = 1000,
        flag_feature: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not self.built:
            self._build(int(x.shape[1]))
            self.to(x.device)

        h, h2 = x, x2
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        for i, layer in enumerate(self.layers):
            if i <= mix_layer:
                h = layer(h)
                if h2 is not None:
                    h2 = layer(h2)
            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2
            if i > mix_layer:
                h = layer(h)

        features = self.classifier_head(h)
        logits = self.final_classifier(features)
        if flag_feature:
            return logits, features
        return logits


# Aliases for consistency with select_model
MixMLP_20News = MixMLP20News
MixMLP_IMDB = MixMLP20News
