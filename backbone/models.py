from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_CIFAR10(nn.Module):
    """Customized CNN for CIFAR-10 (3x32x32), with a modernized layer structure."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer_module in self.layers:
            h = layer_module(h)
        h = torch.flatten(h, 1)
        features = self.classifier_head(h)
        logits = self.final_classifier(features)
        if logits.shape[-1] == 1:
            return logits.view(-1)
        return logits


class LeNet(nn.Module):
    """Base LeNet-style CNN for MNIST/FashionMNIST (1x28x28) with a modernized layer structure."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 10, kernel_size=5),
                    nn.MaxPool2d(kernel_size=2),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(10, 20, kernel_size=5),
                    nn.MaxPool2d(kernel_size=2),
                    nn.ReLU(),
                ),
            ]
        )
        self.classifier_head = nn.Sequential(nn.Linear(320, 50), nn.ReLU())
        self.final_classifier = nn.Linear(50, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        h = torch.flatten(h, 1)
        features = self.classifier_head(h)
        logits = self.final_classifier(features)
        if logits.shape[-1] == 1:
            return logits.view(-1)
        return logits


class CNN_MNIST(LeNet):
    """Customized CNN for MNIST (1x28x28) - LeNet-style."""

    def __init__(self, prior: float = 0.0):
        super().__init__(prior)


class CNN_FashionMNIST(LeNet):
    """Customized CNN for FashionMNIST (1x28x28) - LeNet-style."""

    def __init__(self, prior: float = 0.0):
        super().__init__(prior)


class CNN_AlzheimerMRI(LeNet):
    """Customized CNN for Alzheimer MRI (1x128x128) - LeNet-style with adjusted input."""

    def __init__(self, prior: float = 0.0):
        # Call parent class to initialize basic attributes, then completely override with deeper architecture
        super().__init__(prior)
        # Expected input size (for adaptive resampling in evaluation phase)
        self.expected_image_size = (128, 128)

        # Deep convolutional backbone (deeper than LeNet, with downsampling and BN)
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

        # Global average pooling to avoid huge fully connected layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Unified feature dimension for easy reuse across methods (e.g., HolisticPU binary head, LaGAM projection head)
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.final_classifier = nn.Linear(64, 1)
        self.feature_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        h = self.gap(h)
        h = torch.flatten(h, 1)  # (B, 256)
        features = self.classifier_head(h)  # (B, 64)
        logits = self.final_classifier(features)  # (B, 1)
        if logits.shape[-1] == 1:
            return logits.view(-1)
        return logits


class HolisticPU_CNN_CIFAR10(CNN_CIFAR10):
    """Specialized model for HolisticPU, outputs 2D logits to work with cross-entropy loss."""

    def __init__(self, prior: float = 0.0):
        super().__init__(prior)
        self.final_classifier = nn.Linear(1000, 2)


class HolisticPU_LeNet(LeNet):
    """Specialized LeNet model for HolisticPU, outputs 2D logits."""

    def __init__(self, prior: float = 0.0):
        super().__init__(prior)
        self.final_classifier = nn.Linear(50, 2)


class HolisticPU_CNN_MNIST(HolisticPU_LeNet):
    """Specialized model for HolisticPU on MNIST."""

    def __init__(self, prior: float = 0.0):
        super().__init__(prior)


class HolisticPU_CNN_FashionMNIST(HolisticPU_LeNet):
    """Specialized model for HolisticPU on FashionMNIST."""

    def __init__(self, prior: float = 0.0):
        super().__init__(prior)


class HolisticPU_CNN_AlzheimerMRI(CNN_AlzheimerMRI):
    """Specialized model for HolisticPU on Alzheimer MRI, outputs 2D logits."""

    def __init__(self, prior: float = 0.0):
        super().__init__(prior)
        self.expected_image_size = (128, 128)
        # Inherit GAP + 64-dim feature head, change output to 2 classes
        self.final_classifier = nn.Linear(64, 2)


class _DynamicMLP20News(nn.Module):
    """Dynamic MLP that initializes first layer on first forward based on input dim.

    out_dim controls logits size: 1 for BCE/PU losses, 2 for CE-based methods.
    """

    def __init__(self, out_dim: int = 1, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.out_dim = out_dim
        self.built = False
        # Placeholders
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
        self.final_classifier = nn.Linear(64, self.out_dim)
        self.built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.built:
            self._build(int(x.shape[1]))
            # Ensure newly created parameters/buffers are on the same device as input
            self.to(x.device)
        h = x
        for layer in self.layers:
            h = layer(h)
        features = self.classifier_head(h)
        logits = self.final_classifier(features)
        if logits.shape[-1] == 1:
            return logits.view(-1)
        return logits


class MLP_20News(_DynamicMLP20News):
    def __init__(self, prior: float = 0.0):
        super().__init__(out_dim=1, prior=prior)


class HolisticPU_MLP_20News(_DynamicMLP20News):
    def __init__(self, prior: float = 0.0):
        super().__init__(out_dim=2, prior=prior)


class MLP_IMDB(_DynamicMLP20News):
    def __init__(self, prior: float = 0.0):
        super().__init__(out_dim=1, prior=prior)


class HolisticPU_MLP_IMDB(_DynamicMLP20News):
    def __init__(self, prior: float = 0.0):
        super().__init__(out_dim=2, prior=prior)


__all__ = [
    "CNN_CIFAR10",
    "CNN_FashionMNIST",
    "CNN_MNIST",
    "CNN_AlzheimerMRI",
    "LeNet",
    "HolisticPU_CNN_CIFAR10",
    "HolisticPU_CNN_FashionMNIST",
    "HolisticPU_CNN_MNIST",
    "HolisticPU_CNN_AlzheimerMRI",
    "MLP_20News",
    "HolisticPU_MLP_20News",
    "MLP_IMDB",
    "HolisticPU_MLP_IMDB",
]
