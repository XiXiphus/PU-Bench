"""LaGAM meta backbones adapted to PU-Bench datasets.

Source snapshot:
    author source file(s): lagam/models/
    llong-cs/LaGAM at 362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8

The source implements CIFAR/STL/Alzheimer recipes.  These controlled PU-Bench
variants keep the MetaModule API required by LaGAM's inner update while matching
the benchmark's dataset shapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .meta_layers import *


class MetaCNN_CIFAR10(MetaModule):
    """Customized CNN for CIFAR-10, adapted with MetaLayers for LaGAM."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    MetaConv2d(3, 96, kernel_size=3, padding=1),
                    MetaBatchNorm2d(96),
                    nn.ReLU(inplace=True),
                    MetaConv2d(96, 96, kernel_size=3, padding=1),
                    MetaBatchNorm2d(96),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    MetaConv2d(96, 96, kernel_size=3, padding=1, stride=2),
                    MetaBatchNorm2d(96),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    MetaConv2d(96, 192, kernel_size=3, padding=1),
                    MetaBatchNorm2d(192),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    MetaConv2d(192, 192, kernel_size=3, padding=1),
                    MetaBatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    MetaConv2d(192, 192, kernel_size=3, padding=1, stride=2),
                    MetaBatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                ),
            ]
        )
        # Main classifier head
        self.classifier_head = nn.Sequential(
            MetaLinear(192 * 8 * 8, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            MetaLinear(1000, 1000),
            nn.ReLU(inplace=True),
        )
        self.final_classifier = MetaLinear(1000, 1)

        # LaGAM's projection head for contrastive learning
        self.projection_head = nn.Sequential(
            MetaLinear(1000, 1000), nn.ReLU(), MetaLinear(1000, 128)
        )

    def forward(self, x: torch.Tensor, flag_feature=False):
        h = x
        for layer_module in self.layers:
            h = layer_module(h)
        h = torch.flatten(h, 1)
        features = self.classifier_head(h)
        logits = self.final_classifier(features).view(-1)

        if flag_feature:
            proj_features = F.normalize(self.projection_head(features), dim=1)
            return logits, proj_features

        return logits


class MetaLeNet(MetaModule):
    """Base LeNet-style CNN for MNIST/FashionMNIST, adapted with MetaLayers for LaGAM."""

    def __init__(self, prior: float = 0.0):
        super().__init__()
        self.prior = prior
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    MetaConv2d(1, 10, kernel_size=5),
                    nn.MaxPool2d(kernel_size=2),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    MetaConv2d(10, 20, kernel_size=5),
                    nn.MaxPool2d(kernel_size=2),
                    nn.ReLU(),
                ),
            ]
        )
        self.classifier_head = nn.Sequential(MetaLinear(320, 50), nn.ReLU())
        self.final_classifier = MetaLinear(50, 1)

        # LaGAM's projection head for contrastive learning
        self.projection_head = nn.Sequential(
            MetaLinear(50, 50), nn.ReLU(), MetaLinear(50, 128)
        )

    def forward(self, x: torch.Tensor, flag_feature=False):
        h = x
        for layer in self.layers:
            h = layer(h)
        h = torch.flatten(h, 1)
        features = self.classifier_head(h)
        logits = self.final_classifier(features).view(-1)

        if flag_feature:
            proj_features = F.normalize(self.projection_head(features), dim=1)
            return logits, proj_features

        return logits


MetaCNN_MNIST = MetaLeNet
MetaCNN_FashionMNIST = MetaLeNet


class MetaCNN_AlzheimerMRI(MetaLeNet):
    """MetaLeNet adapted for Alzheimer MRI (1x128x128) with adjusted input dimensions."""

    def __init__(self, prior: float = 0.0):
        # Replace LeNet style with deeper ADNI specialized structure
        super().__init__(prior)
        # Expected input size
        self.expected_image_size = (128, 128)

        # Deeper convolutional backbone (Meta version)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    MetaConv2d(1, 32, kernel_size=3, padding=1),
                    MetaBatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    MetaConv2d(32, 32, kernel_size=3, padding=1),
                    MetaBatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # 128 -> 64
                    nn.Dropout(0.1),
                ),
                nn.Sequential(
                    MetaConv2d(32, 64, kernel_size=3, padding=1),
                    MetaBatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    MetaConv2d(64, 64, kernel_size=3, padding=1),
                    MetaBatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # 64 -> 32
                    nn.Dropout(0.1),
                ),
                nn.Sequential(
                    MetaConv2d(64, 128, kernel_size=3, padding=1),
                    MetaBatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    MetaConv2d(128, 128, kernel_size=3, padding=1),
                    MetaBatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # 32 -> 16
                    nn.Dropout(0.2),
                ),
                nn.Sequential(
                    MetaConv2d(128, 256, kernel_size=3, padding=1),
                    MetaBatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    MetaConv2d(256, 256, kernel_size=3, padding=1),
                    MetaBatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # 16 -> 8
                    nn.Dropout(0.3),
                ),
            ]
        )

        # After GAP, channels are 256, mapped to 64-dim features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_head = nn.Sequential(
            MetaLinear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )
        self.final_classifier = MetaLinear(64, 1)

        # LaGAM contrastive learning projection head (with 64-dim input)
        self.projection_head = nn.Sequential(
            MetaLinear(64, 64), nn.ReLU(), MetaLinear(64, 128)
        )

    def forward(self, x: torch.Tensor, flag_feature=False):
        h = x
        for layer in self.layers:
            h = layer(h)
        h = self.gap(h)
        h = torch.flatten(h, 1)  # (B, 256)
        features = self.classifier_head(h)  # (B, 64)
        logits = self.final_classifier(features).view(-1)

        if flag_feature:
            proj_features = F.normalize(self.projection_head(features), dim=1)
            return logits, proj_features

        return logits


class MetaDynamicMLP(MetaModule):
    """Meta MLP for PU-Bench vector/text/tabular LaGAM runs.

    This is a benchmark adaptation, not an official LaGAM source backbone.  It
    keeps the source-required MetaModule API so the inner label-correction update
    remains differentiable with respect to the perturbation ``eps``.
    """

    def __init__(self, prior: float = 0.0, hidden_dim: int = 300):
        super().__init__()
        self.prior = prior
        self.hidden_dim = int(hidden_dim)
        self.input_dim: int | None = None
        self.classifier_head = None
        self.final_classifier = None
        self.projection_head = None

    def _build(self, input_dim: int, device: torch.device) -> None:
        self.input_dim = int(input_dim)
        self.classifier_head = nn.Sequential(
            MetaLinear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            MetaLinear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.final_classifier = MetaLinear(self.hidden_dim, 1)
        self.projection_head = nn.Sequential(
            MetaLinear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            MetaLinear(self.hidden_dim, 128),
        )
        self.to(device)

    def forward(self, x: torch.Tensor, flag_feature=False):
        h = torch.flatten(x.float(), 1)
        if self.classifier_head is None:
            self._build(h.shape[1], h.device)
        features = self.classifier_head(h)
        logits = self.final_classifier(features).view(-1)

        if flag_feature:
            proj_features = F.normalize(self.projection_head(features), dim=1)
            return logits, proj_features

        return logits


__all__ = [
    "MetaCNN_CIFAR10",
    "MetaCNN_MNIST",
    "MetaCNN_FashionMNIST",
    "MetaCNN_AlzheimerMRI",
    "MetaDynamicMLP",
    "MetaLeNet",
]
