from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.models import CNN_CIFAR10, CNN_MNIST


def _infer_mlp_dims(flat_dim: int) -> list[int]:
    """Choose reasonable MLP hidden sizes based on flat input dimension.

    Keeps sizes comparable across datasets: scales down but not too small.
    """
    # Upper/lower bounds to keep models comparable across datasets
    base = max(256, min(1024, int(2 ** math.ceil(math.log2(max(128, flat_dim // 2))))))
    dims = [base, max(128, base // 2), max(64, base // 4)]
    return dims


class CGenPUDiscriminator(nn.Module):
    """Simple MLP discriminator operating on flattened inputs.

    Outputs a single logit for real/fake discrimination.
    """

    def __init__(self, input_shape: tuple[int, ...]):
        super().__init__()
        self.input_shape = input_shape
        flat_dim = 1
        for d in input_shape:
            flat_dim *= int(d)
        hidden = _infer_mlp_dims(flat_dim)
        layers: list[nn.Module] = []
        last = flat_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, 1)]  # logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        else:
            # Ensure float tensor and device consistency
            if not torch.is_floating_point(x):
                x = x.float()
        logits = self.net(x).view(-1)
        return torch.sigmoid(
            logits
        )  # Return probabilities to match original TF implementation


class CGenPUGenerator(nn.Module):
    """Simple MLP generator producing flattened outputs.

    Input: concatenated noise+class one-hot of shape (B, latent_dim + 2).
    Output: feature tensor shaped like input_shape, activation configurable: 'sigmoid'|'tanh'|'identity'.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        latent_plus_class_dim: int,
        out_activation: str = "sigmoid",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.out_activation = (
            out_activation.lower() if isinstance(out_activation, str) else "sigmoid"
        )
        flat_dim = 1
        for d in input_shape:
            flat_dim *= int(d)
        hidden = _infer_mlp_dims(flat_dim)

        layers: list[nn.Module] = []
        last = latent_plus_class_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, flat_dim)]
        if self.out_activation == "sigmoid":
            layers += [nn.Sigmoid()]
        elif self.out_activation == "tanh":
            layers += [nn.Tanh()]
        else:
            layers += [nn.Identity()]
        self.net = nn.Sequential(*layers)

    def forward(self, z_with_class: torch.Tensor) -> torch.Tensor:
        flat = self.net(z_with_class)
        return flat.view(-1, *self.input_shape)


# ---------------------------- CNN backbones for images ----------------------------


class CGenPUConvDiscriminatorCIFAR(nn.Module):
    """Conv discriminator for 3x32x32 images (CIFAR-like)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),  # 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).view(-1)
        return torch.sigmoid(
            logits
        )  # Return probabilities to match original TF implementation


class CGenPUConvGeneratorCIFAR(nn.Module):
    """Conv generator for 3x32x32 images with Sigmoid output (range [0,1])."""

    def __init__(self, latent_plus_class_dim: int, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_plus_class_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),  # 32x32
            nn.Sigmoid(),
        )

    def forward(self, z_with_class: torch.Tensor) -> torch.Tensor:
        h = self.fc(z_with_class)
        h = h.view(-1, 256, 4, 4)
        img = self.deconv(h)
        return img


class CGenPUConvDiscriminatorMNIST(nn.Module):
    """Conv discriminator for 1x28x28 images (MNIST/FashionMNIST-like)."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),  # 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),  # 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).view(-1)
        return torch.sigmoid(
            logits
        )  # Return probabilities to match original TF implementation


class CGenPUConvGeneratorMNIST(nn.Module):
    """Conv generator for 1x28x28 images with Tanh output (range [-1,1])."""

    def __init__(self, latent_plus_class_dim: int, out_channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(latent_plus_class_dim, 256 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.Tanh(),  # MNIST/FashionMNIST in our loaders are normalized to [-1, 1]
        )

    def forward(self, z_with_class: torch.Tensor) -> torch.Tensor:
        h = self.fc(z_with_class)
        h = h.view(-1, 256, 7, 7)
        img = self.deconv(h)
        return img


class AClassifierCNN(nn.Module):
    """Wrap a CNN classifier backbone to output probabilities in [0,1]."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # Propagate expected input size hint to wrapper so evaluation adapters can respect it
        if hasattr(backbone, "expected_image_size"):
            try:
                self.expected_image_size = getattr(backbone, "expected_image_size")
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if logits.dim() > 1 and logits.shape[-1] > 1:
            # If multi-class, use class-1 as positive by convention
            logits = logits[:, 0]
        return torch.sigmoid(logits).view(-1)


class CGenPUConvDiscriminatorADNI(nn.Module):
    """Conv discriminator for 1x128x128 images (AlzheimerMRI/ADNI-like)."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),  # 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).view(-1)
        return torch.sigmoid(logits)


class CGenPUConvGeneratorADNI(nn.Module):
    """Conv generator for 1x128x128 images with Tanh output (range [-1,1])."""

    def __init__(self, latent_plus_class_dim: int, out_channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(latent_plus_class_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),  # 128x128
            nn.Tanh(),  # ADNI/AlzheimerMRI inputs normalized to [-1, 1]
        )

    def forward(self, z_with_class: torch.Tensor) -> torch.Tensor:
        h = self.fc(z_with_class)
        h = h.view(-1, 512, 4, 4)
        img = self.deconv(h)
        return img


__all__ = [
    "CGenPUDiscriminator",
    "CGenPUGenerator",
    "CGenPUConvDiscriminatorCIFAR",
    "CGenPUConvGeneratorCIFAR",
    "CGenPUConvDiscriminatorMNIST",
    "CGenPUConvGeneratorMNIST",
    "CGenPUConvDiscriminatorADNI",
    "CGenPUConvGeneratorADNI",
    "AClassifierCNN",
]
