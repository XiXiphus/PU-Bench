"""holisticpu_dataset.py

This file provides specialized data transformation and dataset wrapping for HolisticPU method.
Main components:
1. TransformHolisticPU: A data augmentation class that generates a
   weak augmented version and a strong augmented version for each input image.
   This is crucial for applying consistency regularization in the second stage training.
2. RandAugmentMC: Random data augmentation strategy borrowed from FixMatch, used to generate strong augmented images.
"""

import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# =============================================================================
# RandAugment: Strong data augmentation strategy
# Code adapted from https://github.com/google-research/fixmatch
# =============================================================================


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = float(v) * max_v / 10 + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = float(v) * max_v / 10 + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = float(v) * max_v / 10 + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Posterize(img, v, max_v, bias=0):
    v = int(v * max_v / 10) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = int(v * max_v / 10) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = float(v) * max_v / 10 + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = float(v) * max_v / 10 + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = float(v) * max_v / 10 + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = int(v * max_v / 10) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def TranslateX(img, v, max_v, bias=0):
    v = float(v) * max_v / 10 + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = float(v) * max_v / 10 + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)

    # Determine appropriate fill color format based on image channel count:
    # - Single channel (grayscale) images need int or tuple of length 1
    # - Multi-channel (RGB/RGBA etc.) images use tuple of length matching channel count
    if len(img.getbands()) == 1:  # Grayscale image ("L" etc.)
        fill_color = 127
    else:
        fill_color = tuple([127] * len(img.getbands()))

    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fill_color)
    return img


def fixmatch_augment_pool():
    # fmt: off
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    # fmt: on
    return augs


class RandAugmentMC:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        # Default Cutout, consistent with original implementation
        img = CutoutAbs(img, int(32 * 0.5))
        return img


# =============================================================================
# TransformHolisticPU: Generate (weak, strong) data pairs
# =============================================================================


class TransformHolisticPU:
    """Transformer that generates weak and strong augmented data pairs for HolisticPU."""

    def __init__(self, mean, std, image_size=32):
        padding = int(image_size * 0.125)
        # Weak augmentation: standard flip and crop
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=image_size, padding=padding, padding_mode="reflect"
                ),
                transforms.Resize((image_size, image_size)),
            ]
        )
        # Strong augmentation: add RandAugment on top of weak augmentation
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=image_size, padding=padding, padding_mode="reflect"
                ),
                transforms.Resize((image_size, image_size)),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak_aug = self.weak(x)
        strong_aug = self.strong(x)
        return self.normalize(weak_aug), self.normalize(strong_aug)


# =============================================================================
# HolisticPUDatasetWrapper: Dataset wrapper
# =============================================================================


class HolisticPUDatasetWrapper(Dataset):
    """
    A wrapper that converts standard datasets to the format required by HolisticPU.
    It applies TransformHolisticPU to generate strong/weak data pairs.
    """

    def __init__(self, base_dataset: Dataset, transform: callable):
        self.base_dataset = base_dataset
        self.transform = transform

        # Inherit base dataset attributes for easy access
        self.data = getattr(base_dataset, "data", None)
        self.targets = getattr(base_dataset, "targets", None)
        self.pu_labels = getattr(base_dataset, "pu_labels", None)
        self.true_labels = getattr(base_dataset, "true_labels", None)
        self.indices = getattr(base_dataset, "indices", None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # Get raw data from base dataset
        img, target, y_true, idx, u_true = self.base_dataset[index]

        # Ensure img is PIL Image format
        if not isinstance(img, Image.Image):
            # Assume it's a tensor or numpy array of (C, H, W) or (H, W, C)
            # Convert to PIL Image
            if hasattr(img, "numpy"):
                img = img.numpy()
            if img.shape[0] in [1, 3]:  # C, H, W
                img = np.transpose(img, (1, 2, 0))
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            if img.shape[2] == 1:
                img = img.squeeze(2)

            img = Image.fromarray(img)

        # Apply strong/weak data augmentation
        # Return (x_w, x_s) instead of x
        img_transformed = self.transform(img)

        return img_transformed, target, y_true, idx, u_true
