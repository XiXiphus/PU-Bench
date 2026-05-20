"""LaGAM image augmentation wrapper.

Source snapshot:
    author source file(s): lagam/utils/Dataset.py
    author source file(s): lagam/utils/randaugment.py
    llong-cs/LaGAM at 362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# RandAugment implementation from LaGAM's utils/randaugment.py
# (This is a direct copy to make the component self-contained)

import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):
    return img.rotate(v)


def Sharpness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def CutoutAbs(img, v):
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)
    xy = (x0, y0, x1, y1)
    # Use single-value fill for grayscale images; 3-tuple for RGB
    if isinstance(img, Image.Image):
        mode = img.mode
        if mode in ("L", "I", "F", "1", "P") and (
            img.getbands() and len(img.getbands()) == 1
        ):
            fill_color = 125  # single-channel
        else:
            fill_color = (125, 123, 114)
    else:
        fill_color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fill=fill_color)
    return img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
    ]
    return l


class RandomAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        # Cutout is not part of the random choice in original implementation, but applied after
        cutout_val = random.random() * 0.5 * img.size[0]
        return CutoutAbs(img, cutout_val)


def _initial_lagam_targets(base_dataset):
    targets = getattr(base_dataset, "pu_labels")
    targets = torch.as_tensor(targets).float().clone()
    return torch.where(targets == 1, torch.ones_like(targets), torch.zeros_like(targets))


class LaGAMDatasetWrapper(Dataset):
    def __init__(self, base_dataset, image_size=32, mean=None, std=None):
        self.base_dataset = base_dataset
        self.lagam_targets = _initial_lagam_targets(base_dataset)
        self.image_size = image_size
        self.mean = mean or (0.5, 0.5, 0.5)
        self.std = std or (0.5, 0.5, 0.5)

        self.weak_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=self.image_size, padding=int(self.image_size * 0.125)
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        strong_transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=self.image_size, padding=int(self.image_size * 0.125)
            ),
            RandomAugment(3, 5),  # n=3, m=5 based on original paper's code
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]
        self.strong_transform = transforms.Compose(strong_transform_list)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # features, pu_labels, true_labels, indices, pseudo_labels
        img_data, pu_label, true_label, idx, pseudo_label = self.base_dataset[index]
        img_pil = _to_pil_image(img_data)
        img_w = self.weak_transform(img_pil)
        img_s = self.strong_transform(img_pil)
        lagam_target = self.lagam_targets[index]

        return (img_w, img_s), lagam_target, true_label, torch.as_tensor(index), pseudo_label

    def update_targets(self, new_labels, idxes):
        idxes = torch.as_tensor(idxes, dtype=torch.long)
        new_labels = torch.as_tensor(new_labels, dtype=torch.float32)
        self.lagam_targets[idxes] = new_labels


class LaGAMVectorDatasetWrapper(Dataset):
    def __init__(self, base_dataset, weak_aug=None, strong_aug=None):
        self.base_dataset = base_dataset
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        self.lagam_targets = _initial_lagam_targets(base_dataset)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, pu_label, true_label, idx, pseudo_label = self.base_dataset[index]
        x_tensor = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        x_w = self.weak_aug(x_tensor) if self.weak_aug is not None else x_tensor
        x_s = self.strong_aug(x_tensor) if self.strong_aug is not None else x_tensor
        lagam_target = self.lagam_targets[index]
        return (x_w, x_s), lagam_target, true_label, torch.as_tensor(index), pseudo_label

    def update_targets(self, new_labels, idxes):
        idxes = torch.as_tensor(idxes, dtype=torch.long)
        new_labels = torch.as_tensor(new_labels, dtype=torch.float32)
        self.lagam_targets[idxes] = new_labels


class LaGAMEvalDatasetWrapper(Dataset):
    """Deterministic eval-transform view used for LaGAM feature clustering."""

    def __init__(self, base_dataset, mean=None, std=None):
        self.base_dataset = base_dataset
        self.mean = mean or (0.5, 0.5, 0.5)
        self.std = std or (0.5, 0.5, 0.5)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img_data, pu_label, true_label, idx, pseudo_label = self.base_dataset[index]
        img_pil = _to_pil_image(img_data)
        return self.transform(img_pil), pu_label, true_label, torch.as_tensor(index), pseudo_label


class LaGAMVectorEvalDatasetWrapper(Dataset):
    """Deterministic vector view for PU-Bench non-image LaGAM clustering."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, pu_label, true_label, idx, pseudo_label = self.base_dataset[index]
        return x, pu_label, true_label, torch.as_tensor(index), pseudo_label


def _to_pil_image(img_data):
    if isinstance(img_data, Image.Image):
        return img_data

    if isinstance(img_data, torch.Tensor):
        img_data = img_data.detach().cpu().numpy()

    if img_data.shape[0] in [1, 3]:
        img_data = np.transpose(img_data, (1, 2, 0))

    if np.issubdtype(img_data.dtype, np.floating):
        # PU-Bench image tensors are not uniform: CIFAR is stored in [0, 1],
        # while MNIST/Alzheimer are standardized to [-1, 1].  LaGAM's
        # augmentation source expects PIL images before applying its own
        # normalization, so undo the [-1, 1] convention here when present.
        if np.nanmin(img_data) < 0.0:
            img_data = (img_data + 1.0) / 2.0
        img_data = np.clip(img_data, 0.0, 1.0)
        img_data = np.rint(img_data * 255).astype(np.uint8)

    if img_data.shape[-1] == 1:
        img_data = img_data.squeeze(-1)

    return Image.fromarray(img_data)
