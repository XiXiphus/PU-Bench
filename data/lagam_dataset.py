import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import copy

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


class LaGAMDatasetWrapper(Dataset):
    def __init__(self, base_dataset, image_size=32, mean=None, std=None):
        self.base_dataset = base_dataset
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

        # Convert to PIL Image if it's a tensor/numpy array
        if not isinstance(img_data, Image.Image):
            if isinstance(img_data, torch.Tensor):
                img_data = img_data.numpy()

            if img_data.shape[0] in [1, 3]:  # C, H, W -> H, W, C
                img_data = np.transpose(img_data, (1, 2, 0))

            if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                img_data = (img_data * 255).astype(np.uint8)

            if img_data.shape[-1] == 1:
                img_data = img_data.squeeze(-1)

            img_pil = Image.fromarray(img_data)
        else:
            img_pil = img_data

        img_w = self.weak_transform(img_pil)
        img_s = self.strong_transform(img_pil)

        return (img_w, img_s), pu_label, true_label, idx, pseudo_label
