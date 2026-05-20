"""PUL-CPBF-specific weak/strong augmentation.

Source role:
- The image weak/strong pair follows the PUL-CPBF author snapshot under
  ``author source file(s): pulcpbf_author_provided/dataset``.
- RandAugment operators are adapted from that snapshot's
  ``dataset/randaugment.py``.  They stay in this package so PUL-CPBF does not
  borrow wrappers from HolisticPU or shared data preprocessing.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PARAMETER_MAX = 10


def _float_parameter(v: int | float, max_v: float) -> float:
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v: int | float, max_v: float) -> int:
    return int(v * max_v / PARAMETER_MAX)


def AutoContrast(img: Image.Image, **_: object) -> Image.Image:
    return PIL.ImageOps.autocontrast(img)


def Brightness(img: Image.Image, v: int, max_v: float, bias: float = 0) -> Image.Image:
    return PIL.ImageEnhance.Brightness(img).enhance(_float_parameter(v, max_v) + bias)


def Color(img: Image.Image, v: int, max_v: float, bias: float = 0) -> Image.Image:
    return PIL.ImageEnhance.Color(img).enhance(_float_parameter(v, max_v) + bias)


def Contrast(img: Image.Image, v: int, max_v: float, bias: float = 0) -> Image.Image:
    return PIL.ImageEnhance.Contrast(img).enhance(_float_parameter(v, max_v) + bias)


def CutoutAbs(img: Image.Image, v: int, **_: object) -> Image.Image:
    w, h = img.size
    x0 = int(max(0, np.random.uniform(0, w) - v / 2.0))
    y0 = int(max(0, np.random.uniform(0, h) - v / 2.0))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    fill = 127 if len(img.getbands()) == 1 else tuple([127] * len(img.getbands()))
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), fill)
    return img


def Cutout(img: Image.Image, v: int, max_v: float, bias: float = 0) -> Image.Image:
    if v == 0:
        return img
    v_abs = int((_float_parameter(v, max_v) + bias) * min(img.size))
    return CutoutAbs(img, v_abs)


def Equalize(img: Image.Image, **_: object) -> Image.Image:
    return PIL.ImageOps.equalize(img)


def Identity(img: Image.Image, **_: object) -> Image.Image:
    return img


def Invert(img: Image.Image, **_: object) -> Image.Image:
    return PIL.ImageOps.invert(img)


def Posterize(img: Image.Image, v: int, max_v: float, bias: int = 0) -> Image.Image:
    return PIL.ImageOps.posterize(img, _int_parameter(v, max_v) + bias)


def Rotate(img: Image.Image, v: int, max_v: float, bias: int = 0) -> Image.Image:
    degree = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        degree = -degree
    return img.rotate(degree)


def Sharpness(img: Image.Image, v: int, max_v: float, bias: float = 0) -> Image.Image:
    return PIL.ImageEnhance.Sharpness(img).enhance(_float_parameter(v, max_v) + bias)


def ShearX(img: Image.Image, v: int, max_v: float, bias: float = 0) -> Image.Image:
    shear = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        shear = -shear
    return img.transform(img.size, Image.Transform.AFFINE, (1, shear, 0, 0, 1, 0))


def ShearY(img: Image.Image, v: int, max_v: float, bias: float = 0) -> Image.Image:
    shear = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        shear = -shear
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, 0, shear, 1, 0))


def Solarize(img: Image.Image, v: int, max_v: float, bias: int = 0) -> Image.Image:
    threshold = 256 - (_int_parameter(v, max_v) + bias)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img: Image.Image, v: int, max_v: float, bias: float = 0) -> Image.Image:
    shift = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        shift = -shift
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, int(shift * img.size[0]), 0, 1, 0))


def TranslateY(img: Image.Image, v: int, max_v: float, bias: float = 0) -> Image.Image:
    shift = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        shift = -shift
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, 0, 0, 1, int(shift * img.size[1])))


def fixmatch_augment_pool() -> list[tuple[Callable[..., Image.Image], float | None, float | None]]:
    return [
        (AutoContrast, None, None),
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
        (TranslateY, 0.3, 0),
    ]


def simple_augment_pool() -> list[tuple[Callable[..., Image.Image], float | None, float | None]]:
    return [
        (AutoContrast, None, None),
        (Brightness, 1.8, 0.1),
        (Color, 1.8, 0.1),
        (Contrast, 1.8, 0.1),
        (Cutout, 0.2, 0),
        (Equalize, None, None),
        (Invert, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
    ]


class RandAugmentMC:
    """FixMatch-style RandAugment used by the PUL-CPBF image source path."""

    def __init__(self, n: int, m: int, *, simple: bool = False, cutout_size: int = 16):
        if n < 1:
            raise ValueError("n must be >= 1")
        if not (1 <= m <= 10):
            raise ValueError("m must be in [1, 10]")
        self.n = n
        self.m = m
        self.cutout_size = cutout_size
        self.augment_pool = simple_augment_pool() if simple else fixmatch_augment_pool()

    def __call__(self, img: Image.Image) -> Image.Image:
        for op, max_v, bias in random.choices(self.augment_pool, k=self.n):
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        return CutoutAbs(img, self.cutout_size)


def pulcpbf_image_stats(dataset_class: str, fallback_mean, fallback_std):
    """Return source-style image normalization for PU-Bench dataset classes."""

    dataset_class = dataset_class.lower()
    if "cifar" in dataset_class:
        return (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
    if "fashion" in dataset_class:
        return (0.1307,), (0.3081,)
    if "mnist" in dataset_class:
        return (0.5,), (0.5,)
    return tuple(fallback_mean), tuple(fallback_std)


class TransformPULCPBF:
    """Generate weak/strong image pairs following PUL-CPBF source recipes."""

    def __init__(self, mean, std, image_size: int = 32, mode: str = "generic"):
        mode = mode.lower()
        self.mode = mode
        self.image_size = int(image_size)
        padding = 4 if self.image_size <= 32 else max(4, int(self.image_size * 0.125))

        if mode == "alzheimer":
            self.weak = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
            self.strong = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=(1, 3), contrast=(1, 3)),
                    transforms.RandomRotation(degrees=(-30, 30)),
                ]
            )
        else:
            self.weak = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(
                        size=self.image_size,
                        padding=padding,
                        padding_mode="reflect",
                    ),
                ]
            )
            if mode in {"mnist", "fashionmnist"}:
                self.strong = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(
                            size=self.image_size,
                            padding=padding,
                            padding_mode="reflect",
                        ),
                        transforms.RandomAutocontrast(),
                        transforms.RandomEqualize(),
                        transforms.ColorJitter(),
                    ]
                )
            else:
                self.strong = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(
                            size=self.image_size,
                            padding=padding,
                            padding_mode="reflect",
                        ),
                        RandAugmentMC(n=2, m=10, cutout_size=int(self.image_size * 0.5)),
                    ]
                )

        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.normalize(self.weak(x)), self.normalize(self.strong(x))


class TransformPULCPBFEval:
    """Deterministic evaluation transform matching the PUL-CPBF train scale."""

    def __init__(self, mean, std, image_size: int | None = None):
        ops = []
        if image_size is not None:
            ops.append(transforms.Resize(int(image_size)))
        ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        self.transform = transforms.Compose(ops)

    def __call__(self, x: Image.Image) -> torch.Tensor:
        return self.transform(x)


def image_to_pil(x) -> Image.Image:
    """Convert a PU-Bench tensor/array image back to PIL for source augmentations."""

    if isinstance(x, Image.Image):
        return x
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        if np.nanmin(arr) < 0.0:
            arr = (arr + 1.0) / 2.0
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(2)
    return Image.fromarray(arr)


def infer_image_profile(base_dataset: Dataset, params: dict) -> tuple[tuple[float, ...], tuple[float, ...], int, str]:
    dataset_class = str(params.get("dataset_class", "")).lower()
    mean, std = pulcpbf_image_stats(
        dataset_class,
        getattr(base_dataset, "mean", (0.5,)),
        getattr(base_dataset, "std", (0.5,)),
    )

    try:
        sample = base_dataset[0][0]
        if isinstance(sample, torch.Tensor) and sample.dim() >= 3:
            image_size = int(sample.shape[-1])
        else:
            image_size = int(getattr(base_dataset, "image_size", 32))
    except Exception:
        image_size = int(getattr(base_dataset, "image_size", 32))

    if "alzheimer" in dataset_class or "mri" in dataset_class:
        return mean, std, int(getattr(base_dataset, "image_size", image_size)), "alzheimer"
    if "fashion" in dataset_class:
        return mean, std, image_size, "fashionmnist"
    if "mnist" in dataset_class:
        return mean, std, image_size, "mnist"
    return mean, std, image_size, "generic"


class PULCPBFDatasetWrapper(Dataset):
    """Attach PUL-CPBF weak/strong image pairs to a PU-Bench dataset."""

    def __init__(self, base_dataset: Dataset, transform: Callable[[Image.Image], tuple[torch.Tensor, torch.Tensor]]):
        self.base_dataset = base_dataset
        self.transform = transform
        self.data = getattr(base_dataset, "data", None)
        self.targets = getattr(base_dataset, "targets", None)
        self.pu_labels = getattr(base_dataset, "pu_labels", None)
        self.true_labels = getattr(base_dataset, "true_labels", None)
        self.indices = getattr(base_dataset, "indices", None)
        self.pseudo_labels = getattr(base_dataset, "pseudo_labels", None)
        self.pu_metadata = getattr(base_dataset, "pu_metadata", None)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        img, target, y_true, idx, u_true = self.base_dataset[index]
        return self.transform(image_to_pil(img)), target, y_true, idx, u_true


class PULCPBFEvalDatasetWrapper(Dataset):
    """Apply deterministic PUL-CPBF image normalization for evaluation loaders."""

    def __init__(self, base_dataset: Dataset, transform: Callable[[Image.Image], torch.Tensor]):
        self.base_dataset = base_dataset
        self.transform = transform
        self.data = getattr(base_dataset, "data", None)
        self.targets = getattr(base_dataset, "targets", None)
        self.pu_labels = getattr(base_dataset, "pu_labels", None)
        self.true_labels = getattr(base_dataset, "true_labels", None)
        self.indices = getattr(base_dataset, "indices", None)
        self.pseudo_labels = getattr(base_dataset, "pseudo_labels", None)
        self.pu_metadata = getattr(base_dataset, "pu_metadata", None)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        img, target, y_true, idx, u_true = self.base_dataset[index]
        return self.transform(image_to_pil(img)), target, y_true, idx, u_true
