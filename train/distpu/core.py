"""Source-aligned Dist-PU helpers.

Reference implementation:
    Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective
    at commit ``cb74be1a87176fd38270873c06374e53905b7354``.

Active source files:
    ``train.py``, ``customized/mixup.py``, ``dataTools/mixupDataset.py`` and
    ``losses/entropyMinimization.py`` under ``author source file(s): distpu``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def distpu_entropy_loss(scores: torch.Tensor) -> torch.Tensor:
    return -torch.mean(scores * torch.log(scores) + (1 - scores) * torch.log(1 - scores))


def mixup_two_targets(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device | str = "cuda",
    is_bias: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_bce(
    scores: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float
) -> torch.Tensor:
    mixup_loss_a = F.binary_cross_entropy(scores, targets_a)
    mixup_loss_b = F.binary_cross_entropy(scores, targets_b)
    return lam * mixup_loss_a + (1 - lam) * mixup_loss_b


class MixupDataset:
    """Maintain Dist-PU pseudo labels indexed by dataset sample id."""

    def __init__(self) -> None:
        self.indexes: torch.Tensor | None = None
        self.psudo_labels: torch.Tensor | None = None

    def update_psudos(
        self, data_loader: DataLoader, model: torch.nn.Module, device: torch.device
    ) -> None:
        self.indexes, self.psudo_labels = _get_predicted_scores(data_loader, model, device)


def _get_predicted_scores(
    data_loader: DataLoader, model: torch.nn.Module, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predicted_scores = []
    indexes = []

    with torch.no_grad():
        for _batch_idx, (x, _t, _y_true, index, _meta) in enumerate(
            tqdm(data_loader, desc="Pseudo-Labeling")
        ):
            x = x.to(device)
            outputs = model(x).squeeze()
            outputs = torch.sigmoid(outputs)
            predicted_scores.append(outputs.cpu())
            indexes.append(index.squeeze().cpu())

    return torch.cat(indexes), torch.cat(predicted_scores)
