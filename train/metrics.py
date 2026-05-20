"""Shared metric evaluation helpers for PU-Bench trainers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


def _dataset_metadata(loader: DataLoader) -> dict:
    """Return PU metadata from a loader dataset or wrapped base dataset."""

    seen: set[int] = set()
    current = getattr(loader, "dataset", None)
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        metadata = getattr(current, "pu_metadata", None)
        if isinstance(metadata, dict):
            return metadata
        metadata = getattr(current, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
        current = getattr(current, "base_dataset", getattr(current, "dataset", None))
    return {}


def _pu_label_values_from_loader(loader: DataLoader) -> tuple[int, int]:
    metadata = _dataset_metadata(loader)
    return (
        int(metadata.get("pu_labeled_label", 1)),
        int(metadata.get("pu_unlabeled_label", -1)),
    )


def _adapt_input_for_model(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Adapt input tensor to match the model's expected channels and spatial size."""
    if not (isinstance(x, torch.Tensor) and x.dim() == 4):
        return x
    exp_c = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            exp_c = int(mod.in_channels)
            break
    if exp_c is None:
        return x
    in_c = x.size(1)
    out = x
    if exp_c == 3 and in_c == 1:
        out = out.repeat(1, 3, 1, 1)
    elif exp_c == 1 and in_c == 3:
        out = out[:, 0:1, ...]
    h, w = out.size(2), out.size(3)
    target_size = None
    if hasattr(m, "expected_image_size"):
        try:
            sz = getattr(m, "expected_image_size")
            if isinstance(sz, (tuple, list)) and len(sz) == 2:
                target_size = (int(sz[0]), int(sz[1]))
        except Exception:
            target_size = None
    if target_size is None:
        if exp_c == 3:
            target_size = (32, 32)
        elif exp_c == 1:
            target_size = (28, 28)
    if target_size is not None and (h != target_size[0] or w != target_size[1]):
        out = F.interpolate(
            out, size=target_size, mode="bilinear", align_corners=False
        )
    return out


def _model_predict(model: nn.Module, x: torch.Tensor, device: torch.device):
    """Run model forward and return (preds_binary, positive_class_score)."""
    x = x.to(device)
    x = _adapt_input_for_model(model, x)
    outputs = model(x)

    if outputs.dim() > 1 and outputs.shape[1] > 1:
        positive_index = int(getattr(model, "positive_logit_index", 1))
        positive_index = max(0, min(positive_index, outputs.shape[1] - 1))
        predicted_class = torch.argmax(outputs, dim=1).long()
        preds_binary = predicted_class.eq(positive_index).long()
        pos_score = torch.softmax(outputs, dim=1)[:, positive_index]
    else:
        raw = outputs.view(-1)
        if torch.all(raw >= 0) and torch.all(raw <= 1):
            preds_binary = (raw >= 0.5).long()
            pos_score = raw
        else:
            preds_binary = (raw > 0).long()
            pos_score = torch.sigmoid(raw)

    return preds_binary, pos_score


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prior: float,
) -> dict[str, float]:
    """Evaluate oracle metrics using true labels on a PU-formatted loader."""
    y_true_all, y_pred_all, y_scores_all = [], [], []

    model.eval()
    with torch.no_grad():
        for x, _t, y_true, _, _ in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]

            preds_binary, pos_score = _model_predict(model, x, device)

            y_pred_all.extend(preds_binary.cpu().numpy())
            y_true_all.extend(y_true.to(device).cpu().numpy())
            y_scores_all.extend(pos_score.detach().cpu().numpy())

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    y_score_arr = np.array(y_scores_all)

    # Prior-calibrated fallback: if predictions collapse to a single class,
    # recalibrate threshold so predicted positive fraction matches prior.
    try:
        if np.unique(y_pred_arr).size == 1:
            n = len(y_score_arr)
            k = int(round(float(prior) * float(n)))
            if 0 < k < n:
                sorted_scores = np.sort(y_score_arr)
                thr = (sorted_scores[n - k] + sorted_scores[n - k - 1]) / 2.0
                y_pred_arr = (y_score_arr >= thr).astype(int)
    except Exception:
        pass

    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)

    try:
        if len(np.unique(y_true_arr)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except Exception:
        auc = float("nan")

    return {
        "oracle_accuracy": acc,
        "oracle_precision": prec,
        "oracle_recall": rec,
        "oracle_f1": f1,
        "oracle_auc": auc,
    }


def evaluate_proxy_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prior: float,
    scenario: str = "single",
) -> dict[str, float]:
    """Compute Proxy Accuracy (PA) and Proxy AUC (PAUC) using PU labels only."""
    correct_p, correct_u, total_p, total_u = 0, 0, 0, 0
    scores_p: list[float] = []
    scores_u: list[float] = []
    labeled_value, unlabeled_value = _pu_label_values_from_loader(loader)

    model.eval()
    with torch.no_grad():
        for x, t, _, _, _ in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]

            preds_binary, pos_score = _model_predict(model, x, device)
            t = t.to(device)

            p_mask = t == labeled_value
            u_mask = t == unlabeled_value

            # PA: P predicted as positive is correct; U predicted as negative is correct.
            if p_mask.any():
                correct_p += preds_binary[p_mask].eq(1).sum().item()
                total_p += p_mask.sum().item()
                scores_p.extend(pos_score[p_mask].cpu().numpy().tolist())

            if u_mask.any():
                correct_u += preds_binary[u_mask].eq(0).sum().item()
                total_u += u_mask.sum().item()
                scores_u.extend(pos_score[u_mask].cpu().numpy().tolist())

    if total_p == 0 or total_u == 0:
        pa = float("nan")
    elif scenario == "case-control":
        pa = 2 * prior * (correct_p / total_p) + (correct_u / total_u)
    else:
        pa = (
            2 * prior * (correct_p / total_p)
            + (correct_p + correct_u) / (total_p + total_u)
        )

    if len(scores_p) == 0 or len(scores_u) == 0:
        pauc = float("nan")
    else:
        try:
            labels = np.concatenate(
                [np.ones(len(scores_p)), np.zeros(len(scores_u))]
            )
            scores = np.array(scores_p + scores_u)
            pauc = float(roc_auc_score(labels, scores))
        except ValueError:
            pauc = 0.5

    return {
        "proxy_acc": pa,
        "proxy_auc": pauc,
    }
