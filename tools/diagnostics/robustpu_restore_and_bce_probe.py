"""RobustPU pretrain-restore and first weighted-BCE diagnostics.

This script is intentionally outside the trainer path. It reproduces a selected
PU-Bench RobustPU run from config, tracks the source-style pretrain best state,
prints score/logit/weight summaries, and then runs one weighted BCE epoch to
show whether logits collapse immediately.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.experiment_plan import build_plan
from config.method_loader import DEFAULT_METHODS_DIR, list_available_methods
from train.registry import import_trainer_class
from train.robustpu.losses import hardness_values
from train.robustpu.spl import calculate_spl_weights


def _fmt(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: float("nan") for key in ("q00", "q10", "q25", "q50", "q75", "q90", "q100")}
    qs = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "q00": float(qs[0]),
        "q10": float(qs[1]),
        "q25": float(qs[2]),
        "q50": float(qs[3]),
        "q75": float(qs[4]),
        "q90": float(qs[5]),
        "q100": float(qs[6]),
    }


def _array_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "n": 0.0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "n": float(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _print_stats(prefix: str, stats: dict[str, float]) -> None:
    items = " ".join(f"{key}={_fmt(float(value))}" for key, value in stats.items())
    print(f"{prefix} {items}")


def _state_max_abs_diff(current: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> float:
    max_diff = 0.0
    for key, expected_value in expected.items():
        current_value = current[key].detach().cpu()
        expected_value = expected_value.detach().cpu()
        if torch.is_floating_point(current_value):
            diff = torch.max(torch.abs(current_value - expected_value)).item()
        else:
            diff = torch.max((current_value != expected_value).float()).item()
        max_diff = max(max_diff, float(diff))
    return max_diff


def _validation_accuracy_threshold(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(logits)
    sorted_logits = logits[order]
    candidates = [float(sorted_logits[0] - 1.0)]
    candidates.extend(
        float((sorted_logits[idx - 1] + sorted_logits[idx]) * 0.5)
        for idx in range(1, sorted_logits.size)
    )
    candidates.append(float(sorted_logits[-1] + 1.0))
    best_threshold = candidates[0]
    best_accuracy = -1.0
    for threshold in candidates:
        pred = (logits > threshold).astype(np.int64)
        accuracy = float((pred == labels).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold, best_accuracy


def _last_single_logit_linear(model: torch.nn.Module) -> torch.nn.Linear | None:
    found = None
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Linear)
            and int(getattr(module, "out_features", 0)) == 1
            and getattr(module, "bias", None) is not None
        ):
            found = module
    return found


def _calibrate_bias_with_validation(trainer) -> None:
    if trainer.validation_loader is None:
        raise RuntimeError("Validation calibration requires a validation loader.")
    scores = _collect_scores(trainer, trainer.validation_loader)
    threshold, accuracy = _validation_accuracy_threshold(scores["logits"], scores["true"])
    head = _last_single_logit_linear(trainer.model)
    if head is None:
        raise RuntimeError("Validation calibration requires a single-logit linear head.")
    with torch.no_grad():
        head.bias.add_(torch.as_tensor(-threshold, device=head.bias.device, dtype=head.bias.dtype))
    print(
        "CALIBRATION "
        f"threshold={_fmt(threshold)} bias_delta={_fmt(-threshold)} "
        f"val_oracle_accuracy_at_threshold={_fmt(accuracy)}"
    )


def _find_run(args: argparse.Namespace):
    available_methods = list_available_methods(DEFAULT_METHODS_DIR)
    plan = build_plan(
        dataset_config_path=args.dataset_config,
        methods=["robustpu"],
        methods_dir=DEFAULT_METHODS_DIR,
        available_methods=available_methods,
    )
    matches = [
        run
        for run in plan.runs
        if run.method == "robustpu"
        and int(run.seed) == int(args.seed)
        and abs(float(run.labeled_ratio) - float(args.c)) < 1e-12
        and run.scenario == args.scenario
        and run.selection_strategy == args.selection_strategy
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one run match, got {len(matches)}.")
    return matches[0]


def _prepare_trainer(args: argparse.Namespace):
    run = _find_run(args)
    params = copy.deepcopy(run.params)
    params["num_workers"] = int(args.num_workers)
    if args.backbone_policy is not None:
        params["backbone_policy"] = args.backbone_policy
    if args.disable_checkpoint:
        params["checkpoint"] = {"enabled": False}

    trainer_cls = import_trainer_class(run.trainer_key)
    trainer = trainer_cls(
        method=run.method,
        experiment=run.experiment_name,
        params=params,
    )

    if args.device != "auto":
        trainer.device = torch.device(args.device)
        trainer.model.to(trainer.device)

    trainer._sync_prior_from_pu_metadata()
    cfg = trainer._parse_stage_config()
    if args.pre_epochs is not None:
        cfg = replace(cfg, pre_epochs=int(args.pre_epochs))
    if args.pre_monitor is not None:
        cfg = replace(cfg, pre_monitor=args.pre_monitor)
    trainer.robust_cfg = cfg
    trainer.scheduler_p = trainer._make_scheduler("scheduler_p", default_alpha=0.1)
    trainer.scheduler_n = trainer._make_scheduler("scheduler_n", default_alpha=0.11)
    trainer._moving_weights_by_index = None
    trainer.last_weight_stats = {}
    return trainer, run


def _collect_scores(trainer, loader: DataLoader) -> dict[str, np.ndarray]:
    logits_all: list[torch.Tensor] = []
    pu_all: list[torch.Tensor] = []
    true_all: list[torch.Tensor] = []
    was_training = trainer.model.training
    trainer.model.eval()
    with torch.no_grad():
        for x, pu_labels, true_labels, _indices, _pseudo in loader:
            x = trainer._source_input(x).to(trainer.device)
            logits = trainer._positive_logit(trainer.model(x))
            logits_all.append(logits.detach().cpu())
            pu_all.append(pu_labels.detach().cpu())
            true_all.append(true_labels.detach().cpu())
    trainer.model.train(was_training)
    logits = torch.cat(logits_all).numpy()
    return {
        "logits": logits,
        "probs": 1.0 / (1.0 + np.exp(-logits)),
        "pu": torch.cat(pu_all).numpy(),
        "true": torch.cat(true_all).numpy(),
    }


def _print_score_summary(name: str, scores: dict[str, np.ndarray]) -> None:
    logits = scores["logits"]
    probs = scores["probs"]
    pu = scores["pu"]
    true = scores["true"]
    _print_stats(f"LOGITS split={name}", _array_stats(logits))
    _print_stats(f"SIGMOID split={name}", _array_stats(probs))
    groups = {
        "labeled_p": pu == 1,
        "unlabeled": pu != 1,
        "true_pos": true == 1,
        "true_neg": true == 0,
    }
    group_stats = {
        f"{group}_score_mean": float(probs[mask].mean()) if mask.any() else float("nan")
        for group, mask in groups.items()
    }
    _print_stats(f"SCORE_GROUPS split={name}", group_stats)
    rounded_unique = float(np.unique(np.round(logits, 6)).size)
    _print_stats(
        f"CONSTANT_CHECK split={name}",
        {
            "range": float(logits.max() - logits.min()) if logits.size else float("nan"),
            "std": float(logits.std()) if logits.size else float("nan"),
            "unique_rounded_1e6": rounded_unique,
            "is_constant_1e8": float((logits.max() - logits.min()) < 1e-8) if logits.size else float("nan"),
        },
    )


def _pretrain_with_restore_probe(trainer, *, calibrate_validation_threshold: bool = False):
    cfg = trainer.robust_cfg
    optimizer = trainer._make_source_optimizer(
        cfg.pre_optimizer,
        lr=cfg.pre_lr,
        weight_decay=cfg.pre_weight_decay,
    )
    criterion = trainer._make_pu_loss(cfg.pre_loss)
    pretrain_loader = trainer._make_pretrain_loader()

    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_score = float("-inf")

    print(
        "PRETRAIN_CONFIG "
        f"epochs={cfg.pre_epochs} lr={cfg.pre_lr} wd={cfg.pre_weight_decay} "
        f"batch_size={cfg.pre_batch_size} loss={cfg.pre_loss} "
        f"monitor={cfg.pre_monitor} calibration={cfg.pre_calibration}"
    )

    for epoch_idx in range(1, int(cfg.pre_epochs) + 1):
        trainer.model.train()
        for x, pu_labels, _y_true, _idx, _pseudo in pretrain_loader:
            x = trainer._source_input(x).to(trainer.device)
            pu_labels = pu_labels.to(trainer.device)
            logits = trainer._positive_logit(trainer.model(x))
            loss = trainer._stage_loss(criterion, cfg.pre_loss, logits, pu_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trainer.global_epoch += 1
        train_metrics, val_metrics, test_metrics = trainer._evaluate_current_model()
        score = trainer._internal_selection_score(
            train_metrics,
            val_metrics,
            test_metrics,
            monitor_override=cfg.pre_monitor,
        )
        saved = False
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch_idx
            best_state = copy.deepcopy(trainer.model.state_dict())
            saved = True
        print(
            "PRETRAIN_EPOCH "
            f"epoch={epoch_idx} score={_fmt(float(score))} saved_best={int(saved)} "
            f"best_epoch={best_epoch} best_score={_fmt(best_score)} "
            f"train_proxy_auc={_fmt(float(train_metrics.get('proxy_auc', float('nan'))))} "
            f"val_proxy_auc={_fmt(float((val_metrics or {}).get('proxy_auc', float('nan'))))} "
            f"test_oracle_auc={_fmt(float(test_metrics.get('oracle_auc', float('nan'))))}"
        )

    if best_state is None:
        raise RuntimeError("No pretrain best_state was captured.")

    final_state = copy.deepcopy(trainer.model.state_dict())
    final_to_best = _state_max_abs_diff(final_state, best_state)
    trainer.model.load_state_dict(best_state)
    restored_to_best = _state_max_abs_diff(trainer.model.state_dict(), best_state)
    train_metrics, val_metrics, test_metrics = trainer._evaluate_current_model()
    print(
        "PRETRAIN_RESTORE "
        f"best_epoch={best_epoch} best_score={_fmt(best_score)} "
        f"final_to_best_max_abs={_fmt(final_to_best)} "
        f"restored_to_best_max_abs={_fmt(restored_to_best)} "
        f"restored_train_proxy_auc={_fmt(float(train_metrics.get('proxy_auc', float('nan'))))} "
        f"restored_val_proxy_auc={_fmt(float((val_metrics or {}).get('proxy_auc', float('nan'))))} "
        f"restored_test_oracle_auc={_fmt(float(test_metrics.get('oracle_auc', float('nan'))))}"
    )
    if calibrate_validation_threshold:
        _calibrate_bias_with_validation(trainer)


def _weight_probe(trainer, threshold_p: float, threshold_n: float) -> DataLoader:
    cfg = trainer.robust_cfg
    loader = trainer._make_weight_source_loader()
    logits_all: list[torch.Tensor] = []
    weights_all: list[torch.Tensor] = []
    pu_all: list[torch.Tensor] = []
    true_all: list[torch.Tensor] = []

    was_training = trainer.model.training
    trainer.model.eval()
    with torch.no_grad():
        for x, pu_labels, true_labels, _indices, _pseudo in loader:
            x_dev = trainer._source_input(x).to(trainer.device)
            logits = trainer._positive_logit(trainer.model(x_dev))
            pu_dev = pu_labels.to(trainer.device)
            hardness_n = hardness_values(
                cfg.hardness,
                logits / cfg.temper_n,
                -1,
                gamma=cfg.focal_gamma,
            )
            weights = calculate_spl_weights(
                hardness_n,
                threshold_n,
                spl_type=cfg.spl_type,
                mix2_gamma=float(trainer.params.get("mix2_gamma", 1.0)),
                poly_t=float(trainer.params.get("poly_t", 3.0)),
            )
            pos_mask = pu_dev == 1
            if pos_mask.any():
                hardness_p = hardness_values(
                    cfg.hardness,
                    logits[pos_mask] / cfg.temper_p,
                    1,
                    gamma=cfg.focal_gamma,
                )
                weights[pos_mask] = calculate_spl_weights(
                    hardness_p,
                    threshold_p,
                    spl_type=cfg.spl_type,
                    mix2_gamma=float(trainer.params.get("mix2_gamma", 1.0)),
                    poly_t=float(trainer.params.get("poly_t", 3.0)),
                )
            logits_all.append(logits.detach().cpu())
            weights_all.append(weights.detach().cpu())
            pu_all.append(pu_labels.detach().cpu())
            true_all.append(true_labels.detach().cpu())
    trainer.model.train(was_training)

    logits = torch.cat(logits_all).numpy()
    weights = torch.cat(weights_all).numpy()
    pu = torch.cat(pu_all).numpy()
    true = torch.cat(true_all).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    print(
        "WEIGHT_CONFIG "
        f"threshold_p={_fmt(threshold_p)} threshold_n={_fmt(threshold_n)} "
        f"hardness={cfg.hardness} spl_type={cfg.spl_type} "
        f"temper_p={cfg.temper_p} temper_n={cfg.temper_n}"
    )
    for group, mask in {
        "labeled_p": pu == 1,
        "unlabeled": pu != 1,
        "unlabeled_true_pos": (pu != 1) & (true == 1),
        "unlabeled_true_neg": (pu != 1) & (true == 0),
    }.items():
        values = weights[mask]
        stats = _array_stats(values)
        stats.update(_quantiles(values))
        stats["nonzero_ratio"] = float((values > 0).mean()) if values.size else float("nan")
        stats["score_mean"] = float(probs[mask].mean()) if mask.any() else float("nan")
        _print_stats(f"WEIGHTS group={group}", stats)

    return trainer._create_weighted_dataloader(threshold_p, threshold_n)


def _run_one_weighted_epoch(trainer, weighted_loader: DataLoader) -> None:
    cfg = trainer.robust_cfg
    optimizer = trainer._make_source_optimizer(
        cfg.main_optimizer,
        lr=cfg.main_lr,
        weight_decay=cfg.main_weight_decay,
    )
    criterion = trainer._make_pu_loss(cfg.main_loss)
    losses: list[float] = []
    trainer.model.train()
    for x, pu_labels, _true_labels, weights in weighted_loader:
        x = trainer._source_input(x).to(trainer.device)
        pu_labels = pu_labels.to(trainer.device)
        weights = weights.to(trainer.device)
        if weights.sum().item() <= 1e-8:
            continue
        logits = trainer._positive_logit(trainer.model(x))
        loss = trainer._stage_loss(criterion, cfg.main_loss, logits, pu_labels, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    losses_arr = np.asarray(losses, dtype=np.float64)
    stats = _array_stats(losses_arr)
    print(
        "WEIGHTED_BCE_EPOCH "
        f"batches={len(losses)} lr={cfg.main_lr} batch_size={cfg.main_batch_size} loss={cfg.main_loss}"
    )
    _print_stats("WEIGHTED_BCE_LOSS", stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        default="config/datasets_typical/param_sweep_cifar10.yaml",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--scenario", default="case-control")
    parser.add_argument("--selection-strategy", default="random")
    parser.add_argument("--pre-epochs", type=int, default=None)
    parser.add_argument("--pre-monitor", default=None)
    parser.add_argument("--threshold-p", type=float, default=0.1)
    parser.add_argument("--threshold-n", type=float, default=0.11)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--backbone-policy", choices=("controlled", "private"), default=None)
    parser.add_argument("--calibrate-validation-threshold", action="store_true")
    parser.add_argument("--disable-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer, run = _prepare_trainer(args)
    print(
        "RUN "
        f"experiment={run.experiment_name} method={run.method} "
        f"device={trainer.device} prior={_fmt(float(trainer.prior))}"
    )
    _pretrain_with_restore_probe(
        trainer,
        calibrate_validation_threshold=bool(
            args.calibrate_validation_threshold
            or trainer.robust_cfg.pre_calibration == "val_oracle_accuracy_threshold"
        ),
    )
    train_scores_before = _collect_scores(trainer, trainer.train_loader)
    val_scores_before = (
        _collect_scores(trainer, trainer.validation_loader)
        if trainer.validation_loader is not None
        else None
    )
    test_scores_before = _collect_scores(trainer, trainer.test_loader)
    _print_score_summary("train_after_restore", train_scores_before)
    if val_scores_before is not None:
        _print_score_summary("val_after_restore", val_scores_before)
    _print_score_summary("test_after_restore", test_scores_before)

    weighted_loader = _weight_probe(trainer, args.threshold_p, args.threshold_n)
    _run_one_weighted_epoch(trainer, weighted_loader)

    train_scores_after = _collect_scores(trainer, trainer.train_loader)
    val_scores_after = (
        _collect_scores(trainer, trainer.validation_loader)
        if trainer.validation_loader is not None
        else None
    )
    test_scores_after = _collect_scores(trainer, trainer.test_loader)
    _print_score_summary("train_after_one_weighted_bce", train_scores_after)
    if val_scores_after is not None:
        _print_score_summary("val_after_one_weighted_bce", val_scores_after)
    _print_score_summary("test_after_one_weighted_bce", test_scores_after)


if __name__ == "__main__":
    main()
