"""base_trainer.py

Common Trainer base class that encapsulates data loading, model initialization,
evaluation and logging workflows. Each PU method should inherit from this class
and only implement method-specific loss and training loop logic.
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from abc import ABC, abstractmethod

import torch
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from .train_utils import (
    evaluate_metrics,
    prepare_loaders,
    select_model,
    set_global_seed,
    seed_worker,
    ModelCheckpoint,
)
from data.lagam_dataset import LaGAMDatasetWrapper
from data.vector_augment import (
    VectorAugPUDatasetWrapper,
    VectorWeakAugment,
    VectorStrongAugment,
)


class BaseTrainer(ABC):
    """Base class for PU learning trainers."""

    def __init__(self, method: str, experiment: str, params: dict):
        self.method = method
        self.experiment_name = experiment
        self.params = params

        self.console = Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Global random seed
        set_global_seed(self.params.get("seed", 42))

        # Can be overridden in subclass run()
        self.file_console = None
        self.checkpoint_handler = None

        # Data & model preparation
        self._prepare_data()
        self._build_model()

        # Per-seed results and log directories
        seed_value = self.params.get("seed", 42)
        self.results_root = os.path.join("results", f"seed_{seed_value}")
        self.log_dir = os.path.join(self.results_root, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Checkpoint handler
        self.checkpoint_handler = None
        self._init_checkpoint_handler()

        # Global epoch counter
        self.global_epoch = 0

        # Run bookkeeping
        self._run_start_time = None
        self._run_end_time = None
        self._max_gpu_mem_bytes = 0

    # Abstract / overridable interfaces
    @abstractmethod
    def create_criterion(self):
        """Return loss function (or callable object)"""
        raise NotImplementedError

    @abstractmethod
    def train_one_epoch(self, epoch_idx: int):
        """Execute training for one epoch. Implemented by subclasses."""
        raise NotImplementedError

    # Optional hooks (overridden by subclasses as needed)
    def before_training(self):
        """Called before training starts, for additional initialization"""
        self._init_file_console()
        # Mark start time and reset CUDA peak memory stats
        self._run_start_time = time.time()
        if torch.cuda.is_available():
            try:
                # Reset peak stats on all visible devices
                for dev_idx in range(torch.cuda.device_count()):
                    with torch.cuda.device(dev_idx):
                        torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def _init_file_console(self):
        """Initialize file console for logging output to both console and file."""
        # Create log file path with method_experiment name
        log_file_name = f"{self.method}_{self.experiment_name}.log"
        log_file_path = os.path.join(self.log_dir, log_file_name)

        # Initialize file console
        self.file_console = Console(
            file=open(log_file_path, "w", encoding="utf-8"), width=120
        )

        # Log basic training start information
        self.file_console.log(f"=" * 80)
        self.file_console.log(f"{self.method.upper()} Training Started")
        self.file_console.log(f"Method: {self.method}")
        self.file_console.log(f"Experiment: {self.experiment_name}")
        self.file_console.log(f"Device: {self.device}")
        self.file_console.log(
            f"Training epochs: {self.params.get('epochs', self.params.get('num_epochs', 'N/A'))}"
        )
        self.file_console.log(f"Batch size: {self.params.get('batch_size', 128)}")
        self.file_console.log(f"Learning rate: {self.params.get('lr', 1e-3)}")
        self.file_console.log(f"Random seed: {self.params.get('seed', 42)}")
        # --- record all hyperparameters ---
        try:
            import yaml as _yaml

            hyper_yaml = _yaml.safe_dump(
                self.params, allow_unicode=True, sort_keys=False
            )
            self.file_console.log("[bold]All Hyperparameters (YAML):[/bold]")
            self.file_console.log(hyper_yaml)
        except Exception as _e:
            # fallback: print key-value pairs line by line
            self.file_console.log("[bold]All Hyperparameters:[/bold]")
            for k, v in self.params.items():
                self.file_console.log(f"  {k}: {v}")

        self.file_console.log(f"=" * 80)

        # Update checkpoint handler's file_console reference
        if self.checkpoint_handler:
            self.checkpoint_handler.file_console = self.file_console

        self.console.log(f"Log file will be saved to: [cyan]{log_file_path}[/cyan]")

    def after_training(self):
        """Called after training ends"""
        # Persist unified result.json with summary metrics and timing/memory info
        try:
            self._run_end_time = time.time()
            if torch.cuda.is_available():
                try:
                    max_list = []
                    for dev_idx in range(torch.cuda.device_count()):
                        with torch.cuda.device(dev_idx):
                            max_list.append(torch.cuda.max_memory_allocated())
                    self._max_gpu_mem_bytes = int(max(max_list) if max_list else 0)
                except Exception:
                    self._max_gpu_mem_bytes = 0
            else:
                self._max_gpu_mem_bytes = 0

            result = self._compose_result_summary()
            # Write per-experiment JSON into per-seed directory:
            # results/seed_{seed}/{experiment}.json, merging runs by method
            root_dir = self.results_root
            os.makedirs(root_dir, exist_ok=True)
            exp_path = os.path.join(root_dir, f"{self.experiment_name}.json")

            merged = {
                "experiment": self.experiment_name,
                "updated_at": None,
                "runs": {},
            }
            if os.path.exists(exp_path):
                try:
                    with open(exp_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            merged.update(
                                {k: v for k, v in loaded.items() if k != "runs"}
                            )
                            if isinstance(loaded.get("runs"), dict):
                                merged["runs"].update(loaded["runs"])
                except Exception:
                    # ignore malformed existing file
                    pass

            merged["runs"][self.method] = result
            merged["updated_at"] = datetime.utcnow().isoformat() + "Z"

            with open(exp_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)

            if self.file_console:
                self.file_console.log(f"Saved/updated experiment summary: {exp_path}")
        except Exception as _e:
            # Avoid breaking training teardown due to logging
            if self.file_console:
                self.file_console.log(f"Failed to write result.json: {_e}")

    # File console closing
    def _close_file_console(self):
        """Safely close file_console"""
        if (
            hasattr(self, "file_console")
            and self.file_console
            and hasattr(self.file_console, "file")
            and not self.file_console.file.closed
        ):
            try:
                self.file_console.file.close()
            except Exception:
                pass

    # Common implementations
    def _prepare_data(self):
        (
            self.train_loader,
            self.validation_loader,
            self.test_loader,
            self.prior,
            self.update_loader,
        ) = prepare_loaders(
            dataset_name=self.experiment_name,
            data_config=self.params,  # entire params is data_config
            batch_size=self.params.get("batch_size", 128),
            data_dir=self.params.get("data_dir", "data"),
            method=self.method,
        )

        # LaGAM requires a special dataset wrapper for strong/weak augmentations (images only)
        ds_cls = str(self.params.get("dataset_class", "")).lower()
        if self.method.lower() == "lagam" and any(
            token in ds_cls
            for token in ["cifar", "mnist", "fashionmnist", "alzheimer", "mri"]
        ):
            # Configure image size and normalization per dataset
            if "mnist" in ds_cls or "fashionmnist" in ds_cls:
                image_size = 28
                mean = getattr(self.train_loader.dataset, "mean", (0.5,))
                std = getattr(self.train_loader.dataset, "std", (0.5,))
            elif "alzheimer" in ds_cls or "mri" in ds_cls:
                image_size = 128
                mean = getattr(self.train_loader.dataset, "mean", (0.5,))
                std = getattr(self.train_loader.dataset, "std", (0.5,))
            else:
                image_size = 32
                mean = getattr(self.train_loader.dataset, "mean", (0.5, 0.5, 0.5))
                std = getattr(self.train_loader.dataset, "std", (0.5, 0.5, 0.5))

            wrapped_dataset = LaGAMDatasetWrapper(
                self.train_loader.dataset, image_size=image_size, mean=mean, std=std
            )
            self.train_loader = torch.utils.data.DataLoader(
                wrapped_dataset,
                batch_size=self.params.get("batch_size", 128),
                shuffle=True,
                num_workers=self.params.get("num_workers", 4),
                pin_memory=True,
                worker_init_fn=seed_worker,
            )
        elif self.method.lower() == "lagam":
            # Non-image datasets: provide vector strong/weak augmentations for LaGAM
            self.console.log(
                "LaGAM detected non-image dataset; enabling vector weak/strong augmentations.",
                style="yellow",
            )
            base_ds = self.train_loader.dataset
            weak = VectorWeakAugment(
                noise_std=float(self.params.get("vec_weak_noise_std", 0.02)),
                dropout_ratio=float(self.params.get("vec_weak_dropout", 0.0)),
            )
            strong = VectorStrongAugment(
                noise_std=float(self.params.get("vec_strong_noise_std", 0.1)),
                dropout_ratio=float(self.params.get("vec_strong_dropout", 0.1)),
                sign_flip_ratio=float(self.params.get("vec_sign_flip_ratio", 0.05)),
            )
            wrapped_dataset = VectorAugPUDatasetWrapper(
                base_dataset=base_ds, weak_aug=weak, strong_aug=strong
            )
            self.train_loader = torch.utils.data.DataLoader(
                wrapped_dataset,
                batch_size=self.params.get("batch_size", 128),
                shuffle=True,
                num_workers=self.params.get("num_workers", 4),
                pin_memory=True,
                worker_init_fn=seed_worker,
            )

        # Cache input shape
        # We need to handle the tuple case for wrapped datasets
        sample_data = next(iter(self.train_loader))[0]
        if isinstance(sample_data, (list, tuple)):
            self.input_shape = tuple(sample_data[0].shape[1:])
        else:
            self.input_shape = tuple(sample_data.shape[1:])

    def _build_model(self, return_model: bool = False):
        # Model
        model = select_model(
            method=self.method, params=self.params, prior=self.prior
        ).to(self.device)

        # If the selected model is dynamically built on first forward,
        # ensure parameters exist before creating the optimizer by running a dry forward
        try:
            has_params = any(p.requires_grad for p in model.parameters())
        except Exception:
            has_params = False
        if not has_params:
            try:
                sample_batch = next(iter(self.train_loader))
                x_sample = sample_batch[0]
                # Some datasets may yield tuple/list inputs
                if isinstance(x_sample, (list, tuple)):
                    x_sample = x_sample[0]
                with torch.no_grad():
                    _ = model(x_sample.to(self.device))
            except StopIteration:
                pass

        if return_model:
            return model

        self.model = model
        # Optional: initialize final classifier bias from prior for single-logit heads
        if bool(self.params.get("init_bias_from_prior", True)):
            try:
                import math as _math

                def _logit(_p: float) -> float:
                    eps = 1e-6
                    _p = max(min(float(_p), 1 - eps), eps)
                    return _math.log(_p / (1.0 - _p))

                fc = getattr(self.model, "final_classifier", None)
                if (
                    isinstance(fc, torch.nn.Linear)
                    and getattr(fc, "bias", None) is not None
                ):
                    if int(getattr(fc, "out_features", 0)) == 1:
                        with torch.no_grad():
                            fc.bias.fill_(_logit(self.prior))
            except Exception:
                pass
        # Optimizer
        lr = self.params.get("lr", 1e-3)
        wd = self.params.get("weight_decay", 5e-4)

        # For MetaModule, use the custom .params() method
        model_params = (
            self.model.params()
            if hasattr(self.model, "params")
            else self.model.parameters()
        )

        self.optimizer = torch.optim.Adam(model_params, lr=lr, weight_decay=wd)

        # Loss function
        self.criterion = self.create_criterion()

    # Common epoch runner
    def _run_epochs(self, num_epochs: int, stage_name: str = "Training"):
        test_metrics = {}  # Initialize with a default value

        for epoch_idx in tqdm(
            range(1, num_epochs + 1), desc=f"{stage_name} ({self.method.upper()})"
        ):
            self.global_epoch += 1  # Use accumulator for multi-stage training
            self.train_one_epoch(epoch_idx)

            # Evaluation
            train_metrics = evaluate_metrics(
                self.model, self.train_loader, self.device, self.prior
            )
            test_metrics = evaluate_metrics(
                self.model, self.test_loader, self.device, self.prior
            )
            val_metrics = (
                evaluate_metrics(
                    self.model, self.validation_loader, self.device, self.prior
                )
                if self.validation_loader is not None
                else None
            )

            # Rich table (optionally silence early epochs, e.g., to hide warmup with trivial F1)
            silence_before = self.params.get("silence_metrics_before_epoch", 0)
            if (
                self.global_epoch % self.params.get("log_interval", 1) == 0
                and self.global_epoch >= silence_before
            ):
                self._print_metrics(
                    epoch_idx,
                    num_epochs,
                    train_metrics,
                    test_metrics,
                    stage_name,
                    val_metrics=val_metrics,
                )

            # Call checkpoint
            if self.checkpoint_handler:
                all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
                if val_metrics is not None:
                    all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                # Automatic fallback for monitoring metrics: if val_* doesn't exist, fallback to test_*; then to train_*
                try:
                    monitor = getattr(self.checkpoint_handler, "monitor", None)
                    if monitor and monitor not in all_metrics:

                        def _fallback_key(key: str, prefix: str):
                            return prefix + key.split("_", 1)[1] if "_" in key else None

                        # Try test_*
                        test_key = _fallback_key(monitor, "test_")
                        train_key = _fallback_key(monitor, "train_")
                        if test_key and test_key in all_metrics:
                            # Temporarily copy all_metrics[monitor]
                            all_metrics[monitor] = all_metrics[test_key]
                        elif train_key and train_key in all_metrics:
                            all_metrics[monitor] = all_metrics[train_key]
                        # Otherwise keep original logic, enter internal warning
                except Exception:
                    pass

                self.checkpoint_handler(
                    epoch=self.global_epoch,
                    all_metrics=all_metrics,
                    model=self.model,
                    elapsed_seconds=(
                        (time.time() - self._run_start_time)
                        if self._run_start_time
                        else None
                    ),
                )

            # Early stopping check
            if self.checkpoint_handler and self.checkpoint_handler.should_stop:
                self.console.log(
                    f"Early stopping in stage '{stage_name}'.", style="bold red"
                )
                break

        return test_metrics

    def _print_metrics(
        self,
        epoch_idx: int,
        num_epochs: int,
        train_metrics: dict,
        test_metrics: dict,
        stage_name: str,
        val_metrics: dict | None = None,
    ):
        table = Table(
            title=f"Stage: {stage_name} - Epoch {epoch_idx}/{num_epochs} - {self.method.upper()}"
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Train", style="magenta")
        if val_metrics is not None:
            table.add_column("Val", style="yellow")
        table.add_column("Test", style="green")

        for metric in ["accuracy", "error", "f1", "precision", "recall", "auc", "risk"]:
            if metric in train_metrics and metric in test_metrics:
                if val_metrics is not None and metric in val_metrics:
                    table.add_row(
                        metric,
                        f"{train_metrics[metric]:.4f}",
                        f"{val_metrics[metric]:.4f}",
                        f"{test_metrics[metric]:.4f}",
                    )
                else:
                    table.add_row(
                        metric,
                        f"{train_metrics[metric]:.4f}",
                        f"{test_metrics[metric]:.4f}",
                    )

        self.console.print(table)
        if self.file_console:
            self.file_console.print(table)

    # ---------------- Result composition -----------------
    def _compose_result_summary(self) -> dict:
        """Assemble result.json content."""
        start_iso = (
            datetime.fromtimestamp(self._run_start_time).isoformat()
            if self._run_start_time
            else None
        )
        end_iso = (
            datetime.fromtimestamp(self._run_end_time).isoformat()
            if self._run_end_time
            else None
        )
        # Prefer time-to-best for monitored metric if available
        if self.checkpoint_handler and hasattr(
            self.checkpoint_handler, "best_elapsed_seconds"
        ):
            duration_sec = self.checkpoint_handler.best_elapsed_seconds
        else:
            duration_sec = (
                float(self._run_end_time - self._run_start_time)
                if (self._run_start_time is not None and self._run_end_time is not None)
                else None
            )

        dataset_info = self._collect_dataset_stats()

        best = None
        monitor = None
        if self.checkpoint_handler and getattr(
            self.checkpoint_handler, "best_metrics", None
        ):
            best = {
                "epoch": int(self.checkpoint_handler.best_epoch),
                "metrics": self.checkpoint_handler.best_metrics,
            }
            monitor = self.checkpoint_handler.monitor
        else:
            # Fallback: if no checkpoint is configured/available, record the
            # final epoch's metrics as the "best" to avoid nulls in summaries.
            try:
                train_metrics = evaluate_metrics(
                    self.model, self.train_loader, self.device, self.prior
                )
                test_metrics = evaluate_metrics(
                    self.model, self.test_loader, self.device, self.prior
                )
                merged_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                merged_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
                best = {"epoch": int(self.global_epoch), "metrics": merged_metrics}
            except Exception:
                # Keep best as None if evaluation fails during teardown
                pass

        result = {
            "method": self.method,
            "experiment": self.experiment_name,
            "device": str(self.device),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "timing": {
                "start": start_iso,
                "end": end_iso,
                "duration_seconds": duration_sec,
            },
            "max_gpu_memory_bytes": self._max_gpu_mem_bytes,
            "dataset": dataset_info,
            "best": best,
            "monitor": monitor,
            "global_epochs": int(self.global_epoch),
            "hyperparameters": self.params,
        }
        return result

    def _collect_dataset_stats(self) -> dict:
        """Gather dataset-level statistics for train/test splits."""

        # Train stats
        def _split_stats(ds):
            try:
                total = len(ds)
                pos = int((ds.true_labels == 1).sum().item())
                neg = int(total - pos)
                return {
                    "total": int(total),
                    "positives": pos,
                    "negatives": neg,
                    "positive_ratio": (pos / total) if total else None,
                }
            except Exception:
                return {"total": len(ds)}

        train_dataset = getattr(self.train_loader, "dataset", None)
        test_dataset = getattr(self.test_loader, "dataset", None)

        train_detail = None
        if train_dataset is not None:
            try:
                pu = train_dataset.pu_labels
                tl = train_dataset.true_labels
                total = len(train_dataset)
                labeled = int((pu == 1).sum().item())
                unlabeled = int((pu == -1).sum().item())
                pos_in_u = int(((tl == 1) & (pu == -1)).sum().item())
                neg_in_u = int(((tl == 0) & (pu == -1)).sum().item())
                total_pos = int((tl == 1).sum().item())
                prior = (total_pos / total) if total else None
                train_detail = {
                    "total": int(total),
                    "labeled": labeled,
                    "unlabeled": unlabeled,
                    "positives_in_unlabeled": pos_in_u,
                    "negatives_in_unlabeled": neg_in_u,
                    "total_positives": total_pos,
                    "prior": prior,
                }
            except Exception:
                train_detail = _split_stats(train_dataset)

        test_detail = _split_stats(test_dataset) if test_dataset is not None else None

        label_scheme = (
            self.params.get("label_scheme", {}) if isinstance(self.params, dict) else {}
        )
        dataset_class = (
            self.params.get("dataset_class") if isinstance(self.params, dict) else None
        )

        return {
            "class": dataset_class,
            "label_scheme": {
                "positive_classes": label_scheme.get("positive_classes"),
                "negative_classes": label_scheme.get("negative_classes"),
            },
            "train": train_detail,
            "test": test_detail,
        }

    # Checkpoint helper
    def _init_checkpoint_handler(self):
        checkpoint_params = self.params.get("checkpoint")
        if checkpoint_params and checkpoint_params.get("enabled", False):
            save_dir = checkpoint_params.get("save_dir", "checkpoints")
            filename = f"{self.method}_{self.experiment_name}.pth"
            early_stopping_params = checkpoint_params.get("early_stopping")
            self.checkpoint_handler = ModelCheckpoint(
                save_dir=save_dir,
                filename=filename,
                monitor=checkpoint_params.get("monitor", "test_f1"),
                mode=checkpoint_params.get("mode", "max"),
                save_model=checkpoint_params.get("save_model", True),
                verbose=checkpoint_params.get("verbose", True),
                file_console=self.file_console,
                early_stopping_params=early_stopping_params,
            )

    # Public entry
    def run(self):
        """Training entry point. Subclasses can override for multi-stage workflows"""
        self.before_training()
        final_metrics = self._run_epochs(self.params.get("num_epochs", 1))
        self.after_training()

        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()
        # Close file console after logging completion
        self._close_file_console()
        return final_metrics
