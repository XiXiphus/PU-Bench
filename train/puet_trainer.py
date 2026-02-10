from __future__ import annotations
import os
import json
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from rich.console import Console
from rich.table import Table

from .train_utils import prepare_loaders, set_global_seed
from backbone.puet.trees import PUExtraTrees
from .train_utils import ModelCheckpoint


class PUETTrainer:
    """
    Independent trainer for PUExtraTrees, designed to mimic the interface
    of BaseTrainer for seamless integration into the benchmark runner.
    """

    def __init__(self, method: str, experiment: str, params: dict):
        self.method = method
        self.experiment_name = experiment
        self.params = params

        self.console = Console()
        set_global_seed(self.params.get("seed", 42))

        # Per-seed results and logs directory (align with BaseTrainer)
        seed_value = self.params.get("seed", 42)
        self.results_root = os.path.join("results", f"seed_{seed_value}")
        self.log_dir = os.path.join(self.results_root, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        log_file_name = f"{self.method}_{self.experiment_name}.log"
        log_file_path = os.path.join(self.log_dir, log_file_name)
        self.file_console = Console(
            file=open(log_file_path, "w", encoding="utf-8"), width=120
        )

    def run(self):
        """Main entry point for training and evaluation."""
        self.console.log(
            f"Starting PUExtraTrees run for experiment: {self.experiment_name}"
        )
        self._run_start_time = time.time()

        # 1. Load data
        train_loader, _, test_loader, prior, _ = prepare_loaders(
            dataset_name=self.experiment_name,
            data_config=self.params,
            batch_size=self.params.get(
                "batch_size", 1024
            ),  # Larger batch for non-iterative model
            data_dir=self.params.get("data_dir", "data"),
            method=self.method,
        )

        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset

        # PUExtraTrees works with flat numpy arrays
        X_train = train_dataset.features.reshape(len(train_dataset), -1)
        pu_labels_train = train_dataset.pu_labels
        X_test = test_dataset.features.reshape(len(test_dataset), -1)
        y_test_true = test_dataset.true_labels

        # 2. Instantiate model from params
        model = PUExtraTrees(
            n_estimators=self.params.get("n_estimators", 100),
            risk_estimator=self.params.get("risk_estimator", "nnPU"),
            loss=self.params.get("loss", "quadratic"),
            max_depth=self.params.get("max_depth", None),
            min_samples_leaf=self.params.get("min_samples_leaf", 1),
            max_features=self.params.get("max_features", "sqrt"),
            max_candidates=self.params.get("max_candidates", 1),
            n_jobs=self.params.get("n_jobs", -1),
        )

        # 3. Fit model
        self.console.log("Fitting PUExtraTrees model...", style="yellow")
        P_train = X_train[pu_labels_train == 1]
        U_train = X_train[pu_labels_train == -1]

        model.fit(P=P_train, U=U_train, pi=prior)
        self.console.log("Model fitting complete.", style="green")

        # 4. Predict and Evaluate
        self.console.log("Evaluating model...", style="yellow")
        y_pred = model.predict(X_test)

        # The model predicts {1, -1}. Convert to {1, 0} for metric calculation.
        y_pred_binary = (y_pred == 1).astype(int)

        acc = accuracy_score(y_test_true, y_pred_binary)
        f1 = f1_score(y_test_true, y_pred_binary)
        prec = precision_score(y_test_true, y_pred_binary, zero_division=0)
        rec = recall_score(y_test_true, y_pred_binary, zero_division=0)

        # For AUC, we need probability scores, not just binary predictions.
        # Assuming the model has a `predict_proba` method similar to scikit-learn classifiers.
        try:
            y_pred_scores = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test_true, y_pred_scores)
        except (AttributeError, ValueError):
            # Fallback if predict_proba is not available or output is not as expected;
            # AUC will not be calculated.
            auc = None

        metrics = {
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "auc": auc,
        }

        # 5. Log results and integrate checkpoint-like improvement logging
        self._log_results(metrics)

        # Emulate improvement logging using ModelCheckpoint (single-shot)
        ckpt_cfg = self.params.get("checkpoint", {"enabled": False})
        if ckpt_cfg and ckpt_cfg.get("enabled", False):
            save_dir = ckpt_cfg.get("save_dir", "checkpoints")
            filename = f"{self.method}_{self.experiment_name}.pth"
            monitor = ckpt_cfg.get("monitor", "test_f1")
            mode = ckpt_cfg.get("mode", "max")
            ckpt = ModelCheckpoint(
                save_dir=save_dir,
                filename=filename,
                monitor=monitor,
                mode=mode,
                save_model=ckpt_cfg.get("save_model", False),
                verbose=ckpt_cfg.get("verbose", True),
                file_console=self.file_console,
            )
            # Build all_metrics in the same shape
            all_metrics = {f"test_{k}": v for k, v in metrics.items()}
            ckpt(
                epoch=1,
                all_metrics=all_metrics,
                model=None,  # non-torch model; checkpoint will skip saving if save_model=False
                elapsed_seconds=(
                    (self._run_end_time - self._run_start_time)
                    if (
                        hasattr(self, "_run_end_time")
                        and hasattr(self, "_run_start_time")
                        and self._run_end_time
                        and self._run_start_time
                    )
                    else None
                ),
            )
        self._run_end_time = time.time()
        self._write_result_json(metrics, prior, train_dataset, test_dataset)
        self.console.log(f"✔ Completed: {self.experiment_name}")

        # Close file console
        self._close_file_console()

    def _log_results(self, metrics: dict):
        table = Table(
            title=f"Final Metrics - {self.method.upper()} - {self.experiment_name}"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in metrics.items():
            if value is None:
                table.add_row(key, "N/A")
            else:
                table.add_row(key, f"{value:.4f}")

        self.console.print(table)
        if self.file_console:
            self.file_console.print(table)

    def _write_result_json(
        self, metrics: dict, prior: float, train_dataset, test_dataset
    ):
        try:
            results_root = self.results_root
            os.makedirs(results_root, exist_ok=True)
            out_path = os.path.join(results_root, f"{self.experiment_name}.json")
            duration = (
                float(self._run_end_time - self._run_start_time)
                if (self._run_start_time and self._run_end_time)
                else None
            )

            def _stats(ds):
                try:
                    total = len(ds)
                    pos = int((ds.true_labels == 1).sum().item())
                    return {
                        "total": int(total),
                        "positives": pos,
                        "negatives": int(total - pos),
                    }
                except Exception:
                    return {"total": len(ds)}

            single_run = {
                "method": self.method,
                "experiment": self.experiment_name,
                "device": "cpu",
                "gpu_count": 0,
                "timing": {
                    "start": self._run_start_time,
                    "end": self._run_end_time,
                    "duration_seconds": duration,
                },
                "max_gpu_memory_bytes": 0,
                "dataset": {
                    "class": self.params.get("dataset_class"),
                    "train": {
                        "total": len(train_dataset),
                        "total_positives": (
                            int((train_dataset.true_labels == 1).sum().item())
                            if hasattr(train_dataset, "true_labels")
                            else None
                        ),
                        "prior": float(prior),
                    },
                    "test": _stats(test_dataset),
                },
                "best": {
                    "epoch": 1,
                    "metrics": {f"test_{k}": v for k, v in metrics.items()},
                },
                "monitor": self.params.get("checkpoint", {}).get("monitor", "test_f1"),
                "global_epochs": 1,
                "hyperparameters": self.params,
            }
            merged = {
                "experiment": self.experiment_name,
                "updated_at": None,
                "runs": {},
            }
            if os.path.exists(out_path):
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            merged.update(
                                {k: v for k, v in loaded.items() if k != "runs"}
                            )
                            if isinstance(loaded.get("runs"), dict):
                                merged["runs"].update(loaded["runs"])
                except Exception:
                    pass
            merged["runs"][self.method] = single_run
            merged["updated_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._run_end_time or time.time())
            )

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            if self.file_console:
                self.file_console.log(f"Saved/updated experiment summary: {out_path}")
        except Exception as e:
            if self.file_console:
                self.file_console.log(f"Failed to write result.json: {e}")

    def _close_file_console(self):
        """Safely close file_console (same logic as BaseTrainer)"""
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
