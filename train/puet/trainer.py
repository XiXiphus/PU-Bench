from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..utils.checkpointing import ModelCheckpoint
from ..utils.data_factory import prepare_loaders
from ..utils.reproducibility import set_global_seed
from .trees import PUExtraTrees


class PUETTrainer:
    """
    Independent trainer for PUExtraTrees, designed to mimic the interface
    of BaseTrainer for seamless integration into the benchmark runner.

    Source: author source file(s): puet at
    5cb15d7e021a6e24d4cd300278500c38729e2775.  The packaged tree/forest code
    keeps the source estimator semantics; benchmark additions are limited to
    package-relative imports, robust config handling, and probability scores
    needed for PU-Bench AUC reporting.
    """

    def __init__(self, method: str, experiment: str, params: dict):
        self.method = method
        self.experiment_name = experiment
        self.params = params
        label_scheme = self.params.get("label_scheme", {}) or {}
        self.pu_labeled_label = label_scheme.get("pu_labeled_label", 1)
        self.pu_unlabeled_label = label_scheme.get("pu_unlabeled_label", -1)

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
        train_loader, validation_loader, test_loader, prior, _ = prepare_loaders(
            dataset_name=self.experiment_name,
            data_config=self.params,
            batch_size=self.params.get(
                "batch_size", 1024
            ),  # Larger batch for non-iterative model
            data_dir=self.params.get("data_dir", "data"),
            method=self.method,
        )

        train_dataset = train_loader.dataset
        val_dataset = (
            validation_loader.dataset if validation_loader is not None else None
        )
        test_dataset = test_loader.dataset

        # PUExtraTrees works with flat numpy arrays
        X_train = self._features_to_numpy(train_dataset)
        pu_labels_train = self._labels_to_numpy(train_dataset.pu_labels)
        y_train_true = self._labels_to_numpy(train_dataset.true_labels)
        X_val = (
            self._features_to_numpy(val_dataset) if val_dataset is not None else None
        )
        pu_labels_val = (
            self._labels_to_numpy(val_dataset.pu_labels)
            if val_dataset is not None
            else None
        )
        y_val_true = (
            self._labels_to_numpy(val_dataset.true_labels)
            if val_dataset is not None
            else None
        )
        X_test = self._features_to_numpy(test_dataset)
        y_test_true = self._labels_to_numpy(test_dataset.true_labels)

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
        P_train = X_train[pu_labels_train == self.pu_labeled_label]
        U_train = X_train[pu_labels_train != self.pu_labeled_label]

        model.fit(P=P_train, U=U_train, pi=prior)
        self.console.log("Model fitting complete.", style="green")

        # 4. Predict and Evaluate
        self.console.log("Evaluating model...", style="yellow")
        scenario = self.params.get("scenario", "single")
        train_metrics = self._evaluate_oracle(model, X_train, y_train_true)
        train_metrics.update(
            self._evaluate_proxy(model, X_train, pu_labels_train, prior, scenario)
        )
        val_metrics = None
        if val_dataset is not None:
            val_metrics = self._evaluate_oracle(model, X_val, y_val_true)
            val_metrics.update(
                self._evaluate_proxy(model, X_val, pu_labels_val, prior, scenario)
            )
        test_metrics = self._evaluate_oracle(model, X_test, y_test_true)
        all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        if val_metrics is not None:
            all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

        # 5. Log results and integrate checkpoint-like improvement logging
        self._log_results(train_metrics, val_metrics, test_metrics)
        self._run_end_time = time.time()

        # Emulate improvement logging using ModelCheckpoint (single-shot)
        ckpt_cfg = self.params.get(
            "checkpoint",
            {
                "enabled": True,
                "save_model": False,
                "monitor": "val_proxy_acc",
                "mode": "max",
            },
        )
        best_epoch = 1
        best_metrics = all_metrics
        monitor = ckpt_cfg.get("monitor", "val_proxy_acc")
        if ckpt_cfg and ckpt_cfg.get("enabled", False):
            save_dir = ckpt_cfg.get("save_dir", "checkpoints")
            filename = f"{self.method}_{self.experiment_name}.pth"
            mode = ckpt_cfg.get("mode", "max")
            ckpt = ModelCheckpoint(
                save_dir=save_dir,
                filename=filename,
                monitor=monitor,
                mode=mode,
                save_model=False,
                verbose=ckpt_cfg.get("verbose", True),
                file_console=self.file_console,
            )
            ckpt(
                epoch=1,
                all_metrics=all_metrics,
                model=None,  # non-torch model; checkpoint will skip saving if save_model=False
                elapsed_seconds=self._run_end_time - self._run_start_time,
            )
            if ckpt.best_metrics is not None:
                best_epoch = ckpt.best_epoch
                best_metrics = ckpt.best_metrics
                monitor = ckpt.monitor

        self._write_result_json(
            prior=prior,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            monitor=monitor,
        )
        self.console.log(f"✔ Completed: {self.experiment_name}")

        # Close file console
        self._close_file_console()
        return best_metrics

    def _log_results(
        self,
        train_metrics: dict,
        val_metrics: dict | None,
        test_metrics: dict,
    ):
        table = Table(
            title=f"Final Metrics - {self.method.upper()} - {self.experiment_name}"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="magenta")
        if val_metrics is not None:
            table.add_column("Val", style="magenta")
        table.add_column("Test", style="magenta")

        keys = list(train_metrics.keys())
        for key in test_metrics:
            if key not in keys:
                keys.append(key)

        for key in keys:
            row = [key, self._format_metric(train_metrics.get(key))]
            if val_metrics is not None:
                row.append(self._format_metric(val_metrics.get(key)))
            row.append(self._format_metric(test_metrics.get(key)))
            table.add_row(*row)

        self.console.print(table)
        if self.file_console:
            self.file_console.print(table)

    @staticmethod
    def _format_metric(value) -> str:
        if value is None:
            return "--"
        try:
            if not np.isfinite(value):
                return "--"
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    @staticmethod
    def _labels_to_numpy(labels) -> np.ndarray:
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()
        return np.asarray(labels).reshape(-1)

    @staticmethod
    def _features_to_numpy(dataset) -> np.ndarray:
        features = dataset.features
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        features = np.asarray(features)
        return features.reshape(len(dataset), -1)

    def _predict_binary_and_scores(self, model: PUExtraTrees, X: np.ndarray):
        y_pred = model.predict(X)
        y_pred_binary = (np.asarray(y_pred).reshape(-1) == 1).astype(int)
        try:
            y_scores = np.asarray(model.predict_proba(X))[:, 1].reshape(-1)
        except Exception:
            y_scores = y_pred_binary.astype(float)
        return y_pred_binary, y_scores

    def _evaluate_oracle(self, model: PUExtraTrees, X: np.ndarray, y_true) -> dict:
        y_true = self._labels_to_numpy(y_true).astype(int)
        y_pred_binary, y_scores = self._predict_binary_and_scores(model, X)
        try:
            auc = (
                float("nan")
                if len(np.unique(y_true)) < 2
                else float(roc_auc_score(y_true, y_scores))
            )
        except Exception:
            auc = float("nan")
        return {
            "oracle_accuracy": float(accuracy_score(y_true, y_pred_binary)),
            "oracle_precision": float(
                precision_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
            ),
            "oracle_recall": float(
                recall_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
            ),
            "oracle_f1": float(
                f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
            ),
            "oracle_auc": auc,
        }

    def _evaluate_proxy(
        self,
        model: PUExtraTrees,
        X: np.ndarray,
        pu_labels,
        prior: float,
        scenario: str,
    ) -> dict:
        pu = self._labels_to_numpy(pu_labels)
        y_pred_binary, y_scores = self._predict_binary_and_scores(model, X)
        labeled_value = getattr(self, "pu_labeled_label", 1)
        unlabeled_value = getattr(self, "pu_unlabeled_label", -1)
        p_mask = pu == labeled_value
        u_mask = pu == unlabeled_value
        total_p = int(p_mask.sum())
        total_u = int(u_mask.sum())
        if total_p == 0 or total_u == 0:
            pa = float("nan")
        else:
            correct_p = int((y_pred_binary[p_mask] == 1).sum())
            correct_u = int((y_pred_binary[u_mask] == 0).sum())
            if scenario == "case-control":
                pa = 2 * float(prior) * (correct_p / total_p) + (correct_u / total_u)
            else:
                pa = 2 * float(prior) * (correct_p / total_p) + (
                    (correct_p + correct_u) / (total_p + total_u)
                )
        if total_p == 0 or total_u == 0:
            pauc = float("nan")
        else:
            try:
                labels = np.concatenate([np.ones(total_p), np.zeros(total_u)])
                scores = np.concatenate([y_scores[p_mask], y_scores[u_mask]])
                pauc = float(roc_auc_score(labels, scores))
            except Exception:
                pauc = 0.5
        return {"proxy_acc": float(pa), "proxy_auc": float(pauc)}

    def _write_result_json(
        self,
        prior: float,
        train_dataset,
        val_dataset,
        test_dataset,
        best_epoch: int,
        best_metrics: dict,
        monitor: str,
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

            def _stats(ds):
                if ds is None:
                    return None
                try:
                    total = len(ds)
                    labels = self._labels_to_numpy(ds.true_labels)
                    pos = int((labels == 1).sum())
                    return {
                        "total": int(total),
                        "positives": pos,
                        "negatives": int(total - pos),
                        "positive_ratio": (pos / total) if total else None,
                    }
                except Exception:
                    return {"total": len(ds)}

            single_run = {
                "method": self.method,
                "experiment": self.experiment_name,
                "device": "cpu",
                "gpu_count": 0,
                "timing": {
                    "start": start_iso,
                    "end": end_iso,
                    "duration_seconds": duration,
                },
                "max_gpu_memory_bytes": 0,
                "dataset": {
                    "class": self.params.get("dataset_class"),
                    "train": {
                        "total": len(train_dataset),
                        "total_positives": (
                            int(
                                (
                                    self._labels_to_numpy(train_dataset.true_labels)
                                    == 1
                                ).sum()
                            )
                            if hasattr(train_dataset, "true_labels")
                            else None
                        ),
                        "prior": float(prior),
                    },
                    "validation": _stats(val_dataset),
                    "test": _stats(test_dataset),
                },
                "best": {
                    "epoch": int(best_epoch),
                    "metrics": best_metrics,
                },
                "monitor": monitor,
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
