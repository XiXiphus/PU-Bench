"""Result summary helpers for BaseTrainer."""

from __future__ import annotations

from datetime import datetime

import torch

from ..utils.metrics import evaluate_metrics


class ResultSummaryMixin:
    """Build structured per-run result summaries."""

    def _compose_result_summary(self) -> dict:
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

        duration_sec = (
            float(self._run_end_time - self._run_start_time)
            if (self._run_start_time is not None and self._run_end_time is not None)
            else None
        )
        time_to_best_sec = None
        if self.checkpoint_handler and hasattr(
            self.checkpoint_handler, "best_elapsed_seconds"
        ):
            try:
                best_elapsed = self.checkpoint_handler.best_elapsed_seconds
                time_to_best_sec = (
                    float(best_elapsed) if best_elapsed is not None else None
                )
            except Exception:
                time_to_best_sec = None

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
            try:
                prior_calibrated_fallback = self._oracle_prior_calibrated_fallback()
                train_metrics = evaluate_metrics(
                    self.model,
                    self.train_loader,
                    self.device,
                    self.prior,
                    prior_calibrated_fallback=prior_calibrated_fallback,
                )
                test_metrics = evaluate_metrics(
                    self.model,
                    self.test_loader,
                    self.device,
                    self.prior,
                    prior_calibrated_fallback=prior_calibrated_fallback,
                )
                merged_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                merged_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
                best = {"epoch": int(self.global_epoch), "metrics": merged_metrics}
            except Exception:
                pass

        method_metadata = (
            self.params.get("method_metadata", {})
            if isinstance(self.params, dict)
            else {}
        )
        if not isinstance(method_metadata, dict):
            method_metadata = {}

        return {
            "method": self.method,
            "experiment": self.experiment_name,
            "device": str(self.device),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "timing": {
                "start": start_iso,
                "end": end_iso,
                "duration_seconds": duration_sec,
                "time_to_best_seconds": time_to_best_sec,
            },
            "max_gpu_memory_bytes": self._max_gpu_mem_bytes,
            "dataset": dataset_info,
            "best": best,
            "monitor": monitor,
            "global_epochs": int(self.global_epoch),
            "method_metadata": method_metadata,
            "hyperparameters": self.params,
        }

    def _collect_dataset_stats(self) -> dict:
        def _split_stats(ds):
            try:
                metadata = getattr(ds, "pu_metadata", {}) or {}
                true_positive_label = metadata.get("true_positive_label", 1)
                total = len(ds)
                pos = int((ds.true_labels == true_positive_label).sum().item())
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
                metadata = getattr(train_dataset, "pu_metadata", {}) or {}
                true_positive_label = metadata.get("true_positive_label", 1)
                true_negative_label = metadata.get("true_negative_label", 0)
                pu_labeled_label = metadata.get("pu_labeled_label", 1)
                pu_unlabeled_label = metadata.get("pu_unlabeled_label", -1)
                pu = train_dataset.pu_labels
                tl = train_dataset.true_labels
                total = len(train_dataset)
                labeled = int((pu == pu_labeled_label).sum().item())
                unlabeled = int((pu == pu_unlabeled_label).sum().item())
                pos_in_u = int(
                    ((tl == true_positive_label) & (pu == pu_unlabeled_label))
                    .sum()
                    .item()
                )
                neg_in_u = int(
                    ((tl == true_negative_label) & (pu == pu_unlabeled_label))
                    .sum()
                    .item()
                )
                total_pos = int((tl == true_positive_label).sum().item())
                pi_constructed = (total_pos / total) if total else None
                pi_unlabeled = (
                    pos_in_u / (pos_in_u + neg_in_u)
                    if (pos_in_u + neg_in_u) > 0
                    else None
                )
                train_detail = {
                    "total": int(total),
                    "labeled": labeled,
                    "unlabeled": unlabeled,
                    "positives_in_unlabeled": pos_in_u,
                    "negatives_in_unlabeled": neg_in_u,
                    "total_positives": total_pos,
                    "prior": getattr(self, "prior", pi_unlabeled),
                    "risk_prior": getattr(self, "prior", pi_unlabeled),
                    "pi_unlabeled": pi_unlabeled,
                    "pi_constructed_train": pi_constructed,
                    "c_realized": metadata.get("c_realized"),
                    "scenario": metadata.get("scenario"),
                    "selection_strategy": metadata.get("selection_strategy"),
                    "case_control_mode": metadata.get("case_control_mode"),
                    "case_control_semantics": metadata.get("case_control_semantics"),
                    "source_split_policy": metadata.get("source_split_policy"),
                    "prior_source": metadata.get("prior_source", "pi_unlabeled"),
                    "sampling_reference": metadata.get("sampling_reference"),
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
