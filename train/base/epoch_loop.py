"""Shared epoch loop, metrics, and checkpoint calls."""

from __future__ import annotations

import time

from rich.table import Table
from tqdm import tqdm

from ..metrics import evaluate_metrics, evaluate_proxy_metrics


class EpochLoopMixin:
    """Default per-epoch evaluation loop for BaseTrainer subclasses."""

    def _oracle_prior_calibrated_fallback(self) -> bool:
        return bool(self.params.get("oracle_prior_calibrated_fallback", False))

    def _run_epochs(self, num_epochs: int, stage_name: str = "Training"):
        test_metrics = {}

        for epoch_idx in tqdm(
            range(1, num_epochs + 1), desc=f"{stage_name} ({self.method.upper()})"
        ):
            self.global_epoch += 1
            self.train_one_epoch(epoch_idx)

            train_metrics, val_metrics, test_metrics = self.evaluate()

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

            if self.checkpoint_handler:
                all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
                if val_metrics is not None:
                    all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                self.checkpoint_handler(
                    epoch=self.global_epoch,
                    all_metrics=all_metrics,
                    model=self.get_checkpoint_model(),
                    elapsed_seconds=(
                        (time.time() - self._run_start_time)
                        if self._run_start_time
                        else None
                    ),
                )

            if self.checkpoint_handler and self.checkpoint_handler.should_stop:
                self.console.log(
                    f"Early stopping in stage '{stage_name}'.", style="bold red"
                )
                break

        return test_metrics

    def evaluate(self) -> tuple[dict, dict | None, dict]:
        scenario = self.params.get("scenario", "single")
        prior_calibrated_fallback = self._oracle_prior_calibrated_fallback()
        train_oracle = evaluate_metrics(
            self.model,
            self.train_loader,
            self.device,
            self.prior,
            prior_calibrated_fallback=prior_calibrated_fallback,
        )
        test_oracle = evaluate_metrics(
            self.model,
            self.test_loader,
            self.device,
            self.prior,
            prior_calibrated_fallback=prior_calibrated_fallback,
        )
        val_oracle = (
            evaluate_metrics(
                self.model,
                self.validation_loader,
                self.device,
                self.prior,
                prior_calibrated_fallback=prior_calibrated_fallback,
            )
            if self.validation_loader is not None
            else None
        )

        train_proxy = evaluate_proxy_metrics(
            self.model, self.train_loader, self.device, self.prior, scenario
        )
        val_proxy = (
            evaluate_proxy_metrics(
                self.model, self.validation_loader, self.device, self.prior, scenario
            )
            if self.validation_loader is not None
            else None
        )

        train_metrics = {**train_oracle, **train_proxy}
        test_metrics = test_oracle
        val_metrics = None
        if val_oracle is not None:
            val_metrics = {**val_oracle}
            if val_proxy is not None:
                val_metrics.update(val_proxy)

        extra_train, extra_val, extra_test = self.get_extra_epoch_metrics()
        if extra_train:
            train_metrics.update(extra_train)
        if extra_test:
            test_metrics.update(extra_test)
        if extra_val:
            if val_metrics is None:
                val_metrics = {}
            val_metrics.update(extra_val)

        return train_metrics, val_metrics, test_metrics

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

        oracle_keys = [
            "oracle_accuracy",
            "oracle_f1",
            "oracle_precision",
            "oracle_recall",
            "oracle_auc",
        ]
        proxy_keys = ["proxy_acc", "proxy_auc"]

        for metric in oracle_keys:
            train_val = train_metrics.get(metric)
            test_val = test_metrics.get(metric)
            if train_val is None and test_val is None:
                continue
            row = [metric, f"{train_val:.4f}" if train_val is not None else "N/A"]
            if val_metrics is not None:
                v = val_metrics.get(metric)
                row.append(f"{v:.4f}" if v is not None else "N/A")
            row.append(f"{test_val:.4f}" if test_val is not None else "N/A")
            table.add_row(*row)

        for metric in proxy_keys:
            train_val = train_metrics.get(metric)
            if train_val is None:
                continue
            row = [metric, f"{train_val:.4f}"]
            if val_metrics is not None:
                v = val_metrics.get(metric)
                row.append(f"{v:.4f}" if v is not None else "N/A")
            row.append("--")
            table.add_row(*row)

        self.console.print(table)
        if self.file_console:
            self.file_console.print(table)
