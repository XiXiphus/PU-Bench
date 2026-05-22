from __future__ import annotations

import copy
import os

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

console = Console()


class CheckpointBundle(torch.nn.Module):
    """Named module/optimizer payload for multi-component checkpoints."""

    def __init__(
        self,
        modules: dict[str, torch.nn.Module],
        optimizers: dict[str, torch.optim.Optimizer] | None = None,
    ):
        super().__init__()
        self._optimizer_names: list[str] = []
        self._optimizers: dict[str, torch.optim.Optimizer] = {}
        for name, module in modules.items():
            self.add_module(name, module)
        for name, optimizer in (optimizers or {}).items():
            self._optimizer_names.append(name)
            self._optimizers[name] = optimizer

    def state_dict(self, *args, **kwargs):  # noqa: D401
        """Return a nested checkpoint payload."""
        payload = {
            name: module.state_dict(*args, **kwargs)
            for name, module in self._modules.items()
        }
        payload.update(
            {
                name: self._optimizers[name].state_dict()
                for name in self._optimizer_names
            }
        )
        return payload


class ModelCheckpoint:
    """Save the best model during training according to a monitored metric."""

    def __init__(
        self,
        save_dir: str,
        filename: str,
        monitor: str,
        mode: str = "max",
        save_model: bool = True,
        verbose: bool = True,
        file_console: Console | None = None,
        early_stopping_params: dict | None = None,
        keep_best_state: bool = False,
    ):
        """
        Args:
            save_dir (str): Directory to save the model.
            filename (str): Model filename.
            monitor (str): Metric to monitor, formatted as "phase_metric"
                           (e.g., "test_f1", "train_accuracy").
            mode (str):     "max" or "min".
            save_model (bool): Whether to persist model weights.
            verbose (bool):   Whether to log improvements.
            file_console (Console | None): Rich console to also write logs to a file.
            early_stopping_params (dict | None): Parameters for early stopping.
        """
        self.save_dir = save_dir
        self.filename = filename
        self.save_path = os.path.join(self.save_dir, self.filename)
        self.monitor = monitor
        self.mode = mode
        self.save_model = save_model
        self.keep_best_state = keep_best_state
        self.verbose = verbose
        self.file_console = file_console

        if self.mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', but got '{mode}'")

        self.best_score = -np.inf if self.mode == "max" else np.inf
        self.best_epoch = -1
        self.best_metrics = None
        self.best_elapsed_seconds: float | None = None
        self.best_state_dict = None

        # Early stopping attributes
        self.early_stopping_enabled = False
        self.patience = float("inf")
        self.min_delta = 0.0
        self.wait = 0
        self.should_stop = False

        if early_stopping_params and early_stopping_params.get("enabled", False):
            self.early_stopping_enabled = True
            self.patience = early_stopping_params.get("patience", 10)
            self.min_delta = early_stopping_params.get("min_delta", 0)
            if self.verbose:
                self._log(
                    f"Early stopping enabled: patience={self.patience}, min_delta={self.min_delta}",
                    "bold blue",
                )

        if self.save_model:
            os.makedirs(self.save_dir, exist_ok=True)

    def reset_tracking(self) -> None:
        self.best_score = -np.inf if self.mode == "max" else np.inf
        self.best_epoch = -1
        self.best_metrics = None
        self.best_elapsed_seconds = None
        self.best_state_dict = None
        self.wait = 0
        self.should_stop = False
        if hasattr(self, "_warned"):
            delattr(self, "_warned")

    def set_early_stopping(self, enabled: bool, *, reset: bool = False) -> None:
        self.early_stopping_enabled = bool(enabled)
        if reset:
            self.wait = 0
        self.should_stop = False

    def update_best_metrics(self, metrics: dict[str, float]) -> None:
        if self.best_metrics is not None:
            self.best_metrics.update(metrics)

    def _clone_state_dict(self, model: torch.nn.Module) -> dict:
        cloned = {}
        for key, value in model.state_dict().items():
            if torch.is_tensor(value):
                cloned[key] = value.detach().cpu().clone()
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    def _log(self, message: str, style: str = None):
        """Log to stdout and, if provided, to a file-backed Rich Console."""
        if style:
            message = f"[{style}]{message}[/{style}]"
        console.log(message)
        if self.file_console:
            self.file_console.log(message)

    def __call__(
        self,
        epoch: int,
        all_metrics: dict[str, float],
        model: torch.nn.Module,
        elapsed_seconds: float | None = None,
    ):
        """Check after each epoch whether to update 'best' and save the model."""
        current_score = all_metrics.get(self.monitor)
        if current_score is None:
            available = ", ".join(sorted(all_metrics.keys()))
            raise KeyError(
                f"Monitored metric '{self.monitor}' not found. "
                f"Available metrics: {available}"
            )

        improved = False
        if self.mode == "max":
            if current_score > self.best_score + self.min_delta:
                improved = True
        else:
            if current_score < self.best_score - self.min_delta:
                improved = True

        if improved:
            old_best = self.best_score
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_metrics = all_metrics
            # Track time-to-best if provided
            try:
                self.best_elapsed_seconds = (
                    float(elapsed_seconds) if elapsed_seconds is not None else None
                )
            except Exception:
                self.best_elapsed_seconds = None

            if self.verbose:
                old_best_str = f"{old_best:.4f}" if np.isfinite(old_best) else "N/A"
                message = f"Epoch {epoch}: {self.monitor} improved from {old_best_str} to {current_score:.4f}."
                if self.save_model:
                    message += f" Saving model to {self.save_path}"
                self._log(message, "bold green")

            if self.save_model:
                torch.save(model.state_dict(), self.save_path)
            if self.keep_best_state:
                self.best_state_dict = self._clone_state_dict(model)

            # Reset wait counter on improvement
            self.wait = 0
        elif self.early_stopping_enabled:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                if self.verbose:
                    self._log(
                        f"Epoch {epoch}: Early stopping triggered after {self.patience} epochs of no improvement on '{self.monitor}'.",
                        "bold red",
                    )

    def log_best_metrics(self):
        """Render a Rich table with the best metrics recorded so far."""
        if self.best_metrics is None:
            warning_msg = "No best metrics recorded. Perhaps the score never improved from initialization."
            self._log(warning_msg, "bold yellow")
            return

        extra = (
            f", time_to_best={self.best_elapsed_seconds:.2f}s"
            if hasattr(self, "best_elapsed_seconds")
            and self.best_elapsed_seconds is not None
            else ""
        )
        table = Table(
            title=f"Best Metrics ({self.monitor} @ Epoch {self.best_epoch}{extra})"
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Train", style="magenta")
        table.add_column("Val", style="yellow")
        table.add_column("Test", style="green")

        def _extract(prefix):
            return {
                k[len(prefix):]: v
                for k, v in self.best_metrics.items()
                if k.startswith(prefix)
            }

        train_m = _extract("train_")
        val_m = _extract("val_")
        test_m = _extract("test_")

        all_keys = sorted(set(train_m.keys()) | set(val_m.keys()) | set(test_m.keys()))

        def _fmt(d, key):
            v = d.get(key)
            if v is None:
                return "N/A"
            if isinstance(v, float) and v != v:  # nan check
                return "NaN"
            return f"{v:.4f}"

        for key in all_keys:
            table.add_row(key, _fmt(train_m, key), _fmt(val_m, key), _fmt(test_m, key))

        console.print(table)
        if self.file_console:
            self.file_console.print(table)
