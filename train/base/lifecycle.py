"""Training lifecycle, logging, checkpoint setup, and result persistence."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from contextlib import contextmanager

import torch
from rich.console import Console

from ..checkpointing import ModelCheckpoint


class LifecycleMixin:
    """Lifecycle helpers used by BaseTrainer and compatible subclasses."""

    def before_training(self):
        self._init_file_console()
        self._run_start_time = time.time()
        if torch.cuda.is_available():
            try:
                for dev_idx in range(torch.cuda.device_count()):
                    with torch.cuda.device(dev_idx):
                        torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def _init_file_console(self):
        log_file_name = f"{self.method}_{self.experiment_name}.log"
        log_file_path = os.path.join(self.log_dir, log_file_name)

        self.file_console = Console(
            file=open(log_file_path, "w", encoding="utf-8"), width=120
        )

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

        try:
            import yaml as _yaml

            hyper_yaml = _yaml.safe_dump(
                self.params, allow_unicode=True, sort_keys=False
            )
            self.file_console.log("[bold]All Hyperparameters (YAML):[/bold]")
            self.file_console.log(hyper_yaml)
        except Exception:
            self.file_console.log("[bold]All Hyperparameters:[/bold]")
            for k, v in self.params.items():
                self.file_console.log(f"  {k}: {v}")

        self.file_console.log(f"=" * 80)

        if self.checkpoint_handler:
            self.checkpoint_handler.file_console = self.file_console

        self.console.log(f"Log file will be saved to: [cyan]{log_file_path}[/cyan]")

    def after_training(self):
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
                    pass

            merged["runs"][self.method] = result
            merged["updated_at"] = datetime.utcnow().isoformat() + "Z"

            with open(exp_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)

            if self.file_console:
                self.file_console.log(f"Saved/updated experiment summary: {exp_path}")
        except Exception as exc:
            if self.file_console:
                self.file_console.log(f"Failed to write result.json: {exc}")

    def _close_file_console(self):
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

    def _init_checkpoint_handler(self):
        checkpoint_params = self.params.get("checkpoint")
        if checkpoint_params and checkpoint_params.get("enabled", False):
            save_dir = checkpoint_params.get("save_dir", "checkpoints")
            filename = f"{self.method}_{self.experiment_name}.pth"
            early_stopping_params = checkpoint_params.get("early_stopping")
            self.checkpoint_handler = ModelCheckpoint(
                save_dir=save_dir,
                filename=filename,
                monitor=checkpoint_params.get("monitor", "val_proxy_acc"),
                mode=checkpoint_params.get("mode", "max"),
                save_model=checkpoint_params.get("save_model", True),
                verbose=checkpoint_params.get("verbose", True),
                file_console=self.file_console,
                early_stopping_params=early_stopping_params,
                keep_best_state=checkpoint_params.get("keep_best_state", False),
            )
            self._validate_checkpoint_monitor_config()

    @contextmanager
    def suspend_checkpointing(self):
        checkpoint_handler = self.checkpoint_handler
        self.checkpoint_handler = None
        try:
            yield
        finally:
            self.checkpoint_handler = checkpoint_handler

    def reset_checkpoint_tracking(self) -> None:
        if self.checkpoint_handler:
            self.checkpoint_handler.reset_tracking()

    def set_checkpoint_early_stopping(
        self, enabled: bool, *, reset: bool = False
    ) -> None:
        if self.checkpoint_handler:
            self.checkpoint_handler.set_early_stopping(enabled, reset=reset)

    def update_checkpoint_best_metrics(self, metrics: dict[str, float]) -> None:
        if self.checkpoint_handler:
            self.checkpoint_handler.update_best_metrics(metrics)

    def _validate_checkpoint_monitor_config(self) -> None:
        if not self.checkpoint_handler:
            return

        monitor = str(getattr(self.checkpoint_handler, "monitor", ""))
        if monitor.startswith("val_") and getattr(self, "validation_loader", None) is None:
            raise ValueError(
                f"Checkpoint monitor '{monitor}' requires a validation loader. "
                "Set a positive val_ratio or choose an explicit train_/test_ monitor."
            )

    def run_stage(self, stage_name: str, num_epochs: int):
        return self._run_epochs(num_epochs, stage_name=stage_name)

    def finalize(self) -> None:
        try:
            if self._run_start_time is not None:
                self.after_training()
            if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
                self.checkpoint_handler.log_best_metrics()
        finally:
            self._close_file_console()

    def run(self):
        try:
            self.before_training()
            return self.run_stage("Training", self.params.get("num_epochs", 1))
        finally:
            self.finalize()
