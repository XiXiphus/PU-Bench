"""Base trainer facade for PU-Bench methods.

Method trainers inherit this class. Implementation details are split under
``train/base/`` so the run launcher can use a narrow registry while method
packages keep importing a stable ``BaseTrainer`` symbol.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch
from rich.console import Console

from .base.constants import SOURCE_FAITHFUL_NO_BIAS_INIT
from .base.data_model import DataModelMixin
from .base.epoch_loop import EpochLoopMixin
from .base.lifecycle import LifecycleMixin
from .base.results import ResultSummaryMixin
from .reproducibility import set_global_seed


class BaseTrainer(
    DataModelMixin,
    EpochLoopMixin,
    ResultSummaryMixin,
    LifecycleMixin,
    ABC,
):
    """Base class for PU learning trainers."""

    def __init__(self, method: str, experiment: str, params: dict):
        self.method = method
        self.experiment_name = experiment
        self.params = params

        self.setup_context()
        self.configure()
        self.build_data()
        self.build_components()

    def setup_context(self) -> None:
        self.console = Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_global_seed(self.params.get("seed", 42))

        self.file_console = None
        self.checkpoint_handler = None

        seed_value = self.params.get("seed", 42)
        self.results_root = os.path.join("results", f"seed_{seed_value}")
        self.log_dir = os.path.join(self.results_root, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.global_epoch = 0

        self._run_start_time = None
        self._run_end_time = None
        self._max_gpu_mem_bytes = 0

    def configure(self) -> None:
        """Hook for subclasses to normalize params before data/model construction."""

    def build_data(self) -> None:
        self._prepare_data()

    def build_components(self) -> None:
        self._build_model()
        self.checkpoint_handler = None
        self._init_checkpoint_handler()

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
    def get_extra_epoch_metrics(self) -> tuple[dict, dict, dict]:
        """Return optional train/val/test metrics for checkpointing and logs."""
        return {}, {}, {}

    def get_checkpoint_model(self):
        """Return the model object that matches the monitored metrics."""
        return self.model
