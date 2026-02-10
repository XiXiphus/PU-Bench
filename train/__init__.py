"""PU Benchmark New Training Framework (train_)

This package contains the refactored Trainer base class and method-specific trainers,
kept lightweight to avoid hard dependencies during package import.

Note: Individual trainer classes should be imported explicitly, e.g.

    from train.nnpu_trainer import NNPUTrainer
    from train.upu_trainer import UPUTrainer
    from train.vpu_trainer import VPUTrainer
    from train.p3mixc_trainer import P3MIXCTrainer

Lazy import frameworks should use full paths like 'train.nnpu_trainer.NNPUTrainer'
to avoid loading unnecessary modules.
"""

from __future__ import annotations

# flake8: noqa
# isort: skip_file

# fmt: off
# The following imports are intended to simplify the registration of trainers
# in frameworks like `run_task.py`. By exposing them here, they can be
# dynamically imported using their class name, e.g.,
# `getattr(importlib.import_module("train"), "NNPUTrainer")`.

from train.base_trainer import BaseTrainer
from train.pn_trainer import PNTrainer
from train.nnpu_trainer import NNPUTrainer
from train.nnpusb_trainer import NNPUSBTrainer
from train.distpu_trainer import DistPUTrainer
from train.upu_trainer import UPUTrainer
from train.vpu_trainer import VPUTrainer
from train.selfpu_trainer import SelfPUTrainer
from train.robustpu_trainer import RobustPUTrainer
from train.holisticpu_trainer import HolisticPUTrainer
from train.p3mixc_trainer import P3MIXCTrainer
from train.p3mixe_trainer import P3MIXETrainer
from train.lagam_trainer import LaGAMTrainer
from train.pulda_trainer import PULDATrainer
from train.bbepu_trainer import BBEPUTrainer
from train.vaepu_trainer import VAEPUTrainer
from train.puet_trainer import PUETTrainer
from train.pan_trainer import PANTrainer
from train.lbe_trainer import LBETrainer
from train.cgenpu_trainer import CGenPUTrainer
from train.pulcpbf_trainer import PULCPBFTrainer
# fmt: on
