"""Trainer registry used by run_train.

The registry is intentionally separate from BaseTrainer. Trainer modules import
BaseTrainer, so importing trainers from base_trainer.py would create circular
dependencies and make the base class depend on method packages.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


TRAINER_IMPORT_PATHS = {
    "pn": "train.pn_trainer.PNTrainer",
    "nnpu": "train.nnpu.trainer.NNPUTrainer",
    "nnpusb": "train.nnpusb.trainer.NNPUSBTrainer",
    "vpu": "train.vpu.trainer.VPUTrainer",
    "distpu": "train.distpu.trainer.DistPUTrainer",
    "selfpu": "train.selfpu.trainer.SelfPUTrainer",
    "holisticpu": "train.holisticpu.trainer.HolisticPUTrainer",
    "robustpu": "train.robustpu.trainer.RobustPUTrainer",
    "p3mixc": "train.p3mix.c_trainer.P3MIXCTrainer",
    "p3mixe": "train.p3mix.e_trainer.P3MIXETrainer",
    "lagam": "train.lagam.trainer.LaGAMTrainer",
    "pulda": "train.pulda.trainer.PULDATrainer",
    "bbepu": "train.bbepu.trainer.BBEPUTrainer",
    "vaepu": "train.vaepu.trainer.VAEPUTrainer",
    "puet": "train.puet.trainer.PUETTrainer",
    "pan": "train.pan.trainer.PANTrainer",
    "lbe": "train.lbe.trainer.LBETrainer",
    "cgenpu": "train.cgenpu.trainer.CGenPUTrainer",
    "pulcpbf": "train.pulcpbf.trainer.PULCPBFTrainer",
}


def list_registered_methods() -> list[str]:
    return sorted(TRAINER_IMPORT_PATHS)


def get_trainer_import_path(method_name: str) -> str:
    method = method_name.lower()
    try:
        return TRAINER_IMPORT_PATHS[method]
    except KeyError as exc:
        available = ", ".join(list_registered_methods())
        raise KeyError(
            f"Unknown trainer method {method_name!r}. Available: {available}"
        ) from exc


def import_trainer_class(method_name: str) -> type[Any]:
    path = get_trainer_import_path(method_name)
    module_path, attr = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, attr)
