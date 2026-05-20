"""HolisticPU method package.

The trainer import is lazy so compatibility users of ``train.holisticpu.augment``
do not pay for trainer-only dependencies such as ``jenkspy`` until they run the
method.
"""

__all__ = ["HolisticPUTrainer"]


def __getattr__(name: str):
    if name == "HolisticPUTrainer":
        from .trainer import HolisticPUTrainer

        return HolisticPUTrainer
    raise AttributeError(name)
