"""Packaged nnPU trainer.

Source-faithful nnPU trainer.

Primary source:
    https://github.com/kiryor/nnPUlearning
    - `pu_loss.py`: non-negative and unbiased PU risk estimator.
    - `train.py`: Adam-based single-model training loop.
    - `dataset.py`: `prior = n_up / n_u`, the positive fraction in U.

Implementation contract:
    - PU labels are +1 for labeled positives and -1 for unlabeled samples.
    - `self.prior` must be the unlabeled-mixture prior pi_U supplied by the
      data layer, not the positive fraction of the concatenated training set.
    - The default loss is the original sigmoid surrogate; logistic is supported
      because the source CLI exposes it.
    - Source-faithful runs should not initialize the classifier bias from prior;
      BaseTrainer disables that default for `method == "nnpu"` unless explicitly
      overridden.
"""

from __future__ import annotations

import torch

from ..base_trainer import BaseTrainer
from .losses import PULoss


class NNPUTrainer(BaseTrainer):
    """Non-negative PU learning method trainer."""

    def _prepare_data(self):
        super()._prepare_data()
        self._validate_source_data_contract()

    def _validate_source_data_contract(self) -> None:
        """Fail fast if the shared data layer violates nnPU source semantics."""
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "pu_labels"):
            raise ValueError("nnPU requires a PU dataset exposing `pu_labels`.")

        metadata = getattr(dataset, "pu_metadata", {}) or {}
        expected_labels = {
            "pu_labeled_label": 1,
            "pu_unlabeled_label": -1,
        }
        for key, expected in expected_labels.items():
            observed = metadata.get(key, expected)
            if int(observed) != expected:
                raise ValueError(
                    "nnPU source reproduction requires PU labels +1 for "
                    f"positives and -1 for unlabeled samples; got {key}={observed}."
                )

        labels = {
            int(value)
            for value in torch.unique(dataset.pu_labels.detach().cpu()).tolist()
        }
        if 1 not in labels or -1 not in labels:
            raise ValueError(
                "nnPU requires both labeled-positive (+1) and unlabeled (-1) "
                f"examples; observed PU labels {sorted(labels)}."
            )

        pi_unlabeled = metadata.get("pi_unlabeled")
        if (
            pi_unlabeled is not None
            and abs(float(pi_unlabeled) - float(self.prior)) > 1e-8
        ):
            raise ValueError(
                "nnPU prior must be the positive fraction inside U "
                f"(pi_unlabeled={pi_unlabeled}), got prior={self.prior}."
            )

    # Required interfaces
    def create_criterion(self):
        loss_name = self.params.get("loss", "sigmoid")
        gamma = float(self.params.get("gamma", 1.0))
        beta = float(self.params.get("beta", 0.0))
        return PULoss(
            self.prior,
            loss=loss_name,
            nnpu=True,
            gamma=gamma,
            beta=beta,
        )

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (nnPU)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)

            # The source nnPU estimator consumes exactly these +/-1 PU labels.
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
