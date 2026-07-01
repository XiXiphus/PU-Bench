"""Packaged nnPUSB trainer.

Source-faithful neural PUSB/nnPUSB trainer.

Primary source:
    https://github.com/MasaKat0/PUlearning
    Audited HEAD: 3401b77ccdd653d39f4f3a6258a42c7938fa9ede
    - `BiasedPUlearning/nnPUSB/main_nnPUSB_mnist.py`: constructs selected-biased
      labeled positives, then concatenates them with the full training set as U.
    - `BiasedPUlearning/nnPUSB/train_nnPU.py`: trains a logistic nnPU risk with
      class prior `pi` equal to the positive fraction inside U.

Implementation contract:
    - The source-native data setting is selected-biased positives with full-U
      case-control data. PU-Bench preserves the nnPUSB risk formula and label
      mapping, but selected-bias construction is controlled by benchmark splits
      rather than reproducing the source's exact data recipe.
    - Selected bias is a data-construction property. The trainer must not add an
      ad hoc positive-loss weight to compensate for SAR/selected positives.
    - PU-Bench stores unlabeled examples as -1, while the source script uses 0;
      the loss is the direct label-mapped analogue of the source formula.
    - `self.prior` must be the unlabeled-mixture prior pi_U supplied by the data
      layer.
    - Source-faithful runs should not initialize the classifier bias from prior;
      BaseTrainer disables that default for `method == "nnpusb"` unless
      explicitly overridden.
"""

from __future__ import annotations

import torch

from ..base_trainer import BaseTrainer
from ..utils.metrics import _adapt_input_for_model
from .losses import nnPUSBloss


class NNPUSBTrainer(BaseTrainer):
    """nnPUSB learning trainer"""

    def _prepare_data(self):
        super()._prepare_data()
        self._validate_source_data_contract()

    def _validate_source_data_contract(self) -> None:
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "pu_labels"):
            raise ValueError("nnPUSB requires a PU dataset exposing `pu_labels`.")

        metadata = getattr(dataset, "pu_metadata", {}) or {}
        expected_labels = {
            "pu_labeled_label": 1,
            "pu_unlabeled_label": -1,
        }
        for key, expected in expected_labels.items():
            observed = metadata.get(key, expected)
            if int(observed) != expected:
                raise ValueError(
                    "nnPUSB uses PU-Bench's source-mapped labels +1 for "
                    f"positives and -1 for unlabeled samples; got {key}={observed}."
                )

        labels = {
            int(value)
            for value in torch.unique(dataset.pu_labels.detach().cpu()).tolist()
        }
        if 1 not in labels or -1 not in labels:
            raise ValueError(
                "nnPUSB requires both labeled-positive (+1) and unlabeled (-1) "
                f"examples; observed PU labels {sorted(labels)}."
            )

        pi_unlabeled = metadata.get("pi_unlabeled")
        if (
            pi_unlabeled is not None
            and abs(float(pi_unlabeled) - float(self.prior)) > 1e-8
        ):
            raise ValueError(
                "nnPUSB prior must be the positive fraction inside U "
                f"(pi_unlabeled={pi_unlabeled}), got prior={self.prior}."
            )

    def create_criterion(self):
        if float(self.params.get("weight", 1.0)) != 1.0:
            raise ValueError(
                "The MasaKat0 nnPUSB source implementation does not reweight the "
                "positive risk. Express selected bias through the data layer."
            )
        gamma = float(self.params.get("gamma", 1.0))
        beta = float(self.params.get("beta", 0.0))
        return nnPUSBloss(self.prior, nnPU=True, gamma=gamma, beta=beta)

    def _threshold_prior(self) -> float:
        metadata = getattr(
            getattr(self.train_loader, "dataset", None), "pu_metadata", {}
        )
        if not isinstance(metadata, dict):
            metadata = {}

        prior = metadata.get("pi_constructed_train")
        if prior is None:
            n_labeled = metadata.get("n_labeled")
            n_unlabeled = metadata.get("n_unlabeled")
            if n_labeled is not None and n_unlabeled is not None:
                total = int(n_labeled) + int(n_unlabeled)
                if total > 0:
                    prior = (
                        int(n_labeled) + float(self.prior) * int(n_unlabeled)
                    ) / total

        if prior is None:
            prior = self.prior

        prior = float(prior)
        if not 0.0 < prior < 1.0:
            raise ValueError(f"nnPUSB threshold prior must be in (0, 1), got {prior}.")
        return prior

    def calibrate_decision_threshold(self) -> torch.Tensor:
        """Set source-style inference threshold from all training raw scores."""

        was_training = self.model.training
        self.model.eval()

        raw_scores = []
        with torch.no_grad():
            for x, *_ in self.train_loader:  # type: ignore
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = _adapt_input_for_model(self.model, x.to(self.device))
                raw_scores.append(self.model(x).view(-1).detach())

        if was_training:
            self.model.train()

        if not raw_scores:
            raise ValueError("Cannot calibrate nnPUSB threshold from an empty loader.")

        scores = torch.cat(raw_scores, dim=0)
        sorted_scores, _ = torch.sort(scores, dim=0)
        threshold_prior = self._threshold_prior()
        threshold_index = int((1.0 - threshold_prior) * len(sorted_scores))
        threshold_index = max(0, min(threshold_index, len(sorted_scores) - 1))
        threshold = sorted_scores[threshold_index].detach()

        self.decision_threshold = threshold
        self.threshold_prior = threshold_prior
        self.threshold_index = threshold_index
        self.model.pu_score_threshold = threshold
        self.model.pu_threshold_prior = threshold_prior
        return threshold

    def evaluate(self):
        self.calibrate_decision_threshold()
        return super().evaluate()

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)
            # Source formula after mapping source unlabeled label 0 to PU-Bench -1.
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
