"""Data, model, and optimizer setup shared by neural trainers."""

from __future__ import annotations

import torch

from ..utils.data_factory import prepare_loaders
from ..utils.model_factory import select_model
from .constants import SOURCE_FAITHFUL_NO_BIAS_INIT


class DataModelMixin:
    """Shared data loading and default model construction."""

    def _prepare_data(self):
        (
            self.train_loader,
            self.validation_loader,
            self.test_loader,
            self.prior,
            self.update_loader,
        ) = prepare_loaders(
            dataset_name=self.experiment_name,
            data_config=self.params,
            batch_size=self.params.get("batch_size", 128),
            data_dir=self.params.get("data_dir", "data"),
            method=self.method,
        )

        sample_data = next(iter(self.train_loader))[0]
        if isinstance(sample_data, (list, tuple)):
            self.input_shape = tuple(sample_data[0].shape[1:])
        else:
            self.input_shape = tuple(sample_data.shape[1:])

    def _build_model(self, return_model: bool = False):
        model = self.create_model().to(self.device)

        try:
            has_params = any(p.requires_grad for p in model.parameters())
        except Exception:
            has_params = False
        if not has_params:
            try:
                sample_batch = next(iter(self.train_loader))
                x_sample = sample_batch[0]
                if isinstance(x_sample, (list, tuple)):
                    x_sample = x_sample[0]
                with torch.no_grad():
                    _ = model(x_sample.to(self.device))
            except StopIteration:
                pass

        if return_model:
            return model

        self.model = model
        self._maybe_init_final_bias_from_prior()

        lr = float(self.params.get("lr", 1e-3))
        wd = float(self.params.get("weight_decay", 5e-4))

        model_params = (
            self.model.params()
            if hasattr(self.model, "params")
            else self.model.parameters()
        )
        self.optimizer = self._make_optimizer(model_params, lr=lr, weight_decay=wd)
        self.criterion = self.create_criterion()

    def _should_init_bias_from_prior(self) -> bool:
        if "init_bias_from_prior" in self.params:
            return bool(self.params.get("init_bias_from_prior"))
        return self.method.lower() not in SOURCE_FAITHFUL_NO_BIAS_INIT

    def _maybe_init_final_bias_from_prior(self) -> None:
        """Optionally initialize a single-logit final classifier bias."""
        if not self._should_init_bias_from_prior():
            return
        try:
            import math as _math

            def _logit(_p: float) -> float:
                eps = 1e-6
                _p = max(min(float(_p), 1 - eps), eps)
                return _math.log(_p / (1.0 - _p))

            fc = getattr(self.model, "final_classifier", None)
            if (
                isinstance(fc, torch.nn.Linear)
                and getattr(fc, "bias", None) is not None
            ):
                if int(getattr(fc, "out_features", 0)) == 1:
                    with torch.no_grad():
                        fc.bias.fill_(_logit(self.prior))
        except Exception:
            pass

    def _make_optimizer(self, model_params, lr: float, weight_decay: float):
        optimizer_name = str(self.params.get("optimizer", "adam")).lower()
        adam_betas = self._adam_betas()
        if optimizer_name == "adam":
            return torch.optim.Adam(
                model_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=adam_betas,
            )
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                model_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=adam_betas,
            )
        if optimizer_name == "sgd":
            momentum = float(self.params.get("momentum", 0.9))
            return torch.optim.SGD(
                model_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _adam_betas(self) -> tuple[float, float]:
        raw = self.params.get("adam_betas")
        if raw is None:
            return (0.9, 0.999)
        if isinstance(raw, str):
            raw = [part.strip() for part in raw.split(",")]
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"`adam_betas` must contain two values, got {raw!r}.")
        return (float(raw[0]), float(raw[1]))

    def create_model(self):
        """Return the default public benchmark backbone.

        Methods that need private, meta, two-logit, or mixup-compatible models
        override this hook inside their own package.
        """

        return select_model(method=self.method, params=self.params, prior=self.prior)
