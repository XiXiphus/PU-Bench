"""PULDA trainer for PU-Bench.

Primary source:
    author source file(s): pulda
    jiangyangby/PULDA at 7b3dcad95bd7caa0a9477af37a05764fbe6e27bc

Source facts preserved here:
    - PU labels are mapped to ``1`` for labeled positives and ``0`` for U.
    - The warm-up stage uses the same label-distribution alignment objective as
      the source by default; BCE warm-up is only an explicit benchmark option.
    - The alignment stage resets the LDA/EMA loss, initializes pseudo scores on
      the whole PU training set, then trains with binary mixup.
    - Optional resampling repeats P examples and iterates U examples once per
      epoch with fixed P/U batch sizes.

Benchmark boundary:
    The official source only implements CIFAR-10 data/model recipes.  This port
    keeps PULDA's estimator loop and adapts it to PU-Bench's canonical datasets
    and controlled public backbones.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..base_trainer import BaseTrainer
from ..utils.reproducibility import seed_worker
from .losses import (
    PULDALabelDistributionLoss,
    PULDALabelDistributionLossWithEMA,
    TwoWaySigmoidLoss,
    TwoWaySigmoidLossWithEMA,
)
from .mixup import mixup_bce, mixup_two_targets
from .sampler import PULDAResamplingBatchSampler


class PULDATrainer(BaseTrainer):
    """Source-aligned PULDA trainer adapted to PU-Bench datasets."""

    def before_training(self):
        super().before_training()
        self._sync_prior_from_pu_metadata()

        self.use_ema = bool(int(self.params.get("EMA", 1)))
        self.use_two_way = bool(int(self.params.get("two_way", 1)))
        self.temperature = float(self.params.get("tmpr", 3.5))
        self.margin = self._resolve_margin()

        self.warmup_epochs = int(self.params.get("warm_up_epochs", 60))
        self.align_epochs = int(
            self.params.get(
                "pu_epochs",
                max(1, int(self.params.get("num_epochs", 60)) - self.warmup_epochs),
            )
        )
        self.warmup_loss = str(self.params.get("warmup_loss", "lda")).lower()
        if self.warmup_loss not in {"lda", "bce"}:
            raise ValueError("PULDA warmup_loss must be either 'lda' or 'bce'.")

        self.alpha = float(self.params.get("alpha", 11.0))
        self.co_mixup = float(self.params.get("co_mixup", 4.2))
        self.warmup_lr = float(self.params.get("warm_up_lr", 1e-4))
        self.warmup_wd = float(self.params.get("warm_up_weight_decay", 5e-4))
        self.align_lr = float(self.params.get("lr", 1e-3))
        self.align_wd = float(self.params.get("weight_decay", 1e-4))

        self.warmup_base_loss, self.warmup_two_way_loss = self._make_lda_losses()
        self.base_loss = None
        self.two_way_loss = None
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.pseudo_labels = None
        self.alignment_stage_initialized = False

        if int(self.params.get("resample", 1)) == 1:
            self._install_source_resampling_loader()
        else:
            self.train_loader = self._make_loader(
                self.train_loader.dataset,
                batch_size=int(self.params.get("batch_size", 256)),
                shuffle=True,
            )
        self._align_source_eval_update_loaders()
        self._assert_pseudo_label_index_space()

        if self.file_console:
            self.file_console.log("PULDA source alignment:")
            self.file_console.log(
                "  source=jiangyangby/PULDA@7b3dcad95bd7caa0a9477af37a05764fbe6e27bc"
            )
            self.file_console.log(
                f"  model={self.model.__class__.__name__}, "
                f"params={sum(p.numel() for p in self.model.parameters()) / 1e6:.3f}M, "
                "backbone=shared_public"
            )
            self.file_console.log(
                f"  batch_size={self.params.get('batch_size', 256)}, "
                f"test_batch_size={self.params.get('test_batch_size', 128)}, "
                f"num_workers={self.params.get('num_workers', 0)}, "
                f"pin_memory={self.params.get('pin_memory', torch.cuda.is_available())}, "
                f"resample={self.params.get('resample', 1)}, "
                f"P_batch_size={self.params.get('P_batch_size', 16)}, "
                f"U_batch_size={self.params.get('U_batch_size', 128)}"
            )

    def create_criterion(self):
        return torch.nn.Identity()

    def run(self):
        self.before_training()
        original_early_stopping = (
            bool(self.checkpoint_handler.early_stopping_enabled)
            if self.checkpoint_handler
            else False
        )

        if self.warmup_epochs > 0:
            self.set_checkpoint_early_stopping(False)
            self._setup_warmup_stage()
            self.run_stage("Warm-up", self.warmup_epochs)

        if self.align_epochs > 0 and not self.alignment_stage_initialized:
            self._setup_alignment_stage()
            self.alignment_stage_initialized = True

        if self.align_epochs > 0:
            self.set_checkpoint_early_stopping(original_early_stopping, reset=True)
            self.run_stage("Alignment", self.align_epochs)

        self.finalize()

    def train_one_epoch(self, epoch_idx: int):
        if self.global_epoch <= self.warmup_epochs:
            self._train_epoch_warmup()
        else:
            self._train_epoch_align()

    def _sync_prior_from_pu_metadata(self) -> None:
        metadata = getattr(self.train_loader.dataset, "pu_metadata", {}) or {}
        risk_prior = float(metadata.get("pi_unlabeled", self.prior))
        if abs(risk_prior - float(self.prior)) > 1e-4:
            self.console.log(
                f"Using PU-Bench unlabeled-mixture prior pi_U={risk_prior:.4f} "
                f"instead of initial prior {float(self.prior):.4f}.",
                style="bold yellow",
            )
        self.prior = risk_prior

    def _resolve_margin(self) -> float:
        if self.params.get("margin") is not None:
            return float(self.params["margin"])
        return 0.6

    def _make_lda_losses(self):
        if self.use_ema:
            base_loss = PULDALabelDistributionLossWithEMA(
                prior=self.prior,
                temperature=self.temperature,
                alpha_u=float(self.params.get("alpha_U", 0.85)),
            )
        else:
            base_loss = PULDALabelDistributionLoss(
                prior=self.prior,
                temperature=self.temperature,
            )

        two_way_loss = None
        if self.use_two_way:
            if self.use_ema:
                two_way_loss = TwoWaySigmoidLossWithEMA(
                    self.prior,
                    self.margin,
                    self.temperature,
                    float(self.params.get("alpha_CN", 0.5)),
                )
            else:
                two_way_loss = TwoWaySigmoidLoss(
                    self.prior,
                    self.margin,
                    self.temperature,
                )
        return base_loss, two_way_loss

    def _pulda_loss(
        self,
        logits: torch.Tensor,
        pu_binary: torch.Tensor,
        base_loss,
        two_way_loss,
    ) -> torch.Tensor:
        loss = base_loss(logits, pu_binary)
        if self.use_two_way and two_way_loss is not None:
            loss = loss + two_way_loss(logits, pu_binary)
        return loss

    def _install_source_resampling_loader(self) -> None:
        base_ds = self.train_loader.dataset
        pu_labels = getattr(base_ds, "pu_labels", None)
        if pu_labels is None:
            raise RuntimeError(
                "PULDA resampling requires train_loader.dataset.pu_labels. "
                "Set resample=0 explicitly for non-source batching."
            )

        pu_labels = torch.as_tensor(pu_labels)
        p_indices = (pu_labels == 1).nonzero(as_tuple=False).view(-1).cpu().numpy()
        u_indices = (pu_labels != 1).nonzero(as_tuple=False).view(-1).cpu().numpy()
        sampler = PULDAResamplingBatchSampler(
            p_indices,
            u_indices,
            p_batch_size=int(self.params.get("P_batch_size", 16)),
            u_batch_size=int(self.params.get("U_batch_size", 128)),
        )
        self.train_loader = DataLoader(
            base_ds,
            batch_sampler=sampler,
            **self._loader_worker_kwargs(),
        )

    def _loader_worker_kwargs(self) -> dict:
        num_workers = int(self.params.get("num_workers", 0))
        pin_memory = bool(self.params.get("pin_memory", torch.cuda.is_available()))
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "worker_init_fn": seed_worker,
        }
        if num_workers > 0 and "persistent_workers" in self.params:
            kwargs["persistent_workers"] = bool(self.params["persistent_workers"])
        if num_workers > 0 and "prefetch_factor" in self.params:
            kwargs["prefetch_factor"] = int(self.params["prefetch_factor"])
        return kwargs

    def _make_loader(self, dataset, *, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **self._loader_worker_kwargs(),
        )

    def _align_source_eval_update_loaders(self) -> None:
        eval_batch = int(self.params.get("test_batch_size", 128))
        if self.validation_loader is not None:
            self.validation_loader = self._make_loader(
                self.validation_loader.dataset,
                batch_size=eval_batch,
                shuffle=False,
            )
        self.test_loader = self._make_loader(
            self.test_loader.dataset,
            batch_size=eval_batch,
            shuffle=False,
        )
        if self.update_loader is not None:
            self.update_loader = self._make_loader(
                self.update_loader.dataset,
                batch_size=eval_batch,
                shuffle=False,
            )

    def _assert_pseudo_label_index_space(self) -> None:
        if self.update_loader is None:
            raise RuntimeError("PULDA requires update_loader for pseudo-label setup.")

        train_ds = self.train_loader.dataset
        update_ds = self.update_loader.dataset
        if train_ds is update_ds:
            return

        train_indices = getattr(train_ds, "indices", None)
        update_indices = getattr(update_ds, "indices", None)
        if train_indices is None or update_indices is None:
            raise RuntimeError(
                "PULDA train_loader and update_loader must share an index space "
                "for pseudo-label storage."
            )

        train_indices = torch.as_tensor(train_indices)
        update_indices = torch.as_tensor(update_indices)
        if len(train_indices) != len(update_indices) or not torch.equal(
            train_indices.cpu(), update_indices.cpu()
        ):
            raise RuntimeError(
                "PULDA train_loader and update_loader indices differ; pseudo-label "
                "updates would no longer match the source implementation's shared "
                "PU dataset contract."
            )

    def _setup_warmup_stage(self):
        self.console.log("Setting up PULDA warm-up stage...", style="yellow")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.warmup_lr,
            weight_decay=self.warmup_wd,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.align_epochs,
        )

    def _setup_alignment_stage(self):
        self.console.log("Setting up PULDA alignment stage...", style="yellow")
        self._initialize_pseudo_labels()
        self.base_loss, self.two_way_loss = self._make_lda_losses()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.align_lr,
            weight_decay=self.align_wd,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.align_epochs,
            eta_min=0.7 * self.align_lr,
        )

    def _train_epoch_warmup(self):
        self.model.train()
        for x, t, _y_true, _idx, _ in self.train_loader:
            x = x.to(self.device)
            t = t.to(self.device)
            pu_binary = (t == 1).float()
            logits = self.model(x).view(-1)

            if self.warmup_loss == "bce":
                loss = self.bce_loss(logits, pu_binary)
            else:
                loss = self._pulda_loss(
                    logits,
                    pu_binary,
                    self.warmup_base_loss,
                    self.warmup_two_way_loss,
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

    def _initialize_pseudo_labels(self):
        if self.update_loader is None:
            raise RuntimeError("PULDA requires update_loader for pseudo-label setup.")

        self.console.log("Initializing PULDA pseudo-labels...", style="yellow")
        self.model.eval()
        num_samples = len(self.update_loader.dataset)
        self.pseudo_labels = torch.zeros(
            num_samples,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            for x, _, _, idx, _ in self.update_loader:
                x = x.to(self.device)
                idx = idx.to(self.device)
                logits = self.model(x).view(-1)
                self.pseudo_labels[idx] = torch.sigmoid(logits)

        self.model.train()

    def _train_epoch_align(self):
        if self.pseudo_labels is None:
            raise RuntimeError("PULDA pseudo labels are not initialized.")
        if self.base_loss is None:
            raise RuntimeError("PULDA alignment loss is not initialized.")

        self.model.train()
        for x, t, _y_true, idx, _ in self.train_loader:
            x = x.to(self.device)
            t = t.to(self.device)
            idx = idx.to(self.device)

            pseudo_targets = self.pseudo_labels[idx].clone()
            pseudo_targets[t == 1] = 1.0

            x_mix, y_a, y_b, lam = mixup_two_targets(
                x,
                pseudo_targets,
                alpha=self.alpha,
                device=self.device,
            )

            logits_orig = torch.clamp(self.model(x).view(-1), min=-10, max=10)
            logits_mix = torch.clamp(self.model(x_mix).view(-1), min=-10, max=10)

            pu_binary = (t == 1).float()
            loss_align = self._pulda_loss(
                logits_orig,
                pu_binary,
                self.base_loss,
                self.two_way_loss,
            )
            loss_mix = mixup_bce(torch.sigmoid(logits_mix), y_a, y_b, lam)
            loss = loss_align + self.co_mixup * loss_mix

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.pseudo_labels[idx] = torch.sigmoid(logits_orig.detach())

        if self.scheduler:
            self.scheduler.step()
