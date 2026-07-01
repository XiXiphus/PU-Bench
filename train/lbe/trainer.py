from __future__ import annotations

import torch
from torch.optim import Adam
from tqdm import tqdm

from ..base_trainer import BaseTrainer
from ..utils.checkpointing import CheckpointBundle
from ..utils.model_factory import select_model
from .core import (
    m_step_classifier_per_sample_loss,
    m_step_eta_loss,
    observed_label_pretrain_loss,
    posterior_y1,
)


class LBETrainer(BaseTrainer):
    """Controlled LBE-PU trainer.

    Primary source snapshot:
        author source file(s): lbe_author_provided

    Relevant source files:
        - LBE_MLP/em.py: E-step posterior and M-step likelihood formulas.
        - LBE_MLP/train.py: MLP pretraining, 10% M-step subsampling, and
          80% small-loss classifier update.

    PU-Bench keeps the source EM objective while using the benchmark's shared
    controlled backbone and PU data loader.
    """

    def __init__(self, method: str, experiment: str, params: dict):
        super().__init__(method, experiment, params)

        # LBE requires two models: one for P(y=1|x) and one for eta(x)
        self.eta_model = select_model(
            method=self.method, params=self.params, prior=self.prior
        ).to(self.device)

        # Source MLP uses default PyTorch initialization. Prior-bias init is a
        # benchmark convenience and remains opt-in only.
        try:
            import math as _math

            def _logit(_p: float) -> float:
                eps = 1e-6
                _p = max(min(float(_p), 1 - eps), eps)
                return _math.log(_p / (1.0 - _p))

            if bool(self.params.get("init_bias_from_prior", False)):
                fc_eta = getattr(self.eta_model, "final_classifier", None)
                if (
                    isinstance(fc_eta, torch.nn.Linear)
                    and getattr(fc_eta, "bias", None) is not None
                ):
                    if int(getattr(fc_eta, "out_features", 0)) == 1:
                        with torch.no_grad():
                            fc_eta.bias.fill_(_logit(self.prior))
        except Exception:
            pass

        # Ensure dynamic models (e.g., MLPs built on first forward) have parameters
        try:
            has_params_eta = any(p.requires_grad for p in self.eta_model.parameters())
        except Exception:
            has_params_eta = False
        if not has_params_eta:
            try:
                sample_batch = next(iter(self.train_loader))
                x_sample = sample_batch[0]
                if isinstance(x_sample, (list, tuple)):
                    x_sample = x_sample[0]
                with torch.no_grad():
                    _ = self.eta_model(x_sample.to(self.device))
            except Exception:
                pass

        # A separate optimizer for the eta_model
        self.optimizer_eta = self._new_adam(self.eta_model)

        # Fallback: ensure update_loader exists (unshuffled loader over train dataset)
        if getattr(self, "update_loader", None) is None:
            try:
                from data.data_utils import PUDataloader

                self.update_loader = PUDataloader(
                    self.train_loader.dataset,
                    batch_size=self.params.get("batch_size", 128),
                    shuffle=False,
                )
            except Exception:
                # As a last resort, reuse train_loader (may be shuffled)
                self.update_loader = self.train_loader

    def _prepare_data(self):
        super()._prepare_data()
        self._validate_data_contract()

    def _validate_data_contract(self) -> None:
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "pu_labels"):
            raise ValueError("LBE-PU requires a PU dataset exposing `pu_labels`.")

        metadata = getattr(dataset, "pu_metadata", {}) or {}
        expected_labels = {
            "pu_labeled_label": 1,
            "pu_unlabeled_label": -1,
        }
        for key, expected in expected_labels.items():
            observed = metadata.get(key, expected)
            if int(observed) != expected:
                raise ValueError(
                    "LBE-PU requires PU labels +1 for labeled positives and "
                    f"-1 for unlabeled samples; got {key}={observed}."
                )

        labels = {
            int(value)
            for value in torch.unique(dataset.pu_labels.detach().cpu()).tolist()
        }
        if 1 not in labels or -1 not in labels:
            raise ValueError(
                "LBE-PU requires both labeled-positive (+1) and unlabeled (-1) "
                f"examples; observed PU labels {sorted(labels)}."
            )

        indices = getattr(dataset, "indices", None)
        if indices is not None:
            expected = torch.arange(len(dataset), device=indices.device)
            if not torch.equal(indices.long(), expected):
                raise ValueError(
                    "LBE-PU EM storage expects contiguous dataset indices "
                    "0..N-1 for the train dataset."
                )

    def create_criterion(self):
        # M-step loss is custom and implemented in train_one_epoch
        return None

    def get_checkpoint_model(self):
        return CheckpointBundle(
            modules={
                "classifier": self.model,
                "eta_model": self.eta_model,
            }
        )

    def _optimizer_hparams(self) -> tuple[float, float]:
        return (
            float(self.params.get("lr", 1e-3)),
            float(self.params.get("weight_decay", 1e-4)),
        )

    def _trainable_params(self, model: torch.nn.Module):
        return model.params() if hasattr(model, "params") else model.parameters()

    def _new_adam(self, model: torch.nn.Module) -> Adam:
        lr, wd = self._optimizer_hparams()
        return Adam(self._trainable_params(model), lr=lr, weight_decay=wd)

    def _reset_em_optimizers(self) -> None:
        self.optimizer = self._new_adam(self.model)
        self.optimizer_eta = self._new_adam(self.eta_model)

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        self.eta_model.train()

        # --- E-Step ---
        # First, calculate soft labels (posterior P(y|x,q)) for the entire training set
        self.console.log(f"Epoch {epoch_idx}: Performing E-Step...")
        n_total = len(self.train_loader.dataset)
        dataset = self.train_loader.dataset
        storage_device = dataset.features.device
        all_soft_labels_y1 = torch.zeros(n_total, device=storage_device)
        # Switch to eval mode to freeze BN/Dropout behavior during posterior estimation
        was_model_training = self.model.training
        was_eta_training = self.eta_model.training
        self.model.eval()
        self.eta_model.eval()

        with torch.no_grad():
            for x, t, _, indices, _ in tqdm(
                self.update_loader, desc=f"E-Step (epoch {epoch_idx})"
            ):  # Use unshuffled loader (provides indices)
                x, t = x.to(self.device), t.to(self.device)
                q = (t == 1).float()

                p_y1_x = self.model(x).sigmoid().view(-1)
                eta_x = self.eta_model(x).sigmoid().view(-1)
                pst_y1 = posterior_y1(p_y1_x, eta_x, q)

                # map back to storage tensor using dataset indices
                all_soft_labels_y1[indices.to(storage_device)] = pst_y1.detach().to(
                    storage_device
                )

        # Restore training mode for M-step
        if was_model_training:
            self.model.train()
        if was_eta_training:
            self.eta_model.train()

        # --- M-Step ---
        # Update models for m_steps iterations using the fixed soft labels
        m_steps = self.params.get("m_steps", 10)
        subset_ratio = float(self.params.get("subset_ratio", 0.1))
        topk_keep = float(self.params.get("topk_keep", 0.8))
        self.console.log(
            f"Epoch {epoch_idx}: Performing M-Step for {m_steps} iterations (subset_ratio={subset_ratio}, topk_keep={topk_keep})..."
        )

        # Build direct references to dataset tensors (for fast random subset sampling)
        features_all = dataset.features  # on storage_device
        pu_labels_all = dataset.pu_labels  # on storage_device
        train_bs = int(self.params.get("m_step_train_batch_size", 512))

        for _ in tqdm(range(m_steps), desc=f"M-Step (epoch {epoch_idx})"):
            # Match source `randperm(N)[:int(N/10)]`: sample from the whole
            # training set, then apply loss1 small-loss filtering globally
            # across that subset.
            k_subset = max(1, int(n_total * subset_ratio))
            k_subset = min(k_subset, n_total)
            subset_idx = torch.randperm(n_total, device=storage_device)[:k_subset]
            keep_count = max(1, int(k_subset * topk_keep))

            classifier_losses = []
            with torch.no_grad():
                for start in range(0, k_subset, train_bs):
                    end = min(k_subset, start + train_bs)
                    batch_idx = subset_idx[start:end]

                    xs = features_all.index_select(0, batch_idx).to(self.device)
                    pst_chunk = all_soft_labels_y1.index_select(0, batch_idx).to(
                        self.device
                    )
                    z = self.model(xs).view(-1)
                    classifier_losses.append(
                        m_step_classifier_per_sample_loss(z, pst_chunk)
                        .detach()
                        .to(storage_device)
                    )
            classifier_losses_all = torch.cat(classifier_losses, dim=0)
            keep_pos = classifier_losses_all.topk(largest=False, k=keep_count)[1]
            kept_subset_idx = subset_idx.index_select(0, keep_pos)

            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_eta.zero_grad(set_to_none=True)

            for start in range(0, keep_count, train_bs):
                end = min(keep_count, start + train_bs)
                batch_idx = kept_subset_idx[start:end]

                xs = features_all.index_select(0, batch_idx).to(self.device)
                pst_chunk = all_soft_labels_y1.index_select(0, batch_idx).to(
                    self.device
                )

                z = self.model(xs).view(-1)
                ce = m_step_classifier_per_sample_loss(z, pst_chunk).mean()
                (ce * ((end - start) / float(keep_count))).backward()

            for start in range(0, k_subset, train_bs):
                end = min(k_subset, start + train_bs)
                batch_idx = subset_idx[start:end]

                xs = features_all.index_select(0, batch_idx).to(self.device)
                ts = pu_labels_all.index_select(0, batch_idx).to(self.device)
                qs = (ts == 1).float()
                pst_chunk = all_soft_labels_y1.index_select(0, batch_idx).to(
                    self.device
                )

                eta_z = self.eta_model(xs).view(-1)
                loss_eta = m_step_eta_loss(eta_z, qs, pst_chunk.detach())

                chunk_weight = (end - start) / float(k_subset)
                (loss_eta * chunk_weight).backward()

            self.optimizer.step()
            self.optimizer_eta.step()

    def run(self):
        """
        Overrides the base run method to include a pre-training step.
        """
        self.before_training()

        # 1. Pre-training phase. Pretraining is only initialization, so it must
        # not update checkpoint/best state.
        with self.suspend_checkpointing():
            self._pretrain()

        # 2. Main EM training phase
        self.set_checkpoint_early_stopping(
            bool(getattr(self.checkpoint_handler, "early_stopping_enabled", False)),
            reset=True,
        )
        try:
            self.run_stage("EM-Training", self.params.get("num_epochs", 100))
            return (
                self.checkpoint_handler.best_metrics if self.checkpoint_handler else {}
            )
        finally:
            self.finalize()

    def _pretrain(self):
        self.console.log("Starting pre-training phase...", style="bold yellow")
        pretrain_epochs = self.params.get("pretrain_epochs", 100)
        dataset = self.train_loader.dataset
        n_total = len(dataset)
        pu_labels = dataset.pu_labels
        proportion_labeled = (pu_labels == 1).float().mean()

        # Source pretrains classifier first, then eta_model, each with one Adam
        # step per full-training-set pass. Use loader chunks to keep memory bounded.
        optimizer_clf = self._new_adam(self.model)
        for _ in tqdm(range(pretrain_epochs), desc="Pre-training classifier"):
            optimizer_clf.zero_grad(set_to_none=True)
            for x, t, _, _, _ in self.train_loader:
                x, t = x.to(self.device), t.to(self.device)

                # In PU data, t has P (1) and U (-1). For LBE's `q`, we need
                # P (1) and U (0).
                q = (t == 1).float()
                z = self.model(x).view(-1)
                loss_clf = observed_label_pretrain_loss(
                    z, q, proportion_labeled=proportion_labeled
                )
                loss_clf = loss_clf * (x.shape[0] / float(n_total))
                loss_clf.backward()
            optimizer_clf.step()

        optimizer_eta = self._new_adam(self.eta_model)
        for _ in tqdm(range(pretrain_epochs), desc="Pre-training eta"):
            optimizer_eta.zero_grad(set_to_none=True)
            for x, t, _, _, _ in self.train_loader:
                x, t = x.to(self.device), t.to(self.device)
                q = (t == 1).float()
                eta_z = self.eta_model(x).view(-1)
                loss_eta = observed_label_pretrain_loss(
                    eta_z, q, proportion_labeled=proportion_labeled
                )
                loss_eta = loss_eta * (x.shape[0] / float(n_total))
                loss_eta.backward()
            optimizer_eta.step()

        # Source creates a fresh optimizer for the EM stage after pretraining.
        self._reset_em_optimizers()

        self.console.log("Pre-training finished.", style="bold green")
