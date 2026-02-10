from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .train_utils import select_model


class LBETrainer(BaseTrainer):
    """
    Trainer for the LBE (Labeling Bias Estimation) method for PU learning.
    It uses an EM algorithm to jointly train a classifier and a labeling bias model.
    """

    def __init__(self, method: str, experiment: str, params: dict):
        super().__init__(method, experiment, params)

        # LBE requires two models: one for P(y=1|x) and one for eta(x)
        self.eta_model = select_model(
            method=self.method, params=self.params, prior=self.prior
        ).to(self.device)

        # Initialize bias from prior for eta_model if single-logit head
        try:
            import math as _math

            def _logit(_p: float) -> float:
                eps = 1e-6
                _p = max(min(float(_p), 1 - eps), eps)
                return _math.log(_p / (1.0 - _p))

            if bool(self.params.get("init_bias_from_prior", True)):
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
        lr = self.params.get("lr", 1e-3)
        wd = self.params.get("weight_decay", 1e-4)
        self.optimizer_eta = Adam(self.eta_model.parameters(), lr=lr, weight_decay=wd)

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

    def create_criterion(self):
        # M-step loss is custom and implemented in train_one_epoch
        return None

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

                p_q_given_y1_x = ((1 - eta_x) ** (1 - q)) * (eta_x**q)
                p_q_given_y0_x = 1 - q

                p_y1_q_x = p_y1_x * p_q_given_y1_x
                p_y0_q_x = (1 - p_y1_x) * p_q_given_y0_x

                denominator = p_y1_q_x + p_y0_q_x + 1e-8
                pst_y1 = p_y1_q_x / denominator

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
        pos_global_idx_all = (
            (pu_labels_all == 1).nonzero(as_tuple=True)[0].to(storage_device)
        )
        unl_global_idx_all = (
            (pu_labels_all == -1).nonzero(as_tuple=True)[0].to(storage_device)
        )

        for _ in tqdm(range(m_steps), desc=f"M-Step (epoch {epoch_idx})"):
            # Sample candidates from unlabeled pool only
            k_unl = max(1, int(n_total * subset_ratio))
            k_unl = min(k_unl, unl_global_idx_all.numel())
            perm_unl = torch.randperm(
                unl_global_idx_all.numel(), device=storage_device
            )[:k_unl]
            cand_unl_global_idx = unl_global_idx_all.index_select(0, perm_unl)

            # --- 1) Compute per-sample CE for unlabeled candidates (no grad) and keep small-loss ---
            eval_bs = int(self.params.get("m_step_eval_batch_size", 1024))
            eps = 1e-8
            per_sample_ce_parts = []
            with torch.no_grad():
                for start in range(0, k_unl, eval_bs):
                    end = min(k_unl, start + eval_bs)
                    idx_chunk = cand_unl_global_idx[start:end]
                    xs = features_all.index_select(0, idx_chunk).to(self.device)
                    pst_chunk = all_soft_labels_y1.index_select(0, idx_chunk).to(
                        self.device
                    )
                    p = self.model(xs).sigmoid().view(-1)
                    ce = -(
                        pst_chunk * (p + eps).log()
                        + (1 - pst_chunk) * (1 - p + eps).log()
                    )
                    per_sample_ce_parts.append(ce.detach().cpu())

            per_sample_ce = torch.cat(per_sample_ce_parts, dim=0)
            keep_unl = max(1, int(per_sample_ce.numel() * topk_keep))
            kept_unl_local_idx = per_sample_ce.topk(keep_unl, largest=False)[1]
            kept_unl_global_idx = cand_unl_global_idx.index_select(
                0, kept_unl_local_idx.to(cand_unl_global_idx.device)
            )

            # Always include all labeled positives
            kept_global_idx = torch.cat(
                [pos_global_idx_all, kept_unl_global_idx], dim=0
            )

            # --- 2) Gradient update on kept samples with train batches and grad accumulation ---
            train_bs = int(self.params.get("m_step_train_batch_size", 512))
            total_kept = kept_global_idx.numel()
            # Shuffle to mix positives and unlabeled across batches
            if total_kept > 1:
                perm = torch.randperm(total_kept, device=kept_global_idx.device)
                kept_global_idx = kept_global_idx.index_select(0, perm)

            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_eta.zero_grad(set_to_none=True)

            for start in range(0, total_kept, train_bs):
                end = min(total_kept, start + train_bs)
                batch_idx = kept_global_idx[start:end]

                xs = features_all.index_select(0, batch_idx).to(self.device)
                ts = pu_labels_all.index_select(0, batch_idx).to(self.device)
                qs = (ts == 1).float()
                pst_chunk = all_soft_labels_y1.index_select(0, batch_idx).to(
                    self.device
                )

                # Classifier loss with class-balance re-weighting (BCE with logits)
                z = self.model(xs).view(-1)
                pos_cnt = torch.clamp(qs.sum(), min=1.0)
                unl_cnt = torch.clamp((1 - qs).sum(), min=1.0)
                w_pos = (unl_cnt / pos_cnt).detach()
                w_unl = torch.tensor(1.0, device=self.device)

                ce_vec = F.binary_cross_entropy_with_logits(
                    z, pst_chunk, reduction="none"
                )
                weights = torch.where(qs > 0.5, w_pos, w_unl)
                ce = (ce_vec * weights).sum() / (weights.sum() + 1e-8)

                # Labeling model loss (weighted BCE, as per LBE paper)
                eta_z = self.eta_model(xs).view(-1)
                # For U samples (q=0), weight loss by posterior P(y=1|x,q=0)
                # For P samples (q=1), weight is 1.
                loss_eta_vec = F.binary_cross_entropy_with_logits(
                    eta_z, qs, reduction="none"
                )
                weights_eta = torch.where(
                    qs > 0.5, torch.tensor(1.0, device=qs.device), pst_chunk
                )
                # Use .detach() on weights to prevent gradients flowing back to the main classifier
                # through the posteriors, which would violate the EM-step separation.
                loss_eta = (loss_eta_vec * weights_eta.detach()).mean()

                # Normalize to keep overall loss scale roughly constant
                chunk_weight = (end - start) / float(total_kept)
                loss = (ce + loss_eta) * chunk_weight
                loss.backward()

            self.optimizer.step()
            self.optimizer_eta.step()

    def run(self):
        """
        Overrides the base run method to include a pre-training step.
        """
        self.before_training()

        # 1. Pre-training phase
        # Disable early stopping during pre-training to avoid premature stop
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = False
                self.checkpoint_handler.should_stop = False
            except Exception:
                pass
        self._pretrain()

        # 2. Main EM training phase
        # Re-enable early stopping for the final main training
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = True
                self.checkpoint_handler.wait = 0
                self.checkpoint_handler.should_stop = False
            except Exception:
                pass
        self._run_epochs(self.params.get("num_epochs", 100), stage_name="EM-Training")

        self.after_training()

        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()

        self._close_file_console()

        # Return the final metrics from the best epoch
        return self.checkpoint_handler.best_metrics if self.checkpoint_handler else {}

    def _pretrain(self):
        self.console.log("Starting pre-training phase...", style="bold yellow")
        pretrain_epochs = self.params.get("pretrain_epochs", 100)

        # Use the full training data for pre-training steps
        # The original code iterates over the full dataset `pretrain_epochs` times
        for epoch in tqdm(range(pretrain_epochs), desc="Pre-training"):
            for x, t, _, _, _ in self.train_loader:
                x, t = x.to(self.device), t.to(self.device)

                # In PU data, t has P (1) and U (-1). For LBE's `q`, we need P (1) and U (0).
                q = (t == 1).float()

                # Pre-train self.model (classifier)
                self.optimizer.zero_grad(set_to_none=True)
                z = self.model(x).view(-1)
                ce_vec = F.binary_cross_entropy_with_logits(
                    z, q.float(), reduction="none"
                )
                # Class-balance reweighting
                pos_cnt = torch.clamp(q.sum(), min=1.0)
                unl_cnt = torch.clamp((1 - q).sum(), min=1.0)
                w_pos = (unl_cnt / pos_cnt).detach()
                w_unl = torch.tensor(1.0, device=self.device)
                weights = torch.where(q > 0.5, w_pos, w_unl)
                loss_clf = (ce_vec * weights).sum() / (weights.sum() + 1e-8)
                loss_clf.backward()
                self.optimizer.step()

                # Pre-train self.eta_model (labeling bias)
                self.optimizer_eta.zero_grad(set_to_none=True)
                eta_z = self.eta_model(x).view(-1)
                loss_eta = F.binary_cross_entropy_with_logits(eta_z, q.float())
                loss_eta.backward()
                self.optimizer_eta.step()

        self.console.log("Pre-training finished.", style="bold green")
