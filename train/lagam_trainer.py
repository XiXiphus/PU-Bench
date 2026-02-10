from __future__ import annotations
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import faiss

from .base_trainer import BaseTrainer
from backbone.meta_layers import to_var
from .train_utils import select_model
from loss.loss_lagam_bce import LaGAMBCELoss
from loss.loss_lagam_cont import LaGAMContLoss
from data.lagam_dataset import LaGAMDatasetWrapper
from data.data_utils import PUDataset
from .train_utils import mixup_data


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_kmeans(x, num_cluster, gpu_device, temperature=0.07):
    """
    Args:
        x: data to be clustered
        num_cluster: number of clusters
        gpu_device: GPU device index (kept for compatibility)
        temperature: temperature parameter for density normalization
    """
    print("Performing kmeans clustering...")
    results = {"im2cluster": [], "centroids": [], "density": []}

    d = x.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = False
    clus.niter = 20
    clus.nredo = 5
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    # Use CPU index instead of GPU
    index = faiss.IndexFlatL2(d)
    clus.train(x, index)

    D, I = index.search(x, 1)
    im2cluster = [int(n[0]) for n in I]

    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

    # Compute density for each cluster (missing in original implementation)
    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d_val = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d_val

    # Fill empty clusters with max density
    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    # Clip and normalize density
    density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
    density = temperature * density / density.mean()

    # Convert to tensors on available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im2cluster = torch.LongTensor(im2cluster).to(device)
    centroids = torch.tensor(centroids, dtype=torch.float32, device=device)
    density = torch.tensor(density, dtype=torch.float32, device=device)

    results["centroids"] = centroids
    results["im2cluster"] = im2cluster
    results["density"] = density

    return results


class LaGAMTrainer(BaseTrainer):
    """
    Trainer for the LaGAM method. This trainer implements the three-stage training process:
    1. Warm-up phase for initial model training.
    2. K-Means clustering on features from the warm-up model.
    3. Meta-learning phase with pseudo-label correction and contrastive loss, using
    - a validation set for meta-learning updates.
    """

    def _build_model(self):
        """Initialize model with better weight initialization for PU learning."""
        super()._build_model()
        # Initialize final classifier bias to encourage positive predictions
        # Based on prior probability
        if hasattr(self.model, "final_classifier") and hasattr(
            self.model.final_classifier, "bias"
        ):
            if self.model.final_classifier.bias is not None:
                init_bias = float(
                    torch.log(torch.tensor(self.prior / (1 - self.prior)))
                )
                self.model.final_classifier.bias.data.fill_(init_bias)
                self.console.log(
                    f"Initialized final classifier bias to {init_bias:.4f} based on prior={self.prior:.4f}",
                    style="green",
                )

    # _build_model is now handled by BaseTrainer, so the override is removed.

    def create_criterion(self):
        self.bce_loss = LaGAMBCELoss(ent_loss=self.params.get("ent_loss", False)).to(
            self.device
        )
        self.contrastive_loss = LaGAMContLoss(
            temperature=self.params.get("temperature", 0.07),
            cont_cutoff=self.params.get("cont_cutoff", False),
            knn_aug=self.params.get("knn_aug", False),
            num_neighbors=self.params.get("num_neighbors", 10),
            contrastive_clustering=self.params.get("contrastive_clustering", 1),
        )
        # Placeholder, as loss is computed manually
        return torch.nn.Identity()

    def run(self):
        self.before_training()

        warmup_epochs = self.params.get("warmup_epoch", 20)
        num_epochs = self.params.get("num_epochs", 400)

        # Warm-up stage
        self.console.log("\n--- [Stage 1/2] LaGAM: Warm-up ---", style="bold yellow")
        # Disable early stopping during warm-up
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = False
                self.checkpoint_handler.should_stop = False
            except Exception:
                pass
        self.current_stage = "warmup"
        self._run_epochs(warmup_epochs, stage_name="Warm-up")

        # Meta-learning stage (support both image and non-image datasets)
        ds_cls = str(self.params.get("dataset_class", "")).lower()
        is_image_like = any(tok in ds_cls for tok in ["cifar", "mnist", "fashionmnist"])

        # Reset early stopping counter before the main stage
        if self.checkpoint_handler and self.checkpoint_handler.early_stopping_enabled:
            self.console.log(
                "Resetting early stopping counter for Meta-learning stage.",
                style="blue",
            )
            if self.file_console:
                self.file_console.log(
                    "Resetting early stopping counter for Meta-learning stage."
                )
            self.checkpoint_handler.wait = 0
            self.checkpoint_handler.should_stop = False
        # Re-enable early stopping for meta-learning stage
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = True
            except Exception:
                pass

        if not is_image_like:
            self.console.log(
                "\n--- [Stage 2/2] LaGAM: Meta-learning (vector) ---",
                style="bold yellow",
            )
            self.current_stage = "meta"
            self._run_epochs(num_epochs - warmup_epochs, stage_name="Meta-learning")
        else:
            self.console.log(
                "\n--- [Stage 2/2] LaGAM: Meta-learning ---", style="bold yellow"
            )
            self.current_stage = "meta"
            # In LaGAM, epochs in _run_epochs are relative to the stage
            self._run_epochs(num_epochs - warmup_epochs, stage_name="Meta-learning")

        self.after_training()
        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()
        self._close_file_console()

    def train_one_epoch(self, epoch_idx: int):
        if self.current_stage == "warmup":
            self._train_epoch_warmup()
        else:
            self._train_epoch_meta(epoch_idx)

    def _train_epoch_warmup(self):
        self.model.train()
        mix_weight = float(self.params.get("mix_weight", 1.0))
        for batch in self.train_loader:
            # Support both wrapped image datasets: ((x_w,x_s), ...) and tabular: (x, ...)
            if isinstance(batch[0], (list, tuple)):
                x_w, x_s = batch[0]
                x = x_w.to(self.device)
                t, y_true, idx = (
                    batch[1].to(self.device),
                    batch[2].to(self.device),
                    batch[3].to(self.device),
                )
            else:
                x = batch[0].to(self.device)
                t, y_true, idx = (
                    batch[1].to(self.device),
                    batch[2].to(self.device),
                    batch[3].to(self.device),
                )
            t, y_true = t.to(self.device), y_true.to(self.device)

            # For warmup, treat L as 1 and U as 0
            labels_ = (t == 1).float().unsqueeze(1)
            labels = torch.cat([1 - labels_, labels_], dim=1).detach()

            self.optimizer.zero_grad()
            logits = self._forward_logits_no_feature(x)

            loss_cls = self.bce_loss(logits, labels)

            # Mixup (warm-up): follow the original beta(4,4) on weak branch
            x_mix, labels_a, labels_b, lam = mixup_data(
                x, labels, alpha=4.0, device=self.device
            )

            logits_mix = self._forward_logits_no_feature(x_mix)

            # For soft labels, we need a different mixup criterion
            loss_mix = lam * self.bce_loss(logits_mix, labels_a) + (
                1 - lam
            ) * self.bce_loss(logits_mix, labels_b)

            loss_final = loss_cls + mix_weight * loss_mix
            loss_final.backward()
            self.optimizer.step()

    def _forward_logits_no_feature(self, x):
        """Forward helper compatible with Meta models (flag_feature) and plain models."""
        try:
            out = self.model(x, flag_feature=False)
        except TypeError:
            out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _forward_logits_and_features(self, x):
        """Try to get both logits and penultimate features.
        Falls back to using input as feature if the model does not expose features.
        """
        preds = None
        feat = None
        try:
            out = self.model(x, flag_feature=True)
            if isinstance(out, tuple):
                if len(out) >= 2:
                    preds, feat = out[0], out[1]
                else:
                    preds = out[0]
            else:
                preds = out
        except TypeError:
            out = self.model(x)
            if isinstance(out, tuple):
                preds = out[0]
            else:
                preds = out
        # Fallback feature
        if feat is None:
            if isinstance(x, torch.Tensor):
                feat = x.view(x.size(0), -1)
            else:
                # As a last resort, use logits as feature
                feat = preds
                if isinstance(feat, torch.Tensor) and feat.dim() == 1:
                    feat = feat.unsqueeze(1)
        return preds, feat

    def _train_epoch_meta(self, epoch_idx):
        self.model.train()
        mix_weight = float(self.params.get("mix_weight", 1.0))

        # 1. Compute features and run k-means
        features = self.compute_features()
        if features is None:
            return  # Skip epoch if feature computation fails
        cluster_result = run_kmeans(
            features,
            self.params.get("num_cluster", 100),
            self.device.index,
            self.params.get("temperature", 0.07),
        )

        rho_end = self.params.get("rho_end", 0.8)
        rho_start = self.params.get("rho_start", 0.95)

        # EMA parameter calculation (matching original LaGAM)
        # Original uses 0-based epoch indexing, we use 1-based global_epoch
        total_epochs = self.params.get("num_epochs", 400)
        current_epoch_0_based = self.global_epoch - 1  # Convert to 0-based indexing
        ema_param = (
            1.0 * current_epoch_0_based / total_epochs * (rho_end - rho_start)
            + rho_start
        )

        valid_loader_iter = iter(self.validation_loader)

        # Correctly determine dataset size when using a Subset for training
        base_train_dataset = self.train_loader.dataset.base_dataset
        if isinstance(base_train_dataset, torch.utils.data.Subset):
            dataset_size = len(base_train_dataset.dataset)
        else:
            dataset_size = len(base_train_dataset)

        all_updated_labels = torch.zeros(dataset_size, dtype=torch.float32)
        all_indices = torch.zeros(dataset_size, dtype=torch.int64)
        update_mask = torch.zeros(dataset_size, dtype=torch.bool)

        for i, ((images_w, images_s), pu_labels, true_labels, index, _) in enumerate(
            self.train_loader
        ):
            images_w, images_s, pu_labels, true_labels, index = (
                images_w.to(self.device),
                images_s.to(self.device),
                pu_labels.to(self.device),
                true_labels.to(self.device),
                index.to(self.device),
            )

            bs = len(pu_labels)
            labels_ = (pu_labels == 1).float().unsqueeze(1)
            labels = torch.cat([1 - labels_, labels_], dim=1).detach()

            # Meta-learning for label correction
            # Build a fresh copy of current model weights; if model lacks MetaModule
            # interfaces, fall back to a vanilla deepcopy/standard params update.
            try:
                meta_model = select_model(
                    method="lagam", params=self.params, prior=self.prior
                ).to(self.device)
                # Try strict=False in case of minor head/shape mismatch on vector models
                meta_model.load_state_dict(self.model.state_dict(), strict=False)
            except Exception:
                import copy as _copy

                meta_model = _copy.deepcopy(self.model)

            # Safe forward for meta model (no feature flag if unsupported)
            try:
                preds_meta = meta_model(images_w, flag_feature=False)
            except TypeError:
                preds_meta = meta_model(images_w)
            if isinstance(preds_meta, tuple):
                preds_meta = preds_meta[0]

            eps = to_var(torch.zeros(bs, 2, device=self.device))
            labels_meta = labels + eps
            loss = self.bce_loss(preds_meta, labels_meta)

            meta_model.zero_grad()

            # Prefer MetaModule API; otherwise fallback to standard parameter list
            if hasattr(meta_model, "named_params"):
                params = [
                    p
                    for name, p in meta_model.named_params(meta_model)
                    if "classifier" in name and len(p.shape) > 1
                ]
            else:
                params = [p for p in meta_model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                loss, params, create_graph=True, allow_unused=True
            )

            meta_lr = self.params.get("meta_lr", 0.001)
            # Apply parameter update according to available API
            if hasattr(meta_model, "update_params"):
                try:
                    meta_model.update_params(
                        meta_lr, source_params=grads, identifier="classifier"
                    )
                except Exception:
                    meta_model.update_params(meta_lr, source_params=grads)
            else:
                # Manual SGD-style update
                with torch.no_grad():
                    for p, g in zip(params, grads):
                        if g is not None:
                            p.copy_(p - meta_lr * g)

            try:
                images_v, _, labels_v, _, _ = next(valid_loader_iter)
            except StopIteration:
                valid_loader_iter = iter(self.validation_loader)
                images_v, _, labels_v, _, _ = next(valid_loader_iter)

            images_v, labels_v = images_v.to(self.device), labels_v.to(self.device)
            labels_v_one_hot = F.one_hot(labels_v, 2).float()

            try:
                preds_v = meta_model(images_v, flag_feature=False)
            except TypeError:
                preds_v = meta_model(images_v)
            if isinstance(preds_v, tuple):
                preds_v = preds_v[0]

            loss_meta_v = self.bce_loss(preds_v, labels_v_one_hot)
            grad_tuple = torch.autograd.grad(
                loss_meta_v, eps, only_inputs=True, allow_unused=True
            )
            grad_eps = grad_tuple[0]
            if grad_eps is None:
                grad_eps = torch.zeros_like(eps)

            eps = eps - grad_eps
            meta_detected_labels = eps.argmax(dim=1)
            meta_detected_labels[pu_labels == 1] = 1
            meta_detected_labels = F.one_hot(meta_detected_labels, 2).float().detach()

            updated_labels = labels * ema_param + meta_detected_labels * (1 - ema_param)
            labels_final = updated_labels.detach()

            # Store updated labels for dataset update at the end of epoch
            all_updated_labels[index] = labels_final[:, 1].cpu()
            all_indices[index] = index.cpu()
            update_mask[index] = True

            # Main model training (BCE + Contrastive + Mixup)
            self.optimizer.zero_grad()

            preds_final, feat_cont_w = self._forward_logits_and_features(images_w)
            _, feat_cont_s = self._forward_logits_and_features(images_s)

            loss_cls = self.bce_loss(preds_final, labels_final)

            cluster_idxes = cluster_result["im2cluster"][index]
            loss_cont = self.contrastive_loss(
                feat_cont_w,
                feat_cont_s,
                cluster_idxes,
                preds_final,
                start_knn_aug=self.global_epoch > 50,
            )

            # Mixup on weak branch (same as warm-up)
            x_mix, labels_a, labels_b, lam = mixup_data(
                images_w, labels_final, alpha=4.0, device=self.device
            )
            logits_mix = self._forward_logits_no_feature(x_mix)
            if isinstance(logits_mix, tuple):
                logits_mix = logits_mix[0]
            loss_mix = lam * self.bce_loss(logits_mix, labels_a) + (
                1 - lam
            ) * self.bce_loss(logits_mix, labels_b)

            loss_final = (
                loss_cls
                + mix_weight * loss_mix
                + self.params.get("cont_weight", 1.0) * loss_cont
            )
            loss_final.backward()
            self.optimizer.step()

        # Update dataset pseudo-labels at the end of the epoch
        base_dataset = self.train_loader.dataset.base_dataset
        if isinstance(base_dataset, PUDataset):
            final_indices_to_update = all_indices[update_mask]
            final_labels_to_update = all_updated_labels[update_mask]

            # In PU learning, Labeled samples always have label 1.
            # We only update the pseudo-labels for Unlabeled samples.
            original_pu_labels = base_dataset.pu_labels[final_indices_to_update]
            unlabeled_mask_in_batch = original_pu_labels == -1

            indices_to_actually_update = final_indices_to_update[
                unlabeled_mask_in_batch
            ]
            labels_to_actually_update = final_labels_to_update[unlabeled_mask_in_batch]

            base_dataset.pseudo_labels[indices_to_actually_update] = (
                labels_to_actually_update
            )

            # Log accuracy of corrected labels
            true_labels_updated = base_dataset.true_labels[indices_to_actually_update]
            corrected_binary_labels = (labels_to_actually_update > 0.5).long()
            acc = (corrected_binary_labels == true_labels_updated).float().mean()
            self.console.log(
                f"Meta-label correction accuracy on U set: {acc:.4f}", style="cyan"
            )

    def compute_features(self):
        """Use eval_loader (test transforms) to compute features like original LaGAM."""
        self.model.eval()
        # Use the existing eval_loader from BaseTrainer
        if not hasattr(self, "eval_loader") or self.eval_loader is None:
            # Fallback: use train_loader (but this is not preferred)
            loader = self.train_loader
            # Correctly determine dataset size when using a Subset for training
            base_train_dataset = loader.dataset.base_dataset
            if isinstance(base_train_dataset, torch.utils.data.Subset):
                num_samples = len(base_train_dataset.dataset)
            else:
                num_samples = len(base_train_dataset)
        else:
            loader = self.eval_loader
            num_samples = len(self.eval_loader.dataset)

        all_feats = None
        with torch.no_grad():
            running_offset = 0
            for batch in loader:
                # LaGAMDatasetWrapper yields ((x_w, x_s), pu_labels, true_labels, indices, pseudo)
                # Standard PU loaders may yield (x, t, y_true, indices, pseudo) or (x, y)
                images = None
                indices = None

                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    first = batch[0]
                    # Case A: ((x_w, x_s), ...)
                    if (
                        isinstance(first, (list, tuple))
                        and len(first) >= 1
                        and isinstance(first[0], torch.Tensor)
                    ):
                        images = first[0]
                        if len(batch) >= 4 and isinstance(batch[3], torch.Tensor):
                            indices = batch[3]
                    # Case B: (x, y)
                    elif isinstance(first, torch.Tensor) and len(batch) == 2:
                        images = first
                        indices = torch.arange(images.size(0), device=images.device)
                    # Case C: (x, t, y_true, indices, pseudo)
                    elif (
                        isinstance(first, torch.Tensor)
                        and len(batch) >= 4
                        and isinstance(batch[3], torch.Tensor)
                    ):
                        images = first
                        indices = batch[3]

                if images is None:
                    # Skip if batch format is not recognized
                    continue

                images = images.to(self.device)
                _, feat = self._forward_logits_and_features(images)
                if all_feats is None:
                    all_feats = torch.zeros(
                        num_samples, feat.shape[1], device=self.device
                    )

                if indices is not None:
                    try:
                        all_feats[indices] = feat
                    except Exception:
                        # Fallback to sequential fill
                        end = min(running_offset + feat.size(0), num_samples)
                        all_feats[running_offset:end] = feat[: end - running_offset]
                        running_offset = end
                else:
                    end = min(running_offset + feat.size(0), num_samples)
                    all_feats[running_offset:end] = feat[: end - running_offset]
                    running_offset = end

        self.model.train()
        if all_feats is None:
            self.console.log(
                "Failed to compute features. Dataloader might be empty.", style="red"
            )
            return None
        return all_feats.cpu().numpy()
