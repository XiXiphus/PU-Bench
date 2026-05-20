from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from ..base_trainer import BaseTrainer
from ..augmentations.vector import (
    VectorStrongAugment,
    VectorWeakAugment,
)
from ..mixup import mixup_data
from ..reproducibility import seed_worker
from data.data_utils import PUDataset
from .bce import LaGAMBCELoss
from .contrastive import LaGAMContLoss
from .dataset import (
    LaGAMDatasetWrapper,
    LaGAMEvalDatasetWrapper,
    LaGAMVectorDatasetWrapper,
    LaGAMVectorEvalDatasetWrapper,
)
from .meta_layers import to_var
from .model_selector import select_model


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

    k = int(num_cluster)
    k = max(1, min(k, len(x)))
    km = KMeans(n_clusters=k, n_init=5, max_iter=20, random_state=0)
    im2cluster = km.fit_predict(x).astype(int)
    centroids = km.cluster_centers_.astype(np.float32)
    distances = ((x - centroids[im2cluster]) ** 2).sum(axis=1, keepdims=True)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)

    # Compute density for each cluster (missing in original implementation)
    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(float(distances[im][0]))

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
    mean_density = density.mean()
    if mean_density <= 1e-12:
        density = np.full_like(density, float(temperature))
    else:
        density = temperature * density / mean_density

    # Convert to tensors on available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im2cluster = torch.LongTensor(im2cluster.tolist()).to(device)
    centroids = torch.tensor(centroids, dtype=torch.float32, device=device)
    density = torch.tensor(density, dtype=torch.float32, device=device)

    results["centroids"] = centroids
    results["im2cluster"] = im2cluster
    results["density"] = density

    return results


class LaGAMTrainer(BaseTrainer):
    """LaGAM trainer adapted to PU-Bench datasets.

    Primary source:
        author source file(s): lagam
        llong-cs/LaGAM at 362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8

    The official code combines warm-up, meta-label correction, mixup,
    clustering-based contrastive learning, and method-specific augmentations.
    This trainer keeps that estimator loop while placing LaGAM-only components
    under ``train/lagam/`` instead of shared benchmark directories.
    """

    def create_model(self):
        return select_model(params=self.params, prior=self.prior)

    def _prepare_data(self):
        super()._prepare_data()
        self._ensure_validation_split()

        ds_cls = str(self.params.get("dataset_class", "")).lower()
        is_image_like = any(
            token in ds_cls
            for token in ["cifar", "mnist", "fashionmnist", "alzheimer", "mri"]
        )

        if is_image_like:
            if "mnist" in ds_cls or "fashionmnist" in ds_cls:
                image_size = 28
                mean = getattr(self.train_loader.dataset, "mean", (0.5,))
                std = getattr(self.train_loader.dataset, "std", (0.5,))
            elif "alzheimer" in ds_cls or "mri" in ds_cls:
                image_size = 128
                mean = getattr(self.train_loader.dataset, "mean", (0.5,))
                std = getattr(self.train_loader.dataset, "std", (0.5,))
            else:
                image_size = 32
                mean = getattr(self.train_loader.dataset, "mean", (0.5, 0.5, 0.5))
                std = getattr(self.train_loader.dataset, "std", (0.5, 0.5, 0.5))

            wrapped_dataset = LaGAMDatasetWrapper(
                self.train_loader.dataset,
                image_size=image_size,
                mean=mean,
                std=std,
            )
            eval_dataset = LaGAMEvalDatasetWrapper(
                self.train_loader.dataset,
                mean=mean,
                std=std,
            )
        else:
            self.console.log(
                "LaGAM detected non-image dataset; enabling vector weak/strong augmentations.",
                style="yellow",
            )
            weak = VectorWeakAugment(
                noise_std=float(self.params.get("vec_weak_noise_std", 0.02)),
                dropout_ratio=float(self.params.get("vec_weak_dropout", 0.0)),
            )
            strong = VectorStrongAugment(
                noise_std=float(self.params.get("vec_strong_noise_std", 0.1)),
                dropout_ratio=float(self.params.get("vec_strong_dropout", 0.1)),
                sign_flip_ratio=float(self.params.get("vec_sign_flip_ratio", 0.05)),
            )
            wrapped_dataset = LaGAMVectorDatasetWrapper(
                self.train_loader.dataset,
                weak_aug=weak,
                strong_aug=strong,
            )
            eval_dataset = LaGAMVectorEvalDatasetWrapper(self.train_loader.dataset)

        self.train_loader = DataLoader(
            wrapped_dataset,
            batch_size=self.params.get("batch_size", 128),
            shuffle=True,
            num_workers=self.params.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.params.get("batch_size", 128),
            shuffle=True,
            num_workers=self.params.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
        )
        if is_image_like:
            if self.validation_loader is not None:
                self.validation_loader = DataLoader(
                    LaGAMEvalDatasetWrapper(
                        self.validation_loader.dataset,
                        mean=mean,
                        std=std,
                    ),
                    batch_size=self.params.get("batch_size", 128),
                    shuffle=False,
                    num_workers=self.params.get("num_workers", 0),
                    pin_memory=torch.cuda.is_available(),
                    worker_init_fn=seed_worker,
                )
            self.test_loader = DataLoader(
                LaGAMEvalDatasetWrapper(self.test_loader.dataset, mean=mean, std=std),
                batch_size=self.params.get("batch_size", 128),
                shuffle=False,
                num_workers=self.params.get("num_workers", 0),
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=seed_worker,
            )

        sample_data = next(iter(self.train_loader))[0]
        if isinstance(sample_data, (list, tuple)):
            self.input_shape = tuple(sample_data[0].shape[1:])
        else:
            self.input_shape = tuple(sample_data.shape[1:])

    def _ensure_validation_split(self) -> None:
        if self.validation_loader is not None and len(self.validation_loader.dataset) > 0:
            return

        val_ratio = float(self.params.get("lagam_val_ratio", 0.1))
        if val_ratio <= 0:
            raise RuntimeError(
                "LaGAM requires a validation loader for meta-label correction. "
                "Set lagam_val_ratio > 0 or provide a dataset validation split."
            )

        base_dataset = self.train_loader.dataset
        if not isinstance(base_dataset, PUDataset):
            raise RuntimeError(
                "LaGAM validation split expects an unwrapped PU-Bench PUDataset."
            )

        self.console.log(
            "LaGAM validation split missing; creating a method-private split from training data.",
            style="yellow",
        )
        all_indices = np.arange(len(base_dataset))
        new_train_indices, val_indices = train_test_split(
            all_indices,
            test_size=val_ratio,
            stratify=base_dataset.true_labels.cpu().numpy(),
            random_state=self.params.get("seed", 42),
        )

        train_dataset = self._subset_pu_dataset(base_dataset, new_train_indices)
        val_dataset = self._subset_pu_dataset(base_dataset, val_indices)
        self._copy_dataset_hints(base_dataset, train_dataset)
        self._copy_dataset_hints(base_dataset, val_dataset)

        batch_size = self.params.get("batch_size", 128)
        num_workers = self.params.get("num_workers", 0)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
        self.validation_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
        self.update_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

    def _subset_pu_dataset(
        self, dataset: PUDataset, selected_indices: np.ndarray
    ) -> PUDataset:
        selected_indices = np.asarray(selected_indices, dtype=int)
        idx_t = torch.as_tensor(selected_indices, dtype=torch.long)
        local_len = len(selected_indices)
        return PUDataset(
            features=dataset.features[idx_t],
            pu_labels=dataset.pu_labels[idx_t],
            true_labels=dataset.true_labels[idx_t],
            indices=torch.arange(local_len),
            pseudo_labels=dataset.pseudo_labels[idx_t],
            metadata=getattr(dataset, "pu_metadata", None),
            source_indices=getattr(dataset, "source_indices", torch.arange(len(dataset)))[
                idx_t
            ],
            source_roles=getattr(
                dataset,
                "source_roles",
                np.array(["U"] * len(dataset)),
            )[selected_indices],
        )

    @staticmethod
    def _copy_dataset_hints(source: PUDataset, target: PUDataset) -> None:
        for attr in ("mean", "std", "expected_image_size"):
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))

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
        original_early_stopping = (
            bool(self.checkpoint_handler.early_stopping_enabled)
            if self.checkpoint_handler
            else False
        )

        warmup_epochs = self.params.get("warmup_epoch", 20)
        num_epochs = self.params.get("num_epochs", 400)

        # Warm-up stage
        self.console.log("\n--- [Stage 1/2] LaGAM: Warm-up ---", style="bold yellow")
        # Disable early stopping during warm-up
        self.set_checkpoint_early_stopping(False)
        self.current_stage = "warmup"
        self.run_stage("Warm-up", warmup_epochs)

        # Meta-learning stage (support both image and non-image datasets)
        ds_cls = str(self.params.get("dataset_class", "")).lower()
        is_image_like = any(tok in ds_cls for tok in ["cifar", "mnist", "fashionmnist"])

        # Reset early stopping counter before the main stage
        if self.checkpoint_handler and original_early_stopping:
            self.console.log(
                "Resetting early stopping counter for Meta-learning stage.",
                style="blue",
            )
            if self.file_console:
                self.file_console.log(
                    "Resetting early stopping counter for Meta-learning stage."
                )
        # Re-enable early stopping for meta-learning stage
        self.set_checkpoint_early_stopping(original_early_stopping, reset=True)

        if not is_image_like:
            self.console.log(
                "\n--- [Stage 2/2] LaGAM: Meta-learning (vector) ---",
                style="bold yellow",
            )
            self.current_stage = "meta"
            self.run_stage("Meta-learning", num_epochs - warmup_epochs)
        else:
            self.console.log(
                "\n--- [Stage 2/2] LaGAM: Meta-learning ---", style="bold yellow"
            )
            self.current_stage = "meta"
            # In LaGAM, epochs in _run_epochs are relative to the stage
            self.run_stage("Meta-learning", num_epochs - warmup_epochs)

        self.finalize()

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

            # LaGAM trains on method-private targets: initial U=0, P=1, later U soft-corrected.
            labels_ = t.float().view(-1, 1)
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
            if isinstance(batch[0], (list, tuple)) and self.params.get("cont_weight", 1.0) != 0:
                _, feat_cont_w = self._forward_logits_and_features(x)
                _, feat_cont_s = self._forward_logits_and_features(x_s.to(self.device))
                loss_cont = self.contrastive_loss(
                    feat_cont_w,
                    feat_cont_s,
                    None,
                    logits,
                    start_knn_aug=False,
                )
                loss_final = loss_final + self.params.get("cont_weight", 1.0) * loss_cont
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
            labels_ = pu_labels.float().view(-1, 1)
            labels = torch.cat([1 - labels_, labels_], dim=1).detach()

            # Meta-learning for label correction. LaGAM requires a MetaModule
            # clone so the validation loss remains differentiable w.r.t. eps.
            meta_model = select_model(params=self.params, prior=self.prior).to(
                self.device
            )
            try:
                _ = meta_model(images_w[:1], flag_feature=False)
            except TypeError:
                _ = meta_model(images_w[:1])
            meta_model.load_state_dict(self.model.state_dict(), strict=False)
            if not (
                hasattr(meta_model, "named_params")
                and hasattr(meta_model, "update_params")
            ):
                raise RuntimeError(
                    "LaGAM requires a MetaModule backbone with named_params/update_params."
                )

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

            identifier = str(self.params.get("identifier", "classifier"))
            params = [
                p
                for name, p in meta_model.named_params(meta_model)
                if identifier in name and len(p.shape) > 1
            ]
            grads = torch.autograd.grad(
                loss, params, create_graph=True, allow_unused=True
            )

            meta_lr = self.params.get("meta_lr", 0.001)
            meta_model.update_params(
                meta_lr,
                source_params=grads,
                identifier=identifier,
            )

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

        # Update method-private LaGAM targets at the end of the epoch.
        train_dataset = self.train_loader.dataset
        if hasattr(train_dataset, "update_targets"):
            final_indices_to_update = all_indices[update_mask]
            final_labels_to_update = all_updated_labels[update_mask]
            train_dataset.update_targets(final_labels_to_update, final_indices_to_update)

            base_dataset = getattr(train_dataset, "base_dataset", None)
            if isinstance(base_dataset, PUDataset):
                base_dataset.pseudo_labels[final_indices_to_update] = final_labels_to_update
                true_labels_updated = base_dataset.true_labels[final_indices_to_update]
                corrected_binary_labels = (final_labels_to_update > 0.5).long()
                acc = (corrected_binary_labels == true_labels_updated).float().mean()
                self.console.log(
                    f"Meta-label correction accuracy on train targets: {acc:.4f}",
                    style="cyan",
                )

    def compute_features(self):
        """Use eval_loader (test transforms) to compute features like original LaGAM."""
        self.model.eval()
        if not hasattr(self, "eval_loader") or self.eval_loader is None:
            raise RuntimeError("LaGAM requires a deterministic eval_loader for clustering.")
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
