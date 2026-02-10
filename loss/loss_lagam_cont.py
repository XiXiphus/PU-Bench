import torch
import torch.nn.functional as F
import torch.nn as nn


class LaGAMContLoss(nn.Module):
    """
    Contrastive Loss from LaGAM.

    This loss function encourages feature representations to be closer for samples
    that are semantically similar. Similarity can be defined by cluster assignments,
    k-NN, and classifier predictions.
    """

    def __init__(
        self,
        temperature=0.07,
        cont_cutoff=False,
        knn_aug=False,
        num_neighbors=0,
        contrastive_clustering=1,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrastive_clustering = contrastive_clustering
        self.cont_cutoff = cont_cutoff
        self.knn_aug = knn_aug
        self.num_neighbors = num_neighbors

    def forward(self, q, k, cluster_idxes=None, preds=None, start_knn_aug=False):
        """
        Args:
            q (torch.Tensor): Features from the weak augmentation branch (query).
            k (torch.Tensor): Features from the strong augmentation branch (key).
            cluster_idxes (torch.Tensor, optional): Cluster assignments for each sample.
            preds (torch.Tensor, optional): Predictions from the classifier.
            start_knn_aug (bool, optional): Whether to enable k-NN augmentation.
        """
        device = q.device
        batch_size = q.shape[0]

        # Concatenate query and key features for easier matrix operations
        q_and_k = torch.cat([q, k], dim=0)

        # Calculate logits (dot product similarity)
        l_i = torch.einsum("nc,kc->nk", [q, q_and_k]) / self.temperature

        # Standard InfoNCE loss part
        # Create a mask to exclude self-similarity
        self_mask = torch.ones_like(l_i, dtype=torch.float)
        self_mask = (
            torch.scatter(self_mask, 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
            .detach()
        )

        # Positive mask: each sample 'q' is only positive to its augmented counterpart 'k'
        positive_mask_i = torch.zeros_like(l_i, dtype=torch.float)
        positive_mask_i = (
            torch.scatter(
                positive_mask_i,
                1,
                batch_size + torch.arange(batch_size).view(-1, 1).to(device),
                1,
            )
            .detach()
        )

        l_i_exp = torch.exp(l_i)
        l_i_exp_sum = torch.sum((l_i_exp * self_mask), dim=1, keepdim=True)

        # Standard contrastive loss for the augmented pair
        loss = -torch.sum(
            torch.log(l_i_exp / l_i_exp_sum) * positive_mask_i, dim=1
        ).mean()

        # Clustering-based contrastive loss part
        if cluster_idxes is not None and self.contrastive_clustering:
            cluster_idxes = cluster_idxes.view(-1, 1)
            cluster_idxes_kq = torch.cat([cluster_idxes, cluster_idxes], dim=0)
            # Samples in the same cluster are considered positive pairs
            mask = torch.eq(cluster_idxes, cluster_idxes_kq.T).float().to(device)

            # Optional: Cut off by classifier predictions
            if self.cont_cutoff:
                preds = preds.detach()
                pred_labels = (preds > 0.5) * 1
                pred_labels = pred_labels.view(-1, 1)
                pred_labels_kq = torch.cat([pred_labels, pred_labels], dim=0)
                label_mask = torch.eq(pred_labels, pred_labels_kq.T).float().to(device)
                mask = mask * label_mask

            # Optional: Augment positives with k-Nearest Neighbors
            if self.knn_aug and start_knn_aug:
                cosine_corr = q @ q_and_k.T
                available_neighbors = cosine_corr.size(-1)
                # Ensure k is within valid range [1, available_neighbors]
                k_eff = int(min(max(int(self.num_neighbors), 0), available_neighbors))
                if k_eff > 0:
                    _, kNN_index = torch.topk(
                        cosine_corr, k=k_eff, dim=-1, largest=True
                    )
                    mask_kNN = torch.scatter(torch.zeros_like(mask), 1, kNN_index, 1)
                    mask = ((mask + mask_kNN) > 0.5) * 1

            mask = mask.float().detach().to(device)
            anchor_dot_contrast = torch.div(
                torch.matmul(q, q_and_k.T), self.temperature
            )
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # Mask out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
            )
            mask = mask * logits_mask

            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # Compute mean log-likelihood for positive pairs
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

            # The clustering loss is the negative of this mean log-likelihood
            loss_prot = -mean_log_prob_pos.mean()
            loss += loss_prot

        return loss
