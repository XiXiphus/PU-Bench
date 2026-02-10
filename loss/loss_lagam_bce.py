import torch
import torch.nn.functional as F
import torch.nn as nn


class LaGAMBCELoss(nn.Module):
    """
    Custom Binary Cross Entropy loss from LaGAM, designed for soft labels
    and optional entropy regularization.
    """

    def __init__(self, ent_loss=False):
        super().__init__()
        self.ent_loss = ent_loss

    def forward(self, preds, label, weight=None):
        preds = torch.sigmoid(preds)
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)

        logits_ = torch.cat([1.0 - preds, preds], dim=1)
        logits_ = torch.clamp(logits_, 1e-4, 1.0 - 1e-4)

        loss_entries = (-label * logits_.log()).sum(dim=0)
        # Normalize by the sum of labels in each class to handle soft labels
        label_sum = label.sum(dim=0)
        # Add a small epsilon to avoid division by zero if a class has no samples
        label_num_reverse = 1.0 / (label_sum + 1e-8)
        loss = (loss_entries * label_num_reverse).sum()

        if self.ent_loss:
            # Encourages confident predictions
            loss_ent = -(logits_ * logits_.log()).sum(1).mean()
            loss = loss + loss_ent * 0.1
        return loss
