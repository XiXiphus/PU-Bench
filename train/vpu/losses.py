import math
import torch
from torch import nn


class VPULoss(nn.Module):
    """Source-faithful VPU variational objective.

    Primary source:
        https://github.com/HC-Feynman/vpu/blob/master/vpu.py

    The source implementation trains a log-softmax model Phi and uses column 1 as
    ``log phi``.  Its per-step objective is
    ``logmeanexp(log_phi_x) - mean(log_phi_p) + lam * mixup_regularizer``.
    """

    def __init__(self, args):
        super(VPULoss, self).__init__()
        self.mix_alpha = float(args.mix_alpha)
        self.lam = float(args.lam)
        self.name = "vpu"

    def forward(self, log_phi_x, log_phi_p, out_log_phi_mix, mix_target):
        var_loss = (
            torch.logsumexp(log_phi_x, dim=0)
            - math.log(len(log_phi_x))
            - torch.mean(log_phi_p)
        )
        reg_mix_log = ((torch.log(mix_target) - out_log_phi_mix) ** 2).mean()

        phi_loss = var_loss + self.lam * reg_mix_log

        return phi_loss, var_loss, reg_mix_log


__all__ = ["VPULoss"]
