import math
import torch
from torch import nn
import torch.nn.functional as F


class VPULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, args):
        super(VPULoss, self).__init__()
        self.mix_alpha = args.mix_alpha
        self.name = "vpu"

    def forward(self, output_phi_all, targets, out_log_phi_all, sam_target, lam):
        log_phi_x = output_phi_all
        log_phi_p = output_phi_all[targets == 1]
        var_loss = (
            torch.logsumexp(log_phi_x, dim=0)
            - math.log(len(log_phi_x))
            - 1 * torch.mean(log_phi_p)
        )
        reg_mix_log = ((torch.log(sam_target) - out_log_phi_all) ** 2).mean()

        phi_loss = var_loss + lam * reg_mix_log

        return phi_loss
