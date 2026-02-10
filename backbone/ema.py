from copy import deepcopy

import torch


class ModelEMA(object):
    """Exponential Moving Average wrapper for model parameters."""

    def __init__(self, args_or_trainer, model, decay: float = 0.999):
        """Args can be Trainer or object containing device attribute."""
        # Compatible with passed trainer or args
        device = getattr(args_or_trainer, "device", "cpu")
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, "module")
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        needs_module = hasattr(model, "module") and not self.ema_has_module
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in self.param_keys:
            j = f"module.{k}" if needs_module else k
            model_v = msd[j].detach()
            ema_v = esd[k]
            esd[k].copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)

        for k in self.buffer_keys:
            j = f"module.{k}" if needs_module else k
            esd[k].copy_(msd[j])
