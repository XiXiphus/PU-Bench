"""CGenPU trainer package for PU-Bench.

Reference implementation:
    apapich/CGenPU, mirrored under
    ``author source file(s): cgenpu``.
    Active source files are ``pu/CGenPU.py`` and ``pu/misc/utils.py``.

This is a PyTorch adaptation for PU-Bench.  The source loop updates the
discriminator, auxiliary classifier, then generator for each unlabeled batch;
positive latent class is one-hot index 0 and negative latent class is index 1.
"""

from __future__ import annotations

import torch
from torch import nn

from ..base_trainer import BaseTrainer
from ..utils.checkpointing import CheckpointBundle
from .models import (
    AClassifierCNN,
    CGenPUConvDiscriminatorADNI,
    CGenPUConvDiscriminatorCIFAR,
    CGenPUConvDiscriminatorMNIST,
    CGenPUConvGeneratorADNI,
    CGenPUConvGeneratorCIFAR,
    CGenPUConvGeneratorMNIST,
    CGenPUDiscriminator,
    CGenPUGenerator,
)


def _kld(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7, reduce_mean: bool = True
) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    v = p * torch.log(p / q)
    return v.mean() if reduce_mean else v.sum()


class CGenPUTrainer(BaseTrainer):
    """CGenPU D/A/G trainer with PU-Bench evaluation and checkpoint hooks."""

    def before_training(self):
        super().before_training()
        self.latent_dim = int(self.params.get("latent_dim", 128))
        self.aux_strength = float(self.params.get("aux_strength", 1.0))
        self.z_type = str(self.params.get("z_type", "uniform"))  # "uniform" | "normal"
        self.reduce_mean = bool(
            self.params.get("reduce_mean", True)
        )  # Match original TF implementation
        # Benchmark-only convenience, disabled by default for source-like runs.
        self.prior_reg = float(self.params.get("prior_reg", 0.0))
        # ADNI lightweight augmentation is not in source; keep opt-in only.
        self.aug_flip_prob = float(self.params.get("adni_aug_flip_prob", 0.0))
        self.aug_noise_std = float(self.params.get("adni_aug_noise_std", 0.0))
        if self.file_console:
            self.file_console.log("CGenPU source alignment:")
            self.file_console.log("  source=apapich/CGenPU")
            self.file_console.log(
                f"  latent_dim={self.latent_dim}, z_type={self.z_type}, "
                f"aux_strength={self.aux_strength}, reduce_mean={self.reduce_mean}"
            )

    def create_criterion(self):
        # BCE for D with probabilities; for A we use KLD terms directly
        return nn.BCELoss(reduction="mean")

    def get_checkpoint_model(self):
        return CheckpointBundle(
            modules={
                "discriminator": self.D,
                "auxiliary": self.A,
                "generator": self.G,
            },
            optimizers={
                "optimizer_d": self.D_opt,
                "optimizer_a": self.A_opt,
                "optimizer_g": self.G_opt,
            },
        )

    def _build_model(self):
        # Ensure hyperparams are available even though before_training() hasn't run yet
        if not hasattr(self, "latent_dim"):
            self.latent_dim = int(self.params.get("latent_dim", 128))
        if not hasattr(self, "aux_strength"):
            self.aux_strength = float(self.params.get("aux_strength", 1.0))
        if not hasattr(self, "z_type"):
            self.z_type = str(self.params.get("z_type", "uniform"))
        if not hasattr(self, "reduce_mean"):
            self.reduce_mean = bool(self.params.get("reduce_mean", True))

        # Skip BaseTrainer model; CGenPU has three networks
        # Use input shape from BaseTrainer
        input_shape = self.input_shape

        # Treat any 3D tensor as image; specialize by common shapes
        is_image = len(input_shape) == 3
        is_mnist_like = is_image and input_shape[0] == 1 and input_shape[-1] == 28
        is_cifar_like = is_image and input_shape[0] == 3 and input_shape[-1] == 32
        is_adni_like = is_image and input_shape[0] == 1 and input_shape[-1] == 128
        # Attach to instance for augmentation detection
        self.is_adni_like = bool(is_adni_like)

        if is_cifar_like:
            # CNN discriminators/generators
            self.D = CGenPUConvDiscriminatorCIFAR(in_channels=3).to(self.device)
            self.G = CGenPUConvGeneratorCIFAR(self.latent_dim + 2, out_channels=3).to(
                self.device
            )
            # A uses CNN classifier backbone wrapped with sigmoid
            from backbone.models import CNN_CIFAR10

            self.A = AClassifierCNN(CNN_CIFAR10()).to(self.device)
        elif is_mnist_like:
            self.D = CGenPUConvDiscriminatorMNIST(in_channels=1).to(self.device)
            self.G = CGenPUConvGeneratorMNIST(self.latent_dim + 2, out_channels=1).to(
                self.device
            )
            from backbone.models import CNN_MNIST

            self.A = AClassifierCNN(CNN_MNIST()).to(self.device)
        elif is_adni_like:
            # Alzheimer MRI (ADNI-like) grayscale 1x128x128
            self.D = CGenPUConvDiscriminatorADNI(in_channels=1).to(self.device)
            self.G = CGenPUConvGeneratorADNI(self.latent_dim + 2, out_channels=1).to(
                self.device
            )
            from backbone.models import CNN_AlzheimerMRI

            self.A = AClassifierCNN(CNN_AlzheimerMRI()).to(self.device)
            if bool(self.params.get("prior_bias_init", False)):
                try:
                    from math import log

                    def _logit(p: float) -> float:
                        eps = 1e-6
                        p = max(min(float(p), 1 - eps), eps)
                        return log(p / (1 - p))

                    if hasattr(self.A, "backbone") and hasattr(
                        self.A.backbone, "final_classifier"
                    ):
                        fc = self.A.backbone.final_classifier
                        if hasattr(fc, "bias") and fc.bias is not None:
                            with torch.no_grad():
                                fc.bias.fill_(_logit(self.prior))
                except Exception:
                    pass
        else:
            # Fallback to MLP for vectors/text/tabular
            self.D = CGenPUDiscriminator(input_shape).to(self.device)
            # Source examples use the same sigmoid MLP shape for D and A.
            self.A = CGenPUDiscriminator(input_shape).to(self.device)
            # Choose generator output activation based on config (defaults: tanh for vectors)
            gen_out_activation = str(
                self.params.get("gen_out_activation", "tanh")
            ).lower()
            self.G = CGenPUGenerator(
                input_shape, self.latent_dim + 2, out_activation=gen_out_activation
            ).to(self.device)

        # Optimizers (allow different lrs via params)
        lr_d = float(self.params.get("lr_d", self.params.get("lr", 1e-4)))
        lr_a = float(self.params.get("lr_a", self.params.get("lr", 1e-4)))
        lr_g = float(self.params.get("lr_g", self.params.get("lr", 1e-4)))
        betas = (
            float(self.params.get("beta1", 0.5)),
            float(self.params.get("beta2", 0.999)),
        )

        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=betas)
        self.A_opt = torch.optim.Adam(self.A.parameters(), lr=lr_a, betas=betas)
        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=betas)

        # BCE for D real/fake (now expects probabilities, not logits)
        self.bce_loss = nn.BCELoss(reduction="mean")

        # Expose A as the evaluation model so that BaseTrainer can compute metrics
        # (A outputs probabilities in [0,1], evaluate_metrics handles this case.)
        self.model = self.A
        # Provide a default criterion to satisfy BaseTrainer expectations
        self.criterion = self.create_criterion()

    def _augment_adni(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "is_adni_like", False):
            return x
        # Use A model's training flag as reference, Trainer itself doesn't inherit nn.Module
        try:
            if not (hasattr(self, "A") and getattr(self.A, "training", False)):
                return x
        except Exception:
            return x
        # Horizontal flip
        if self.aug_flip_prob > 0:
            b = x.size(0)
            flip_mask = (torch.rand(b, device=x.device) < self.aug_flip_prob).view(
                b, 1, 1, 1
            )
            x_flipped = torch.flip(x, dims=[3])
            x = torch.where(flip_mask, x_flipped, x)
        # Light Gaussian noise and clip back to [-1, 1]
        if self.aug_noise_std > 0:
            noise = torch.randn_like(x) * self.aug_noise_std
            x = torch.clamp(x + noise, -1.0, 1.0)
        return x

    # Sampling helpers
    def _sample_noise(self, shape: tuple[int, ...]) -> torch.Tensor:
        if self.z_type == "normal":
            return torch.randn(*shape, device=self.device)
        # default uniform in [-1, 1]
        return torch.rand(*shape, device=self.device) * 2.0 - 1.0

    def _concat_class(self, z: torch.Tensor, label_idx: int) -> torch.Tensor:
        b = z.size(0)
        one_hot = torch.zeros(b, 2, device=self.device)
        one_hot[:, label_idx] = 1.0
        return torch.cat([z, one_hot], dim=1)

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(enabled)

    def train_one_epoch(self, epoch_idx: int):
        self.D.train()
        self.A.train()
        self.G.train()

        # Build P and U loaders following original TF implementation logic
        full_ds = self.train_loader.dataset
        pos_indices = (full_ds.pu_labels == 1).nonzero().squeeze()
        unl_indices = (full_ds.pu_labels == -1).nonzero().squeeze()

        batch_size = int(self.params.get("batch_size", 128))

        # U loader: drop_last=True (equivalent to TF's drop_remainder=True)
        u_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full_ds, unl_indices),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # P loader is consumed as an infinite stream before batching, matching
        # source ``xp_dataset.repeat().batch(batch_size)``.
        p_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full_ds, pos_indices),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        p_iter = iter(p_loader)
        p_pending: torch.Tensor | None = None

        def get_next_p_batch() -> torch.Tensor:
            nonlocal p_iter, p_pending
            parts: list[torch.Tensor] = []
            remaining = batch_size
            while remaining > 0:
                if p_pending is None:
                    try:
                        batch = next(p_iter)
                    except StopIteration:
                        p_iter = iter(p_loader)
                        batch = next(p_iter)
                    chunk = batch[0] if isinstance(batch, (tuple, list)) else batch
                else:
                    chunk = p_pending
                    p_pending = None

                if chunk.size(0) > remaining:
                    parts.append(chunk[:remaining])
                    p_pending = chunk[remaining:]
                    remaining = 0
                else:
                    parts.append(chunk)
                    remaining -= chunk.size(0)
            return torch.cat(parts, dim=0)

        # Loop based on U data batches (matching original TF implementation)
        for x_u, *_ in u_loader:
            x_pl = get_next_p_batch()
            # Ensure tensors are float and on correct device for MLP case
            x_pl = x_pl.to(self.device)
            x_u = x_u.to(self.device)
            if x_pl.dim() == 2 and not torch.is_floating_point(x_pl):
                x_pl = x_pl.float()
            if x_u.dim() == 2 and not torch.is_floating_point(x_u):
                x_u = x_u.float()
            # ADNI simple augmentation (training only)
            x_pl = self._augment_adni(x_pl)
            x_u = self._augment_adni(x_u)

            # 1) Prepare conditional noise and generate synthetic P/N
            z_p = self._sample_noise((x_pl.size(0), self.latent_dim))
            z_n = self._sample_noise((x_u.size(0), self.latent_dim))
            zc_p = self._concat_class(
                z_p, 0
            )  # class index 0 -> positive (align with original)
            zc_n = self._concat_class(
                z_n, 1
            )  # class index 1 -> negative (align with original)

            self.G.eval()
            with torch.no_grad():
                gp = self.G(zc_p)
                gn = self.G(zc_n)

            # 2) Update D (real=1, fake=0)
            self.D.train()
            self.D_opt.zero_grad()
            d_xp = self.D(x_pl)
            d_xu = self.D(x_u)
            d_gp = self.D(gp)
            d_gn = self.D(gn)
            loss_r = self.bce_loss(d_xp, torch.ones_like(d_xp)) + self.bce_loss(
                d_xu, torch.ones_like(d_xu)
            )
            loss_f = self.bce_loss(d_gp, torch.zeros_like(d_gp)) + self.bce_loss(
                d_gn, torch.zeros_like(d_gn)
            )
            loss_d = 0.5 * (loss_r + loss_f)
            loss_d.backward()
            self.D_opt.step()

            # 3) Update A (KL terms on probabilities)
            self.A.train()
            self.A_opt.zero_grad()
            a_xp = self.A(x_pl).view(-1)
            a_gp = self.A(gp).view(-1)
            a_gn = self.A(gn).view(-1)
            ones = torch.ones_like(a_xp)
            # KLD terms as in TF implementation
            la = (
                _kld(ones, a_xp, reduce_mean=self.reduce_mean)
                + _kld(a_xp, a_gp, reduce_mean=self.reduce_mean)
                + _kld(a_gp, 1.0 - a_gn, reduce_mean=self.reduce_mean)
            )
            if self.prior_reg > 0:
                a_xu = self.A(x_u).view(-1)
                pi = torch.as_tensor(self.prior, device=self.device, dtype=a_xu.dtype)
                mean_u = a_xu.mean()
                pi = torch.clamp(pi, 1e-6, 1 - 1e-6)
                mean_u = torch.clamp(mean_u, 1e-6, 1 - 1e-6)
                prior_kl = pi * torch.log(pi / mean_u) + (1 - pi) * torch.log(
                    (1 - pi) / (1 - mean_u)
                )
                la = la + self.prior_reg * prior_kl
            (self.aux_strength * la).backward()
            self.A_opt.step()

            # 4) Update G (fool D + align A)
            self.G.train()
            self.D.eval()
            self.A.eval()
            self._set_requires_grad(self.D, False)
            self._set_requires_grad(self.A, False)
            self.G_opt.zero_grad()
            z_p = self._sample_noise((x_pl.size(0), self.latent_dim))
            z_n = self._sample_noise((x_u.size(0), self.latent_dim))
            zc_p = self._concat_class(z_p, 0)
            zc_n = self._concat_class(z_n, 1)
            try:
                gp = self.G(zc_p)
                gn = self.G(zc_n)

                # Source uses A/D with training=False during the G step.
                with torch.no_grad():
                    a_xp = self.A(x_pl).view(-1)
                d_gp = self.D(gp)
                d_gn = self.D(gn)
                a_gp = self.A(gp).view(-1)
                a_gn = self.A(gn).view(-1)

                # GAN generator loss (make D think fakes are real)
                lg = self.bce_loss(d_gp, torch.ones_like(d_gp)) + self.bce_loss(
                    d_gn, torch.ones_like(d_gn)
                )
                # Auxiliary alignment
                la_g = _kld(a_xp, a_gp, reduce_mean=self.reduce_mean) + _kld(
                    a_gp, 1.0 - a_gn, reduce_mean=self.reduce_mean
                )
                (lg + self.aux_strength * la_g).backward()
                self.G_opt.step()
            finally:
                self._set_requires_grad(self.D, True)
                self._set_requires_grad(self.A, True)

            # Restore training mode for next iteration
            self.D.train()
            self.A.train()
            self.G.train()

    # Override run to reuse BaseTrainer epoch runner for evaluation & checkpointing
    def run(self):
        self.before_training()
        final = self._run_epochs(
            self.params.get("num_epochs", self.params.get("epochs", 200))
        )
        self.after_training()

        # Log best metrics summary like other methods
        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()
        # Close file console after logging completion
        self._close_file_console()
        return final
