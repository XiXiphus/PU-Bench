from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from backbone.vaepu_models import (
    VAEencoder,
    VAEdecoder,
    Discriminator,
    ClassifierO,
    VAEConvEncoder,
    VAEConvDecoder,
    VAEConvDiscriminator,
)
from .train_utils import select_model, _zero_one_loss
from data.lagam_dataset import LaGAMDatasetWrapper


class VAEPUTrainer(BaseTrainer):
    """
    Trainer for VAE-PU method with adaptive configuration for different dataset types.
    """

    def before_training(self):
        super().before_training()

        # Get dataset name and apply adaptive configuration
        self.dataset_name = self._get_dataset_name()
        self._adapt_config_to_dataset()

        # Initialize PN decision threshold (will be updated on validation set)
        self.pn_decision_threshold = float(
            self.params.get("pn_decision_threshold", 0.0)
        )
        self.threshold_selection = str(
            self.params.get("threshold_selection", "f1")
        ).lower()

        # VAE-PU has a complex multi-stage process not driven by a single num_epochs
        self.pretrain_epochs = self.params.get("num_epoch_pre", 100)
        self.main_epochs = self.params.get("num_epoch", 800)
        self.step1_end = self.params.get("num_epoch_step1", 400)
        self.step2_end = self.params.get("num_epoch_step2", 500)
        self.step3_end = self.params.get("num_epoch_step3", 700)
        self.step_pn1_end = self.params.get("num_epoch_step_pn1", 500)
        self.step_pn2_end = self.params.get("num_epoch_step_pn2", 600)

    def _get_dataset_name(self):
        """Get dataset name for adaptive configuration"""
        dataset_name = ""

        if hasattr(self, "params") and "dataset" in self.params:
            dataset_name = str(self.params["dataset"])

        if (
            not dataset_name
            and hasattr(self, "train_loader")
            and hasattr(self.train_loader, "dataset")
        ):
            dataset = self.train_loader.dataset
            if hasattr(dataset, "dataset_name"):
                dataset_name = dataset.dataset_name
            elif hasattr(dataset, "__class__"):
                dataset_name = dataset.__class__.__name__

        if not dataset_name:
            dataset = getattr(self.train_loader, "dataset", None)
            if hasattr(dataset, "base_dataset"):
                base_dataset = dataset.base_dataset
                if hasattr(base_dataset, "__class__"):
                    dataset_name = base_dataset.__class__.__name__

        return dataset_name.lower() if dataset_name else ""

    def _adapt_config_to_dataset(self):
        """Adaptively select optimal configuration parameters based on dataset characteristics"""
        if not self.params.get("adaptive_config", True):
            self.console.log(
                "Adaptive config disabled, using default config", style="yellow"
            )
            return

        input_shape = self.input_shape
        total_features = int(np.prod(input_shape))

        is_image = (
            len(input_shape) == 3
            and input_shape[-1] in [28, 32]
            and input_shape[0] in [1, 3]
        )
        is_large_image = is_image and input_shape[-1] > 64
        is_high_dim = total_features > 1000
        is_text_like = total_features > 100 and len(input_shape) == 1

        dataset_name = self.dataset_name.lower()
        config_applied = self._apply_dataset_specific_config(dataset_name)

        if not config_applied:
            if is_image or is_large_image:
                self._apply_image_config(input_shape)
                dataset_type = "image"
            elif is_text_like:
                self._apply_text_config()
                dataset_type = "text"
            elif is_high_dim:
                self._apply_large_scale_config()
                dataset_type = "large-scale"
            else:
                self._apply_tabular_config()
                dataset_type = "tabular"
        else:
            dataset_type = "dataset-specific"

        self._apply_manual_overrides()

        self.console.log(
            f"✅ Adaptive config applied: dataset='{dataset_name}', type={dataset_type}, "
            f"features={total_features}, shape={input_shape}",
            style="green",
        )

    def _apply_dataset_specific_config(self, dataset_name: str) -> bool:
        """Apply optimal config based on specific dataset name"""
        if "mnist" in dataset_name and "fashion" not in dataset_name:
            self.params.update(
                {
                    "batch_size_l": 50,
                    "batch_size_u": 100,  # Original paper configuration
                    "n_hidden_vae_e": [500, 500],
                    "n_h_y": 100,
                    "n_h_o": 100,
                    "n_hidden_vae_d": [500, 500],
                    "n_hidden_disc": [256],
                    "n_hidden_cl": [],
                    "lr_pu": 3e-4,
                    "lr_disc": 3e-4,
                    "lr_pn": 1e-5,  # Original paper exact configuration
                    "alpha_gen": 0.1,
                    "alpha_disc": 0.1,
                    "alpha_gen2": 3.0,  # Random Labelling
                    "num_epoch_pre": 100,
                    "num_epoch": 800,
                    "num_epoch_step1": 400,
                    "num_epoch_step2": 500,
                    "num_epoch_step3": 700,
                }
            )
            self.console.log("📊 Applied MNIST config from paper", style="blue")
            return True

        elif "fashion" in dataset_name:
            self.params.update(
                {
                    "batch_size_l": 64,
                    "batch_size_u": 128,
                    "n_h_y": 128,
                    "n_h_o": 64,  # Slightly larger latent space
                    "lr_pu": 2e-4,
                    "lr_disc": 2e-4,
                    "lr_pn": 5e-5,
                    "alpha_gen": 0.1,
                    "alpha_disc": 0.1,
                    "alpha_gen2": 2.0,
                    "num_epoch": 600,
                }
            )
            self.console.log("👗 Applied Fashion-MNIST config", style="blue")
            return True

        elif "cifar" in dataset_name:
            self.params.update(
                {
                    "batch_size_l": 64,
                    "batch_size_u": 128,
                    "n_h_y": 128,
                    "n_h_o": 64,
                    "lr_pu": 2e-4,
                    "lr_disc": 2e-4,
                    "lr_pn": 1e-4,
                    "alpha_gen": 0.3,
                    "alpha_disc": 0.3,
                    "alpha_gen2": 1.0,
                    "num_epoch_pre": 100,
                    "num_epoch": 600,
                    "num_epoch_step1": 240,
                    "num_epoch_step2": 360,
                    "num_epoch_step3": 480,
                }
            )
            self.console.log("🖼️ Applied CIFAR-10 config from paper", style="blue")
            return True

        elif "20news" in dataset_name or "newsgroups" in dataset_name:
            self.params.update(
                {
                    "batch_size_l": 64,
                    "batch_size_u": 128,
                    "n_hidden_vae_e": [512, 256],
                    "n_h_y": 64,
                    "n_h_o": 64,
                    "n_hidden_vae_d": [256, 512],
                    "n_hidden_disc": [128, 64],
                    "n_hidden_cl": [32],
                    "lr_pu": 1e-4,
                    "lr_disc": 1e-4,
                    "lr_pn": 1e-4,
                    "alpha_gen": 0.01,
                    "alpha_disc": 0.01,
                    "alpha_gen2": 1.0,
                    "num_epoch_pre": 50,
                    "num_epoch": 200,
                    "num_epoch_step1": 80,
                    "num_epoch_step2": 120,
                    "num_epoch_step3": 160,
                    "num_epoch_step_pn1": 120,
                    "num_epoch_step_pn2": 140,
                }
            )
            self.console.log("📰 Applied 20News config from paper", style="blue")
            return True

        elif "imdb" in dataset_name:
            self.params.update(
                {
                    "batch_size_l": 64,
                    "batch_size_u": 128,
                    "n_hidden_vae_e": [512, 256],
                    "n_h_y": 64,
                    "n_h_o": 64,
                    "lr_pu": 1e-4,
                    "lr_disc": 1e-4,
                    "lr_pn": 1e-4,
                    "alpha_gen": 0.01,
                    "alpha_disc": 0.01,
                    "alpha_gen2": 1.0,
                    "num_epoch": 200,
                }
            )
            self.console.log("🎬 Applied IMDB text config", style="blue")
            return True

        elif "mushroom" in dataset_name:
            self.params.update(
                {
                    "batch_size_l": 128,
                    "batch_size_u": 256,
                    "n_hidden_vae_e": [512, 256, 128],
                    "n_h_y": 64,
                    "n_h_o": 32,
                    "n_hidden_vae_d": [128, 256, 512],
                    "n_hidden_disc": [256, 128],
                    "alpha_gen": 0.05,
                    "alpha_disc": 0.05,
                    "alpha_gen2": 1.0,
                    "num_epoch": 150,
                }
            )
            self.console.log("🍄 Applied Mushrooms tabular config", style="blue")
            return True

        elif "connect" in dataset_name:
            self.params.update(
                {
                    "batch_size_l": 128,
                    "batch_size_u": 256,
                    "n_hidden_vae_e": [512, 256],
                    "n_h_y": 64,
                    "n_h_o": 32,
                    "alpha_gen": 0.05,
                    "alpha_disc": 0.05,
                    "num_epoch": 150,
                }
            )
            self.console.log("🔴 Applied Connect4 config", style="blue")
            return True

        elif "spam" in dataset_name:
            self.params.update(
                {
                    "batch_size_l": 128,
                    "batch_size_u": 256,
                    "n_hidden_vae_e": [256, 128],
                    "n_h_y": 32,
                    "n_h_o": 16,
                    "alpha_gen": 0.1,
                    "alpha_disc": 0.1,
                    "num_epoch": 200,
                }
            )
            self.console.log("📧 Applied Spambase config", style="blue")
            return True

        elif "alzheimer" in dataset_name or "mri" in dataset_name:
            self.params.update(
                {
                    "batch_size_l": 32,
                    "batch_size_u": 64,
                    "n_h_y": 128,
                    "n_h_o": 64,
                    "lr_pu": 1e-4,
                    "lr_disc": 1e-4,
                    "lr_pn": 5e-5,
                    "alpha_gen": 0.1,
                    "alpha_disc": 0.1,
                    "alpha_gen2": 2.0,
                    "num_epoch": 400,
                }
            )
            self.console.log("🧠 Applied Alzheimer MRI config", style="blue")
            return True

        return False

    def _apply_manual_overrides(self):
        """Apply manual parameter overrides if specified in config"""
        overrides = self.params.get("manual_override", {})
        if overrides:
            applied_overrides = []
            for key, value in overrides.items():
                if value is not None:
                    self.params[key] = value
                    applied_overrides.append(f"{key}={value}")

            if applied_overrides:
                self.console.log(
                    f"⚠️ Manual overrides applied: {', '.join(applied_overrides)}",
                    style="yellow",
                )

    def _apply_image_config(self, input_shape):
        """Optimized config for image datasets"""
        self.params.update(
            {
                "n_h_y": 128,
                "n_h_o": 64,
                "lr_pu": 1e-4,
                "lr_disc": 1e-4,
                "lr_pn": 5e-5,
                "alpha_gen": 0.1,
                "alpha_disc": 0.1,
                "alpha_gen2": 3.0,
                "num_epoch_pre": 100,
                "num_epoch": 300,
                "num_epoch_step1": 120,
                "num_epoch_step2": 180,
                "num_epoch_step3": 240,
                "batch_size_l": 64,
                "batch_size_u": 128,
            }
        )

    def _apply_mnist_config(self):
        """MNIST specific config from paper"""
        self.params.update(
            {
                "n_h_y": 100,
                "n_h_o": 100,
                "alpha_gen": 0.1,
                "alpha_disc": 0.1,
                "alpha_gen2": 3.0,
                "lr_pu": 3e-4,
                "lr_pn": 1e-5,
            }
        )

    def _apply_cifar_config(self):
        """CIFAR-10 specific config from paper"""
        self.params.update(
            {
                "alpha_gen": 0.3,
                "alpha_disc": 0.3,
                "alpha_gen2": 1.0,
                "lr_pu": 2e-4,
                "lr_disc": 2e-4,
            }
        )

    def _apply_text_config(self):
        """Optimized config for text datasets based on 20News paper"""
        self.params.update(
            {
                "n_hidden_vae_e": [512, 256],
                "n_h_y": 64,
                "n_h_o": 64,
                "lr_pu": 1e-4,
                "lr_disc": 1e-4,
                "alpha_gen": 0.01,
                "alpha_disc": 0.01,
                "alpha_gen2": 1.0,
                "num_epoch": 200,
                "batch_size_l": 64,
                "batch_size_u": 128,
            }
        )

    def _apply_tabular_config(self):
        """Optimized config for tabular data"""
        self.params.update(
            {
                "n_hidden_vae_e": [512, 256, 128],
                "n_h_y": 64,
                "n_h_o": 32,
                "n_hidden_vae_d": [128, 256, 512],
                "n_hidden_disc": [256, 128],
                "alpha_gen": 0.05,
                "alpha_disc": 0.05,
                "alpha_gen2": 1.0,
                "batch_size_l": 128,
                "batch_size_u": 256,
            }
        )

    def _apply_large_scale_config(self):
        """Optimized config for large-scale data"""
        self.params.update(
            {
                "n_hidden_vae_e": [1024, 512, 256],
                "n_h_y": 128,
                "n_h_o": 64,
                "lr": 5e-5,
                "lr_pu": 5e-5,
                "lr_pn": 1e-5,
                "alpha_gen": 0.01,
                "alpha_gen2": 0.5,
                "batch_size_l": 32,
                "batch_size_u": 64,
                "num_epoch": 400,
            }
        )

    def create_criterion(self):
        return nn.Identity()

    def _build_model(self):
        input_shape = self.input_shape
        # Treat Alzheimer MRI (1x128x128) as image as well for fair comparison
        self.is_image = len(input_shape) == 3 and input_shape[-1] in [28, 32, 128]
        if self.is_image:
            in_ch, h, _w = input_shape
            n_h_y = int(self.params.get("n_h_y", 128))
            n_h_o = int(self.params.get("n_h_o", 128))
            self.model_en = VAEConvEncoder(
                in_channels=int(in_ch), n_h_y=n_h_y, n_h_o=n_h_o
            ).to(self.device)
            self.model_de = VAEConvDecoder(
                out_channels=int(in_ch), n_h_y=n_h_y, n_h_o=n_h_o, img_size=int(h)
            ).to(self.device)
            self.model_disc = VAEConvDiscriminator(in_channels=int(in_ch)).to(
                self.device
            )
        else:
            flat_input_dim = int(np.prod(self.input_shape))
            vae_config = {
                "n_h_y": int(self.params.get("n_h_y", 64)),
                "n_h_o": int(self.params.get("n_h_o", 64)),
                "n_o": int(self.params.get("n_o", 2)),
                "n_hidden_vae_e": self.params.get("n_hidden_vae_e", [512, 256]),
                "n_hidden_vae_d": self.params.get("n_hidden_vae_d", [256, 512]),
                "n_hidden_disc": self.params.get("n_hidden_disc", [128, 64]),
            }
            self.model_en = VAEencoder(vae_config, input_dim=flat_input_dim).to(
                self.device
            )
            self.model_de = VAEdecoder(vae_config, input_dim=flat_input_dim).to(
                self.device
            )
            self.model_disc = Discriminator(vae_config, input_dim=flat_input_dim).to(
                self.device
            )

        n_h_o = int(self.params.get("n_h_o", 64))
        n_hidden_cl = self.params.get("n_hidden_cl", [])
        self.model_cl = ClassifierO(n_h_o=n_h_o, n_hidden=n_hidden_cl).to(self.device)

        self.model = select_model(self.method, self.params, self.prior).to(self.device)

        try:
            has_params = any(p.requires_grad for p in self.model.parameters())
        except Exception:
            has_params = False
        if not has_params:
            try:
                sample_batch = next(iter(self.train_loader))
                x_sample = sample_batch[0]
                if isinstance(x_sample, (list, tuple)):
                    x_sample = x_sample[0]
                with torch.no_grad():
                    _ = self.model(x_sample.to(self.device))
            except Exception:
                pass

        lr_vae = float(self.params.get("lr_pu", self.params.get("lr", 2e-4)))
        lr_disc = float(self.params.get("lr_disc", self.params.get("lr", 2e-4)))
        lr_pn = float(self.params.get("lr_pn", self.params.get("lr", 2e-4)))

        try:
            with torch.no_grad():
                if self.is_image:
                    dummy_x = torch.zeros(2, *self.input_shape, device=self.device)
                else:
                    dummy_x = torch.zeros(
                        2, int(np.prod(self.input_shape)), device=self.device
                    )
                o_dummy = F.one_hot(torch.tensor([1, 0], device=self.device), 2).float()
                _y_mu, _y_lss, _o_mu, _o_lss = self.model_en(
                    dummy_x if self.is_image else dummy_x, o_dummy
                )
                h_o_dim = _o_mu.shape[1]
                from torch.nn import Linear

                first_linear_in = None
                for m in self.model_cl.net:
                    if isinstance(m, Linear):
                        first_linear_in = m.in_features
                        break
                if first_linear_in is not None and first_linear_in != h_o_dim:
                    n_hidden_cl = self.params.get("n_hidden_cl", [])
                    self.console.log(
                        f"[Fix] Rebuilding ClassifierO: h_o_dim={h_o_dim} != classifier_in={first_linear_in}",
                        style="yellow",
                    )
                    self.model_cl = ClassifierO(n_h_o=h_o_dim, n_hidden=n_hidden_cl).to(
                        self.device
                    )
        except Exception:
            pass

        self.optimizer_vae = torch.optim.Adam(
            list(self.model_en.parameters()) + list(self.model_de.parameters()),
            lr=lr_vae,
        )
        self.optimizer_disc = torch.optim.Adam(self.model_disc.parameters(), lr=lr_disc)
        self.optimizer_cl = torch.optim.Adam(self.model_cl.parameters(), lr=lr_vae)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_pn)

    def reparameterization(self, mu, lss):
        eps = torch.randn_like(mu)
        return mu + torch.exp(lss / 2.0) * eps

    def _pn_pos_logit(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return the positive-class logit from PN classifier outputs.
        - If single-logit output: return flattened logit.
        - If multi-logit output: select configured positive index (default 0).
        """
        if outputs.dim() > 1 and outputs.shape[1] > 1:
            pos_index = int(self.params.get("pn_positive_index", 0))
            return outputs[:, pos_index].view(-1)
        return outputs.view(-1)

    def run(self):
        self.before_training()

        self.console.log(
            "\n--- [Stage 1/3] VAE-PU: Pre-training VAE ---", style="bold yellow"
        )
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = False
                self.checkpoint_handler.should_stop = False
            except Exception:
                pass
        for epoch in tqdm(
            range(1, self.pretrain_epochs + 1),
            desc=f"Stage 1/3 (VAE Pretrain) [{self.method.upper()}]",
            leave=False,
        ):
            self._train_pretrain_epoch(epoch)

        self.console.log(
            "\n--- [Stage 2/3] VAE-PU: Finding Prior with GMM ---", style="bold yellow"
        )
        self._find_prior()

        self.console.log(
            "\n--- [Stage 3/3] VAE-PU: Main Training ---", style="bold yellow"
        )
        if self.checkpoint_handler:
            try:
                self.checkpoint_handler.early_stopping_enabled = True
                self.checkpoint_handler.wait = 0
                self.checkpoint_handler.should_stop = False
            except Exception:
                pass
        for epoch in range(1, self.main_epochs + 1):
            self.global_epoch += 1
            self.train_one_epoch(epoch)
            if self.validation_loader is not None:
                try:
                    new_thr = self._compute_optimal_threshold_on_val(
                        self.threshold_selection
                    )
                    if np.isfinite(new_thr):
                        self.pn_decision_threshold = float(new_thr)
                        self.console.log(
                            f"[Threshold] Updated PN decision threshold on val: {self.pn_decision_threshold:.6f} (method={self.threshold_selection})",
                            style="cyan",
                        )
                except Exception:
                    pass

            train_metrics = self.evaluate_metrics_pn(self.train_loader)
            test_metrics = self.evaluate_metrics_pn(self.test_loader)
            val_metrics = (
                self.evaluate_metrics_pn(self.validation_loader)
                if self.validation_loader is not None
                else None
            )
            self._print_metrics(
                epoch,
                self.main_epochs,
                train_metrics,
                test_metrics,
                "Main Training",
                val_metrics=val_metrics,
            )

            if hasattr(self, "checkpoint_handler") and self.checkpoint_handler:
                all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
                if val_metrics is not None:
                    all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                import time as _time

                self.checkpoint_handler(
                    epoch=self.global_epoch,
                    all_metrics=all_metrics,
                    model=self.model,
                    elapsed_seconds=(
                        _time.time() - self._run_start_time
                        if self._run_start_time
                        else None
                    ),
                )

            if self.checkpoint_handler and self.checkpoint_handler.should_stop:
                self.console.log(
                    "Early stopping in main training stage.", style="bold red"
                )
                break

        self.after_training()

        if self.checkpoint_handler and self.checkpoint_handler.best_metrics:
            self.checkpoint_handler.log_best_metrics()
        self._close_file_console()

        return self.checkpoint_handler.best_metrics if self.checkpoint_handler else {}

    def train_one_epoch(self, epoch_idx: int):
        self.model_en.train()
        self.model_de.train()
        self.model_disc.train()
        self.model.train()
        self.model_cl.train()

        p_loader = self.get_positive_loader()
        u_loader = self.get_unlabeled_loader()

        p_iter = iter(p_loader)

        def get_next_p_batch():
            nonlocal p_iter
            try:
                return next(p_iter)
            except StopIteration:
                p_iter = iter(p_loader)
                return next(p_iter)

        for b_idx, (x_u, *_) in enumerate(u_loader, 1):
            x_pl, *_ = get_next_p_batch()
            x_pl, x_u = x_pl.to(self.device), x_u.to(self.device)

            disc_loss = None
            vade_vals = None
            pn_vals = None

            if epoch_idx <= self.step3_end:
                if not (self.step1_end < epoch_idx <= self.step2_end):
                    disc_loss = self._train_step_disc(x_pl, x_u)
                    vade_vals = self._train_step_vae(x_pl, x_u, epoch_idx)

            if epoch_idx > self.step1_end:
                if not (self.step_pn1_end < epoch_idx <= self.step_pn2_end):
                    pn_vals = self._train_step_pn(x_pl, x_u)

            if self.params.get("log_every_step", False):
                try:
                    parts = [f"[Epoch {epoch_idx} Step {b_idx}]"]
                    if disc_loss is not None:
                        parts.append(f"disc={disc_loss:.6f}")
                    if vade_vals is not None:
                        v_vade, v_gan, v_gan2, v_total = vade_vals
                        parts.append(
                            f"vade={v_vade:.6f} gan={v_gan:.6f} gan2={v_gan2:.6f} total={v_total:.6f}"
                        )
                    if pn_vals is not None:
                        pn_total, pn_pl, pn_pu1, pn_neg = pn_vals
                        parts.append(
                            f"pn_total={pn_total:.6f} pl={pn_pl:.6f} pu1={pn_pu1:.6f} negRisk={pn_neg:.6f}"
                        )
                    self.console.log(" ".join(parts), style="dim")
                except Exception:
                    pass

    def _train_pretrain_epoch(self, epoch):
        self.model_en.train()
        self.model_de.train()

        p_loader = self.get_positive_loader()
        u_loader = self.get_unlabeled_loader()

        p_iter = iter(p_loader)

        def get_next_p_batch():
            nonlocal p_iter
            try:
                return next(p_iter)
            except StopIteration:
                p_iter = iter(p_loader)
                return next(p_iter)

        progress_bar = tqdm(
            u_loader,
            desc=f"Pretrain Epoch {epoch}/{self.pretrain_epochs}",
            leave=False,
        )
        for b_idx, (x_u, *_) in enumerate(progress_bar, 1):
            x_pl, *_ = get_next_p_batch()
            x_pl, x_u = x_pl.to(self.device), x_u.to(self.device)

            if self.is_image:
                x = torch.cat([x_pl, x_u], dim=0)
            else:
                x_pl = x_pl.view(x_pl.size(0), -1)
                x_u = x_u.view(x_u.size(0), -1)
                x = torch.cat([x_pl, x_u], dim=0)

            o_pl = (
                F.one_hot(torch.ones(x_pl.shape[0], dtype=torch.long), 2)
                .float()
                .to(self.device)
            )
            o_u = (
                F.one_hot(torch.zeros(x_u.shape[0], dtype=torch.long), 2)
                .float()
                .to(self.device)
            )
            o = torch.cat([o_pl, o_u], dim=0)

            self.optimizer_vae.zero_grad()

            h_y_mu, h_y_lss, h_o_mu, h_o_lss = self.model_en(x, o)
            h_y = self.reparameterization(h_y_mu, h_y_lss)
            h_o = self.reparameterization(h_o_mu, h_o_lss)

            recon_x = self.model_de(h_y, h_o)
            if self.is_image:
                # MNIST/FashionMNIST: 1x28x28; ADNI(1x128x128) should not take BCE path
                is_mnist = x.dim() == 4 and x.shape[1] == 1 and x.shape[-1] <= 32
                if is_mnist:
                    bce = F.binary_cross_entropy_with_logits(
                        recon_x, x, reduction="none"
                    )
                    loss = bce.view(bce.size(0), -1).sum(dim=1).mean()
                else:
                    mse = F.mse_loss(recon_x, x, reduction="none")
                    loss = mse.view(mse.size(0), -1).sum(dim=1).mean()
            else:
                # For vector/continuous features, use MSE as in original implementation
                mse = F.mse_loss(recon_x, x, reduction="none")
                loss = 0.5 * mse.sum(dim=1).mean()

            loss.backward()
            self.optimizer_vae.step()

            try:
                progress_bar.set_postfix(recon_loss=f"{loss.detach().item():.6f}")
            except Exception:
                pass

            if self.params.get("log_every_step", False):
                try:
                    self.console.log(
                        f"[Pretrain][Epoch {epoch} Step {b_idx}] recon_loss={loss.detach().item():.6f}",
                        style="dim",
                    )
                except Exception:
                    pass

    def _find_prior(self):
        self.model_en.eval()
        # Ensure we are using the base dataset to get original features
        base_dataset = self.train_loader.dataset
        if isinstance(base_dataset, torch.utils.data.Subset):
            base_dataset = base_dataset.dataset
        if isinstance(base_dataset, LaGAMDatasetWrapper):  # Or any other wrapper
            base_dataset = base_dataset.base_dataset

        x_tr_l = base_dataset.features[base_dataset.pu_labels == 1]
        x_tr_u = base_dataset.features[base_dataset.pu_labels == -1]

        if self.is_image:
            x_tr_l_flat = x_tr_l
            x_tr_u_flat = x_tr_u
        else:
            x_tr_l_flat = x_tr_l.view(x_tr_l.size(0), -1)
            x_tr_u_flat = x_tr_u.view(x_tr_u.size(0), -1)

        o_pl = (
            F.one_hot(torch.ones(x_tr_l.shape[0], dtype=torch.long), 2)
            .float()
            .to(self.device)
        )
        o_u = (
            F.one_hot(torch.zeros(x_tr_u.shape[0], dtype=torch.long), 2)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            h_y_u_mu, _, _, _ = self.model_en(x_tr_u_flat.to(self.device), o_u)
            h_y_l_mu, _, _, _ = self.model_en(x_tr_l_flat.to(self.device), o_pl)

        h_y = torch.cat([h_y_u_mu, h_y_l_mu], dim=0).cpu().numpy()

        gmm = GaussianMixture(n_components=2, covariance_type="diag")
        gmm.fit(h_y)

        h_y_l_np = h_y_l_mu.cpu().numpy()
        from scipy.stats import multivariate_normal

        c0 = multivariate_normal.logpdf(
            h_y_l_np, gmm.means_[0], np.diag(gmm.covariances_[0])
        )
        c1 = multivariate_normal.logpdf(
            h_y_l_np, gmm.means_[1], np.diag(gmm.covariances_[1])
        )

        if np.mean(c0) > np.mean(c1):
            self.p = torch.tensor(gmm.weights_[0], device=self.device)
            self.mu = torch.tensor(gmm.means_[[1, 0]], device=self.device)
            self.var = torch.tensor(gmm.covariances_[[1, 0]], device=self.device)
        else:
            self.p = torch.tensor(gmm.weights_[1], device=self.device)
            self.mu = torch.tensor(gmm.means_, device=self.device)
            self.var = torch.tensor(gmm.covariances_, device=self.device)

        self.console.log(f"Estimated prior p: {self.p.item():.4f}", style="green")

    def _generate(self, x_pl, x_u):
        if self.is_image:
            x_pl_flat = x_pl
            x_u_flat = x_u
        else:
            x_pl_flat = x_pl.view(x_pl.size(0), -1)
            x_u_flat = x_u.view(x_u.size(0), -1)

        o_pl = (
            F.one_hot(torch.ones(x_pl.shape[0], dtype=torch.long), 2)
            .float()
            .to(self.device)
        )
        o_u = (
            F.one_hot(torch.zeros(x_u.shape[0], dtype=torch.long), 2)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            h_y_mu, h_y_lss, h_o_mu, h_o_lss = self.model_en(x_pl_flat, o_pl)
            h_y = self.reparameterization(h_y_mu, h_y_lss)

            _, _, h_o_mu_x, h_o_lss_x = self.model_en(x_u_flat, o_u)
            h_o_x = self.reparameterization(h_o_mu_x, h_o_lss_x)

            h_o_pl = self.reparameterization(h_o_mu, h_o_lss)
            dist = torch.cdist(h_o_pl, h_o_x)
            nearest_idx = torch.argmin(dist, dim=1)
            ne_h_o = h_o_x[nearest_idx]

            is_mnist = self.is_image and self.input_shape[0] == 1
            x_gen = self.model_de(h_y, ne_h_o, sigmoid=is_mnist)

        if self.is_image:
            return x_gen
        return x_gen.view(-1, *self.input_shape)

    def _train_step_disc(self, x_pl, x_u):
        self.optimizer_disc.zero_grad()
        with torch.no_grad():
            x_pu = self._generate(x_pl, x_u)

        if self.is_image:
            d_x_pu = self.model_disc(x_pu)
            d_x_u = self.model_disc(x_u)
        else:
            x_pu_flat = x_pu.view(x_pu.size(0), -1)
            x_u_flat = x_u.view(x_u.size(0), -1)
            d_x_pu = self.model_disc(x_pu_flat)
            d_x_u = self.model_disc(x_u_flat)

        loss_pu = F.binary_cross_entropy_with_logits(d_x_pu, torch.zeros_like(d_x_pu))
        loss_u = F.binary_cross_entropy_with_logits(d_x_u, torch.ones_like(d_x_u))

        loss = self.params.get("alpha_disc", 1.0) * (loss_pu + loss_u)
        loss.backward()
        self.optimizer_disc.step()
        return float(loss.detach().item())

    def _train_step_vae(self, x_pl, x_u, epoch):
        self.optimizer_vae.zero_grad()
        self.optimizer_cl.zero_grad()

        if self.is_image:
            x_flat = torch.cat([x_pl, x_u], dim=0)
        else:
            x_pl_flat = x_pl.view(x_pl.size(0), -1)
            x_u_flat = x_u.view(x_u.size(0), -1)
            x_flat = torch.cat([x_pl_flat, x_u_flat], dim=0)

        p = self.p.item()
        o_pl = (
            F.one_hot(torch.ones(x_pl.shape[0], dtype=torch.long), 2)
            .float()
            .to(self.device)
        )
        o_u = (
            F.one_hot(torch.zeros(x_u.shape[0], dtype=torch.long), 2)
            .float()
            .to(self.device)
        )
        o = torch.cat([o_pl, o_u], dim=0)

        h_y_mu, h_y_lss, h_o_mu, h_o_lss = self.model_en(x_flat, o)
        h_y = self.reparameterization(h_y_mu, h_y_lss)
        h_o = self.reparameterization(h_o_mu, h_o_lss)

        c0 = -0.5 * torch.sum(
            ((h_y - self.mu[0]) ** 2 / self.var[0]) + torch.log(self.var[0] + 1e-9),
            dim=1,
        ) + torch.log(torch.tensor(1 - p + 1e-9))
        c1 = -0.5 * torch.sum(
            ((h_y - self.mu[1]) ** 2 / self.var[1]) + torch.log(self.var[1] + 1e-9),
            dim=1,
        ) + torch.log(torch.tensor(p + 1e-9))
        c = F.softmax(torch.stack([c0, c1], dim=1), dim=1)[:, 1].unsqueeze(1)

        loss1_0 = -0.5 * torch.sum(
            torch.log(self.var[0] + 1e-9)
            + (torch.exp(h_y_lss) + (h_y_mu - self.mu[0]) ** 2) / self.var[0],
            dim=1,
            keepdim=True,
        )
        loss1_1 = -0.5 * torch.sum(
            torch.log(self.var[1] + 1e-9)
            + (torch.exp(h_y_lss) + (h_y_mu - self.mu[1]) ** 2) / self.var[1],
            dim=1,
            keepdim=True,
        )
        loss1 = ((1 - c) * loss1_0 + c * loss1_1).mean()

        loss2 = -0.5 * torch.sum(torch.exp(h_o_lss) + h_o_mu**2, dim=1).mean()

        recon_x = self.model_de(h_y, h_o)
        if self.is_image:
            # MNIST/FashionMNIST: 1x28x28; ADNI(1x128x128) should not take BCE path
            is_mnist = (
                len(self.input_shape) == 3
                and self.input_shape[0] == 1
                and self.input_shape[-1] <= 32
            )
            if is_mnist:
                bce = F.binary_cross_entropy_with_logits(
                    recon_x, x_flat, reduction="none"
                )
                loss3 = -bce.view(bce.size(0), -1).sum(dim=1).mean()
            else:
                mse = F.mse_loss(recon_x, x_flat, reduction="none")
                loss3 = -0.5 * mse.view(mse.size(0), -1).sum(dim=1).mean()
        else:
            loss3 = (
                -0.5 * F.mse_loss(recon_x, x_flat, reduction="none").sum(dim=1).mean()
            )

        loss4 = 0.5 * torch.sum(1 + h_y_lss, dim=1).mean()
        loss5 = 0.5 * torch.sum(1 + h_o_lss, dim=1).mean()
        loss6 = (
            -c * torch.log(c / (p + 1e-9) + 1e-9)
            - (1 - c) * torch.log((1 - c) / (1 - p + 1e-9) + 1e-9)
        ).mean()

        c_o = self.model_cl(h_o)
        label_o = o[:, 0].unsqueeze(1)  # o is [[1,0], [1,0]...[0,1], [0,1]...]
        loss7 = -F.binary_cross_entropy_with_logits(c_o, label_o)

        alpha_vade = float(self.params.get("alpha_vade", 1.0))
        vade_loss = -alpha_vade * (
            loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
        )

        x_pu = self._generate(x_pl, x_u)
        if self.is_image:
            d_x_pu = self.model_disc(x_pu)
        else:
            d_x_pu = self.model_disc(x_pu.view(x_pu.size(0), -1))
        loss_gan = F.binary_cross_entropy_with_logits(d_x_pu, torch.ones_like(d_x_pu))

        loss_gan2 = torch.tensor(0.0, device=self.device)
        if epoch > self.step1_end:
            d_x_pu2 = self.model(x_pu)
            pos_logit = self._pn_pos_logit(d_x_pu2)
            loss_gan2 = F.binary_cross_entropy_with_logits(
                pos_logit, torch.ones_like(pos_logit)
            )

        total_loss = (
            vade_loss
            + self.params.get("alpha_gen", 1.0) * loss_gan
            + self.params.get("alpha_gen2", 1.0) * loss_gan2
        )
        total_loss.backward()
        self.optimizer_vae.step()
        self.optimizer_cl.step()
        return (
            float(vade_loss.detach().item()),
            float(loss_gan.detach().item()),
            float(loss_gan2.detach().item()),
            float(total_loss.detach().item()),
        )

    def _train_step_pn(self, x_pl, x_u):
        self.optimizer.zero_grad()

        pi_pl = float(self.params.get("pi_pl", 0.01))
        est_p = (
            float(self.p.item()) if hasattr(self, "p") else (pi_pl + 0.5 * (1 - pi_pl))
        )
        pi_pu = max(0.0, min(1.0, est_p) - pi_pl)
        pi_u = max(0.0, 1.0 - pi_pl)

        x_pu = self._generate(x_pl, x_u)

        if self.is_image:
            if x_pl.dim() == 2:
                x_pl = x_pl.view(-1, *self.input_shape)
            if x_u.dim() == 2:
                x_u = x_u.view(-1, *self.input_shape)
            if x_pu.dim() == 2:
                x_pu = x_pu.view(-1, *self.input_shape)

        pn_x_pl = self._pn_pos_logit(self.model(x_pl))
        pn_x_pu = self._pn_pos_logit(self.model(x_pu.detach()))
        pn_x_u = self._pn_pos_logit(self.model(x_u))

        def sigmoid_loss(t, y):
            return torch.sigmoid(-t * y)

        pl_loss = (pi_pl * sigmoid_loss(pn_x_pl, torch.ones_like(pn_x_pl))).mean()
        pu1_loss = (pi_pu * sigmoid_loss(pn_x_pu, torch.ones_like(pn_x_pu))).mean()

        negative_risk = (
            -pi_pu * sigmoid_loss(pn_x_pu, -torch.ones_like(pn_x_pu))
        ).mean() + (pi_u * sigmoid_loss(pn_x_u, -torch.ones_like(pn_x_u))).mean()

        if negative_risk < 0:
            loss = -negative_risk
        else:
            loss = pl_loss + pu1_loss + negative_risk

        loss.backward()
        self.optimizer.step()
        return (
            float(loss.detach().item()),
            float(pl_loss.detach().item()),
            float(pu1_loss.detach().item()),
            float(negative_risk.detach().item()),
        )

    def get_positive_loader(self):
        pos_indices = (self.train_loader.dataset.pu_labels == 1).nonzero().squeeze()
        p_dataset = torch.utils.data.Subset(self.train_loader.dataset, pos_indices)
        return DataLoader(
            p_dataset, batch_size=self.params.get("batch_size_l", 50), shuffle=True
        )

    def get_unlabeled_loader(self):
        unl_indices = (self.train_loader.dataset.pu_labels == -1).nonzero().squeeze()
        u_dataset = torch.utils.data.Subset(self.train_loader.dataset, unl_indices)
        return DataLoader(
            u_dataset, batch_size=self.params.get("batch_size_u", 100), shuffle=True
        )

    def evaluate_metrics_pn(self, loader):
        y_true_all, y_pred_all = [], []
        total_risk_sum = 0.0
        self.model.eval()
        with torch.no_grad():
            for x, t, y_true, _, _ in loader:
                x, t, y_true = (
                    x.to(self.device),
                    t.to(self.device),
                    y_true.to(self.device),
                )

                if x.dim() == 2:
                    x = x.view(-1, *self.input_shape)

                outputs = self.model(x)
                pos_logit = self._pn_pos_logit(outputs)
                thr = getattr(self, "pn_decision_threshold", 0.0)
                preds_binary = (pos_logit > thr).long()
                y_true_all.extend(y_true.cpu().numpy())
                y_pred_all.extend(preds_binary.cpu().numpy())

                eval_outputs = self._pn_pos_logit(outputs)
                pos_mask, unl_mask = (t == 1), (t == -1)
                risk_pos_term = _zero_one_loss(eval_outputs[pos_mask]).sum()
                risk_neg_term = _zero_one_loss(-eval_outputs[pos_mask]).sum()
                risk_unl_term = _zero_one_loss(-eval_outputs[unl_mask]).sum()
                batch_risk = (
                    self.prior * (risk_pos_term - risk_neg_term) + risk_unl_term
                )
                total_risk_sum += batch_risk.item()
        self.model.train()

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        acc = accuracy_score(y_true_all, y_pred_all)
        risk = total_risk_sum / max(1, len(y_true_all))
        f1 = f1_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, zero_division=0)
        rec = recall_score(y_true_all, y_pred_all, zero_division=0)
        try:
            if len(set(y_true_all)) < 2:
                auc = float("nan")
            else:
                scores = []
                with torch.no_grad():
                    for x, _, y_true, _, _ in loader:
                        if x.dim() == 2:
                            x = x.view(-1, *self.input_shape)
                        x = x.to(self.device)
                        s = self._pn_pos_logit(self.model(x)).detach().cpu().numpy()
                        scores.append(s)
                y_scores = np.concatenate(scores, axis=0)
                auc = float(roc_auc_score(y_true_all, y_scores))
        except Exception:
            auc = float("nan")

        return {
            "error": 1 - acc,
            "risk": risk,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        }

    def _compute_optimal_threshold_on_val(self, mode: str = "f1") -> float:
        """Compute optimal PN decision threshold on validation set"""
        if self.validation_loader is None:
            return getattr(self, "pn_decision_threshold", 0.0)

        scores, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for x, _t, y_true, _, _ in self.validation_loader:
                if x.dim() == 2:
                    x = x.view(-1, *self.input_shape)
                x = x.to(self.device)
                s = self._pn_pos_logit(self.model(x)).detach().cpu().numpy()
                scores.append(s)
                labels.append(y_true.numpy())
        if not scores:
            return getattr(self, "pn_decision_threshold", 0.0)
        import numpy as np
        from sklearn.metrics import f1_score, roc_curve

        y_scores = np.concatenate(scores)
        y_true = np.concatenate(labels)

        if mode == "f1":
            uniq = np.unique(y_scores)
            if uniq.size > 512:
                qs = np.linspace(0.0, 1.0, 512)
                thr_list = np.quantile(y_scores, qs)
            else:
                thr_list = uniq
            best_thr, best_f1 = 0.0, -1.0
            for thr in thr_list:
                preds = (y_scores > thr).astype(int)
                f1 = f1_score(y_true, preds)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)
            return float(best_thr)
        else:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            j = tpr - fpr
            idx = int(np.argmax(j))
            return float(thresholds[idx])
