from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base_trainer import BaseTrainer
from ..utils.checkpointing import CheckpointBundle
from ..utils.metrics import _pu_label_values_from_loader
from ..utils.model_factory import select_model
from .models import (
    ClassifierO,
    ClassifierPN,
    Discriminator,
    VAEConvDecoder,
    VAEConvDiscriminator,
    VAEConvEncoder,
    VAEdecoder,
    VAEencoder,
)


class VAEPUTrainer(BaseTrainer):
    """VAE-PU trainer adapted to PU-Bench datasets.

    Primary source:
        author source file(s): vaepu
        byeonghu-na/vae-pu at 79ddb7a1b30cf63c2f1b065b2075126051c3a6aa

    Source facts preserved here:
        - three-stage loop: VAE pretrain, GMM prior over h_y, then VAE/adversarial/PN training;
        - generated PU examples combine h_y from labeled positives with nearest h_o from U;
        - PN risk weights are the empirical PU split weights
          (pi_pl, pi_pu, pi_u), matching the source objective's convention;
        - MNIST-style binary image inputs are mapped back to the source [0, 1] domain before BCE and PN use.

    Benchmark boundary:
        This is a PyTorch PU-Bench adaptation. MNIST uses the author-code flat
        MLP recipe; other PU-Bench image datasets may still use benchmark CNN
        branches and should not be reported as exact source recipes.
    """

    def _prepare_data(self):
        super()._prepare_data()
        self.dataset_name = self._get_dataset_name()
        self._adapt_config_to_dataset()
        self._set_pu_mixture_weights()
        self._vaepu_config_adapted = True

    def before_training(self):
        super().before_training()

        if not getattr(self, "_vaepu_config_adapted", False):
            self.dataset_name = self._get_dataset_name()
            self._adapt_config_to_dataset()
            self._set_pu_mixture_weights()
            self._vaepu_config_adapted = True

        # Initialize PN decision threshold (will be updated on validation set)
        self.pn_decision_threshold = float(
            self.params.get("pn_decision_threshold", 0.0)
        )
        self.threshold_selection = str(
            self.params.get("threshold_selection", "fixed")
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
            and hasattr(self, "params")
            and "dataset_class" in self.params
        ):
            dataset_name = str(self.params["dataset_class"])

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

    def _base_train_dataset(self):
        dataset = self.train_loader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        if hasattr(dataset, "base_dataset"):
            dataset = dataset.base_dataset
        return dataset

    def _set_pu_mixture_weights(self) -> None:
        """Set VAE-PU PN-risk weights from the constructed PU-Bench split."""
        dataset = self._base_train_dataset()
        metadata = getattr(dataset, "pu_metadata", {}) or {}

        n_total = float(metadata.get("n_total", len(dataset)))
        if n_total <= 0:
            raise ValueError(
                "VAE-PU cannot infer PN-risk weights from an empty dataset."
            )

        pi_pl_default = float(metadata.get("n_labeled", 0)) / n_total
        pi_pu_default = float(metadata.get("n_pos_unlabeled", 0)) / n_total
        pi_u_default = float(metadata.get("n_unlabeled", 0)) / n_total

        pi_given = self.params.get("pi_given")
        self.pi_pl = float(self.params.get("pi_pl", pi_pl_default))
        if "pi_pu" in self.params:
            self.pi_pu = float(self.params["pi_pu"])
        elif pi_given is not None:
            self.pi_pu = max(0.0, float(pi_given) - self.pi_pl)
        else:
            self.pi_pu = float(pi_pu_default)
        self.pi_u = float(self.params.get("pi_u", pi_u_default))
        self.pi_p = float(pi_given) if pi_given is not None else self.pi_pl + self.pi_pu

        self.params.update(
            {
                "pi_pl": self.pi_pl,
                "pi_pu": self.pi_pu,
                "pi_u": self.pi_u,
                "pi_given": self.pi_p,
            }
        )

    def _adapt_config_to_dataset(self):
        """Adaptively select optimal configuration parameters based on dataset characteristics"""
        if not self.params.get("adaptive_config", True):
            self._apply_manual_overrides()
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
                    "mnist_source_recipe": True,
                    "batch_size_l": 10,
                    "batch_size_u": 990,
                    "batch_size_l_pn": 10,
                    "batch_size_u_pn": 990,
                    "n_hidden_vae_e": [500, 500],
                    "n_h_y": 10,
                    "n_h_o": 2,
                    "n_hidden_vae_d": [500, 500],
                    "n_hidden_disc": [256],
                    "n_hidden_cl": [],
                    "n_hidden_pn": [300, 300, 300, 300],
                    "lr_pu": 3e-4,
                    "lr_disc": 3e-4,
                    "lr_pn": 1e-5,
                    "alpha_gen": 0.1,
                    "alpha_disc": 0.1,
                    "alpha_gen2": 3.0,
                    "alpha_disc2": 3.0,
                    "num_epoch_pre": 100,
                    "num_epoch": 800,
                    "num_epoch_step1": 400,
                    "num_epoch_step2": 500,
                    "num_epoch_step3": 700,
                    "num_epoch_step_pn1": 500,
                    "num_epoch_step_pn2": 600,
                }
            )
            self.console.log("📊 Applied MNIST source recipe", style="blue")
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
            self.console.log(
                "🖼️ Applied CIFAR-10 alpha-guided benchmark config", style="blue"
            )
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
            self.console.log(
                "📰 Applied 20News alpha-guided benchmark config", style="blue"
            )
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
        """MNIST source-style config."""
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
        """CIFAR-10 alpha-guided benchmark config."""
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
        """Text benchmark config using source 20News alpha constants."""
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

    def _use_mnist_source_recipe(self) -> bool:
        dataset_name = getattr(self, "dataset_name", "").lower()
        return (
            bool(self.params.get("mnist_source_recipe", True))
            and "mnist" in dataset_name
            and "fashion" not in dataset_name
        )

    def _build_model(self):
        input_shape = self.input_shape
        # Treat Alzheimer MRI (1x128x128) as image as well for fair comparison
        self.is_image = len(input_shape) == 3 and input_shape[-1] in [28, 32, 128]
        self.use_mnist_source_recipe = self._use_mnist_source_recipe()
        self.vae_input_is_flat = (not self.is_image) or self.use_mnist_source_recipe
        self.pn_input_is_flat = self.use_mnist_source_recipe
        flat_input_dim = int(np.prod(self.input_shape))
        vae_config = {
            "n_h_y": int(self.params.get("n_h_y", 64)),
            "n_h_o": int(self.params.get("n_h_o", 64)),
            "n_o": int(self.params.get("n_o", 2)),
            "n_hidden_vae_e": self.params.get("n_hidden_vae_e", [512, 256]),
            "n_hidden_vae_d": self.params.get("n_hidden_vae_d", [256, 512]),
            "n_hidden_disc": self.params.get("n_hidden_disc", [128, 64]),
            "n_hidden_pn": self.params.get("n_hidden_pn", [300, 300, 300, 300]),
        }

        if self.vae_input_is_flat:
            self.model_en = VAEencoder(vae_config, input_dim=flat_input_dim).to(
                self.device
            )
            self.model_de = VAEdecoder(vae_config, input_dim=flat_input_dim).to(
                self.device
            )
            self.model_disc = Discriminator(vae_config, input_dim=flat_input_dim).to(
                self.device
            )
        else:
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

        n_h_o = int(self.params.get("n_h_o", 64))
        n_hidden_cl = self.params.get("n_hidden_cl", [])
        self.model_cl = ClassifierO(n_h_o=n_h_o, n_hidden=n_hidden_cl).to(self.device)

        if self.use_mnist_source_recipe:
            self.model = ClassifierPN(vae_config, input_dim=flat_input_dim).to(
                self.device
            )
        else:
            self.model = select_model(self.method, self.params, self.prior).to(
                self.device
            )

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
                if self.vae_input_is_flat:
                    dummy_x = torch.zeros(
                        2, int(np.prod(self.input_shape)), device=self.device
                    )
                else:
                    dummy_x = torch.zeros(2, *self.input_shape, device=self.device)
                o_dummy = torch.stack(
                    [
                        self._observation_code(1, observed=True).squeeze(0),
                        self._observation_code(1, observed=False).squeeze(0),
                    ],
                    dim=0,
                )
                _y_mu, _y_lss, _o_mu, _o_lss = self.model_en(dummy_x, o_dummy)
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

    def get_checkpoint_model(self):
        return CheckpointBundle(
            modules={
                "model_en": self.model_en,
                "model_de": self.model_de,
                "model_disc": self.model_disc,
                "model_cl": self.model_cl,
                "model_pn": self.model,
            },
            optimizers={
                "optimizer_vae": self.optimizer_vae,
                "optimizer_disc": self.optimizer_disc,
                "optimizer_cl": self.optimizer_cl,
                "optimizer_pn": self.optimizer,
            },
        )

    def reparameterization(self, mu, lss):
        eps = torch.randn_like(mu)
        return mu + torch.exp(lss / 2.0) * eps

    @staticmethod
    def _source_sigmoid_loss(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-t * y)

    def _uses_binary_image_likelihood(self) -> bool:
        return (
            bool(getattr(self, "is_image", False))
            and len(self.input_shape) == 3
            and int(self.input_shape[0]) == 1
            and int(self.input_shape[-1]) <= 32
        )

    def _source_input(self, x: torch.Tensor) -> torch.Tensor:
        """Map binary image inputs to the VAE-PU source domain when needed."""
        if self._uses_binary_image_likelihood():
            if torch.is_floating_point(x) and (
                x.detach().min() < 0 or x.detach().max() > 1
            ):
                return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
            return x.clamp(0.0, 1.0)
        return x

    def _vae_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self._source_input(x)
        if getattr(self, "vae_input_is_flat", False):
            return x.view(x.size(0), -1)
        if getattr(self, "is_image", False) and x.dim() == 2:
            return x.view(-1, *self.input_shape)
        return x

    def _pn_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self._source_input(x)
        if getattr(self, "pn_input_is_flat", False):
            return x.view(x.size(0), -1)
        if getattr(self, "is_image", False) and x.dim() == 2:
            return x.view(-1, *self.input_shape)
        if not getattr(self, "is_image", False) and x.dim() > 2:
            return x.view(x.size(0), -1)
        return x

    def _observation_code(self, batch_size: int, observed: bool) -> torch.Tensor:
        # Source convention: labeled positives -> [1, 0], unlabeled -> [0, 1].
        class_idx = 0 if observed else 1
        idx = torch.full(
            (int(batch_size),), class_idx, dtype=torch.long, device=self.device
        )
        return F.one_hot(idx, 2).float()

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
        main_early_stopping_enabled = False
        if self.checkpoint_handler:
            main_early_stopping_enabled = bool(
                self.checkpoint_handler.early_stopping_enabled
            )
            self.set_checkpoint_early_stopping(False)
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
        self.set_checkpoint_early_stopping(main_early_stopping_enabled, reset=True)
        for epoch in range(1, self.main_epochs + 1):
            self.global_epoch += 1
            self.train_one_epoch(epoch)
            if self.validation_loader is not None and self.threshold_selection not in {
                "fixed",
                "source",
                "none",
            }:
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
            scenario = self.params.get("scenario", "single")
            proxy_train = self.evaluate_proxy_metrics_pn(self.train_loader, scenario)
            train_metrics = {**train_metrics, **proxy_train}
            if self.validation_loader is not None:
                proxy_val = self.evaluate_proxy_metrics_pn(
                    self.validation_loader,
                    scenario,
                )
                val_metrics = {**val_metrics, **proxy_val}
            self._print_metrics(
                epoch,
                self.main_epochs,
                train_metrics,
                test_metrics,
                "Main Training",
                val_metrics=val_metrics,
            )

            if (
                hasattr(self, "checkpoint_handler")
                and self.checkpoint_handler
                and epoch > self.step1_end
            ):
                all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                if val_metrics is not None:
                    all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                import time as _time

                self.checkpoint_handler(
                    epoch=self.global_epoch,
                    all_metrics=all_metrics,
                    model=self.get_checkpoint_model(),
                    elapsed_seconds=(
                        _time.time() - self._run_start_time
                        if self._run_start_time
                        else None
                    ),
                )
                if self.checkpoint_handler.best_epoch == self.global_epoch:
                    # Attach test metrics after selection so test labels cannot
                    # influence checkpoint choice but result summaries still
                    # report the selected epoch's held-out performance.
                    self.update_checkpoint_best_metrics(
                        {f"test_{k}": v for k, v in test_metrics.items()}
                    )

            if self.checkpoint_handler and self.checkpoint_handler.should_stop:
                self.console.log(
                    "Early stopping in main training stage.", style="bold red"
                )
                break

        self.finalize()

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

            x_pl_in = self._vae_input(x_pl)
            x_u_in = self._vae_input(x_u)
            x = torch.cat([x_pl_in, x_u_in], dim=0)

            o_pl = self._observation_code(x_pl.shape[0], observed=True)
            o_u = self._observation_code(x_u.shape[0], observed=False)
            o = torch.cat([o_pl, o_u], dim=0)

            self.optimizer_vae.zero_grad()

            h_y_mu, h_y_lss, h_o_mu, h_o_lss = self.model_en(x, o)
            h_y = self.reparameterization(h_y_mu, h_y_lss)
            h_o = self.reparameterization(h_o_mu, h_o_lss)

            recon_x = self.model_de(h_y, h_o)
            if self._uses_binary_image_likelihood():
                bce = F.binary_cross_entropy_with_logits(recon_x, x, reduction="none")
                loss = bce.view(bce.size(0), -1).sum(dim=1).mean()
            else:
                mse = F.mse_loss(recon_x, x, reduction="none")
                loss = 0.5 * mse.view(mse.size(0), -1).sum(dim=1).mean()

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
        if hasattr(base_dataset, "base_dataset"):
            base_dataset = base_dataset.base_dataset

        x_tr_l = base_dataset.features[base_dataset.pu_labels == 1]
        x_tr_u = base_dataset.features[base_dataset.pu_labels == -1]

        o_pl = self._observation_code(x_tr_l.shape[0], observed=True)
        o_u = self._observation_code(x_tr_u.shape[0], observed=False)

        with torch.no_grad():
            h_y_u_mu, _, _, _ = self.model_en(
                self._vae_input(x_tr_u.to(self.device)), o_u
            )
            h_y_l_mu, _, _, _ = self.model_en(
                self._vae_input(x_tr_l.to(self.device)), o_pl
            )

        h_y = torch.cat([h_y_u_mu, h_y_l_mu], dim=0).cpu().numpy()

        gmm = GaussianMixture(n_components=2, covariance_type="diag")
        gmm.fit(h_y)

        h_y_l_np = h_y_l_mu.cpu().numpy()
        var = np.maximum(gmm.covariances_, 1e-9)
        c0 = np.mean(
            -0.5 * np.square(h_y_l_np - gmm.means_[0]) / var[0] - 0.5 * np.log(var[0]),
            axis=1,
        )
        c1 = np.mean(
            -0.5 * np.square(h_y_l_np - gmm.means_[1]) / var[1] - 0.5 * np.log(var[1]),
            axis=1,
        )

        if float(np.mean(c0 > c1)) > 0.5:
            self.p = torch.tensor(
                gmm.weights_[0], device=self.device, dtype=torch.float32
            )
            self.mu = torch.tensor(
                gmm.means_[[1, 0]], device=self.device, dtype=torch.float32
            )
            self.var = torch.tensor(
                var[[1, 0]], device=self.device, dtype=torch.float32
            )
        else:
            self.p = torch.tensor(
                gmm.weights_[1], device=self.device, dtype=torch.float32
            )
            self.mu = torch.tensor(gmm.means_, device=self.device, dtype=torch.float32)
            self.var = torch.tensor(var, device=self.device, dtype=torch.float32)

        self.console.log(f"Estimated prior p: {self.p.item():.4f}", style="green")

    def _generate(self, x_pl, x_u, enable_grad: bool = False):
        def _forward_generate():
            x_pl_in = self._vae_input(x_pl)
            x_u_in = self._vae_input(x_u)

            o_pl = self._observation_code(x_pl.shape[0], observed=True)
            o_u = self._observation_code(x_u.shape[0], observed=False)

            h_y_mu, h_y_lss, h_o_mu, h_o_lss = self.model_en(x_pl_in, o_pl)
            h_y = self.reparameterization(h_y_mu, h_y_lss)

            _, _, h_o_mu_x, h_o_lss_x = self.model_en(x_u_in, o_u)
            h_o_x = self.reparameterization(h_o_mu_x, h_o_lss_x)

            h_o_pl = self.reparameterization(h_o_mu, h_o_lss)
            dist = torch.cdist(h_o_pl, h_o_x)
            nearest_idx = torch.argmin(dist, dim=1)
            ne_h_o = h_o_x[nearest_idx]

            is_mnist = self.is_image and self.input_shape[0] == 1
            x_gen = self.model_de(h_y, ne_h_o, sigmoid=is_mnist)

            return x_gen

        if enable_grad:
            return _forward_generate()
        with torch.no_grad():
            return _forward_generate()

    def _train_step_disc(self, x_pl, x_u):
        self.optimizer_disc.zero_grad()
        with torch.no_grad():
            x_pu = self._generate(x_pl, x_u)

        if getattr(self, "vae_input_is_flat", False):
            d_x_pu = self.model_disc(x_pu.view(x_pu.size(0), -1))
            d_x_u = self.model_disc(self._vae_input(x_u))
        else:
            d_x_pu = self.model_disc(x_pu)
            d_x_u = self.model_disc(self._source_input(x_u))

        loss_pu = F.binary_cross_entropy_with_logits(d_x_pu, torch.zeros_like(d_x_pu))
        loss_u = F.binary_cross_entropy_with_logits(d_x_u, torch.ones_like(d_x_u))

        loss = self.params.get("alpha_disc", 1.0) * (loss_pu + loss_u)
        loss.backward()
        self.optimizer_disc.step()
        return float(loss.detach().item())

    def _train_step_vae(self, x_pl, x_u, epoch):
        self.optimizer_vae.zero_grad()
        self.optimizer_cl.zero_grad()

        x_pl_in = self._vae_input(x_pl)
        x_u_in = self._vae_input(x_u)
        x_flat = torch.cat([x_pl_in, x_u_in], dim=0)

        p = float(getattr(self, "pi_p", self.params.get("pi_given", 0.5)))
        o_pl = self._observation_code(x_pl.shape[0], observed=True)
        o_u = self._observation_code(x_u.shape[0], observed=False)
        o = torch.cat([o_pl, o_u], dim=0)

        h_y_mu, h_y_lss, h_o_mu, h_o_lss = self.model_en(x_flat, o)
        h_y = self.reparameterization(h_y_mu, h_y_lss)
        h_o = self.reparameterization(h_o_mu, h_o_lss)
        log_prior_neg = torch.log(
            torch.as_tensor(1 - p + 1e-9, device=self.device, dtype=h_y.dtype)
        )
        log_prior_pos = torch.log(
            torch.as_tensor(p + 1e-9, device=self.device, dtype=h_y.dtype)
        )

        c0 = (
            -0.5
            * torch.sum(
                ((h_y - self.mu[0]) ** 2 / self.var[0]) + torch.log(self.var[0] + 1e-9),
                dim=1,
            )
            + log_prior_neg
        )
        c1 = (
            -0.5
            * torch.sum(
                ((h_y - self.mu[1]) ** 2 / self.var[1]) + torch.log(self.var[1] + 1e-9),
                dim=1,
            )
            + log_prior_pos
        )
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
        if self._uses_binary_image_likelihood():
            bce = F.binary_cross_entropy_with_logits(recon_x, x_flat, reduction="none")
            loss3 = -bce.view(bce.size(0), -1).sum(dim=1).mean()
        else:
            mse = F.mse_loss(recon_x, x_flat, reduction="none")
            loss3 = -0.5 * mse.view(mse.size(0), -1).sum(dim=1).mean()

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

        x_pu = self._generate(x_pl, x_u, enable_grad=True)
        if getattr(self, "vae_input_is_flat", False):
            d_x_pu = self.model_disc(x_pu.view(x_pu.size(0), -1))
        else:
            d_x_pu = self.model_disc(x_pu)
        loss_gan = F.binary_cross_entropy_with_logits(d_x_pu, torch.ones_like(d_x_pu))

        loss_gan2 = torch.tensor(0.0, device=self.device)
        if epoch > self.step1_end:
            d_x_pu2 = self.model(self._pn_input(x_pu))
            pos_logit = self._pn_pos_logit(d_x_pu2)
            loss_gan2 = self._source_sigmoid_loss(
                pos_logit, torch.ones_like(pos_logit)
            ).mean()

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

        pi_pl = float(getattr(self, "pi_pl", self.params.get("pi_pl", 0.0)))
        pi_pu = float(getattr(self, "pi_pu", self.params.get("pi_pu", 0.0)))
        pi_u = float(getattr(self, "pi_u", self.params.get("pi_u", 1.0 - pi_pl)))

        x_pu = self._generate(x_pl, x_u)

        pn_x_pl = self._pn_pos_logit(self.model(self._pn_input(x_pl)))
        pn_x_pu = self._pn_pos_logit(self.model(self._pn_input(x_pu.detach())))
        pn_x_u = self._pn_pos_logit(self.model(self._pn_input(x_u)))

        pl_loss = (
            pi_pl * self._source_sigmoid_loss(pn_x_pl, torch.ones_like(pn_x_pl))
        ).mean()
        pu1_loss = (
            pi_pu * self._source_sigmoid_loss(pn_x_pu, torch.ones_like(pn_x_pu))
        ).mean()

        negative_risk = (
            -pi_pu * self._source_sigmoid_loss(pn_x_pu, -torch.ones_like(pn_x_pu))
        ).mean() + (
            pi_u * self._source_sigmoid_loss(pn_x_u, -torch.ones_like(pn_x_u))
        ).mean()

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
        batch_size = int(self.params.get("batch_size_l", 50))
        return DataLoader(
            p_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=bool(getattr(self, "use_mnist_source_recipe", False))
            and len(p_dataset) >= batch_size,
        )

    def get_unlabeled_loader(self):
        unl_indices = (self.train_loader.dataset.pu_labels == -1).nonzero().squeeze()
        u_dataset = torch.utils.data.Subset(self.train_loader.dataset, unl_indices)
        batch_size = int(self.params.get("batch_size_u", 100))
        return DataLoader(
            u_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=bool(getattr(self, "use_mnist_source_recipe", False))
            and len(u_dataset) >= batch_size,
        )

    def evaluate_metrics_pn(self, loader):
        y_true_all, y_pred_all, y_score_all = [], [], []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for x, t, y_true, _, _ in loader:
                    x, t, y_true = (
                        x.to(self.device),
                        t.to(self.device),
                        y_true.to(self.device),
                    )

                    outputs = self.model(self._pn_input(x))
                    pos_logit = self._pn_pos_logit(outputs)
                    thr = getattr(self, "pn_decision_threshold", 0.0)
                    preds_binary = (pos_logit > thr).long()
                    y_true_all.extend(y_true.cpu().numpy())
                    y_pred_all.extend(preds_binary.cpu().numpy())
                    y_score_all.extend(pos_logit.detach().cpu().numpy().tolist())
        finally:
            self.model.train(was_training)

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        acc = accuracy_score(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, zero_division=0)
        rec = recall_score(y_true_all, y_pred_all, zero_division=0)
        try:
            if len(set(y_true_all)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_true_all, y_score_all))
        except Exception:
            auc = float("nan")

        return {
            "oracle_accuracy": acc,
            "oracle_precision": prec,
            "oracle_recall": rec,
            "oracle_f1": f1,
            "oracle_auc": auc,
        }

    def evaluate_proxy_metrics_pn(
        self,
        loader,
        scenario: str = "single",
    ) -> dict[str, float]:
        correct_p, correct_u, total_p, total_u = 0, 0, 0, 0
        scores_p: list[float] = []
        scores_u: list[float] = []
        threshold = float(getattr(self, "pn_decision_threshold", 0.0))
        labeled_value, unlabeled_value = _pu_label_values_from_loader(loader)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for x, t, _y_true, _, _ in loader:
                    x = x.to(self.device)
                    t = t.to(self.device)

                    pos_logit = self._pn_pos_logit(self.model(self._pn_input(x)))
                    preds_binary = (pos_logit > threshold).long()
                    pos_score = torch.sigmoid(pos_logit)

                    p_mask = t == labeled_value
                    u_mask = t == unlabeled_value
                    if p_mask.any():
                        correct_p += preds_binary[p_mask].eq(1).sum().item()
                        total_p += p_mask.sum().item()
                        scores_p.extend(pos_score[p_mask].cpu().numpy().tolist())
                    if u_mask.any():
                        correct_u += preds_binary[u_mask].eq(0).sum().item()
                        total_u += u_mask.sum().item()
                        scores_u.extend(pos_score[u_mask].cpu().numpy().tolist())
        finally:
            self.model.train(was_training)

        if total_p == 0 or total_u == 0:
            pa = float("nan")
        elif scenario == "case-control":
            pa = 2 * self.prior * (correct_p / total_p) + (correct_u / total_u)
        else:
            pa = 2 * self.prior * (correct_p / total_p) + (correct_p + correct_u) / (
                total_p + total_u
            )

        if len(scores_p) == 0 or len(scores_u) == 0:
            pauc = float("nan")
        else:
            try:
                from sklearn.metrics import roc_auc_score

                labels = np.concatenate(
                    [np.ones(len(scores_p)), np.zeros(len(scores_u))]
                )
                scores = np.asarray(scores_p + scores_u)
                pauc = float(roc_auc_score(labels, scores))
            except ValueError:
                pauc = 0.5

        return {"proxy_acc": pa, "proxy_auc": pauc}

    def _compute_optimal_threshold_on_val(self, mode: str = "f1") -> float:
        """Compute optimal PN decision threshold on validation set"""
        if self.validation_loader is None:
            return getattr(self, "pn_decision_threshold", 0.0)

        scores, labels = [], []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for x, _t, y_true, _, _ in self.validation_loader:
                    x = x.to(self.device)
                    s = (
                        self._pn_pos_logit(self.model(self._pn_input(x)))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    scores.append(s)
                    labels.append(y_true.numpy())
        finally:
            self.model.train(was_training)
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
