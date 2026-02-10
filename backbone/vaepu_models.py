import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.cgenpu_models import _infer_mlp_dims


class VAEencoder(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        hidden_dims = _infer_mlp_dims(input_dim)

        # Encoder for y (label-related latent variable)
        en_y_layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            en_y_layers.extend([nn.Linear(last_dim, h_dim), nn.LeakyReLU(0.2)])
            last_dim = h_dim
        self.vae_en_y = nn.Sequential(*en_y_layers)

        # Encoder for o (observation-related latent variable)
        en_o_layers = []
        last_dim = input_dim + config["n_o"]
        for h_dim in hidden_dims:
            en_o_layers.extend([nn.Linear(last_dim, h_dim), nn.LeakyReLU(0.2)])
            last_dim = h_dim
        self.vae_en_o = nn.Sequential(*en_o_layers)

        self.vae_en_y_mu = nn.Linear(hidden_dims[-1], config["n_h_y"])
        self.vae_en_y_lss = nn.Linear(hidden_dims[-1], config["n_h_y"])
        self.vae_en_o_mu = nn.Linear(hidden_dims[-1], config["n_h_o"])
        self.vae_en_o_lss = nn.Linear(hidden_dims[-1], config["n_h_o"])

    def forward(self, x, o):
        hidden_y = self.vae_en_y(x)
        y_mu = self.vae_en_y_mu(hidden_y)
        y_lss = self.vae_en_y_lss(hidden_y)

        hidden_o = self.vae_en_o(torch.cat([x, o], dim=1))
        o_mu = self.vae_en_o_mu(hidden_o)
        o_lss = self.vae_en_o_lss(hidden_o)
        return y_mu, y_lss, o_mu, o_lss


class VAEdecoder(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        hidden_dims = _infer_mlp_dims(input_dim)

        de_layers = []
        last_dim = config["n_h_y"] + config["n_h_o"]
        for h_dim in reversed(hidden_dims):
            de_layers.extend([nn.Linear(last_dim, h_dim), nn.LeakyReLU(0.2)])
            last_dim = h_dim

        de_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.vae_de = nn.Sequential(*de_layers)

    def forward(self, h_y, h_o, sigmoid=False):
        recon = self.vae_de(torch.cat([h_y, h_o], dim=1))
        if sigmoid:
            recon = torch.sigmoid(recon)
        return recon


class Discriminator(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        hidden_dims = _infer_mlp_dims(input_dim)

        disc_layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            disc_layers.extend(
                [nn.Linear(last_dim, h_dim), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
            )
            last_dim = h_dim

        disc_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.disc_u = nn.Sequential(*disc_layers)

    def forward(self, x, sigmoid=False):
        disc = self.disc_u(x)
        if sigmoid:
            disc = torch.sigmoid(disc)
        return disc


class ClassifierO(nn.Module):
    """Classifier for the observation latent variable h_o."""

    def __init__(self, n_h_o: int, n_hidden: list[int] | None = None):
        super().__init__()
        if n_hidden is None:
            n_hidden = []  # Can be an empty list for a direct linear classifier

        layers: list[nn.Module] = []
        last_dim = n_h_o
        for h_dim in n_hidden:
            layers.extend(
                [
                    nn.Linear(last_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigmoid=False):
        c = self.net(x)
        if sigmoid:
            c = torch.sigmoid(c)
        return c


# ----------------------------------------------------------------------------
# CNN variants for image inputs (28x28x1 or 32x32x3)
# ----------------------------------------------------------------------------


class VAEConvEncoder(nn.Module):
    """CNN encoder producing (mu, logvar) for y and o branches.

    For simplicity, we share initial conv trunk and split into two heads.
    """

    def __init__(self, in_channels: int, n_h_y: int, n_h_o: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  # /2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # /4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # /8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # compute flat size at runtime; use adaptive pooling to 4x4
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(256 * 4 * 4, 512)
        self.y_mu = nn.Linear(512, n_h_y)
        self.y_lss = nn.Linear(512, n_h_y)
        self.o_mu = nn.Linear(512, n_h_o)
        self.o_lss = nn.Linear(512, n_h_o)

    def forward(self, x, o_onehot):
        h = self.trunk(x)
        h = self.pool(h)
        h = h.reshape(h.size(0), -1)
        h = F.leaky_relu(self.fc(h), 0.2)
        y_mu = self.y_mu(h)
        y_lss = self.y_lss(h)
        o_mu = self.o_mu(h)
        o_lss = self.o_lss(h)
        return y_mu, y_lss, o_mu, o_lss


class VAEConvDecoder(nn.Module):
    """CNN decoder mapping (h_y, h_o) to image with optional Sigmoid.
    For MNIST use Sigmoid; for CIFAR use Tanh or clamp in Trainer.
    """

    def __init__(self, out_channels: int, n_h_y: int, n_h_o: int, img_size: int):
        super().__init__()
        self.img_size = int(img_size)

        # Build decoder branches: 32x32 (MNIST/Fashion/CIFAR-like) and 128x128 (ADNI-like)
        if self.img_size >= 64:  # ADNI 1x128x128 branch (4→8→16→32→64→128)
            self.fc = nn.Linear(n_h_y + n_h_o, 512 * 4 * 4)
            self.deconv = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 8x8
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16x16
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32x32
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x64
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, out_channels, 4, 2, 1),  # 128x128
            )
        else:  # 32x32 branch (4→8→16→32). 28x28 handled via center-crop below
            self.fc = nn.Linear(n_h_y + n_h_o, 256 * 4 * 4)
            self.deconv = nn.Sequential(
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, out_channels, 4, 2, 1),  # 32x32
            )

    def forward(self, h_y, h_o, sigmoid=False):
        h = torch.cat([h_y, h_o], dim=1)
        h = self.fc(h)
        if self.img_size >= 64:
            h = h.reshape(-1, 512, 4, 4)
        else:
            h = h.reshape(-1, 256, 4, 4)
        x = self.deconv(h)
        if self.img_size == 28:
            # center-crop to 28x28
            x = x[:, :, 2:-2, 2:-2]
        if sigmoid:
            x = torch.sigmoid(x)
        return x


class VAEConvDiscriminator(nn.Module):
    """Simple CNN discriminator producing logits."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Robust head for both 28x28 and 32x32 inputs: global pool then 1x1 conv
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1, 1, 1, 0),
        )

    def forward(self, x, sigmoid=False):
        disc = self.net(x)
        disc = disc.reshape(-1, 1)
        if sigmoid:
            disc = torch.sigmoid(disc)
        return disc


# Note: ClassifierPN is removed as we now use the standardized models from models.py

__all__ = [
    "VAEencoder",
    "VAEdecoder",
    "Discriminator",
    "ClassifierO",
    "VAEConvEncoder",
    "VAEConvDecoder",
    "VAEConvDiscriminator",
]
