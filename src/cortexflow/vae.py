"""Lightweight Variational Autoencoder for latent space compression.

Maps images (B, 3, H, W) → latents (B, C_latent, H/f, W/f) and back.
Also supports 1D signals (audio mel spectrograms).

This is a simple convolutional VAE — for production use, swap in a
pretrained SD/FLUX VAE with ``from_pretrained()``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortexflow._types import VAEConfig


class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    """Convolutional encoder: image → (mu, logvar) in latent space."""

    def __init__(self, in_channels: int, latent_channels: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        # Input conv
        prev_ch = in_channels
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_ch, h_dim, 3, stride=2, padding=1),
                nn.GroupNorm(min(32, h_dim), h_dim),
                nn.SiLU(),
                ResBlock(h_dim),
            ])
            prev_ch = h_dim

        self.encoder = nn.Sequential(*layers)
        self.to_mu = nn.Conv2d(prev_ch, latent_channels, 1)
        self.to_logvar = nn.Conv2d(prev_ch, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)


class Decoder(nn.Module):
    """Convolutional decoder: latent → image."""

    def __init__(self, latent_channels: int, out_channels: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        # Reverse the hidden dims
        dims = list(reversed(hidden_dims))
        prev_ch = dims[0]
        layers.append(nn.Conv2d(latent_channels, prev_ch, 1))

        for h_dim in dims[1:]:
            layers.extend([
                ResBlock(prev_ch),
                nn.ConvTranspose2d(prev_ch, h_dim, 4, stride=2, padding=1),
                nn.GroupNorm(min(32, h_dim), h_dim),
                nn.SiLU(),
            ])
            prev_ch = h_dim

        # Final upsample + output conv
        layers.extend([
            ResBlock(prev_ch),
            nn.ConvTranspose2d(prev_ch, prev_ch, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, prev_ch), prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, out_channels, 3, padding=1),
        ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class LatentVAE(nn.Module):
    """Variational Autoencoder for image ↔ latent compression.

    Default configuration compresses by 2^len(hidden_dims) spatially.
    With ``hidden_dims=[64, 128, 256, 512]`` → 16× downsampling
    (256×256 image → 16×16×4 latent).

    Usage::

        vae = LatentVAE()
        z, mu, logvar = vae.encode(images)
        reconstructed = vae.decode(z)
        loss = vae.loss(images, reconstructed, mu, logvar)
    """

    def __init__(self, config: VAEConfig | None = None) -> None:
        super().__init__()
        cfg = config or VAEConfig()
        self.config = cfg

        self.encoder = Encoder(cfg.in_channels, cfg.latent_channels, cfg.hidden_dims)
        self.decoder = Decoder(cfg.latent_channels, cfg.in_channels, cfg.hidden_dims)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images to latent space.

        Returns:
            (z, mu, logvar) where z is the reparameterized sample.
        """
        mu, logvar = self.encoder(x)
        # Reparameterization trick
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents back to image space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute VAE loss = reconstruction + KL divergence.

        Returns:
            (total_loss, {"recon": ..., "kl": ...})
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.config.kl_weight * kl_loss
        return total, {"recon": recon_loss.item(), "kl": kl_loss.item()}


# ── 1D VAE for Audio Spectrograms ────────────────────────────────────────


class ResBlock1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AudioVAE(nn.Module):
    """1D VAE for mel spectrogram latent compression.

    Input: ``(B, n_mels, T)`` → Latent: ``(B, latent_channels, T // stride)``
    """

    def __init__(
        self,
        n_mels: int = 80,
        latent_channels: int = 8,
        hidden_dim: int = 128,
        n_downsample: int = 3,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder
        enc: list[nn.Module] = [nn.Conv1d(n_mels, hidden_dim, 3, padding=1)]
        ch = hidden_dim
        for _ in range(n_downsample):
            enc.extend([
                ResBlock1D(ch),
                nn.Conv1d(ch, ch * 2, 4, stride=2, padding=1),
                nn.SiLU(),
            ])
            ch *= 2
        enc.append(ResBlock1D(ch))
        self.encoder = nn.Sequential(*enc)
        self.to_mu = nn.Conv1d(ch, latent_channels, 1)
        self.to_logvar = nn.Conv1d(ch, latent_channels, 1)

        # Decoder
        dec: list[nn.Module] = [nn.Conv1d(latent_channels, ch, 1)]
        for _ in range(n_downsample):
            dec.extend([
                ResBlock1D(ch),
                nn.ConvTranspose1d(ch, ch // 2, 4, stride=2, padding=1),
                nn.SiLU(),
            ])
            ch //= 2
        dec.extend([ResBlock1D(ch), nn.Conv1d(ch, n_mels, 1)])
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        std = (0.5 * logvar).exp()
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
