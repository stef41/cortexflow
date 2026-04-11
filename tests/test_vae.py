"""Tests for the VAE modules."""

import torch
import pytest

from cortexflow._types import VAEConfig
from cortexflow.vae import AudioVAE, Decoder, Encoder, LatentVAE, ResBlock, ResBlock1D
from conftest import BATCH


class TestResBlock:
    def test_shape_preserved(self):
        block = ResBlock(32)
        x = torch.randn(BATCH, 32, 8, 8)
        assert block(x).shape == x.shape

    def test_residual_connection(self):
        block = ResBlock(16)
        x = torch.randn(1, 16, 4, 4)
        out = block(x)
        # Output should be different (block isn't zero-initialized)
        assert not torch.equal(out, x)


class TestResBlock1D:
    def test_shape_preserved(self):
        block = ResBlock1D(32)
        x = torch.randn(BATCH, 32, 64)
        assert block(x).shape == x.shape


class TestEncoder:
    def test_output_shapes(self):
        enc = Encoder(3, 4, [16, 32])
        x = torch.randn(BATCH, 3, 16, 16)
        mu, logvar = enc(x)
        # 2 downsamples: 16 → 8 → 4
        assert mu.shape == (BATCH, 4, 4, 4)
        assert logvar.shape == (BATCH, 4, 4, 4)


class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(4, 3, [16, 32])
        z = torch.randn(BATCH, 4, 4, 4)
        out = dec(z)
        # 2 upsamples: 4 → 8 → 16
        assert out.shape == (BATCH, 3, 16, 16)


class TestLatentVAE:
    @pytest.fixture
    def vae(self, vae_config):
        return LatentVAE(vae_config)

    def test_encode_shape(self, vae):
        x = torch.randn(BATCH, 3, 16, 16)
        z, mu, logvar = vae.encode(x)
        assert z.shape[0] == BATCH
        assert z.shape[1] == 4  # latent channels
        assert mu.shape == z.shape
        assert logvar.shape == z.shape

    def test_decode_shape(self, vae):
        z = torch.randn(BATCH, 4, 4, 4)
        out = vae.decode(z)
        assert out.shape[0] == BATCH
        assert out.shape[1] == 3  # RGB

    def test_forward_roundtrip(self, vae):
        x = torch.randn(BATCH, 3, 16, 16)
        x_recon, mu, logvar = vae(x)
        assert x_recon.shape == x.shape

    def test_loss_components(self, vae):
        x = torch.randn(BATCH, 3, 16, 16)
        x_recon, mu, logvar = vae(x)
        total_loss, components = vae.loss(x, x_recon, mu, logvar)
        assert torch.isfinite(total_loss)
        assert "recon" in components
        assert "kl" in components
        assert components["recon"] > 0
        assert components["kl"] >= 0

    def test_gradient_flow(self, vae):
        x = torch.randn(BATCH, 3, 16, 16, requires_grad=True)
        x_recon, mu, logvar = vae(x)
        loss, _ = vae.loss(x, x_recon, mu, logvar)
        loss.backward()
        assert x.grad is not None

    def test_reparameterization(self, vae):
        """Encoding same input twice should give different z (stochastic)."""
        x = torch.randn(1, 3, 16, 16)
        z1, _, _ = vae.encode(x)
        z2, _, _ = vae.encode(x)
        assert not torch.equal(z1, z2)

    def test_custom_config(self):
        cfg = VAEConfig(in_channels=1, latent_channels=2, hidden_dims=[8, 16], kl_weight=0.1)
        vae = LatentVAE(cfg)
        x = torch.randn(1, 1, 8, 8)
        x_recon, mu, logvar = vae(x)
        assert x_recon.shape[1] == 1


class TestAudioVAE:
    @pytest.fixture
    def audio_vae(self):
        return AudioVAE(n_mels=16, latent_channels=4, hidden_dim=16, n_downsample=2)

    def test_encode_shape(self, audio_vae):
        x = torch.randn(BATCH, 16, 32)  # (B, n_mels, T)
        z, mu, logvar = audio_vae.encode(x)
        assert z.shape[0] == BATCH
        assert z.shape[1] == 4  # latent_channels
        assert z.shape[2] == 8  # 32 / 2^2

    def test_decode_shape(self, audio_vae):
        z = torch.randn(BATCH, 4, 8)
        out = audio_vae.decode(z)
        assert out.shape == (BATCH, 16, 32)

    def test_encode_decode_roundtrip(self, audio_vae):
        x = torch.randn(BATCH, 16, 32)
        z, _, _ = audio_vae.encode(x)
        x_recon = audio_vae.decode(z)
        assert x_recon.shape == x.shape

    def test_gradient_flow(self, audio_vae):
        x = torch.randn(BATCH, 16, 32, requires_grad=True)
        z, mu, logvar = audio_vae.encode(x)
        z.sum().backward()
        assert x.grad is not None
