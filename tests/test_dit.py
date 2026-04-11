"""Tests for the Diffusion Transformer backbone."""

import torch
import pytest

from cortexflow._types import DiTConfig
from cortexflow.dit import (
    DiffusionTransformer,
    DiTBlock,
    FinalLayer,
    PatchEmbed,
    SwiGLU,
    TimestepEmbedding,
    modulate,
)
from conftest import BATCH, HIDDEN_DIM, NUM_HEADS, DEPTH


class TestModulate:
    def test_shape_preserved(self):
        x = torch.randn(2, 8, HIDDEN_DIM)
        shift = torch.randn(2, HIDDEN_DIM)
        scale = torch.randn(2, HIDDEN_DIM)
        out = modulate(x, shift, scale)
        assert out.shape == x.shape

    def test_identity_when_zero(self):
        x = torch.randn(2, 8, 16)
        shift = torch.zeros(2, 16)
        scale = torch.zeros(2, 16)
        out = modulate(x, shift, scale)
        torch.testing.assert_close(out, x)


class TestSwiGLU:
    def test_shape(self):
        m = SwiGLU(HIDDEN_DIM, HIDDEN_DIM * 2)
        x = torch.randn(2, 8, HIDDEN_DIM)
        assert m(x).shape == (2, 8, HIDDEN_DIM)

    def test_different_hidden(self):
        m = SwiGLU(16, 64)
        x = torch.randn(1, 4, 16)
        assert m(x).shape == (1, 4, 16)


class TestTimestepEmbedding:
    def test_scalar_input(self):
        emb = TimestepEmbedding(HIDDEN_DIM)
        t = torch.tensor(0.5)
        out = emb(t)
        assert out.shape == (1, HIDDEN_DIM)

    def test_batch_input(self):
        emb = TimestepEmbedding(HIDDEN_DIM)
        t = torch.rand(4)
        out = emb(t)
        assert out.shape == (4, HIDDEN_DIM)

    def test_different_timesteps_different_embeddings(self):
        emb = TimestepEmbedding(HIDDEN_DIM)
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])
        e1 = emb(t1)
        e2 = emb(t2)
        assert not torch.allclose(e1, e2)


class TestPatchEmbed:
    def test_output_shape(self):
        pe = PatchEmbed(4, HIDDEN_DIM, patch_size=2)
        x = torch.randn(BATCH, 4, 8, 8)
        out = pe(x)
        assert out.shape == (BATCH, 16, HIDDEN_DIM)  # (8/2)^2 = 16

    def test_different_patch_size(self):
        pe = PatchEmbed(3, 64, patch_size=4)
        x = torch.randn(1, 3, 16, 16)
        out = pe(x)
        assert out.shape == (1, 16, 64)  # (16/4)^2 = 16


class TestDiTBlock:
    def test_forward_shape(self):
        block = DiTBlock(HIDDEN_DIM, NUM_HEADS, qk_norm=True, use_cross_attn=True, cond_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, 16, HIDDEN_DIM)
        c = torch.randn(BATCH, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = block(x, c, bt)
        assert out.shape == x.shape

    def test_without_cross_attn(self):
        block = DiTBlock(HIDDEN_DIM, NUM_HEADS, qk_norm=False, use_cross_attn=False)
        x = torch.randn(BATCH, 8, HIDDEN_DIM)
        c = torch.randn(BATCH, HIDDEN_DIM)
        out = block(x, c)
        assert out.shape == x.shape

    def test_without_qk_norm(self):
        block = DiTBlock(HIDDEN_DIM, NUM_HEADS, qk_norm=False, use_cross_attn=True, cond_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, 8, HIDDEN_DIM)
        c = torch.randn(BATCH, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = block(x, c, bt)
        assert out.shape == x.shape

    def test_zero_init_gating(self):
        block = DiTBlock(HIDDEN_DIM, NUM_HEADS, use_cross_attn=True, cond_dim=HIDDEN_DIM)
        # Gating should be zero-initialized → output ≈ input at initialization
        x = torch.randn(BATCH, 8, HIDDEN_DIM)
        c = torch.zeros(BATCH, HIDDEN_DIM)
        out = block(x, c)
        # With zero conditioning, the modulation is zero, and zero-init gates
        # mean the residual contribution should be near-zero
        diff = (out - x).abs().mean()
        assert diff < 0.5  # loose check — zero init dampens but not exact zero

    def test_gradient_flow(self):
        block = DiTBlock(HIDDEN_DIM, NUM_HEADS, use_cross_attn=True, cond_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, 8, HIDDEN_DIM, requires_grad=True)
        c = torch.randn(BATCH, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = block(x, c, bt)
        out.sum().backward()
        assert x.grad is not None


class TestFinalLayer:
    def test_shape(self):
        fl = FinalLayer(HIDDEN_DIM, patch_size=2, out_channels=4)
        x = torch.randn(BATCH, 16, HIDDEN_DIM)
        c = torch.randn(BATCH, HIDDEN_DIM)
        out = fl(x, c)
        assert out.shape == (BATCH, 16, 2 * 2 * 4)

    def test_zero_init(self):
        fl = FinalLayer(HIDDEN_DIM, patch_size=2, out_channels=4)
        x = torch.randn(BATCH, 8, HIDDEN_DIM)
        c = torch.zeros(BATCH, HIDDEN_DIM)
        out = fl(x, c)
        assert out.abs().mean() < 0.1  # near-zero output at init


class TestDiffusionTransformer:
    def test_forward_shape(self, dit_config):
        model = DiffusionTransformer(dit_config, img_size=8)
        x = torch.randn(BATCH, 4, 8, 8)
        t = torch.rand(BATCH)
        bg = torch.randn(BATCH, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = model(x, t, bg, bt)
        assert out.shape == x.shape

    def test_different_image_sizes(self):
        cfg = DiTConfig(hidden_dim=HIDDEN_DIM, depth=1, num_heads=NUM_HEADS, cond_dim=HIDDEN_DIM, patch_size=2)
        for size in [4, 8, 16]:
            model = DiffusionTransformer(cfg, img_size=size)
            x = torch.randn(1, 4, size, size)
            t = torch.rand(1)
            bg = torch.randn(1, HIDDEN_DIM)
            out = model(x, t, bg)
            assert out.shape == x.shape

    def test_no_brain_tokens(self, dit_config):
        model = DiffusionTransformer(dit_config, img_size=8)
        x = torch.randn(BATCH, 4, 8, 8)
        t = torch.rand(BATCH)
        bg = torch.randn(BATCH, HIDDEN_DIM)
        out = model(x, t, bg)  # no brain_tokens
        assert out.shape == x.shape

    def test_gradient_flow(self, dit_config):
        model = DiffusionTransformer(dit_config, img_size=8)
        x = torch.randn(BATCH, 4, 8, 8, requires_grad=True)
        t = torch.rand(BATCH)
        bg = torch.randn(BATCH, HIDDEN_DIM)
        out = model(x, t, bg)
        out.sum().backward()
        assert x.grad is not None

    def test_param_count(self, dit_config):
        model = DiffusionTransformer(dit_config, img_size=8)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        assert n_params < 10_000_000  # small test config
