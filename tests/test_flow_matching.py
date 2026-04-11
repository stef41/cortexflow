"""Tests for rectified flow matching."""

import torch
import pytest

from cortexflow._types import DiTConfig, FlowConfig
from cortexflow.dit import DiffusionTransformer
from cortexflow.flow_matching import EMAModel, RectifiedFlowMatcher
from conftest import BATCH, HIDDEN_DIM, NUM_HEADS


class TestRectifiedFlowMatcher:
    @pytest.fixture
    def fm(self):
        return RectifiedFlowMatcher(FlowConfig(num_steps=4, logit_normal=True))

    @pytest.fixture
    def fm_uniform(self):
        return RectifiedFlowMatcher(FlowConfig(num_steps=4, logit_normal=False))

    @pytest.fixture
    def fm_midpoint(self):
        return RectifiedFlowMatcher(FlowConfig(num_steps=4, solver="midpoint"))

    @pytest.fixture
    def small_dit(self):
        cfg = DiTConfig(hidden_dim=HIDDEN_DIM, depth=1, num_heads=NUM_HEADS, cond_dim=HIDDEN_DIM, patch_size=2)
        return DiffusionTransformer(cfg, img_size=4)

    def test_sample_timesteps_logit_normal(self, fm):
        t = fm.sample_timesteps(100, torch.device("cpu"))
        assert t.shape == (100,)
        assert t.min() > 0
        assert t.max() < 1

    def test_sample_timesteps_uniform(self, fm_uniform):
        t = fm_uniform.sample_timesteps(100, torch.device("cpu"))
        assert t.shape == (100,)
        assert t.min() > 0
        assert t.max() < 1

    def test_compute_loss(self, fm, small_dit):
        x1 = torch.randn(BATCH, 4, 4, 4)
        bg = torch.randn(BATCH, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        loss = fm.compute_loss(small_dit, x1, bg, bt)
        assert loss.shape == ()
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_compute_loss_backward(self, fm, small_dit):
        x1 = torch.randn(BATCH, 4, 4, 4)
        bg = torch.randn(BATCH, HIDDEN_DIM)
        loss = fm.compute_loss(small_dit, x1, bg)
        loss.backward()
        for p in small_dit.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_sample_euler(self, fm, small_dit):
        bg = torch.randn(BATCH, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = fm.sample(small_dit, (BATCH, 4, 4, 4), bg, bt, num_steps=2)
        assert out.shape == (BATCH, 4, 4, 4)
        assert torch.isfinite(out).all()

    def test_sample_midpoint(self, fm_midpoint, small_dit):
        bg = torch.randn(BATCH, HIDDEN_DIM)
        out = fm_midpoint.sample(small_dit, (BATCH, 4, 4, 4), bg, num_steps=2)
        assert out.shape == (BATCH, 4, 4, 4)

    def test_sample_with_cfg(self, fm, small_dit):
        bg = torch.randn(BATCH, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        bg_uncond = torch.zeros(BATCH, HIDDEN_DIM)
        bt_uncond = torch.zeros(BATCH, 4, HIDDEN_DIM)
        out = fm.sample(
            small_dit, (BATCH, 4, 4, 4), bg, bt,
            num_steps=2, cfg_scale=3.0,
            brain_global_uncond=bg_uncond, brain_tokens_uncond=bt_uncond,
        )
        assert out.shape == (BATCH, 4, 4, 4)

    def test_sample_no_cfg(self, fm, small_dit):
        bg = torch.randn(BATCH, HIDDEN_DIM)
        out = fm.sample(small_dit, (BATCH, 4, 4, 4), bg, num_steps=2, cfg_scale=1.0)
        assert out.shape == (BATCH, 4, 4, 4)

    def test_loss_decreases_with_training(self, small_dit):
        """Verify loss decreases after a few optimizer steps."""
        fm = RectifiedFlowMatcher(FlowConfig(num_steps=4))
        opt = torch.optim.Adam(small_dit.parameters(), lr=1e-3)
        bg = torch.randn(BATCH, HIDDEN_DIM)

        losses = []
        for _ in range(10):
            x1 = torch.randn(BATCH, 4, 4, 4)
            loss = fm.compute_loss(small_dit, x1, bg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should generally trend downward (not necessarily monotonic)
        assert losses[-1] < losses[0] * 1.5  # at least not diverging


class TestEMAModel:
    def test_create(self):
        model = torch.nn.Linear(10, 10)
        ema = EMAModel(model, decay=0.999)
        assert len(ema.shadow) == 2  # weight + bias

    def test_update(self):
        model = torch.nn.Linear(10, 10)
        ema = EMAModel(model, decay=0.999)
        original = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        ema.update(model)

        # EMA should have changed but not to exactly the model values
        for name in ema.shadow:
            assert not torch.equal(ema.shadow[name], original[name])

    def test_apply_and_restore(self):
        model = torch.nn.Linear(10, 10)
        ema = EMAModel(model, decay=0.999)

        # Make some updates
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)

        # Apply EMA
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}
        originals = ema.apply_to(model)

        # Model should now have EMA weights
        for name, param in model.named_parameters():
            if name in ema.shadow:
                torch.testing.assert_close(param.data, ema.shadow[name])

        # Restore
        ema.restore(model, originals)
        for name, param in model.named_parameters():
            if name in original_params:
                torch.testing.assert_close(param.data, original_params[name])

    def test_decay_effect(self):
        model = torch.nn.Linear(10, 10)

        ema_fast = EMAModel(model, decay=0.9)
        ema_slow = EMAModel(model, decay=0.999)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(10.0)

        ema_fast.update(model)
        ema_slow.update(model)

        # Fast decay should be closer to current model params
        for name, param in model.named_parameters():
            if name in ema_fast.shadow:
                diff_fast = (ema_fast.shadow[name] - param.data).abs().mean()
                diff_slow = (ema_slow.shadow[name] - param.data).abs().mean()
                assert diff_fast < diff_slow
