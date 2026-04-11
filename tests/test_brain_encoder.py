"""Tests for brain encoder modules."""

import torch
import pytest

from cortexflow.brain_encoder import (
    BrainEncoder,
    ROIBrainEncoder,
    SubjectAdapter,
    make_synthetic_brain_data,
)
from conftest import BATCH, HIDDEN_DIM, N_BRAIN_TOKENS, N_VOXELS


class TestBrainEncoder:
    @pytest.fixture
    def encoder(self):
        return BrainEncoder(N_VOXELS, cond_dim=HIDDEN_DIM, n_tokens=N_BRAIN_TOKENS)

    def test_output_shapes(self, encoder):
        voxels = torch.randn(BATCH, N_VOXELS)
        bg, bt = encoder(voxels)
        assert bg.shape == (BATCH, HIDDEN_DIM)
        assert bt.shape == (BATCH, N_BRAIN_TOKENS, HIDDEN_DIM)

    def test_single_sample(self, encoder):
        voxels = torch.randn(N_VOXELS)  # no batch dim
        bg, bt = encoder(voxels)
        assert bg.shape == (1, HIDDEN_DIM)
        assert bt.shape == (1, N_BRAIN_TOKENS, HIDDEN_DIM)

    def test_gradient_flow(self, encoder):
        voxels = torch.randn(BATCH, N_VOXELS, requires_grad=True)
        bg, bt = encoder(voxels)
        (bg.sum() + bt.sum()).backward()
        assert voxels.grad is not None

    def test_different_inputs_different_outputs(self, encoder):
        v1 = torch.randn(1, N_VOXELS)
        v2 = torch.randn(1, N_VOXELS)
        bg1, _ = encoder(v1)
        bg2, _ = encoder(v2)
        assert not torch.equal(bg1, bg2)

    def test_custom_dims(self):
        enc = BrainEncoder(128, cond_dim=64, n_tokens=8, hidden_dim=48)
        bg, bt = enc(torch.randn(1, 128))
        assert bg.shape == (1, 64)
        assert bt.shape == (1, 8, 64)

    def test_dropout_inference(self):
        enc = BrainEncoder(N_VOXELS, cond_dim=HIDDEN_DIM, dropout=0.5)
        enc.eval()
        v = torch.randn(1, N_VOXELS)
        bg1, _ = enc(v)
        bg2, _ = enc(v)
        torch.testing.assert_close(bg1, bg2)  # deterministic at eval


class TestROIBrainEncoder:
    @pytest.fixture
    def roi_encoder(self):
        return ROIBrainEncoder(
            roi_sizes={"V1": 20, "FFA": 15, "A1": 10},
            cond_dim=HIDDEN_DIM,
            n_tokens=N_BRAIN_TOKENS,
            per_roi_dim=16,
        )

    def test_output_shapes(self, roi_encoder):
        roi_voxels = {
            "V1": torch.randn(BATCH, 20),
            "FFA": torch.randn(BATCH, 15),
            "A1": torch.randn(BATCH, 10),
        }
        bg, bt = roi_encoder(roi_voxels)
        assert bg.shape == (BATCH, HIDDEN_DIM)
        assert bt.shape == (BATCH, N_BRAIN_TOKENS, HIDDEN_DIM)

    def test_single_sample(self, roi_encoder):
        roi_voxels = {
            "V1": torch.randn(20),
            "FFA": torch.randn(15),
            "A1": torch.randn(10),
        }
        bg, bt = roi_encoder(roi_voxels)
        assert bg.shape == (1, HIDDEN_DIM)

    def test_gradient_flow(self, roi_encoder):
        roi_voxels = {
            "V1": torch.randn(BATCH, 20, requires_grad=True),
            "FFA": torch.randn(BATCH, 15, requires_grad=True),
            "A1": torch.randn(BATCH, 10, requires_grad=True),
        }
        bg, bt = roi_encoder(roi_voxels)
        (bg.sum() + bt.sum()).backward()
        for v in roi_voxels.values():
            assert v.grad is not None


class TestSubjectAdapter:
    @pytest.fixture
    def adapter(self):
        return SubjectAdapter(cond_dim=HIDDEN_DIM, rank=4, n_subjects=5)

    def test_output_shape(self, adapter):
        bg = torch.randn(BATCH, HIDDEN_DIM)
        subj = torch.tensor([0, 1])
        out = adapter(bg, subj)
        assert out.shape == (BATCH, HIDDEN_DIM)

    def test_zero_init(self, adapter):
        """At initialization, adapter should be near-identity."""
        bg = torch.randn(BATCH, HIDDEN_DIM)
        subj = torch.tensor([0, 1])
        out = adapter(bg, subj)
        torch.testing.assert_close(out, bg, atol=1e-5, rtol=1e-5)

    def test_different_subjects_after_training(self, adapter):
        """After a gradient step, different subjects should give different outputs."""
        bg = torch.randn(1, HIDDEN_DIM)
        s0 = adapter(bg, torch.tensor([0]))
        s1 = adapter(bg, torch.tensor([1]))
        # Before training, both near-identical (zero init)
        torch.testing.assert_close(s0, s1, atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self, adapter):
        bg = torch.randn(BATCH, HIDDEN_DIM, requires_grad=True)
        subj = torch.tensor([0, 1])
        out = adapter(bg, subj)
        out.sum().backward()
        assert bg.grad is not None


class TestMakeSyntheticBrainData:
    def test_output_shape(self):
        data = make_synthetic_brain_data(batch_size=4, n_voxels=128)
        assert data.shape == (4, 128)

    def test_statistics(self):
        data = make_synthetic_brain_data(batch_size=16, n_voxels=256)
        # Should be roughly zero-mean, unit-var
        assert data.mean().abs() < 0.5
        assert data.std() > 0.3

    def test_device(self):
        data = make_synthetic_brain_data(batch_size=2, n_voxels=64, device="cpu")
        assert data.device.type == "cpu"

    def test_different_batches_different(self):
        d1 = make_synthetic_brain_data(batch_size=2, n_voxels=64)
        d2 = make_synthetic_brain_data(batch_size=2, n_voxels=64)
        # Different random seeds → different data
        assert not torch.equal(d1, d2)
