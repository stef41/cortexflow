"""Tests for core data types."""

import torch
import pytest

from cortexflow._types import (
    BrainData,
    CortexFlowError,
    DiTConfig,
    FlowConfig,
    Modality,
    ReconstructionResult,
    TrainingConfig,
    VAEConfig,
)


class TestBrainData:
    def test_single_sample(self):
        voxels = torch.randn(100)
        bd = BrainData(voxels=voxels)
        assert bd.n_voxels == 100
        assert bd.batch_size == 1

    def test_batch(self):
        voxels = torch.randn(4, 200)
        bd = BrainData(voxels=voxels, subject_id="sub-01")
        assert bd.n_voxels == 200
        assert bd.batch_size == 4
        assert bd.subject_id == "sub-01"

    def test_with_roi_mask(self):
        voxels = torch.randn(2, 50)
        mask = torch.ones(50, dtype=torch.bool)
        bd = BrainData(voxels=voxels, roi_mask=mask)
        assert bd.roi_mask is not None
        assert bd.roi_mask.shape == (50,)

    def test_default_tr(self):
        bd = BrainData(voxels=torch.randn(10))
        assert bd.tr == 2.0


class TestModality:
    def test_values(self):
        assert Modality.IMAGE == "image"
        assert Modality.AUDIO == "audio"
        assert Modality.TEXT == "text"

    def test_is_string_enum(self):
        assert isinstance(Modality.IMAGE, str)


class TestReconstructionResult:
    def test_defaults(self):
        result = ReconstructionResult(
            modality=Modality.IMAGE,
            output=torch.randn(1, 3, 64, 64),
            brain_condition=torch.randn(1, 768),
        )
        assert result.n_steps == 50
        assert result.cfg_scale == 1.0
        assert result.metadata == {}

    def test_with_metadata(self):
        result = ReconstructionResult(
            modality=Modality.TEXT,
            output=torch.zeros(1, 10),
            brain_condition=torch.randn(1, 256),
            metadata={"texts": ["hello"]},
        )
        assert result.metadata["texts"] == ["hello"]


class TestConfigs:
    def test_dit_config_defaults(self):
        cfg = DiTConfig()
        assert cfg.in_channels == 4
        assert cfg.hidden_dim == 768
        assert cfg.depth == 12
        assert cfg.num_heads == 12
        assert cfg.qk_norm is True

    def test_vae_config_defaults(self):
        cfg = VAEConfig()
        assert cfg.latent_channels == 4
        assert cfg.hidden_dims == [64, 128, 256, 512]

    def test_flow_config_defaults(self):
        cfg = FlowConfig()
        assert cfg.num_steps == 50
        assert cfg.logit_normal is True
        assert cfg.solver == "euler"
        assert cfg.cfg_scale == 4.0

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate == 1e-4
        assert cfg.batch_size == 16
        assert cfg.ema_decay == 0.9999

    def test_dit_config_custom(self):
        cfg = DiTConfig(hidden_dim=512, depth=8, num_heads=8)
        assert cfg.hidden_dim == 512
        assert cfg.depth == 8

    def test_flow_config_midpoint(self):
        cfg = FlowConfig(solver="midpoint")
        assert cfg.solver == "midpoint"


class TestCortexFlowError:
    def test_is_exception(self):
        assert issubclass(CortexFlowError, Exception)

    def test_raise(self):
        with pytest.raises(CortexFlowError):
            raise CortexFlowError("test error")
