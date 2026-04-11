"""Tests for the Brain → Image pipeline."""

import torch
import pytest

from cortexflow._types import BrainData, DiTConfig, FlowConfig, VAEConfig
from cortexflow.brain2img import Brain2Image, build_brain2img
from conftest import BATCH, HIDDEN_DIM, IMG_SIZE, N_VOXELS, NUM_HEADS


class TestBrain2Image:
    @pytest.fixture
    def model(self):
        return build_brain2img(
            n_voxels=N_VOXELS,
            img_size=IMG_SIZE,
            hidden_dim=HIDDEN_DIM,
            depth=1,
            num_heads=NUM_HEADS,
        )

    @pytest.fixture
    def brain_data(self):
        return BrainData(voxels=torch.randn(BATCH, N_VOXELS))

    def test_encode_brain(self, model, brain_data):
        bg, bt = model.encode_brain(brain_data)
        assert bg.shape[0] == BATCH
        assert bg.shape[1] == HIDDEN_DIM
        assert bt.ndim == 3

    def test_training_loss(self, model, brain_data):
        images = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        loss = model.training_loss(images, brain_data)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_training_loss_no_cfg_dropout(self, model, brain_data):
        images = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        loss = model.training_loss(images, brain_data, cfg_dropout=0.0)
        assert torch.isfinite(loss)

    def test_training_loss_full_cfg_dropout(self, model, brain_data):
        images = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        model.train()
        loss = model.training_loss(images, brain_data, cfg_dropout=1.0)
        assert torch.isfinite(loss)

    def test_training_loss_backward(self, model, brain_data):
        images = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        loss = model.training_loss(images, brain_data)
        loss.backward()
        # Check gradients exist on DiT params
        for p in model.dit.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_reconstruct(self, model, brain_data):
        model.eval()
        result = model.reconstruct(brain_data, num_steps=2, cfg_scale=2.0)
        assert result.modality.value == "image"
        assert result.output.shape == (BATCH, 3, IMG_SIZE, IMG_SIZE)
        assert result.n_steps == 2
        assert result.cfg_scale == 2.0
        assert torch.isfinite(result.output).all()

    def test_reconstruct_clamped(self, model, brain_data):
        model.eval()
        result = model.reconstruct(brain_data, num_steps=2)
        assert result.output.min() >= 0
        assert result.output.max() <= 1

    def test_reconstruct_single_sample(self, model):
        bd = BrainData(voxels=torch.randn(1, N_VOXELS))
        model.eval()
        result = model.reconstruct(bd, num_steps=2)
        assert result.output.shape[0] == 1


class TestBuildBrain2Img:
    def test_default_build(self):
        model = build_brain2img(n_voxels=64, img_size=8, hidden_dim=16, depth=1, num_heads=4)
        assert isinstance(model, Brain2Image)

    def test_param_count(self):
        model = build_brain2img(n_voxels=64, img_size=8, hidden_dim=16, depth=1, num_heads=4)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1000  # non-trivial model
