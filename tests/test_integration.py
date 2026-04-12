"""End-to-end integration tests for cortexflow."""

import torch
import pytest

from cortexflow._types import BrainData, Modality
from cortexflow.brain2audio import build_brain2audio
from cortexflow.brain2img import Brain2Image, build_brain2img
from cortexflow.brain2text import build_brain2text
from cortexflow.brain_encoder import ROIBrainEncoder, make_synthetic_brain_data
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from conftest import HIDDEN_DIM, N_VOXELS


class TestEndToEndImage:
    """Full pipeline: synthetic brain → image reconstruction."""

    def test_full_pipeline(self):
        model = build_brain2img(n_voxels=N_VOXELS, img_size=8, hidden_dim=16, depth=1, num_heads=4)
        fmri = make_synthetic_brain_data(batch_size=2, n_voxels=N_VOXELS)
        brain = BrainData(voxels=fmri)

        # Training step
        images = torch.rand(2, 3, 8, 8)
        model.train()
        loss = model.training_loss(images, brain)
        assert torch.isfinite(loss)
        loss.backward()

        # Inference
        model.eval()
        result = model.reconstruct(brain, num_steps=2)
        assert result.modality == Modality.IMAGE
        assert result.output.shape == (2, 3, 8, 8)


class TestEndToEndAudio:
    """Full pipeline: synthetic brain → audio reconstruction."""

    def test_full_pipeline(self):
        model = build_brain2audio(n_voxels=N_VOXELS, n_mels=16, audio_len=16, hidden_dim=16, depth=1)
        fmri = make_synthetic_brain_data(batch_size=2, n_voxels=N_VOXELS)
        brain = BrainData(voxels=fmri)

        # Training step
        mel = torch.randn(2, 16, 16)
        model.train()
        loss = model.training_loss(mel, brain)
        assert torch.isfinite(loss)
        loss.backward()

        # Inference
        model.eval()
        result = model.reconstruct(brain, num_steps=2)
        assert result.modality == Modality.AUDIO
        assert result.output.shape == (2, 16, 16)


class TestEndToEndText:
    """Full pipeline: synthetic brain → text reconstruction."""

    def test_full_pipeline(self):
        model = build_brain2text(n_voxels=N_VOXELS, max_len=16, hidden_dim=16, depth=1)
        fmri = make_synthetic_brain_data(batch_size=2, n_voxels=N_VOXELS)
        brain = BrainData(voxels=fmri)

        # Training step
        tokens = torch.randint(32, 127, (2, 16))
        model.train()
        loss = model.training_loss(tokens, brain)
        assert torch.isfinite(loss)
        loss.backward()

        # Inference
        model.eval()
        result = model.reconstruct(brain, max_len=8)
        assert result.modality == Modality.TEXT
        assert len(result.metadata["texts"]) == 2

    def test_text_generation_quality(self):
        """Generated text should be valid UTF-8 strings."""
        model = build_brain2text(n_voxels=N_VOXELS, max_len=32, hidden_dim=16, depth=1)
        model.eval()
        fmri = make_synthetic_brain_data(batch_size=4, n_voxels=N_VOXELS)
        brain = BrainData(voxels=fmri)
        result = model.reconstruct(brain, max_len=16, temperature=1.0)
        for text in result.metadata["texts"]:
            assert isinstance(text, str)


class TestMultiModalPipeline:
    """Test that all three pipelines can coexist and share brain data."""

    def test_shared_brain_data(self):
        fmri = make_synthetic_brain_data(batch_size=2, n_voxels=N_VOXELS)
        brain = BrainData(voxels=fmri)

        img_model = build_brain2img(n_voxels=N_VOXELS, img_size=8, hidden_dim=16, depth=1, num_heads=4)
        audio_model = build_brain2audio(n_voxels=N_VOXELS, n_mels=16, audio_len=16, hidden_dim=16, depth=1)
        text_model = build_brain2text(n_voxels=N_VOXELS, max_len=16, hidden_dim=16, depth=1)

        img_model.eval()
        audio_model.eval()
        text_model.eval()

        img_result = img_model.reconstruct(brain, num_steps=2)
        audio_result = audio_model.reconstruct(brain, num_steps=2)
        text_result = text_model.reconstruct(brain, max_len=8)

        assert img_result.modality == Modality.IMAGE
        assert audio_result.modality == Modality.AUDIO
        assert text_result.modality == Modality.TEXT

    def test_different_brain_samples(self):
        """Different fMRI inputs should produce different outputs."""
        model = build_brain2img(n_voxels=N_VOXELS, img_size=8, hidden_dim=16, depth=1, num_heads=4)
        model.eval()

        torch.manual_seed(42)
        b1 = BrainData(voxels=torch.randn(1, N_VOXELS))
        b2 = BrainData(voxels=torch.randn(1, N_VOXELS))

        r1 = model.reconstruct(b1, num_steps=2)
        r2 = model.reconstruct(b2, num_steps=2)

        # Outputs should differ (brain condition differs)
        assert not torch.equal(r1.output, r2.output)


class TestROIBrainEncoding:
    """Test ROI-aware encoder integration with pipelines."""

    @pytest.fixture
    def roi_encoder(self):
        return ROIBrainEncoder(
            roi_sizes={"V1": 20, "FFA": 15, "A1": 10},
            cond_dim=16, n_tokens=4, per_roi_dim=8,
        )

    @pytest.fixture
    def roi_brain_data(self):
        return BrainData(
            voxels=torch.randn(2, 45),
            roi_voxels={"V1": torch.randn(2, 20), "FFA": torch.randn(2, 15), "A1": torch.randn(2, 10)},
        )

    def test_roi_brain2img(self, roi_encoder, roi_brain_data):
        dit_cfg = DiTConfig(hidden_dim=16, depth=1, num_heads=4, cond_dim=16)
        vae_cfg = VAEConfig(hidden_dims=[8, 16])
        model = Brain2Image(
            img_size=8, dit_config=dit_cfg, vae_config=vae_cfg,
            n_brain_tokens=4, brain_encoder=roi_encoder,
        )
        model.eval()
        result = model.reconstruct(roi_brain_data, num_steps=2)
        assert result.output.shape == (2, 3, 8, 8)

    def test_roi_ablation_changes_output(self, roi_encoder, roi_brain_data):
        """Zeroing an ROI should change the brain encoding."""
        bg1, bt1 = roi_encoder(roi_brain_data.roi_voxels)

        ablated = {k: v.clone() for k, v in roi_brain_data.roi_voxels.items()}
        ablated["FFA"] = torch.zeros_like(ablated["FFA"])
        bg2, bt2 = roi_encoder(ablated)

        assert not torch.allclose(bg1, bg2), "Ablating a region should change global embedding"
        assert not torch.allclose(bt1, bt2), "Ablating a region should change brain tokens"
