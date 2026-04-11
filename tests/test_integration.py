"""End-to-end integration tests for cortexflow."""

import torch
import pytest

from cortexflow._types import BrainData, Modality
from cortexflow.brain2audio import build_brain2audio
from cortexflow.brain2img import build_brain2img
from cortexflow.brain2text import build_brain2text
from cortexflow.brain_encoder import make_synthetic_brain_data
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
