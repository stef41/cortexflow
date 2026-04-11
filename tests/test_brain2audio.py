"""Tests for the Brain → Audio pipeline."""

import torch
import pytest

from cortexflow._types import BrainData
from cortexflow.brain2audio import AudioDiT, Brain2Audio, DiTBlock1D, build_brain2audio
from conftest import BATCH, HIDDEN_DIM, N_MELS, N_VOXELS, AUDIO_LEN, NUM_HEADS


class TestDiTBlock1D:
    def test_forward_shape(self):
        block = DiTBlock1D(HIDDEN_DIM, NUM_HEADS, cond_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, 16, HIDDEN_DIM)
        c = torch.randn(BATCH, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = block(x, c, bt)
        assert out.shape == x.shape

    def test_without_brain_tokens(self):
        block = DiTBlock1D(HIDDEN_DIM, NUM_HEADS, cond_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, 16, HIDDEN_DIM)
        c = torch.randn(BATCH, HIDDEN_DIM)
        out = block(x, c, None)
        assert out.shape == x.shape


class TestAudioDiT:
    @pytest.fixture
    def model(self):
        return AudioDiT(
            n_mels=N_MELS, seq_len=AUDIO_LEN, hidden_dim=HIDDEN_DIM,
            depth=1, num_heads=NUM_HEADS, cond_dim=HIDDEN_DIM,
        )

    def test_forward_shape(self, model):
        x = torch.randn(BATCH, N_MELS, AUDIO_LEN)
        t = torch.rand(BATCH)
        bg = torch.randn(BATCH, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = model(x, t, bg, bt)
        assert out.shape == (BATCH, N_MELS, AUDIO_LEN)

    def test_gradient_flow(self, model):
        x = torch.randn(BATCH, N_MELS, AUDIO_LEN, requires_grad=True)
        t = torch.rand(BATCH)
        bg = torch.randn(BATCH, HIDDEN_DIM)
        out = model(x, t, bg)
        out.sum().backward()
        assert x.grad is not None


class TestBrain2Audio:
    @pytest.fixture
    def model(self):
        return build_brain2audio(
            n_voxels=N_VOXELS, n_mels=N_MELS, audio_len=AUDIO_LEN,
            hidden_dim=HIDDEN_DIM, depth=1,
        )

    @pytest.fixture
    def brain_data(self):
        return BrainData(voxels=torch.randn(BATCH, N_VOXELS))

    def test_training_loss(self, model, brain_data):
        mel = torch.randn(BATCH, N_MELS, AUDIO_LEN)
        loss = model.training_loss(mel, brain_data)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_training_loss_backward(self, model, brain_data):
        mel = torch.randn(BATCH, N_MELS, AUDIO_LEN)
        loss = model.training_loss(mel, brain_data)
        loss.backward()
        for p in model.dit.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_reconstruct(self, model, brain_data):
        model.eval()
        result = model.reconstruct(brain_data, num_steps=2, cfg_scale=2.0)
        assert result.modality.value == "audio"
        assert result.output.shape == (BATCH, N_MELS, AUDIO_LEN)
        assert torch.isfinite(result.output).all()

    def test_reconstruct_single_sample(self, model):
        bd = BrainData(voxels=torch.randn(1, N_VOXELS))
        model.eval()
        result = model.reconstruct(bd, num_steps=2)
        assert result.output.shape[0] == 1


class TestMelToWaveform:
    def test_output_shape(self):
        mel = torch.rand(1, N_MELS, 16).abs() + 0.01
        waveform = Brain2Audio.mel_to_waveform(mel, n_fft=64, hop_length=16)
        assert waveform.ndim == 2
        assert waveform.shape[0] == 1
        assert waveform.shape[1] > 0

    def test_batch(self):
        mel = torch.rand(BATCH, N_MELS, 16).abs() + 0.01
        waveform = Brain2Audio.mel_to_waveform(mel, n_fft=64, hop_length=16)
        assert waveform.shape[0] == BATCH


class TestBuildBrain2Audio:
    def test_default_build(self):
        model = build_brain2audio(n_voxels=64, n_mels=16, audio_len=32, hidden_dim=16, depth=1)
        assert isinstance(model, Brain2Audio)
