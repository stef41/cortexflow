"""Example: Reconstruct audio from synthetic fMRI data.

Demonstrates the Brain2Audio pipeline with mel spectrogram generation
and optional Griffin-Lim waveform conversion.
"""

import torch

from cortexflow import BrainData, Brain2Audio, build_brain2audio, make_synthetic_brain_data

# Build model
model = build_brain2audio(
    n_voxels=1024,
    n_mels=80,
    audio_len=128,
    hidden_dim=256,
    depth=6,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate synthetic fMRI
fmri = make_synthetic_brain_data(batch_size=2, n_voxels=1024)
brain = BrainData(voxels=fmri)

# Training step
mel_target = torch.randn(2, 80, 128).abs()  # replace with real mel spectrograms
model.train()
loss = model.training_loss(mel_target, brain)
print(f"Training loss: {loss.item():.4f}")

# Reconstruct mel spectrogram
model.eval()
result = model.reconstruct(brain, num_steps=20, cfg_scale=3.0)
mel = result.output
print(f"Reconstructed mel shape: {mel.shape}")

# Convert to waveform via Griffin-Lim
waveform = Brain2Audio.mel_to_waveform(mel.abs(), n_fft=1024, hop_length=256)
print(f"Waveform shape: {waveform.shape}")
print(f"Duration: {waveform.shape[-1] / 16000:.2f}s at 16kHz")
