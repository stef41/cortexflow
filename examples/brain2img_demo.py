"""Example: Reconstruct an image from synthetic fMRI data.

Demonstrates the full Brain2Image pipeline with a small model.
"""

import torch

from cortexflow import BrainData, build_brain2img, make_synthetic_brain_data

# Build a small model (use larger dims for real data)
model = build_brain2img(
    n_voxels=1024,
    img_size=64,
    hidden_dim=256,
    depth=6,
    num_heads=8,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate synthetic fMRI data
fmri = make_synthetic_brain_data(batch_size=2, n_voxels=1024)
brain = BrainData(voxels=fmri, subject_id="sub-01")

# Training step
images = torch.rand(2, 3, 64, 64)  # replace with real images
model.train()
loss = model.training_loss(images, brain)
print(f"Training loss: {loss.item():.4f}")

# Reconstruct
model.eval()
result = model.reconstruct(brain, num_steps=20, cfg_scale=4.0)
print(f"Reconstructed image shape: {result.output.shape}")
print(f"Pixel value range: [{result.output.min():.3f}, {result.output.max():.3f}]")
