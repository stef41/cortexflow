"""Example: ROI-targeted diverse brain decoding.

Demonstrates generating multiple semantically diverse reconstructions
from the same brain signal, using ROI-aware encoding that processes
different cortical regions (V1, FFA, A1) independently.

Shows that brain_noise produces genuinely different outputs by
measuring pairwise L2 distances between samples.

NOTE: Models are briefly trained (overfit) on synthetic data so that
brain conditioning has a measurable effect — freshly initialized DiT
models predict zero velocity regardless of input.
"""

import torch

from cortexflow import (
    BrainData,
    Brain2Image,
    ROIBrainEncoder,
    build_brain2img,
    build_brain2audio,
    build_brain2text,
    Brain2Text,
)
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig


def quick_train(model, brain, targets, steps=30, lr=1e-2):
    """Quick overfit so conditioning has an effect."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for i in range(steps):
        if hasattr(model, 'training_loss'):
            loss = model.training_loss(targets, brain)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return loss.item()


# ── 1. Diverse image generation with brain_noise ────────────────────────

print("=" * 60)
print("1. Diverse Image Generation (brain_noise vs generation noise)")
print("=" * 60)

img_model = build_brain2img(
    n_voxels=64, img_size=8, hidden_dim=32, depth=1, num_heads=4,
)

brain_img = BrainData(voxels=torch.randn(2, 64))
images = torch.rand(2, 3, 8, 8)
loss = quick_train(img_model, brain_img, images, steps=20)
print(f"Training loss: {loss:.4f}")

# Single brain input for diversity comparison
single_brain = BrainData(voxels=brain_img.voxels[:1])

# With brain_noise — explores different brain interpretations
result_bn = img_model.reconstruct(single_brain, num_steps=5, num_samples=4, brain_noise=0.5)
samples_bn = result_bn.output[0]  # (4, 3, 8, 8)

# Without brain_noise — diversity only from generation noise
result_no = img_model.reconstruct(single_brain, num_steps=5, num_samples=4, brain_noise=0.0)
samples_no = result_no.output[0]

def pairwise_l2(samples):
    n = samples.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append((samples[i] - samples[j]).pow(2).mean().sqrt().item())
    return sum(dists) / len(dists)

d_bn = pairwise_l2(samples_bn)
d_no = pairwise_l2(samples_no)
print(f"Diversity WITH brain_noise:    {d_bn:.4f}")
print(f"Diversity WITHOUT brain_noise: {d_no:.4f}")
print(f"Amplification: {d_bn / max(d_no, 1e-8):.1f}x")

# ── 2. ROI-aware generation + ablation ──────────────────────────────────

print("\n" + "=" * 60)
print("2. ROI Ablation — Effect of silencing brain regions")
print("=" * 60)

roi_sizes = {"V1": 30, "FFA": 20, "A1": 14}
hidden_dim = 32
n_tokens = 4

roi_encoder = ROIBrainEncoder(
    roi_sizes=roi_sizes, cond_dim=hidden_dim, n_tokens=n_tokens, per_roi_dim=16,
)

dit_cfg = DiTConfig(hidden_dim=hidden_dim, depth=1, num_heads=4, cond_dim=hidden_dim)
vae_cfg = VAEConfig(hidden_dims=[16, 32])
flow_cfg = FlowConfig(num_steps=5)
roi_model = Brain2Image(
    img_size=8, dit_config=dit_cfg, vae_config=vae_cfg, flow_config=flow_cfg,
    n_brain_tokens=n_tokens, brain_encoder=roi_encoder,
)

# Train with ROI data
roi_voxels = {name: torch.randn(2, n) for name, n in roi_sizes.items()}
roi_brain = BrainData(voxels=torch.randn(2, sum(roi_sizes.values())), roi_voxels=roi_voxels)
images = torch.rand(2, 3, 8, 8)
loss = quick_train(roi_model, roi_brain, images, steps=20)

# Generate from single input
single_roi = {k: v[:1] for k, v in roi_voxels.items()}
brain_full = BrainData(voxels=torch.randn(1, 64), roi_voxels=single_roi)

# Ablate each region and measure effect
torch.manual_seed(0)
r_full = roi_model.reconstruct(brain_full, num_steps=5)

for ablate_name in roi_sizes:
    ablated_roi = {k: v.clone() for k, v in single_roi.items()}
    ablated_roi[ablate_name] = torch.zeros_like(ablated_roi[ablate_name])
    brain_abl = BrainData(voxels=torch.randn(1, 64), roi_voxels=ablated_roi)
    torch.manual_seed(0)  # same generation noise
    r_abl = roi_model.reconstruct(brain_abl, num_steps=5)
    d = (r_full.output - r_abl.output).pow(2).mean().sqrt().item()
    print(f"  Silence {ablate_name:>3s} ({roi_sizes[ablate_name]:>3d} voxels) → L2 shift = {d:.4f}")

# ── 3. Diverse audio generation ─────────────────────────────────────────

print("\n" + "=" * 60)
print("3. Diverse Audio Generation")
print("=" * 60)

audio_model = build_brain2audio(
    n_voxels=64, n_mels=16, audio_len=32, hidden_dim=32, depth=1,
)
brain_audio = BrainData(voxels=torch.randn(2, 64))
mel_target = torch.rand(2, 16, 32)
loss = quick_train(audio_model, brain_audio, mel_target, steps=30)
print(f"Training loss: {loss:.4f}")

single_audio = BrainData(voxels=brain_audio.voxels[:1])
result_audio = audio_model.reconstruct(
    single_audio, num_steps=5, num_samples=3, brain_noise=0.5,
)
mels = result_audio.output[0]  # (3, 16, 32)
print(f"Audio output shape: {result_audio.output.shape}")
print(f"Mean pairwise mel distance: {pairwise_l2(mels):.4f}")

# Compare without brain_noise
result_audio_no = audio_model.reconstruct(
    single_audio, num_steps=5, num_samples=3, brain_noise=0.0,
)
mels_no = result_audio_no.output[0]
print(f"Without brain_noise:          {pairwise_l2(mels_no):.4f}")

# ── 4. Diverse text generation ──────────────────────────────────────────

print("\n" + "=" * 60)
print("4. Diverse Text Generation")
print("=" * 60)

text_model = build_brain2text(
    n_voxels=64, max_len=16, hidden_dim=32, depth=1,
)
# Train on ambiguous data: same brain → different texts
# This creates genuine multi-modality in the posterior
brain_text = BrainData(voxels=torch.randn(1, 64).expand(4, -1).clone())
tokens_batch = torch.zeros(4, 16, dtype=torch.long)
for i, w in enumerate(["cat", "car", "cap", "can"]):
    t = Brain2Text.text_to_tokens(w)
    tokens_batch[i, :len(t)] = t

opt = torch.optim.Adam(text_model.parameters(), lr=3e-3)
text_model.train()
for step in range(30):
    loss = text_model.training_loss(tokens_batch, brain_text)
    opt.zero_grad()
    loss.backward()
    opt.step()
text_model.eval()
print(f"Text training loss: {loss.item():.6f}")

single_text = BrainData(voxels=brain_text.voxels[:1])
result_text = text_model.reconstruct(
    single_text, max_len=8, num_samples=5,
    temperature=1.2, brain_noise=0.5,
)

texts = result_text.metadata["texts"][0]
print(f"Generated {len(texts)} diverse texts from same brain input:")
for i, t in enumerate(texts):
    print(f"  [{i}] {repr(t)}")
print(f"Unique: {len(set(texts))}/{len(texts)}")
