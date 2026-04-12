"""Training demo: train brain→image/audio/text models until they converge.

This demonstrates that cortexflow models can actually learn the
brain → stimulus mapping. Uses structured synthetic data where each
brain pattern maps to a distinct stimulus.

Results are saved to train_outputs/ with loss curves and reconstructions.
"""

import os
import time

import torch
import torch.nn.functional as F

from cortexflow import (
    BrainData,
    build_brain2img,
    build_brain2audio,
    build_brain2text,
    Brain2Text,
)

OUT = "train_outputs"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════════
# STRUCTURED SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════
# 8 distinct brain patterns → 8 distinct stimuli
# Each brain pattern is a unique random vector; each stimulus is
# a structured (not random) target that the model should learn to
# reconstruct from the corresponding brain input.

N_PATTERNS = 8
N_VOXELS = 64
IMG_SIZE = 8

# Fixed brain patterns (the "subjects watching different stimuli")
brain_patterns = torch.randn(N_PATTERNS, N_VOXELS)
# Normalize to unit norm for stability
brain_patterns = F.normalize(brain_patterns, dim=1) * 3.0

# Fixed target images: each is a distinct color/pattern
target_images = torch.zeros(N_PATTERNS, 3, IMG_SIZE, IMG_SIZE)
colors = [
    (1.0, 0.0, 0.0),  # red
    (0.0, 1.0, 0.0),  # green
    (0.0, 0.0, 1.0),  # blue
    (1.0, 1.0, 0.0),  # yellow
    (1.0, 0.0, 1.0),  # magenta
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 1.0, 1.0),  # white
    (0.0, 0.0, 0.0),  # black
]
for i, (r, g, b) in enumerate(colors):
    target_images[i, 0] = r
    target_images[i, 1] = g
    target_images[i, 2] = b
    # Add spatial pattern: darken quadrant based on index
    qi = i % 4
    row = (qi // 2) * (IMG_SIZE // 2)
    col = (qi % 2) * (IMG_SIZE // 2)
    target_images[i, :, row:row + IMG_SIZE // 2, col:col + IMG_SIZE // 2] *= 0.4

# Fixed target audio: each is a distinct frequency sine on a mel-like grid
N_MELS = 16
AUDIO_LEN = 16
target_mels = torch.zeros(N_PATTERNS, N_MELS, AUDIO_LEN)
for i in range(N_PATTERNS):
    # Each pattern activates a different mel band
    band = (i * N_MELS) // N_PATTERNS
    width = max(1, N_MELS // N_PATTERNS)
    target_mels[i, band:band + width, :] = 1.0
    # Add temporal pattern: energy in a specific time window
    t_start = (i * AUDIO_LEN) // N_PATTERNS
    t_width = max(1, AUDIO_LEN // N_PATTERNS)
    target_mels[i, :, t_start:t_start + t_width] += 0.5

# Fixed target texts: each is a distinct 4-letter word
words = ["fire", "lake", "moon", "star", "wind", "rain", "tree", "bird"]
target_tokens = torch.zeros(N_PATTERNS, 8, dtype=torch.long)
for i, w in enumerate(words):
    t = Brain2Text.text_to_tokens(w)
    target_tokens[i, :len(t)] = t


def make_batch(indices, modality="image"):
    """Create a training batch from pattern indices."""
    brains = BrainData(voxels=brain_patterns[indices])
    if modality == "image":
        return brains, target_images[indices]
    elif modality == "audio":
        return brains, target_mels[indices]
    elif modality == "text":
        return brains, target_tokens[indices]


def train_loop(model, modality, n_steps=2000, lr=1e-3, batch_size=8,
               cached_latents=None):
    """Generic training loop with progress reporting.
    
    For image/audio modalities using flow matching, pass cached_latents
    to avoid re-encoding targets every step (huge CPU speedup).
    """
    # When using cached latents, only train DiT + brain encoder (not VAE)
    if cached_latents is not None:
        params = [p for n, p in model.named_parameters()
                  if not n.startswith("vae.")]
    else:
        params = model.parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=lr * 0.01)
    model.train()
    losses = []
    t0 = time.time()

    for step in range(n_steps):
        # Sample random patterns
        idx = torch.randint(0, N_PATTERNS, (batch_size,))
        brain, target = make_batch(idx, modality)

        if cached_latents is not None and modality in ("image", "audio"):
            # Use pre-encoded latents → skip VAE, go straight to flow matching
            z = cached_latents[idx]
            bg, bt = model.encode_brain(brain)
            loss = model.flow_matcher.compute_loss(model.dit, z, bg, bt)
        elif modality == "image":
            loss = model.training_loss(target, brain, cfg_dropout=0.0)
        elif modality == "audio":
            loss = model.training_loss(target, brain)
        elif modality == "text":
            loss = model.training_loss(target, brain)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        losses.append(loss.item())

        if step % 200 == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            avg = sum(losses[-50:]) / min(50, len(losses))
            print(f"  Step {step:5d}/{n_steps}: loss={loss.item():.4f} avg50={avg:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.0f}s)")

    return losses


# ═══════════════════════════════════════════════════════════════
# TRAIN BRAIN → IMAGE
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("TRAINING BRAIN → IMAGE")
print("=" * 70)

img_model = build_brain2img(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    hidden_dim=32, depth=2, num_heads=4,
)

# Step 1: Pre-train the VAE to encode/decode our target images faithfully
print("  Pre-training VAE on target images...")
vae_opt = torch.optim.Adam(img_model.vae.parameters(), lr=1e-3)
img_model.vae.train()
t0 = time.time()
for step in range(300):
    # Always train on all 8 target images (small enough to fit in one batch)
    recon, mu, logvar = img_model.vae(target_images)
    loss, info = img_model.vae.loss(target_images, recon, mu, logvar)
    vae_opt.zero_grad()
    loss.backward()
    vae_opt.step()
    if step % 100 == 0:
        print(f"    VAE step {step}: recon={info['recon']:.6f} kl={info['kl']:.4f} ({time.time()-t0:.0f}s)")
img_model.vae.eval()

# Verify VAE reconstruction quality
with torch.no_grad():
    vae_recon, _, _ = img_model.vae(target_images)
    vae_err = (target_images - vae_recon).pow(2).mean().sqrt().item()
    print(f"  VAE reconstruction error: {vae_err:.4f}")

# Step 2: Pre-encode to latents for fast flow matching training
print("  Encoding target images to VAE latents...")
with torch.no_grad():
    img_latents, _, _ = img_model.vae.encode(target_images)
print(f"  Latent shape: {img_latents.shape}")

# Step 3: Train the flow matching (DiT + BrainEncoder) with cached latents
print("  Training flow matching (DiT + BrainEncoder)...")
img_losses = train_loop(img_model, "image", n_steps=2000, lr=3e-3, cached_latents=img_latents)

# Evaluate: reconstruct each pattern
img_model.eval()
print("\n  Reconstruction evaluation:")
img_results = {}
for i in range(N_PATTERNS):
    brain = BrainData(voxels=brain_patterns[i:i + 1])
    torch.manual_seed(0)
    out = img_model.reconstruct(brain, num_steps=20, cfg_scale=3.0)
    # Compare dominant color channel
    out_rgb = out.output[0].mean(dim=(1, 2))  # (3,)
    tgt_rgb = target_images[i].mean(dim=(1, 2))
    l2 = (out.output - target_images[i:i + 1]).pow(2).mean().sqrt().item()
    cos = F.cosine_similarity(
        out.output.flatten().unsqueeze(0),
        target_images[i:i + 1].flatten().unsqueeze(0),
    ).item()
    ch_names = ["R", "G", "B"]
    dom_out = ch_names[out_rgb.argmax().item()]
    dom_tgt = ch_names[tgt_rgb.argmax().item()]
    # For black (all zeros), mark as special
    if tgt_rgb.max() < 0.01:
        dom_tgt = "K"  # black
        dom_out = "K" if out_rgb.max() < 0.2 else dom_out
    img_results[i] = {"l2": l2, "cos": cos, "dom_out": dom_out, "dom_tgt": dom_tgt}
    match = "✓" if dom_out == dom_tgt else "✗"
    color = colors[i]
    print(f"  Pattern {i} ({words[i]:4s}, color={color}): "
          f"L2={l2:.3f} cos={cos:.3f} dom={dom_out}(expect {dom_tgt}) {match}")

correct = sum(1 for r in img_results.values() if r["dom_out"] == r["dom_tgt"])
mean_cos = sum(r["cos"] for r in img_results.values()) / N_PATTERNS
print(f"\n  Dominant channel correct: {correct}/{N_PATTERNS}")
print(f"  Mean cosine similarity: {mean_cos:.3f}")

# ═══════════════════════════════════════════════════════════════
# TRAIN BRAIN → AUDIO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING BRAIN → AUDIO")
print("=" * 70)

audio_model = build_brain2audio(
    n_voxels=N_VOXELS, n_mels=N_MELS, audio_len=AUDIO_LEN,
    hidden_dim=32, depth=2,
)
audio_losses = train_loop(audio_model, "audio", n_steps=2000, lr=3e-3)

# Evaluate
audio_model.eval()
print("\n  Reconstruction evaluation:")
audio_results = {}
for i in range(N_PATTERNS):
    brain = BrainData(voxels=brain_patterns[i:i + 1])
    torch.manual_seed(0)
    out = audio_model.reconstruct(brain, num_steps=20, cfg_scale=3.0)
    # Check which mel band is most active
    mel_energy = out.output[0].mean(dim=-1)  # (n_mels,)
    peak_band = mel_energy.argmax().item()
    expected_band = (i * N_MELS) // N_PATTERNS
    l2 = (out.output - target_mels[i:i + 1]).pow(2).mean().sqrt().item()
    cos = F.cosine_similarity(
        out.output.flatten().unsqueeze(0),
        target_mels[i:i + 1].flatten().unsqueeze(0),
    ).item()
    near = abs(peak_band - expected_band) <= 4
    audio_results[i] = {"l2": l2, "cos": cos, "peak": peak_band, "expected": expected_band}
    match = "✓" if near else "✗"
    print(f"  Pattern {i} ({words[i]:4s}): L2={l2:.3f} cos={cos:.3f} "
          f"peak_band={peak_band} (expect ~{expected_band}) {match}")

correct_audio = sum(1 for r in audio_results.values()
                    if abs(r["peak"] - r["expected"]) <= 4)
mean_cos_audio = sum(r["cos"] for r in audio_results.values()) / N_PATTERNS
print(f"\n  Peak band correct (±4): {correct_audio}/{N_PATTERNS}")
print(f"  Mean cosine similarity: {mean_cos_audio:.3f}")

# ═══════════════════════════════════════════════════════════════
# TRAIN BRAIN → TEXT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING BRAIN → TEXT")
print("=" * 70)

text_model = build_brain2text(
    n_voxels=N_VOXELS, max_len=8, hidden_dim=32, depth=2,
)
text_losses = train_loop(text_model, "text", n_steps=500, lr=3e-3)

# Evaluate
text_model.eval()
print("\n  Reconstruction evaluation:")
text_results = {}
for i in range(N_PATTERNS):
    brain = BrainData(voxels=brain_patterns[i:i + 1])
    out = text_model.reconstruct(brain, max_len=6, temperature=0.3)
    generated = out.metadata["texts"][0]
    exact = generated.strip()[:4] == words[i]
    prefix = generated.strip()[:2] == words[i][:2]
    text_results[i] = {"generated": generated, "target": words[i], "exact": exact}
    match = "✓" if exact else ("~" if prefix else "✗")
    print(f"  Pattern {i}: brain → {repr(generated):12s} (target: {words[i]:4s}) {match}")

correct_text = sum(1 for r in text_results.values() if r["exact"])
print(f"\n  Exact match (first 4 chars): {correct_text}/{N_PATTERNS}")

# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

import json

results = {
    "image": {
        "final_loss": img_losses[-1],
        "min_loss": min(img_losses),
        "initial_loss": img_losses[0],
        "dominant_channel_correct": correct,
        "mean_cosine": mean_cos,
    },
    "audio": {
        "final_loss": audio_losses[-1],
        "min_loss": min(audio_losses),
        "initial_loss": audio_losses[0],
        "peak_band_correct": correct_audio,
        "mean_cosine": mean_cos_audio,
    },
    "text": {
        "final_loss": text_losses[-1],
        "min_loss": min(text_losses),
        "initial_loss": text_losses[0],
        "exact_match": correct_text,
        "results": {words[i]: text_results[i]["generated"] for i in range(N_PATTERNS)},
    },
}

with open(f"{OUT}/results.json", "w") as f:
    json.dump({k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                    for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

# Save loss curves as PNG
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, losses, title in [
        (axes[0], img_losses, "Brain → Image"),
        (axes[1], audio_losses, "Brain → Audio"),
        (axes[2], text_losses, "Brain → Text"),
    ]:
        # Smooth with moving average
        window = 50
        smoothed = [sum(losses[max(0, i - window):i + 1]) / min(i + 1, window)
                    for i in range(len(losses))]
        ax.plot(smoothed, linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training Loss Curves (smoothed)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/loss_curves.png")

    # Save image reconstructions grid
    fig, axes = plt.subplots(2, N_PATTERNS, figsize=(2.5 * N_PATTERNS, 5))
    for i in range(N_PATTERNS):
        # Top row: target
        t = target_images[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(t)
        axes[0, i].set_title(f"Target ({words[i]})", fontsize=8)
        axes[0, i].axis("off")
        # Bottom row: reconstruction
        brain = BrainData(voxels=brain_patterns[i:i + 1])
        torch.manual_seed(0)
        out = img_model.reconstruct(brain, num_steps=20, cfg_scale=3.0)
        r = out.output[0].detach().clamp(0, 1).permute(1, 2, 0).numpy()
        axes[1, i].imshow(r)
        cos = img_results[i]["cos"]
        axes[1, i].set_title(f"Recon (cos={cos:.2f})", fontsize=8)
        axes[1, i].axis("off")
    fig.suptitle("Brain → Image: Target vs Reconstruction", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/image_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/image_reconstructions.png")

    # Save audio reconstructions
    fig, axes = plt.subplots(2, N_PATTERNS, figsize=(2.5 * N_PATTERNS, 5))
    for i in range(N_PATTERNS):
        axes[0, i].imshow(target_mels[i].numpy(), aspect="auto", origin="lower")
        axes[0, i].set_title(f"Target ({words[i]})", fontsize=8)
        axes[0, i].axis("off")
        brain = BrainData(voxels=brain_patterns[i:i + 1])
        torch.manual_seed(0)
        out = audio_model.reconstruct(brain, num_steps=20, cfg_scale=3.0)
        axes[1, i].imshow(out.output[0].detach().numpy(), aspect="auto", origin="lower")
        cos = audio_results[i]["cos"]
        axes[1, i].set_title(f"Recon (cos={cos:.2f})", fontsize=8)
        axes[1, i].axis("off")
    fig.suptitle("Brain → Audio (Mel): Target vs Reconstruction", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/audio_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/audio_reconstructions.png")

    # Text results figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    lines = ["Brain → Text: Reconstruction Results\n"]
    for i in range(N_PATTERNS):
        gen = text_results[i]["generated"]
        tgt = words[i]
        mark = "✓" if text_results[i]["exact"] else "✗"
        lines.append(f"  Pattern {i}: brain → {repr(gen):12s}  target: {tgt:4s}  {mark}")
    lines.append(f"\n  Exact match: {correct_text}/{N_PATTERNS}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace")
    fig.savefig(f"{OUT}/text_reconstructions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT}/text_reconstructions.png")

except ImportError:
    print("  matplotlib not available, skipping visualization")

# Save model checkpoints
torch.save(img_model.state_dict(), f"{OUT}/brain2img.pt")
torch.save(audio_model.state_dict(), f"{OUT}/brain2audio.pt")
torch.save(text_model.state_dict(), f"{OUT}/brain2text.pt")
print(f"  Saved model checkpoints to {OUT}/")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"  Brain → Image:  loss {img_losses[0]:.3f} → {img_losses[-1]:.3f} "
      f"(min: {min(img_losses):.3f}), {correct}/{N_PATTERNS} dominant channel correct, "
      f"cos={mean_cos:.3f}")
print(f"  Brain → Audio:  loss {audio_losses[0]:.3f} → {audio_losses[-1]:.3f} "
      f"(min: {min(audio_losses):.3f}), {correct_audio}/{N_PATTERNS} peak band correct, "
      f"cos={mean_cos_audio:.3f}")
print(f"  Brain → Text:   loss {text_losses[0]:.3f} → {text_losses[-1]:.3f} "
      f"(min: {min(text_losses):.3f}), {correct_text}/{N_PATTERNS} exact match")

all_pass = correct >= N_PATTERNS // 2 and correct_audio >= N_PATTERNS // 2 and correct_text >= N_PATTERNS // 2
print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'} "
      f"(need ≥{N_PATTERNS // 2}/{N_PATTERNS} correct per modality)")
