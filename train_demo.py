"""Training demo: train cortexflow decoders using neuroprobe's forward model.

Uses neuroprobe's brain encoding model (stimulus → predicted BOLD) to
generate realistic (brain_activity, stimulus) pairs, then trains
cortexflow's brain→image/audio/text pipelines on this data.

This simulates the actual neuroscience workflow:
  1. Subject sees/hears a stimulus
  2. fMRI records brain activity (simulated by neuroprobe's forward model)
  3. Decoder reconstructs the stimulus from brain activity (cortexflow)

Results saved to train_outputs/ with loss curves and reconstruction PNGs.
"""

import os
import time
import json

import numpy as np
import torch
import torch.nn.functional as F

from cortexflow import (
    BrainData,
    build_brain2img,
    build_brain2audio,
    build_brain2text,
    Brain2Text,
)
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from cortexflow.brain2img import Brain2Image

# neuroprobe provides the forward encoding model (stimulus → brain)
from neuroprobe.media import (
    build_brain_model,
    synthesize_audio,
)

OUT = "train_outputs"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
N_SAMPLES = 16      # number of (brain, stimulus) pairs per modality
N_VOXELS = 128      # brain activity dimensionality
IMG_SIZE = 32       # cortexflow output image size (32x32 → recognizable shapes)
N_MELS = 16         # mel spectrogram bands
AUDIO_LEN = 16      # mel spectrogram time steps

# ═══════════════════════════════════════════════════════════════
# GENERATE DATA VIA NEUROPROBE FORWARD MODEL
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("GENERATING DATA VIA NEUROPROBE FORWARD ENCODER")
print("=" * 70)

# Build forward brain encoding models: stimulus → predicted BOLD activity
# This simulates how the brain responds to visual/auditory stimuli
vision_forward = build_brain_model(
    modality="video", feature_dim=256, n_vertices=N_VOXELS,
    hidden_dim=128, seed=42,
)
audio_forward = build_brain_model(
    modality="audio", feature_dim=256, n_vertices=N_VOXELS,
    hidden_dim=128, seed=77,
)
text_forward = build_brain_model(
    modality="text", feature_dim=256, n_vertices=N_VOXELS,
    hidden_dim=128, seed=99, vocab_size=256,  # byte-level
)

print(f"  Forward models: stimulus → {N_VOXELS}-dim brain activity")

# --- Image data: clean synthetic images with recognizable content ---
print(f"\n  Generating {N_SAMPLES} clean synthetic images ({IMG_SIZE}x{IMG_SIZE})...")


def make_synthetic_image(idx, size):
    """Generate a clean, recognizable image with shapes on solid background."""
    img = torch.zeros(3, size, size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    yy_f = yy.float()
    xx_f = xx.float()
    cy, cx = size // 2, size // 2

    if idx == 0:    # red circle on white
        img[:] = 1.0
        mask = ((yy_f - cy) ** 2 + (xx_f - cx) ** 2) < (size // 4) ** 2
        img[0][mask] = 1.0; img[1][mask] = 0.0; img[2][mask] = 0.0
    elif idx == 1:  # blue square on black
        s = size // 4
        img[:, cy - s:cy + s, cx - s:cx + s] = 0.0
        img[2, cy - s:cy + s, cx - s:cx + s] = 1.0
    elif idx == 2:  # green circle on gray
        img[:] = 0.5
        mask = ((yy_f - cy) ** 2 + (xx_f - cx) ** 2) < (size // 3) ** 2
        img[0][mask] = 0.0; img[1][mask] = 0.9; img[2][mask] = 0.0
    elif idx == 3:  # yellow horizontal bar on dark blue
        img[2] = 0.3
        h = size // 6
        img[0, cy - h:cy + h, :] = 1.0; img[1, cy - h:cy + h, :] = 1.0
        img[2, cy - h:cy + h, :] = 0.0
    elif idx == 4:  # white circle on red
        img[0] = 0.8
        mask = ((yy_f - cy) ** 2 + (xx_f - cx) ** 2) < (size // 4) ** 2
        img[:, :, :][0][mask] = 1.0; img[1][mask] = 1.0; img[2][mask] = 1.0
    elif idx == 5:  # magenta square on dark green
        img[1] = 0.3
        s = size // 3
        img[0, cy - s:cy + s, cx - s:cx + s] = 0.9
        img[2, cy - s:cy + s, cx - s:cx + s] = 0.9
    elif idx == 6:  # cyan horizontal stripe on white
        img[:] = 1.0
        h = size // 5
        img[0, cy - h:cy + h, :] = 0.0; img[1, cy - h:cy + h, :] = 1.0
        img[2, cy - h:cy + h, :] = 1.0
    elif idx == 7:  # orange vertical bar on blue
        img[2] = 0.7
        w = size // 5
        img[0, :, cx - w:cx + w] = 1.0; img[1, :, cx - w:cx + w] = 0.5
        img[2, :, cx - w:cx + w] = 0.0
    elif idx == 8:  # left red, right blue
        img[0, :, :cx] = 0.9
        img[2, :, cx:] = 0.9
    elif idx == 9:  # top green, bottom yellow
        img[1, :cy, :] = 0.8
        img[0, cy:, :] = 1.0; img[1, cy:, :] = 1.0
    elif idx == 10: # bright center (white circle on black)
        mask = ((yy_f - cy) ** 2 + (xx_f - cx) ** 2) < (size // 3) ** 2
        img[:] = 0.0; img[0][mask] = 1.0; img[1][mask] = 1.0; img[2][mask] = 1.0
    elif idx == 11: # dark center (black circle on white)
        img[:] = 1.0
        mask = ((yy_f - cy) ** 2 + (xx_f - cx) ** 2) < (size // 3) ** 2
        img[0][mask] = 0.0; img[1][mask] = 0.0; img[2][mask] = 0.0
    elif idx == 12: # red gradient left→right
        img[0] = xx_f / size
    elif idx == 13: # blue gradient top→bottom
        img[2] = yy_f / size
    elif idx == 14: # checkerboard (4x4 blocks)
        block = size // 4
        for bi in range(4):
            for bj in range(4):
                if (bi + bj) % 2 == 0:
                    img[:, bi*block:(bi+1)*block, bj*block:(bj+1)*block] = 1.0
    elif idx == 15: # diagonal split: purple top-left, green bottom-right
        for y in range(size):
            for x in range(size):
                if x + y < size:
                    img[0, y, x] = 0.6; img[2, y, x] = 0.8
                else:
                    img[1, y, x] = 0.7
    return img.clamp(0, 1)


image_brains, image_targets = [], []
for i in range(N_SAMPLES):
    clean_img = make_synthetic_image(i, IMG_SIZE)  # (3, H, W)
    # Feed to neuroprobe as single-frame video for brain encoding
    video = clean_img.unsqueeze(0)  # (1, 3, H, W)
    with torch.no_grad():
        bold = vision_forward.predict(video)    # (1, V)
        brain_vec = bold.mean(dim=0)            # (V,)
    image_brains.append(brain_vec)
    image_targets.append(clean_img)

brain_patterns_img = torch.stack(image_brains)  # (N, V)
target_images = torch.stack(image_targets)      # (N, 3, H, W)
print(f"  Brain: {brain_patterns_img.shape}, range "
      f"[{brain_patterns_img.min():.2f}, {brain_patterns_img.max():.2f}]")
print(f"  Images: {target_images.shape} — clean geometric shapes")

# --- Audio data: synthesize waveforms → mel-like target ---
print(f"\n  Generating {N_SAMPLES} audio stimuli via synthesize_audio...")
audio_brains, audio_targets = [], []
for i in range(N_SAMPLES):
    wav = synthesize_audio(duration=0.1, sample_rate=4000, seed=i * 17 + 3)
    with torch.no_grad():
        bold = audio_forward.predict(wav)       # (T, V)
        brain_vec = bold.mean(dim=0)            # (V,)
    # Build a mel-like target from the waveform via short-time FFT
    n_fft = N_MELS * 2
    hop = max(1, wav.shape[0] // AUDIO_LEN)
    padded = F.pad(wav, (0, n_fft))
    frames = padded.unfold(0, n_fft, hop)[:AUDIO_LEN]
    if frames.shape[0] < AUDIO_LEN:
        frames = F.pad(frames, (0, 0, 0, AUDIO_LEN - frames.shape[0]))
    spec = torch.fft.rfft(frames, dim=-1).abs()[:, :N_MELS]  # (T, M)
    mel = spec.T / spec.max().clamp(min=1e-6)                 # (M, T) in [0,1]
    audio_brains.append(brain_vec)
    audio_targets.append(mel)

brain_patterns_aud = torch.stack(audio_brains)
target_mels = torch.stack(audio_targets)
print(f"  Brain: {brain_patterns_aud.shape}")
print(f"  Mels: {target_mels.shape}")

# --- Text data: words → forward model → brain patterns ---
print(f"\n  Generating {N_SAMPLES} text stimuli...")
words = ["fire", "lake", "moon", "star", "wind", "rain", "tree", "bird",
         "gold", "iron", "dust", "salt", "bone", "silk", "jade", "rust"]
words = words[:N_SAMPLES]
text_brains, text_tokens_list = [], []
for i, word in enumerate(words):
    tokens = Brain2Text.text_to_tokens(word)
    token_t = torch.tensor(tokens, dtype=torch.long).clamp(max=255)
    with torch.no_grad():
        bold = text_forward.predict(token_t)    # (L, V)
        brain_vec = bold.mean(dim=0)            # (V,)
    padded = torch.zeros(8, dtype=torch.long)
    padded[:len(tokens)] = torch.tensor(tokens)
    text_brains.append(brain_vec)
    text_tokens_list.append(padded)

brain_patterns_txt = torch.stack(text_brains)
target_tokens = torch.stack(text_tokens_list)
print(f"  Brain: {brain_patterns_txt.shape}")
print(f"  Words: {words}")


def make_batch(indices, modality="image"):
    """Create a training batch from pattern indices."""
    if modality == "image":
        return BrainData(voxels=brain_patterns_img[indices]), target_images[indices]
    elif modality == "audio":
        return BrainData(voxels=brain_patterns_aud[indices]), target_mels[indices]
    elif modality == "text":
        return BrainData(voxels=brain_patterns_txt[indices]), target_tokens[indices]


def train_loop(model, modality, n_steps=2000, lr=1e-3, batch_size=8,
               cached_latents=None):
    """Training loop. cached_latents skips VAE encode (huge CPU speedup)."""
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
        idx = torch.randint(0, N_SAMPLES, (batch_size,))
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
print("\n" + "=" * 70)
print("TRAINING BRAIN → IMAGE")
print("=" * 70)

img_model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=48, depth=3, num_heads=4, cond_dim=48),
    vae_config=VAEConfig(hidden_dims=[32, 64, 128]),  # 3 downsamples → 4x4 latent
    flow_config=FlowConfig(),
)

# Pre-train VAE on target images
print("  Pre-training VAE on target images...")
vae_opt = torch.optim.Adam(img_model.vae.parameters(), lr=1e-3)
img_model.vae.train()
t0 = time.time()
for step in range(300):
    recon, mu, logvar = img_model.vae(target_images)
    loss, info = img_model.vae.loss(target_images, recon, mu, logvar)
    vae_opt.zero_grad()
    loss.backward()
    vae_opt.step()
    if step % 100 == 0:
        print(f"    VAE step {step}: recon={info['recon']:.6f} kl={info['kl']:.4f} ({time.time()-t0:.0f}s)")
img_model.vae.eval()

with torch.no_grad():
    vae_recon, _, _ = img_model.vae(target_images)
    vae_err = (target_images - vae_recon).pow(2).mean().sqrt().item()
    print(f"  VAE reconstruction error: {vae_err:.4f}")

# Cache latents for fast training
print("  Encoding target images to VAE latents...")
with torch.no_grad():
    img_latents, _, _ = img_model.vae.encode(target_images)
print(f"  Latent shape: {img_latents.shape}")

# Train flow matching
print("  Training flow matching (DiT + BrainEncoder)...")
img_losses = train_loop(img_model, "image", n_steps=2000, lr=3e-3, cached_latents=img_latents)

# Evaluate
img_model.eval()
print("\n  Reconstruction evaluation:")
img_results = {}
for i in range(N_SAMPLES):
    brain = BrainData(voxels=brain_patterns_img[i:i + 1])
    torch.manual_seed(0)
    out = img_model.reconstruct(brain, num_steps=50, cfg_scale=3.0)
    l2 = (out.output - target_images[i:i + 1]).pow(2).mean().sqrt().item()
    cos = F.cosine_similarity(
        out.output.flatten().unsqueeze(0),
        target_images[i:i + 1].flatten().unsqueeze(0),
    ).item()
    img_results[i] = {"l2": l2, "cos": cos}
    quality = "✓" if cos > 0.7 else ("~" if cos > 0.3 else "✗")
    print(f"  Sample {i:2d}: L2={l2:.3f} cos={cos:.3f} {quality}")

mean_cos_img = sum(r["cos"] for r in img_results.values()) / N_SAMPLES
good_img = sum(1 for r in img_results.values() if r["cos"] > 0.5)
print(f"\n  Mean cosine: {mean_cos_img:.3f}, Good (cos>0.5): {good_img}/{N_SAMPLES}")

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

audio_model.eval()
print("\n  Reconstruction evaluation:")
audio_results = {}
for i in range(N_SAMPLES):
    brain = BrainData(voxels=brain_patterns_aud[i:i + 1])
    torch.manual_seed(0)
    out = audio_model.reconstruct(brain, num_steps=20, cfg_scale=3.0)
    l2 = (out.output - target_mels[i:i + 1]).pow(2).mean().sqrt().item()
    cos = F.cosine_similarity(
        out.output.flatten().unsqueeze(0),
        target_mels[i:i + 1].flatten().unsqueeze(0),
    ).item()
    audio_results[i] = {"l2": l2, "cos": cos}
    quality = "✓" if cos > 0.7 else ("~" if cos > 0.3 else "✗")
    print(f"  Sample {i:2d}: L2={l2:.3f} cos={cos:.3f} {quality}")

mean_cos_aud = sum(r["cos"] for r in audio_results.values()) / N_SAMPLES
good_aud = sum(1 for r in audio_results.values() if r["cos"] > 0.5)
print(f"\n  Mean cosine: {mean_cos_aud:.3f}, Good (cos>0.5): {good_aud}/{N_SAMPLES}")

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

text_model.eval()
print("\n  Reconstruction evaluation:")
text_results = {}
for i in range(N_SAMPLES):
    brain = BrainData(voxels=brain_patterns_txt[i:i + 1])
    out = text_model.reconstruct(brain, max_len=6, temperature=0.3)
    generated = out.metadata["texts"][0]
    exact = generated.strip()[:4] == words[i][:4]
    text_results[i] = {"generated": generated, "target": words[i], "exact": exact}
    match = "✓" if exact else "✗"
    print(f"  Sample {i:2d}: brain → {repr(generated):12s} (target: {words[i]:4s}) {match}")

correct_text = sum(1 for r in text_results.values() if r["exact"])
print(f"\n  Exact match (first 4 chars): {correct_text}/{N_SAMPLES}")

# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

import json

results = {
    "data_source": "neuroprobe forward encoding model (stimulus -> simulated BOLD)",
    "n_samples": N_SAMPLES,
    "n_voxels": N_VOXELS,
    "image": {
        "final_loss": img_losses[-1],
        "min_loss": min(img_losses),
        "initial_loss": img_losses[0],
        "mean_cosine": mean_cos_img,
        "good_reconstructions": good_img,
    },
    "audio": {
        "final_loss": audio_losses[-1],
        "min_loss": min(audio_losses),
        "initial_loss": audio_losses[0],
        "mean_cosine": mean_cos_aud,
        "good_reconstructions": good_aud,
    },
    "text": {
        "final_loss": text_losses[-1],
        "min_loss": min(text_losses),
        "initial_loss": text_losses[0],
        "exact_match": correct_text,
        "results": {words[i]: text_results[i]["generated"] for i in range(N_SAMPLES)},
    },
}

with open(f"{OUT}/results.json", "w") as f:
    json.dump({k: (round(v, 6) if isinstance(v, float) else
                    {kk: round(vv, 6) if isinstance(vv, float) else vv
                     for kk, vv in v.items()} if isinstance(v, dict) else v)
                for k, v in results.items()}, f, indent=2)

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
    fig.suptitle("Training Loss Curves — neuroprobe forward model data", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/loss_curves.png")

    # Save image reconstructions grid
    n_show = min(8, N_SAMPLES)
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    for i in range(n_show):
        t = target_images[i].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, i].imshow(t, interpolation="nearest")
        axes[0, i].set_title(f"Target {i}", fontsize=8)
        axes[0, i].axis("off")
        brain = BrainData(voxels=brain_patterns_img[i:i + 1])
        torch.manual_seed(0)
        out = img_model.reconstruct(brain, num_steps=50, cfg_scale=3.0)
        r = out.output[0].detach().clamp(0, 1).permute(1, 2, 0).numpy()
        axes[1, i].imshow(r, interpolation="nearest")
        cos = img_results[i]["cos"]
        axes[1, i].set_title(f"Recon (cos={cos:.2f})", fontsize=8)
        axes[1, i].axis("off")
    fig.suptitle("Brain → Image: Target vs Reconstruction (neuroprobe data)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/image_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/image_reconstructions.png")

    # Save audio reconstructions
    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
    for i in range(n_show):
        axes[0, i].imshow(target_mels[i].numpy(), aspect="auto", origin="lower")
        axes[0, i].set_title(f"Target {i}", fontsize=8)
        axes[0, i].axis("off")
        brain = BrainData(voxels=brain_patterns_aud[i:i + 1])
        torch.manual_seed(0)
        out = audio_model.reconstruct(brain, num_steps=50, cfg_scale=3.0)
        axes[1, i].imshow(out.output[0].detach().numpy(), aspect="auto", origin="lower")
        cos = audio_results[i]["cos"]
        axes[1, i].set_title(f"Recon (cos={cos:.2f})", fontsize=8)
        axes[1, i].axis("off")
    fig.suptitle("Brain → Audio (Mel): Target vs Reconstruction (neuroprobe data)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/audio_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/audio_reconstructions.png")

    # Text results figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axis("off")
    lines = ["Brain → Text: Reconstruction Results (neuroprobe data)\n"]
    for i in range(N_SAMPLES):
        gen = text_results[i]["generated"]
        tgt = words[i]
        mark = "✓" if text_results[i]["exact"] else "✗"
        lines.append(f"  {i:2d}: brain → {repr(gen):12s}  target: {tgt:4s}  {mark}")
    lines.append(f"\n  Exact match: {correct_text}/{N_SAMPLES}")
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
print(f"  Data: neuroprobe forward encoding model → {N_SAMPLES} (brain, stimulus) pairs")
print(f"  Brain → Image:  loss {img_losses[0]:.3f} → {img_losses[-1]:.3f} "
      f"(min: {min(img_losses):.3f}), cos={mean_cos_img:.3f}, "
      f"good: {good_img}/{N_SAMPLES}")
print(f"  Brain → Audio:  loss {audio_losses[0]:.3f} → {audio_losses[-1]:.3f} "
      f"(min: {min(audio_losses):.3f}), cos={mean_cos_aud:.3f}, "
      f"good: {good_aud}/{N_SAMPLES}")
print(f"  Brain → Text:   loss {text_losses[0]:.3f} → {text_losses[-1]:.3f} "
      f"(min: {min(text_losses):.3f}), exact: {correct_text}/{N_SAMPLES}")

all_pass = (good_img >= N_SAMPLES // 2 and good_aud >= N_SAMPLES // 2
            and correct_text >= N_SAMPLES // 2)
print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
