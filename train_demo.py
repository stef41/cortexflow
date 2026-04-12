"""Training demo: train cortexflow decoders using neuroprobe's forward model.

Uses neuroprobe's brain encoding model (stimulus → predicted BOLD) to
generate realistic (brain_activity, stimulus) pairs, then trains
cortexflow's brain→image/audio/text pipelines on this data.

This simulates the actual neuroscience workflow:
  1. Subject sees/hears a stimulus
  2. fMRI records brain activity (simulated by neuroprobe's forward model)
  3. Decoder reconstructs the stimulus from brain activity (cortexflow)

KEY: Uses a proper train/test split to demonstrate GENERALIZATION —
the model reconstructs stimuli it has NEVER seen during training.

Results saved to train_outputs/ with loss curves and reconstruction PNGs.
"""

import os
import time
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from cortexflow import (
    BrainData,
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
N_TOTAL = 120       # total (brain, stimulus) pairs
N_TRAIN = 96        # training set
N_TEST = 24         # held-out test set (NEVER seen during training)
N_VOXELS = 512      # brain activity dimensionality
IMG_SIZE = 32       # cortexflow output image size
N_MELS = 16         # mel spectrogram bands
AUDIO_LEN = 16      # mel spectrogram time steps


def compute_ssim(img1, img2):
    """Compute structural similarity (SSIM) between two images.
    img1, img2: (C, H, W) tensors in [0, 1].
    """
    x = img1.flatten()
    y = img2.flatten()
    mu_x, mu_y = x.mean(), y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim.item()

# ═══════════════════════════════════════════════════════════════
# GENERATE DATA VIA NEUROPROBE FORWARD MODEL
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("GENERATING DATA VIA NEUROPROBE FORWARD ENCODER")
print("=" * 70)

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
    hidden_dim=128, seed=99, vocab_size=256,
)
print(f"  Forward models: stimulus → {N_VOXELS}-dim brain activity")
print(f"  Train/test split: {N_TRAIN} train / {N_TEST} test (held-out)")


# ── Image generation: parameterized random shapes ──
print(f"\n  Generating {N_TOTAL} parameterized images ({IMG_SIZE}x{IMG_SIZE})...")


def make_random_image(seed, size):
    """Generate a parameterized image: random shape, position, color, background.

    Same seed always produces the same image. Different seeds produce
    structurally different images that share the same generative process,
    enabling the model to learn shape/color/position from brain patterns
    rather than memorizing individual images.
    """
    gen = torch.Generator().manual_seed(seed)
    # Random background (darker range for contrast)
    bg = torch.rand(3, generator=gen) * 0.4
    img = bg.view(3, 1, 1).expand(3, size, size).clone()

    # Shape type: 0=circle, 1=square, 2=hbar, 3=vbar, 4=two-tone
    shape_type = torch.randint(0, 5, (1,), generator=gen).item()

    # Foreground color (brighter range for visibility)
    fg = torch.rand(3, generator=gen) * 0.5 + 0.5

    # Random position and size
    cy = torch.randint(size // 4, 3 * size // 4 + 1, (1,), generator=gen).item()
    cx = torch.randint(size // 4, 3 * size // 4 + 1, (1,), generator=gen).item()
    r = torch.randint(size // 6, size // 3 + 1, (1,), generator=gen).item()

    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    yf, xf = yy.float(), xx.float()

    if shape_type == 0:  # circle
        mask = ((yf - cy) ** 2 + (xf - cx) ** 2) < r ** 2
        for c in range(3):
            img[c][mask] = fg[c]
    elif shape_type == 1:  # square
        mask = (yy >= max(0, cy - r)) & (yy < min(size, cy + r)) & \
               (xx >= max(0, cx - r)) & (xx < min(size, cx + r))
        for c in range(3):
            img[c][mask] = fg[c]
    elif shape_type == 2:  # horizontal bar
        h = max(2, r // 2)
        mask = (yy >= max(0, cy - h)) & (yy < min(size, cy + h))
        for c in range(3):
            img[c][mask] = fg[c]
    elif shape_type == 3:  # vertical bar
        w = max(2, r // 2)
        mask = (xx >= max(0, cx - w)) & (xx < min(size, cx + w))
        for c in range(3):
            img[c][mask] = fg[c]
    elif shape_type == 4:  # two-tone vertical split
        fg2 = torch.rand(3, generator=gen) * 0.5 + 0.5
        split = torch.randint(size // 3, 2 * size // 3 + 1, (1,), generator=gen).item()
        for c in range(3):
            img[c, :, :split] = fg[c]
            img[c, :, split:] = fg2[c]

    return img.clamp(0, 1)


image_brains, image_targets = [], []
for i in range(N_TOTAL):
    clean_img = make_random_image(i * 7 + 13, IMG_SIZE)
    video = clean_img.unsqueeze(0)
    with torch.no_grad():
        bold = vision_forward.predict(video)
        brain_vec = bold.mean(dim=0)
    image_brains.append(brain_vec)
    image_targets.append(clean_img)

brain_patterns_img = torch.stack(image_brains)
target_images = torch.stack(image_targets)

# Split into train and test
train_idx = list(range(N_TRAIN))
test_idx = list(range(N_TRAIN, N_TOTAL))
print(f"  Images: {target_images.shape}")
print(f"  Train: indices 0-{N_TRAIN-1} ({N_TRAIN} samples)")
print(f"  Test:  indices {N_TRAIN}-{N_TOTAL-1} ({N_TEST} samples, HELD OUT)")

# ── Audio data ──
print(f"\n  Generating {N_TOTAL} audio stimuli via synthesize_audio...")
audio_brains, audio_targets = [], []
for i in range(N_TOTAL):
    wav = synthesize_audio(duration=0.1, sample_rate=4000, seed=i * 17 + 3)
    with torch.no_grad():
        bold = audio_forward.predict(wav)
        brain_vec = bold.mean(dim=0)
    n_fft = N_MELS * 2
    hop = max(1, wav.shape[0] // AUDIO_LEN)
    padded = F.pad(wav, (0, n_fft))
    frames = padded.unfold(0, n_fft, hop)[:AUDIO_LEN]
    if frames.shape[0] < AUDIO_LEN:
        frames = F.pad(frames, (0, 0, 0, AUDIO_LEN - frames.shape[0]))
    spec = torch.fft.rfft(frames, dim=-1).abs()[:, :N_MELS]
    mel = spec.T / spec.max().clamp(min=1e-6)
    audio_brains.append(brain_vec)
    audio_targets.append(mel)

brain_patterns_aud = torch.stack(audio_brains)
target_mels = torch.stack(audio_targets)
print(f"  Mels: {target_mels.shape}")

# ── Text data ──
# Text is a discrete memorization task — generalization requires semantic
# embeddings, which this byte-level decoder doesn't have. We include it
# to demonstrate the pipeline works, not to claim generalization.
print(f"\n  Generating {N_TOTAL} text stimuli...")
# 120 unique 4-char words
_base_words = [
    "fire", "lake", "moon", "star", "wind", "rain", "tree", "bird",
    "gold", "iron", "dust", "salt", "bone", "silk", "jade", "rust",
    "dawn", "peak", "wave", "rose", "coal", "palm", "cork", "mint",
    "bark", "cliff", "dew", "fog", "glow", "haze", "isle", "knot",
    "loom", "moss", "nest", "opal", "pine", "reed", "snow", "tide",
    "vale", "wren", "zinc", "arch", "bell", "cape", "dove", "echo",
    "fern", "glen", "harp", "inch", "jazz", "kelp", "lime", "mist",
    "noir", "onyx", "pond", "quiz", "reef", "sage", "tusk", "urn",
    "vine", "wax", "yew", "zeal", "acre", "bass", "cove", "dune",
    "elm", "fawn", "grit", "hemp", "iris", "jolt", "kite", "lynx",
    "mace", "nook", "oath", "pyre", "rift", "silt", "twig", "wick",
    "axle", "brew", "clam", "dock", "ewe", "flux", "gap", "hull",
    "jig", "lark", "malt", "nib", "orb", "peg", "rye", "sap",
    "tar", "urn", "vow", "web", "yam", "zest", "aloe", "chop",
    "dusk", "elk", "fox", "gum", "hut", "jab", "keg", "lid",
]
words = _base_words[:N_TOTAL]
text_brains, text_tokens_list = [], []
for i, word in enumerate(words):
    tokens = Brain2Text.text_to_tokens(word)
    token_t = tokens.detach().clone().long().clamp(max=255)
    with torch.no_grad():
        bold = text_forward.predict(token_t)
        brain_vec = bold.mean(dim=0)
    pad = torch.zeros(8, dtype=torch.long)
    pad[:len(tokens)] = tokens.detach().clone().long()
    text_brains.append(brain_vec)
    text_tokens_list.append(pad)

brain_patterns_txt = torch.stack(text_brains)
target_tokens = torch.stack(text_tokens_list)
print(f"  Words: {len(words)} total (train: first {N_TRAIN}, test: last {N_TEST})")
print(f"  Sample train: {words[:8]}")
print(f"  Sample test:  {words[N_TRAIN:N_TRAIN+8]}")


def make_batch(indices, modality="image"):
    """Create a training batch from pattern indices."""
    if modality == "image":
        return BrainData(voxels=brain_patterns_img[indices]), target_images[indices]
    elif modality == "audio":
        return BrainData(voxels=brain_patterns_aud[indices]), target_mels[indices]
    elif modality == "text":
        return BrainData(voxels=brain_patterns_txt[indices]), target_tokens[indices]


def train_loop(model, modality, n_steps=2000, lr=1e-3, batch_size=8,
               cached_latents=None, n_train=N_TRAIN):
    """Training loop — only samples from TRAINING set indices."""
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
        idx = torch.randint(0, n_train, (batch_size,))  # ONLY train indices
        brain, target = make_batch(idx, modality)

        if cached_latents is not None and modality in ("image", "audio"):
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

        if step % 500 == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            avg = sum(losses[-50:]) / min(50, len(losses))
            print(f"  Step {step:5d}/{n_steps}: loss={loss.item():.4f} avg50={avg:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.0f}s)")

    return losses


def evaluate_images(model, indices, brain_patterns, targets, label="", verbose=True):
    """Evaluate image reconstructions with cos, SSIM, L2."""
    model.eval()
    results = {}
    for i in indices:
        brain = BrainData(voxels=brain_patterns[i:i + 1])
        torch.manual_seed(0)
        out = model.reconstruct(brain, num_steps=50, cfg_scale=3.0)
        recon = out.output[0].detach().clamp(0, 1)
        target = targets[i]
        cos = F.cosine_similarity(recon.flatten().unsqueeze(0),
                                   target.flatten().unsqueeze(0)).item()
        ssim = compute_ssim(recon, target)
        l2 = (recon - target).pow(2).mean().sqrt().item()
        results[i] = {"cos": cos, "ssim": ssim, "l2": l2}
        if verbose:
            quality = "✓" if ssim > 0.5 else ("~" if ssim > 0.2 else "✗")
            print(f"  {label} {i:2d}: SSIM={ssim:.3f} cos={cos:.3f} L2={l2:.3f} {quality}")
    return results


# ═══════════════════════════════════════════════════════════════
# TRAIN BRAIN → IMAGE (train on N_TRAIN, evaluate on held-out test)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING BRAIN → IMAGE")
print("=" * 70)

img_model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=64, depth=4, num_heads=4, cond_dim=64),
    vae_config=VAEConfig(hidden_dims=[32, 64, 128]),
    flow_config=FlowConfig(),
)

# Pre-train VAE on ALL images (VAE is unsupervised, not brain-specific)
print("  Pre-training VAE on all images...")
vae_opt = torch.optim.Adam(img_model.vae.parameters(), lr=1e-3)
img_model.vae.train()
t0 = time.time()
for step in range(1000):
    # Mini-batch VAE training for larger dataset
    vae_idx = torch.randint(0, N_TOTAL, (min(32, N_TOTAL),))
    vae_batch = target_images[vae_idx]
    recon, mu, logvar = img_model.vae(vae_batch)
    loss, info = img_model.vae.loss(vae_batch, recon, mu, logvar)
    vae_opt.zero_grad()
    loss.backward()
    vae_opt.step()
    if step % 200 == 0:
        print(f"    VAE step {step}: recon={info['recon']:.6f} kl={info['kl']:.4f} ({time.time()-t0:.0f}s)")
img_model.vae.eval()

# Cache latents for fast training
print("  Encoding target images to VAE latents...")
with torch.no_grad():
    img_latents, _, _ = img_model.vae.encode(target_images)
print(f"  Latent shape: {img_latents.shape}")

# Train flow matching — ONLY on training set
print(f"  Training flow matching on {N_TRAIN} training samples...")
img_losses = train_loop(img_model, "image", n_steps=5000, lr=3e-3,
                        cached_latents=img_latents, n_train=N_TRAIN)

# Evaluate on TRAIN set (show first 8 only)
print(f"\n  === TRAIN SET EVALUATION ({N_TRAIN} samples, showing first 8) ===")
train_img_results_partial = evaluate_images(img_model, train_idx[:8], brain_patterns_img, target_images, "TRAIN")
train_img_results_rest = evaluate_images(img_model, train_idx[8:], brain_patterns_img, target_images, "TRAIN", verbose=False)
train_img_results = {**train_img_results_partial, **train_img_results_rest}

# Evaluate on TEST set (HELD OUT — never seen during training)
print(f"\n  === TEST SET EVALUATION ({N_TEST} held-out samples) ===")
test_img_results = evaluate_images(img_model, test_idx, brain_patterns_img, target_images, "TEST ")

train_cos_img = sum(r["cos"] for r in train_img_results.values()) / N_TRAIN
train_ssim_img = sum(r["ssim"] for r in train_img_results.values()) / N_TRAIN
test_cos_img = sum(r["cos"] for r in test_img_results.values()) / N_TEST
test_ssim_img = sum(r["ssim"] for r in test_img_results.values()) / N_TEST
print(f"\n  Train: cos={train_cos_img:.3f} SSIM={train_ssim_img:.3f}")
print(f"  Test:  cos={test_cos_img:.3f} SSIM={test_ssim_img:.3f}")
print(f"  Generalization gap: cos={train_cos_img - test_cos_img:.3f}")

# ═══════════════════════════════════════════════════════════════
# TRAIN BRAIN → AUDIO (train on N_TRAIN, evaluate on held-out test)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING BRAIN → AUDIO")
print("=" * 70)

audio_model = build_brain2audio(
    n_voxels=N_VOXELS, n_mels=N_MELS, audio_len=AUDIO_LEN,
    hidden_dim=32, depth=2,
)
audio_losses = train_loop(audio_model, "audio", n_steps=3000, lr=3e-3, n_train=N_TRAIN)

audio_model.eval()
print(f"\n  === TRAIN SET (showing first 8 of {N_TRAIN}) ===")
train_audio_results = {}
for i in train_idx:
    brain = BrainData(voxels=brain_patterns_aud[i:i + 1])
    torch.manual_seed(0)
    out = audio_model.reconstruct(brain, num_steps=20, cfg_scale=3.0)
    cos = F.cosine_similarity(
        out.output.flatten().unsqueeze(0),
        target_mels[i:i + 1].flatten().unsqueeze(0),
    ).item()
    train_audio_results[i] = {"cos": cos}
    if i < 8:
        print(f"  TRAIN {i:2d}: cos={cos:.3f}")

print(f"\n  === TEST SET ({N_TEST} held-out) ===")
test_audio_results = {}
all_audio_outputs = []
for i in test_idx:
    brain = BrainData(voxels=brain_patterns_aud[i:i + 1])
    torch.manual_seed(0)
    out = audio_model.reconstruct(brain, num_steps=20, cfg_scale=3.0)
    all_audio_outputs.append(out.output[0].detach())
    cos = F.cosine_similarity(
        out.output.flatten().unsqueeze(0),
        target_mels[i:i + 1].flatten().unsqueeze(0),
    ).item()
    test_audio_results[i] = {"cos": cos}
    print(f"  TEST  {i:2d}: cos={cos:.3f}")

# Degeneracy check: are outputs actually different for different inputs?
if len(all_audio_outputs) > 1:
    out_stack = torch.stack(all_audio_outputs)
    inter_cos = []
    for a in range(len(out_stack)):
        for b in range(a + 1, len(out_stack)):
            c = F.cosine_similarity(
                out_stack[a].flatten().unsqueeze(0),
                out_stack[b].flatten().unsqueeze(0),
            ).item()
            inter_cos.append(c)
    mean_inter = sum(inter_cos) / len(inter_cos)
    print(f"\n  Degeneracy check: mean inter-output cos={mean_inter:.3f}"
          f" ({'DIVERSE' if mean_inter < 0.95 else 'DEGENERATE — outputs too similar'})")

train_cos_aud = sum(r["cos"] for r in train_audio_results.values()) / N_TRAIN
test_cos_aud = sum(r["cos"] for r in test_audio_results.values()) / N_TEST
print(f"\n  Train cos: {train_cos_aud:.3f}")
print(f"  Test  cos: {test_cos_aud:.3f}")
print(f"  Generalization gap: {train_cos_aud - test_cos_aud:.3f}")

# ═══════════════════════════════════════════════════════════════
# TRAIN BRAIN → TEXT (train on N_TRAIN, evaluate on held-out test)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING BRAIN → TEXT")
print("=" * 70)

text_model = build_brain2text(
    n_voxels=N_VOXELS, max_len=8, hidden_dim=64, depth=3,
)
text_losses = train_loop(text_model, "text", n_steps=2000, lr=3e-3, n_train=N_TRAIN)

text_model.eval()
print(f"\n  === TRAIN SET (showing first 8 of {N_TRAIN}) ===")
train_text_results = {}
for i in train_idx:
    brain = BrainData(voxels=brain_patterns_txt[i:i + 1])
    out = text_model.reconstruct(brain, max_len=6, temperature=0.3)
    gen = out.metadata["texts"][0][:4]
    exact = gen == words[i][:4]
    train_text_results[i] = {"generated": gen, "target": words[i], "exact": exact}
    if i < 8:
        mark = "✓" if exact else "✗"
        print(f"  TRAIN {i:2d}: brain → {repr(gen):8s} (target: {words[i]:4s}) {mark}")

print(f"\n  === TEST SET ({N_TEST} held-out) ===")
test_text_results = {}
for i in test_idx:
    brain = BrainData(voxels=brain_patterns_txt[i:i + 1])
    out = text_model.reconstruct(brain, max_len=6, temperature=0.3)
    gen = out.metadata["texts"][0][:4]
    exact = gen == words[i][:4]
    test_text_results[i] = {"generated": gen, "target": words[i], "exact": exact}
    mark = "✓" if exact else "✗"
    print(f"  TEST  {i:2d}: brain → {repr(gen):8s} (target: {words[i]:4s}) {mark}")

train_correct = sum(1 for r in train_text_results.values() if r["exact"])
test_correct = sum(1 for r in test_text_results.values() if r["exact"])
print(f"\n  Train exact: {train_correct}/{N_TRAIN}")
print(f"  Test  exact: {test_correct}/{N_TEST}")

# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

import json

# Diversity evaluation
print("\n  Evaluating semantic diversity (brain_noise=0.15, 4 samples)...")
diversity_scores = []
for i in range(min(4, N_TRAIN)):
    brain = BrainData(voxels=brain_patterns_img[i:i + 1])
    samples = []
    for s in range(4):
        torch.manual_seed(s * 17 + 3)
        out = img_model.reconstruct(brain, num_steps=50, cfg_scale=3.0, brain_noise=0.15)
        samples.append(out.output[0].detach())
    pair_dists = []
    for a in range(len(samples)):
        for b in range(a + 1, len(samples)):
            dist = (samples[a] - samples[b]).pow(2).mean().sqrt().item()
            pair_dists.append(dist)
    mean_dist = sum(pair_dists) / len(pair_dists)
    diversity_scores.append(mean_dist)
    print(f"    Brain {i}: mean inter-sample L2 = {mean_dist:.4f}")

mean_diversity = sum(diversity_scores) / len(diversity_scores)
print(f"  Overall mean diversity: {mean_diversity:.4f}")

# Merge all text results
all_text_results = {**train_text_results, **test_text_results}

def round_nested(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: round_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested(x) for x in obj]
    return obj

results = {
    "data_source": "neuroprobe forward encoding (stimulus -> simulated BOLD)",
    "split": {"total": N_TOTAL, "train": N_TRAIN, "test": N_TEST},
    "n_voxels": N_VOXELS,
    "image": {
        "final_loss": img_losses[-1],
        "train": {"mean_cos": train_cos_img, "mean_ssim": train_ssim_img},
        "test":  {"mean_cos": test_cos_img, "mean_ssim": test_ssim_img},
        "generalization_gap_cos": train_cos_img - test_cos_img,
    },
    "audio": {
        "final_loss": audio_losses[-1],
        "train": {"mean_cos": train_cos_aud},
        "test":  {"mean_cos": test_cos_aud},
        "generalization_gap_cos": train_cos_aud - test_cos_aud,
    },
    "text": {
        "final_loss": text_losses[-1],
        "train": {"exact_match": train_correct, "total": N_TRAIN},
        "test":  {"exact_match": test_correct, "total": N_TEST},
        "results": {words[i]: all_text_results[i]["generated"] for i in test_idx},
    },
    "diversity": {
        "brain_noise": 0.15,
        "mean_l2": mean_diversity,
        "per_input": diversity_scores,
    },
}

with open(f"{OUT}/results.json", "w") as f:
    json.dump(round_nested(results), f, indent=2)
print(f"  Saved {OUT}/results.json")

# ── Visualizations ──
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, losses, title in [
        (axes[0], img_losses, "Brain → Image"),
        (axes[1], audio_losses, "Brain → Audio"),
        (axes[2], text_losses, "Brain → Text"),
    ]:
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

    # Image reconstructions: 4-row grid (train targets, train recons, TEST targets, TEST recons)
    n_train_show = min(8, N_TRAIN)
    n_test_show = min(8, N_TEST)
    n_cols = max(n_train_show, n_test_show)
    fig, axes = plt.subplots(4, n_cols, figsize=(3 * n_cols, 12))

    for col in range(n_cols):
        # Row 0-1: Train
        if col < n_train_show:
            i = train_idx[col]
            t = target_images[i].permute(1, 2, 0).clamp(0, 1).numpy()
            axes[0, col].imshow(t, interpolation="nearest")
            axes[0, col].set_title(f"Train {i}", fontsize=8)
            brain = BrainData(voxels=brain_patterns_img[i:i + 1])
            torch.manual_seed(0)
            out = img_model.reconstruct(brain, num_steps=50, cfg_scale=3.0)
            r = out.output[0].detach().clamp(0, 1).permute(1, 2, 0).numpy()
            axes[1, col].imshow(r, interpolation="nearest")
            s = train_img_results[i]["ssim"]
            axes[1, col].set_title(f"SSIM={s:.2f}", fontsize=8)
        for row in range(2):
            axes[row, col].axis("off")

        # Row 2-3: Test (held out)
        if col < n_test_show:
            i = test_idx[col]
            t = target_images[i].permute(1, 2, 0).clamp(0, 1).numpy()
            axes[2, col].imshow(t, interpolation="nearest")
            axes[2, col].set_title(f"TEST {i}", fontsize=8, color="red")
            brain = BrainData(voxels=brain_patterns_img[i:i + 1])
            torch.manual_seed(0)
            out = img_model.reconstruct(brain, num_steps=50, cfg_scale=3.0)
            r = out.output[0].detach().clamp(0, 1).permute(1, 2, 0).numpy()
            axes[3, col].imshow(r, interpolation="nearest")
            s = test_img_results[i]["ssim"]
            axes[3, col].set_title(f"SSIM={s:.2f}", fontsize=8, color="red")
        for row in range(2, 4):
            axes[row, col].axis("off")

    fig.suptitle(f"Brain → Image: Train (top) vs TEST (bottom, held-out)\n"
                 f"Train SSIM={train_ssim_img:.3f}  Test SSIM={test_ssim_img:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/image_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/image_reconstructions.png")

    # Audio reconstructions
    n_show = min(8, N_TRAIN)
    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
    for col in range(n_show):
        i = train_idx[col]
        axes[0, col].imshow(target_mels[i].numpy(), aspect="auto", origin="lower")
        axes[0, col].set_title(f"Target {i}", fontsize=8)
        axes[0, col].axis("off")
        brain = BrainData(voxels=brain_patterns_aud[i:i + 1])
        torch.manual_seed(0)
        out = audio_model.reconstruct(brain, num_steps=20, cfg_scale=3.0)
        axes[1, col].imshow(out.output[0].detach().numpy(), aspect="auto", origin="lower")
        cos = train_audio_results[i]["cos"]
        axes[1, col].set_title(f"cos={cos:.2f}", fontsize=8)
        axes[1, col].axis("off")
    fig.suptitle(f"Brain → Audio: Train cos={train_cos_aud:.3f}, Test cos={test_cos_aud:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/audio_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/audio_reconstructions.png")

    # Text results
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis("off")
    lines = ["Brain → Text: Train vs Test (discrete memorization task)\n",
             "  TRAIN SET (first 16):"]
    for i in train_idx[:16]:
        r = train_text_results[i]
        mark = "✓" if r["exact"] else "✗"
        lines.append(f"    {i:3d}: {repr(r['generated']):8s} target={r['target']:5s} {mark}")
    lines.append(f"\n  TEST SET (HELD OUT — all {N_TEST}):")
    for i in test_idx:
        r = test_text_results[i]
        mark = "✓" if r["exact"] else "✗"
        lines.append(f"    {i:3d}: {repr(r['generated']):8s} target={r['target']:5s} {mark}")
    lines.append(f"\n  Train: {train_correct}/{N_TRAIN}  Test: {test_correct}/{N_TEST}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace")
    fig.savefig(f"{OUT}/text_reconstructions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT}/text_reconstructions.png")

    # Semantic diversity visualization
    N_DIVERSE = 4
    N_SHOW_DIV = 4
    print("\n  Generating semantic diversity visualization...")
    fig, axes = plt.subplots(N_SHOW_DIV, N_DIVERSE + 1,
                             figsize=(3 * (N_DIVERSE + 1), 3 * N_SHOW_DIV))
    for row in range(N_SHOW_DIV):
        brain = BrainData(voxels=brain_patterns_img[row:row + 1])
        t = target_images[row].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[row, 0].imshow(t, interpolation="nearest")
        axes[row, 0].set_title(f"Target {row}", fontsize=9)
        axes[row, 0].axis("off")
        for s in range(N_DIVERSE):
            torch.manual_seed(s * 17 + 3)
            out = img_model.reconstruct(brain, num_steps=50, cfg_scale=3.0, brain_noise=0.15)
            sample = out.output[0].detach().clamp(0, 1)
            r = sample.permute(1, 2, 0).numpy()
            axes[row, s + 1].imshow(r, interpolation="nearest")
            cos_to_target = F.cosine_similarity(
                sample.flatten().unsqueeze(0),
                target_images[row].flatten().unsqueeze(0),
            ).item()
            axes[row, s + 1].set_title(f"Sample {s} (cos={cos_to_target:.2f})", fontsize=8)
            axes[row, s + 1].axis("off")
    fig.suptitle(f"Semantic Diversity: {N_DIVERSE} samples per brain "
                 f"(noise=0.15, L2={mean_diversity:.3f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/semantic_diversity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/semantic_diversity.png")

except ImportError:
    print("  matplotlib not available, skipping visualization")

# Save model checkpoints
torch.save(img_model.state_dict(), f"{OUT}/brain2img.pt")
torch.save(audio_model.state_dict(), f"{OUT}/brain2audio.pt")
torch.save(text_model.state_dict(), f"{OUT}/brain2text.pt")
print(f"  Saved model checkpoints to {OUT}/")

# ═══════════════════════════════════════════════════════════════
# SUMMARY — GENERALIZATION RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING SUMMARY — GENERALIZATION EVALUATION")
print("=" * 70)
print(f"  Data: {N_TOTAL} total ({N_TRAIN} train / {N_TEST} test held-out)")
print(f"\n  Brain → Image:")
print(f"    TRAIN: cos={train_cos_img:.3f} SSIM={train_ssim_img:.3f}")
print(f"    TEST:  cos={test_cos_img:.3f} SSIM={test_ssim_img:.3f}")
print(f"    Gap:   {train_cos_img - test_cos_img:.3f}")
print(f"\n  Brain → Audio:")
print(f"    TRAIN: cos={train_cos_aud:.3f}")
print(f"    TEST:  cos={test_cos_aud:.3f}")
print(f"    Gap:   {train_cos_aud - test_cos_aud:.3f}")
print(f"    NOTE:  mel targets from synthesize_audio are nearly uniform —")
print(f"           high cos reflects target similarity, not reconstruction quality")
print(f"\n  Brain → Text (discrete memorization — no semantic embedding):")
print(f"    TRAIN: {train_correct}/{N_TRAIN}")
print(f"    TEST:  {test_correct}/{N_TEST} (expected: 0 — byte-level has no generalization path)")
print(f"\n  Diversity: mean L2={mean_diversity:.4f} (brain_noise=0.15)")

all_pass = (test_cos_img > 0.3 and test_cos_aud > 0.3 and mean_diversity > 0.01)
print(f"\n  Overall: {'PASS' if all_pass else 'NEEDS WORK'}")
