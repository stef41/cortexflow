"""Noise robustness study: how does reconstruction degrade with brain signal noise?

Trains one clean model, then evaluates on test data with increasing
levels of Gaussian noise added to the brain activity patterns.

This tests a neuroscience-relevant question: fMRI data is inherently
noisy. How robust is DIit vs linear regression under realistic noise?

Noise levels: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 (as fraction of brain signal std)
Both DiT and linear are evaluated at each noise level.

Results saved to train_outputs/noise_robustness.json and noise_robustness.png.
~25 min total (10k training steps + 7 evaluation rounds).
"""

import os
import time
import json
import math

import torch
import torch.nn.functional as F

from cortexflow import BrainData
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from cortexflow.brain2img import Brain2Image
from cortexflow.flow_matching import EMAModel
from neuroprobe.media import build_brain_model

OUT = "train_outputs"
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
N_TOTAL = 500
N_TRAIN = 400
N_TEST = 100
N_VOXELS = 512
IMG_SIZE = 32
N_STEPS = 10000
N_AVG = 4
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def compute_ssim(img1, img2):
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
# GENERATE DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("NOISE ROBUSTNESS STUDY")
print("=" * 70)
print(f"  Noise levels: {NOISE_LEVELS}")
print(f"  N_TRAIN={N_TRAIN}, N_TEST={N_TEST}")
print()

torch.manual_seed(42)

vision_forward = build_brain_model(
    modality="video", feature_dim=256, n_vertices=N_VOXELS,
    hidden_dim=128, seed=42,
)


def make_random_image(seed, size):
    gen = torch.Generator().manual_seed(seed)
    bg = torch.rand(3, generator=gen) * 0.4
    img = bg.view(3, 1, 1).expand(3, size, size).clone()
    shape_type = torch.randint(0, 5, (1,), generator=gen).item()
    fg = torch.rand(3, generator=gen) * 0.5 + 0.5
    cy = torch.randint(size // 4, 3 * size // 4 + 1, (1,), generator=gen).item()
    cx = torch.randint(size // 4, 3 * size // 4 + 1, (1,), generator=gen).item()
    r = torch.randint(size // 6, size // 3 + 1, (1,), generator=gen).item()
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    yf, xf = yy.float(), xx.float()
    if shape_type == 0:
        mask = ((yf - cy) ** 2 + (xf - cx) ** 2) < r ** 2
        for c in range(3):
            img[c][mask] = fg[c]
    elif shape_type == 1:
        mask = (yy >= max(0, cy - r)) & (yy < min(size, cy + r)) & \
               (xx >= max(0, cx - r)) & (xx < min(size, cx + r))
        for c in range(3):
            img[c][mask] = fg[c]
    elif shape_type == 2:
        h = max(2, r // 2)
        mask = (yy >= max(0, cy - h)) & (yy < min(size, cy + h))
        for c in range(3):
            img[c][mask] = fg[c]
    elif shape_type == 3:
        w = max(2, r // 2)
        mask = (xx >= max(0, cx - w)) & (xx < min(size, cx + w))
        for c in range(3):
            img[c][mask] = fg[c]
    elif shape_type == 4:
        fg2 = torch.rand(3, generator=gen) * 0.5 + 0.5
        split = torch.randint(size // 3, 2 * size // 3 + 1, (1,), generator=gen).item()
        for c in range(3):
            img[c, :, :split] = fg[c]
            img[c, :, split:] = fg2[c]
    return img.clamp(0, 1)


print("  Generating data...")
image_brains, image_targets = [], []
for i in range(N_TOTAL):
    clean_img = make_random_image(i * 7 + 13, IMG_SIZE)
    video = clean_img.unsqueeze(0)
    with torch.no_grad():
        bold = vision_forward.predict(video)
        brain_vec = bold.mean(dim=0)
    image_brains.append(brain_vec)
    image_targets.append(clean_img)

brain_patterns = torch.stack(image_brains)
target_images = torch.stack(image_targets)

train_idx = list(range(N_TRAIN))
test_idx = list(range(N_TRAIN, N_TOTAL))

# Brain signal statistics (for calibrating noise)
brain_std = brain_patterns.std().item()
print(f"  Brain signal std: {brain_std:.3f}")

# Random baseline
rand_cos = []
for a in range(20):
    for b in range(a + 1, 20):
        c = F.cosine_similarity(
            target_images[a].flatten().unsqueeze(0),
            target_images[b].flatten().unsqueeze(0),
        ).item()
        rand_cos.append(c)
baseline_cos = sum(rand_cos) / len(rand_cos)
print(f"  Random baseline cos: {baseline_cos:.3f}")
print()


# ═══════════════════════════════════════════════════════════════
# TRAIN MODEL (once, on clean data)
# ═══════════════════════════════════════════════════════════════
print("  Training model on clean data...")
torch.manual_seed(42)
t0 = time.time()

model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=64, depth=4, num_heads=4, cond_dim=64),
    vae_config=VAEConfig(hidden_dims=[32, 64, 128]),
    flow_config=FlowConfig(),
)

# Train VAE
vae_opt = torch.optim.Adam(model.vae.parameters(), lr=1e-3)
model.vae.train()
for step in range(2000):
    idx = torch.randint(0, N_TRAIN, (min(32, N_TRAIN),))
    batch = target_images[train_idx][idx]
    recon, mu, logvar = model.vae(batch)
    loss, _ = model.vae.loss(batch, recon, mu, logvar)
    vae_opt.zero_grad()
    loss.backward()
    vae_opt.step()
model.vae.eval()

# Cache latents
all_idx = train_idx + test_idx
with torch.no_grad():
    all_latents, _, _ = model.vae.encode(target_images[all_idx])
train_latents = all_latents[:N_TRAIN]
test_latents = all_latents[N_TRAIN:]

# Linear mapping
train_z = train_latents.flatten(1)
X_b = torch.cat([brain_patterns[train_idx], torch.ones(N_TRAIN, 1)], dim=1)
W_lin = torch.linalg.lstsq(X_b, train_z).solution
latent_shape = tuple(train_latents.shape[1:])

# Linear predictions for training (for residuals)
X_train = torch.cat([brain_patterns[train_idx], torch.ones(N_TRAIN, 1)], dim=1)
lin_preds_train = (X_train @ W_lin).view(N_TRAIN, *latent_shape)
residuals_train = train_latents - lin_preds_train

# Brain encoder pre-training
enc_opt = torch.optim.Adam(model.brain_encoder.parameters(), lr=3e-3)
for step in range(2000):
    idx = torch.randint(0, N_TRAIN, (min(32, N_TRAIN),))
    brain = BrainData(voxels=brain_patterns[train_idx][idx])
    bg, _ = model.encode_brain(brain)
    target_flat = train_latents[idx].flatten(1)
    if not hasattr(model, '_warmup_proj'):
        model._warmup_proj = torch.nn.Linear(bg.shape[-1], target_flat.shape[-1])
    pred = model._warmup_proj(bg)
    loss = F.mse_loss(pred, target_flat)
    enc_opt.zero_grad()
    loss.backward()
    enc_opt.step()
if hasattr(model, '_warmup_proj'):
    del model._warmup_proj

# Train flow matching on residuals
params = [p for n, p in model.named_parameters() if not n.startswith("vae.")]
opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS, eta_min=3e-5)
model.train()
ema = EMAModel(model, decay=0.999)

for step in range(N_STEPS):
    idx = torch.randint(0, N_TRAIN, (min(32, N_TRAIN),))
    brain = BrainData(
        voxels=brain_patterns[train_idx][idx] + 0.1 * torch.randn(len(idx), N_VOXELS)
    )
    z = residuals_train[idx]
    bg, bt = model.encode_brain(brain)
    loss = model.flow_matcher.compute_loss(model.dit, z, bg, bt)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()
    ema.update(model)
    if step % 2000 == 0 or step == N_STEPS - 1:
        print(f"    step {step:5d}/{N_STEPS}: loss={loss.item():.4f}")

ema.apply_to(model)
model.eval()
print(f"  Training complete ({time.time() - t0:.0f}s)")


# ═══════════════════════════════════════════════════════════════
# EVALUATE AT EACH NOISE LEVEL
# ═══════════════════════════════════════════════════════════════
results = []

for noise_level in NOISE_LEVELS:
    print(f"\n{'─' * 70}")
    print(f"  Noise level: {noise_level} (absolute noise std = {noise_level * brain_std:.4f})")
    print(f"{'─' * 70}")
    t0 = time.time()

    # Add noise to test brain patterns
    torch.manual_seed(123)  # same noise realization for all levels for comparability
    noise = torch.randn(N_TEST, N_VOXELS)
    noisy_test_brains = brain_patterns[test_idx] + noise_level * brain_std * noise

    # Linear predictions on noisy data
    X_noisy = torch.cat([noisy_test_brains, torch.ones(N_TEST, 1)], dim=1)
    lin_preds_noisy = (X_noisy @ W_lin).view(N_TEST, *latent_shape)

    # Linear baseline decode
    with torch.no_grad():
        lin_decoded = model.vae.decode(lin_preds_noisy).clamp(0, 1)
    lin_cos_list = [F.cosine_similarity(lin_decoded[i].flatten().unsqueeze(0),
                                         target_images[test_idx[i]].flatten().unsqueeze(0)).item()
                    for i in range(N_TEST)]
    lin_ssim_list = [compute_ssim(lin_decoded[i], target_images[test_idx[i]])
                     for i in range(N_TEST)]
    linear_cos = sum(lin_cos_list) / len(lin_cos_list)
    linear_ssim = sum(lin_ssim_list) / len(lin_ssim_list)

    # DiT evaluation on noisy data
    dit_cos_list, dit_ssim_list = [], []
    for i in range(N_TEST):
        brain = BrainData(voxels=noisy_test_brains[i:i + 1])
        bg, bt = model.encode_brain(brain)
        shape_1 = (1,) + latent_shape
        latent_sum = None
        for s in range(N_AVG):
            torch.manual_seed(s)
            z = model.flow_matcher.sample(
                model.dit, shape_1, bg, bt, num_steps=50, cfg_scale=1.0,
            )
            z = z + lin_preds_noisy[i:i + 1]
            if latent_sum is None:
                latent_sum = z
            else:
                latent_sum = latent_sum + z
        avg_z = latent_sum / N_AVG
        with torch.no_grad():
            recon = model.vae.decode(avg_z)[0].detach().clamp(0, 1)
        target = target_images[test_idx[i]]
        cos = F.cosine_similarity(recon.flatten().unsqueeze(0),
                                   target.flatten().unsqueeze(0)).item()
        ssim = compute_ssim(recon, target)
        dit_cos_list.append(cos)
        dit_ssim_list.append(ssim)

    dit_cos = sum(dit_cos_list) / len(dit_cos_list)
    dit_ssim = sum(dit_ssim_list) / len(dit_ssim_list)
    elapsed = time.time() - t0

    # Paired comparison (DiT vs linear per sample)
    diffs = [dit_cos_list[i] - lin_cos_list[i] for i in range(N_TEST)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    # Paired t-statistic
    t_stat = mean_diff / (std_diff / N_TEST ** 0.5) if std_diff > 0 else 0

    res = {
        "noise_level": noise_level,
        "noise_std_abs": round(noise_level * brain_std, 4),
        "dit_cos": round(dit_cos, 4),
        "dit_ssim": round(dit_ssim, 4),
        "linear_cos": round(linear_cos, 4),
        "linear_ssim": round(linear_ssim, 4),
        "gap_cos": round(dit_cos - linear_cos, 4),
        "gap_ssim": round(dit_ssim - linear_ssim, 4),
        "paired_t": round(t_stat, 3),
        "time_s": round(elapsed),
    }
    results.append(res)
    print(f"  DiT: cos={dit_cos:.3f} SSIM={dit_ssim:.3f}")
    print(f"  Lin: cos={linear_cos:.3f} SSIM={linear_ssim:.3f}")
    print(f"  Gap: {dit_cos - linear_cos:+.3f} (paired t={t_stat:.2f}) ({elapsed:.0f}s)")


# ═══════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("NOISE ROBUSTNESS RESULTS")
print(f"{'=' * 70}")
print(f"  Brain signal std: {brain_std:.3f}")
print(f"  Random baseline cos: {baseline_cos:.3f}")
print()

header = f"  {'Noise':>5s} {'DiT cos':>8s} {'SSIM':>6s} {'Lin cos':>8s} {'Gap':>7s} {'t-stat':>7s}"
print(header)
print(f"  {'─' * 44}")
for r in results:
    line = (f"  {r['noise_level']:>5.1f} {r['dit_cos']:>8.3f} {r['dit_ssim']:>6.3f} "
            f"{r['linear_cos']:>8.3f} {r['gap_cos']:>+7.3f} {r['paired_t']:>7.2f}")
    print(line)

# Compute degradation rates
if len(results) >= 2:
    clean_dit = results[0]["dit_cos"]
    clean_lin = results[0]["linear_cos"]
    noisiest_dit = results[-1]["dit_cos"]
    noisiest_lin = results[-1]["linear_cos"]
    dit_drop = clean_dit - noisiest_dit
    lin_drop = clean_lin - noisiest_lin
    print(f"\n  Degradation (noise=0 → noise={NOISE_LEVELS[-1]}):")
    print(f"    DiT:    {clean_dit:.3f} → {noisiest_dit:.3f} (drop {dit_drop:.3f})")
    print(f"    Linear: {clean_lin:.3f} → {noisiest_lin:.3f} (drop {lin_drop:.3f})")
    if lin_drop > 0:
        print(f"    DiT retains {1 - dit_drop / lin_drop:.0%} of linear's noise sensitivity"
              if dit_drop < lin_drop else
              f"    DiT degrades {dit_drop / lin_drop:.0%} as much as linear")

# Save
with open(f"{OUT}/noise_robustness.json", "w") as f:
    json.dump({
        "brain_std": brain_std,
        "baseline_cos": baseline_cos,
        "results": results,
    }, f, indent=2)
print(f"\n  Saved {OUT}/noise_robustness.json")


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ns = [r["noise_level"] for r in results]
    dit_cos = [r["dit_cos"] for r in results]
    lin_cos = [r["linear_cos"] for r in results]
    dit_ssim = [r["dit_ssim"] for r in results]
    lin_ssim = [r["linear_ssim"] for r in results]
    gaps = [r["gap_cos"] for r in results]

    # Panel 1: Cosine vs noise
    ax = axes[0]
    ax.plot(ns, dit_cos, "o-", color="#1565c0", linewidth=2, markersize=8, label="Residual DiT")
    ax.plot(ns, lin_cos, "s--", color="#d32f2f", linewidth=2, markersize=7, label="Linear baseline")
    ax.axhline(y=baseline_cos, color="gray", linestyle=":", linewidth=1,
               label=f"Random ({baseline_cos:.3f})")
    ax.set_xlabel("Noise Level (× brain signal std)", fontsize=11)
    ax.set_ylabel("Test Cosine Similarity", fontsize=11)
    ax.set_title("Reconstruction vs Brain Signal Noise", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Gap vs noise
    ax = axes[1]
    colors = ["#d32f2f" if g < -0.01 else "#f57c00" if g < 0 else "#388e3c" for g in gaps]
    ax.bar(range(len(ns)), gaps, color=colors, alpha=0.8,
           tick_label=[f"{n:.1f}" for n in ns])
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("Noise Level", fontsize=11)
    ax.set_ylabel("DiT cos − Linear cos", fontsize=11)
    ax.set_title("Gap Under Noise", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: SSIM vs noise
    ax = axes[2]
    ax.plot(ns, dit_ssim, "o-", color="#388e3c", linewidth=2, markersize=8, label="Residual DiT")
    ax.plot(ns, lin_ssim, "s--", color="#d32f2f", linewidth=2, markersize=7, label="Linear baseline")
    ax.set_xlabel("Noise Level (× brain signal std)", fontsize=11)
    ax.set_ylabel("Test SSIM", fontsize=11)
    ax.set_title("Structural Similarity Under Noise", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Noise Robustness — How Does Brain Signal Noise Affect Reconstruction?",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/noise_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT}/noise_robustness.png")

except ImportError:
    print("  matplotlib not available, skipping visualization")
