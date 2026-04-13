"""Scaling study: how do metrics change with the number of training samples?

Trains the image pipeline at different data scales:
  50, 100, 150, 200, 300, 400 train samples (test fixed at 100)

For each scale, measures:
  - DiT test cos / SSIM
  - Linear baseline test cos
  - Gap (DiT - linear)

Each config: 10k training steps (fast), n_avg=4 for speed.
Total: ~20-25 min for 6 scales.

Results saved to train_outputs/scaling_results.json and scaling_curve.png.
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
N_TEST = 100         # fixed test set across all scales
N_VOXELS = 512
IMG_SIZE = 32
N_STEPS = 10000      # training steps per scale (reduced for speed)
N_AVG = 4            # sample averaging (reduced for speed)
SCALES = [50, 100, 150, 200, 300, 400]  # training set sizes


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
# GENERATE ALL DATA (once, shared across all scales)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("SCALING STUDY: Test Metrics vs Training Set Size")
print("=" * 70)
print(f"  Scales: {SCALES}")
print(f"  Test set: {N_TEST} (fixed, same across all scales)")
print(f"  Steps per scale: {N_STEPS}, n_avg={N_AVG}")
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


N_MAX = max(SCALES) + N_TEST
print(f"  Generating {N_MAX} images...")
image_brains, image_targets = [], []
for i in range(N_MAX):
    clean_img = make_random_image(i * 7 + 13, IMG_SIZE)
    video = clean_img.unsqueeze(0)
    with torch.no_grad():
        bold = vision_forward.predict(video)
        brain_vec = bold.mean(dim=0)
    image_brains.append(brain_vec)
    image_targets.append(clean_img)

brain_patterns = torch.stack(image_brains)
target_images = torch.stack(image_targets)

# Test set is ALWAYS the last N_TEST samples
test_idx = list(range(N_MAX - N_TEST, N_MAX))

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
# RUN EACH SCALE
# ═══════════════════════════════════════════════════════════════
def run_scale(n_train):
    """Train and evaluate at a specific data scale."""
    torch.manual_seed(42)
    t0 = time.time()

    train_idx = list(range(n_train))

    model = Brain2Image(
        n_voxels=N_VOXELS, img_size=IMG_SIZE,
        dit_config=DiTConfig(hidden_dim=64, depth=4, num_heads=4, cond_dim=64),
        vae_config=VAEConfig(hidden_dims=[32, 64, 128]),
        flow_config=FlowConfig(),
    )

    # Train VAE on train images only
    vae_opt = torch.optim.Adam(model.vae.parameters(), lr=1e-3)
    model.vae.train()
    for step in range(2000):
        idx = torch.randint(0, n_train, (min(32, n_train),))
        batch = target_images[idx]
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
    # Map back: first n_train are train, last N_TEST are test
    train_latents = all_latents[:n_train]
    test_latents = all_latents[n_train:]

    # Linear mapping (train only)
    train_z = train_latents.flatten(1)
    X_b = torch.cat([brain_patterns[train_idx], torch.ones(n_train, 1)], dim=1)
    W_lin = torch.linalg.lstsq(X_b, train_z).solution
    latent_shape = tuple(train_latents.shape[1:])

    # Linear predictions for all
    X_all = torch.cat([brain_patterns[all_idx], torch.ones(len(all_idx), 1)], dim=1)
    lin_preds_all = (X_all @ W_lin).view(len(all_idx), *latent_shape)
    lin_preds_train = lin_preds_all[:n_train]
    lin_preds_test = lin_preds_all[n_train:]

    # Linear baseline: decode test predictions
    with torch.no_grad():
        lin_decoded = model.vae.decode(lin_preds_test).clamp(0, 1)
    lin_cos_list = [F.cosine_similarity(lin_decoded[i].flatten().unsqueeze(0),
                                         target_images[test_idx[i]].flatten().unsqueeze(0)).item()
                    for i in range(N_TEST)]
    linear_cos = sum(lin_cos_list) / len(lin_cos_list)

    # Residuals for training
    residuals_train = train_latents - lin_preds_train

    # Brain encoder pre-training
    enc_opt = torch.optim.Adam(model.brain_encoder.parameters(), lr=3e-3)
    for step in range(2000):
        idx = torch.randint(0, n_train, (min(32, n_train),))
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
        idx = torch.randint(0, n_train, (min(32, n_train),))
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

    # Evaluate on test set
    test_cos_list, test_ssim_list = [], []
    for i in range(N_TEST):
        brain = BrainData(voxels=brain_patterns[test_idx[i]:test_idx[i] + 1])
        bg, bt = model.encode_brain(brain)
        shape_1 = (1,) + latent_shape
        latent_sum = None
        for s in range(N_AVG):
            torch.manual_seed(s)
            z = model.flow_matcher.sample(
                model.dit, shape_1, bg, bt, num_steps=50, cfg_scale=1.0,
            )
            z = z + lin_preds_test[i:i + 1]
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
        test_cos_list.append(cos)
        test_ssim_list.append(ssim)

    test_cos = sum(test_cos_list) / len(test_cos_list)
    test_ssim = sum(test_ssim_list) / len(test_ssim_list)
    elapsed = time.time() - t0

    return {
        "n_train": n_train,
        "test_cos": round(test_cos, 3),
        "test_ssim": round(test_ssim, 3),
        "linear_cos": round(linear_cos, 3),
        "gap": round(test_cos - linear_cos, 3),
        "above_random": round(test_cos - baseline_cos, 3),
        "time_s": round(elapsed),
    }


results = []
for i, n_train in enumerate(SCALES):
    print(f"{'─' * 70}")
    print(f"  [{i+1}/{len(SCALES)}] n_train={n_train}")
    print(f"{'─' * 70}")
    res = run_scale(n_train)
    results.append(res)
    print(f"  → test cos={res['test_cos']:.3f} SSIM={res['test_ssim']:.3f} "
          f"linear={res['linear_cos']:.3f} gap={res['gap']:+.3f} ({res['time_s']}s)")
    print()


# ═══════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SCALING RESULTS")
print(f"{'=' * 70}")
print(f"  Random baseline cos: {baseline_cos:.3f}")
print()

header = f"  {'N_train':>7s} {'DiT cos':>8s} {'SSIM':>6s} {'Linear':>7s} {'Gap':>7s} {'> Random':>8s} {'Time':>5s}"
print(header)
print(f"  {'─' * 52}")
for r in results:
    line = (f"  {r['n_train']:>7d} {r['test_cos']:>8.3f} {r['test_ssim']:>6.3f} "
            f"{r['linear_cos']:>7.3f} {r['gap']:>+7.3f} {r['above_random']:>+8.3f} {r['time_s']:>4d}s")
    print(line)

# Save
with open(f"{OUT}/scaling_results.json", "w") as f:
    json.dump({"results": results, "baseline_cos": baseline_cos}, f, indent=2)
print(f"\n  Saved {OUT}/scaling_results.json")


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ns = [r["n_train"] for r in results]
    dit_cos = [r["test_cos"] for r in results]
    lin_cos = [r["linear_cos"] for r in results]
    gaps = [r["gap"] for r in results]
    ssims = [r["test_ssim"] for r in results]

    # Panel 1: Test cos vs n_train
    ax = axes[0]
    ax.plot(ns, dit_cos, "o-", color="#1565c0", linewidth=2, markersize=8, label="Residual DiT")
    ax.plot(ns, lin_cos, "s--", color="#d32f2f", linewidth=2, markersize=7, label="Linear baseline")
    ax.axhline(y=baseline_cos, color="gray", linestyle=":", linewidth=1, label=f"Random ({baseline_cos:.3f})")
    ax.set_xlabel("Number of Training Samples", fontsize=11)
    ax.set_ylabel("Test Cosine Similarity", fontsize=11)
    ax.set_title("Scaling: DiT vs Linear", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Gap vs n_train
    ax = axes[1]
    colors = ["#d32f2f" if g < -0.01 else "#f57c00" if g < 0 else "#388e3c" for g in gaps]
    ax.bar(range(len(ns)), gaps, color=colors, alpha=0.8, tick_label=[str(n) for n in ns])
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("N_train", fontsize=11)
    ax.set_ylabel("DiT cos − Linear cos", fontsize=11)
    ax.set_title("Gap vs Linear Baseline", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: SSIM vs n_train
    ax = axes[2]
    ax.plot(ns, ssims, "o-", color="#388e3c", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Training Samples", fontsize=11)
    ax.set_ylabel("Test SSIM", fontsize=11)
    ax.set_title("Structural Similarity vs Data Scale", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle("Scaling Study — Image Reconstruction Quality vs Training Data", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/scaling_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT}/scaling_curve.png")

except ImportError:
    print("  matplotlib not available, skipping visualization")
