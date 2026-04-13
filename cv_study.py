"""Cross-validation study: 5-fold CV with confidence intervals.

Trains the full image pipeline (residual DiT + EMA) 5 times,
each time with a different 400/100 train/test split.

Reports mean ± std for all metrics across folds.
This gives proper error bars for scientific credibility.

Each fold: 10k training steps, n_avg=4 for speed.
Total: ~50 min for 5 folds.

Results saved to train_outputs/cv_results.json and cv_summary.png.
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
N_FOLDS = 5
N_VOXELS = 512
IMG_SIZE = 32
N_STEPS = 10000      # training steps per fold
N_AVG = 4            # sample averaging (reduced for speed)


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
# GENERATE ALL DATA (once)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("CROSS-VALIDATION STUDY: 5-Fold CV with Confidence Intervals")
print("=" * 70)
print(f"  N_TOTAL={N_TOTAL}, N_FOLDS={N_FOLDS}")
print(f"  Steps per fold: {N_STEPS}, n_avg={N_AVG}")
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


print(f"  Generating {N_TOTAL} images...")
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

# Create fold indices (deterministic shuffle)
perm = torch.randperm(N_TOTAL, generator=torch.Generator().manual_seed(42))
fold_size = N_TOTAL // N_FOLDS
folds = []
for f in range(N_FOLDS):
    test_idx = perm[f * fold_size : (f + 1) * fold_size].tolist()
    train_idx = [i for i in perm.tolist() if i not in set(test_idx)]
    folds.append((train_idx, test_idx))
    print(f"  Fold {f+1}: {len(train_idx)} train, {len(test_idx)} test")
print()


# ═══════════════════════════════════════════════════════════════
# RUN EACH FOLD
# ═══════════════════════════════════════════════════════════════
def run_fold(fold_idx, train_idx, test_idx):
    """Train and evaluate on one fold."""
    torch.manual_seed(42 + fold_idx)
    t0 = time.time()
    n_train = len(train_idx)
    n_test = len(test_idx)

    model = Brain2Image(
        n_voxels=N_VOXELS, img_size=IMG_SIZE,
        dit_config=DiTConfig(hidden_dim=64, depth=4, num_heads=4, cond_dim=64),
        vae_config=VAEConfig(hidden_dims=[32, 64, 128]),
        flow_config=FlowConfig(),
    )

    # Train VAE on train images only
    train_images = target_images[train_idx]
    vae_opt = torch.optim.Adam(model.vae.parameters(), lr=1e-3)
    model.vae.train()
    for step in range(2000):
        idx = torch.randint(0, n_train, (min(32, n_train),))
        batch = train_images[idx]
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

    # Linear baseline on test
    with torch.no_grad():
        lin_decoded = model.vae.decode(lin_preds_test).clamp(0, 1)
    lin_cos_list = [F.cosine_similarity(lin_decoded[i].flatten().unsqueeze(0),
                                         target_images[test_idx[i]].flatten().unsqueeze(0)).item()
                    for i in range(n_test)]
    lin_ssim_list = [compute_ssim(lin_decoded[i], target_images[test_idx[i]])
                     for i in range(n_test)]
    linear_cos = sum(lin_cos_list) / len(lin_cos_list)
    linear_ssim = sum(lin_ssim_list) / len(lin_ssim_list)

    # Residuals for training
    residuals_train = train_latents - lin_preds_train

    # Brain encoder pre-training
    train_brains = brain_patterns[train_idx]
    enc_opt = torch.optim.Adam(model.brain_encoder.parameters(), lr=3e-3)
    for step in range(2000):
        idx = torch.randint(0, n_train, (min(32, n_train),))
        brain = BrainData(voxels=train_brains[idx])
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
            voxels=train_brains[idx] + 0.1 * torch.randn(len(idx), N_VOXELS)
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
    for i in range(n_test):
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
        "fold": fold_idx + 1,
        "n_train": n_train,
        "n_test": n_test,
        "test_cos": round(test_cos, 4),
        "test_ssim": round(test_ssim, 4),
        "linear_cos": round(linear_cos, 4),
        "linear_ssim": round(linear_ssim, 4),
        "gap_cos": round(test_cos - linear_cos, 4),
        "gap_ssim": round(test_ssim - linear_ssim, 4),
        "above_random": round(test_cos - baseline_cos, 4),
        "time_s": round(elapsed),
        "per_sample_cos": [round(c, 4) for c in test_cos_list],
        "per_sample_ssim": [round(s, 4) for s in test_ssim_list],
    }


results = []
for f, (train_idx, test_idx) in enumerate(folds):
    print(f"{'─' * 70}")
    print(f"  FOLD {f+1}/{N_FOLDS} ({len(train_idx)} train, {len(test_idx)} test)")
    print(f"{'─' * 70}")
    res = run_fold(f, train_idx, test_idx)
    results.append(res)
    print(f"  → cos={res['test_cos']:.4f} SSIM={res['test_ssim']:.4f} "
          f"linear={res['linear_cos']:.4f} gap={res['gap_cos']:+.4f} ({res['time_s']}s)")
    print()


# ═══════════════════════════════════════════════════════════════
# COMPUTE SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════
def mean_std(values):
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, var ** 0.5


cos_vals = [r["test_cos"] for r in results]
ssim_vals = [r["test_ssim"] for r in results]
lin_cos_vals = [r["linear_cos"] for r in results]
lin_ssim_vals = [r["linear_ssim"] for r in results]
gap_cos_vals = [r["gap_cos"] for r in results]
gap_ssim_vals = [r["gap_ssim"] for r in results]

cos_m, cos_s = mean_std(cos_vals)
ssim_m, ssim_s = mean_std(ssim_vals)
lin_cos_m, lin_cos_s = mean_std(lin_cos_vals)
lin_ssim_m, lin_ssim_s = mean_std(lin_ssim_vals)
gap_cos_m, gap_cos_s = mean_std(gap_cos_vals)
gap_ssim_m, gap_ssim_s = mean_std(gap_ssim_vals)

# 95% CI: mean ± 1.96 * std / sqrt(n)  (or t-distribution for n=5: ~2.776)
t_crit = 2.776  # t(df=4, alpha=0.025) for 95% CI with n=5
n = N_FOLDS
cos_ci = t_crit * cos_s / n ** 0.5
ssim_ci = t_crit * ssim_s / n ** 0.5
gap_cos_ci = t_crit * gap_cos_s / n ** 0.5

print(f"\n{'=' * 70}")
print("5-FOLD CROSS-VALIDATION RESULTS")
print(f"{'=' * 70}")
print(f"  Random baseline cos: {baseline_cos:.3f}")
print()

header = f"  {'Fold':>4s} {'DiT cos':>8s} {'SSIM':>7s} {'Linear cos':>10s} {'Gap':>8s}"
print(header)
print(f"  {'─' * 40}")
for r in results:
    line = f"  {r['fold']:>4d} {r['test_cos']:>8.4f} {r['test_ssim']:>7.4f} {r['linear_cos']:>10.4f} {r['gap_cos']:>+8.4f}"
    print(line)
print(f"  {'─' * 40}")
print(f"  {'Mean':>4s} {cos_m:>8.4f} {ssim_m:>7.4f} {lin_cos_m:>10.4f} {gap_cos_m:>+8.4f}")
print(f"  {'±Std':>4s} {cos_s:>8.4f} {ssim_s:>7.4f} {lin_cos_s:>10.4f} {gap_cos_s:>+8.4f}")
print()
print(f"  95% Confidence Intervals (t-distribution, df=4):")
print(f"    DiT cos:  {cos_m:.3f} ± {cos_ci:.3f}  [{cos_m - cos_ci:.3f}, {cos_m + cos_ci:.3f}]")
print(f"    SSIM:     {ssim_m:.3f} ± {ssim_ci:.3f}  [{ssim_m - ssim_ci:.3f}, {ssim_m + ssim_ci:.3f}]")
print(f"    Gap:      {gap_cos_m:+.3f} ± {gap_cos_ci:.3f}  [{gap_cos_m - gap_cos_ci:+.3f}, {gap_cos_m + gap_cos_ci:+.3f}]")

# Save results
summary = {
    "n_folds": N_FOLDS,
    "n_total": N_TOTAL,
    "n_steps": N_STEPS,
    "n_avg": N_AVG,
    "baseline_cos": baseline_cos,
    "summary": {
        "dit_cos": {"mean": round(cos_m, 4), "std": round(cos_s, 4), "ci95": round(cos_ci, 4)},
        "ssim": {"mean": round(ssim_m, 4), "std": round(ssim_s, 4), "ci95": round(ssim_ci, 4)},
        "linear_cos": {"mean": round(lin_cos_m, 4), "std": round(lin_cos_s, 4)},
        "linear_ssim": {"mean": round(lin_ssim_m, 4), "std": round(lin_ssim_s, 4)},
        "gap_cos": {"mean": round(gap_cos_m, 4), "std": round(gap_cos_s, 4), "ci95": round(gap_cos_ci, 4)},
        "gap_ssim": {"mean": round(gap_ssim_m, 4), "std": round(gap_ssim_s, 4)},
    },
    "folds": results,
}
with open(f"{OUT}/cv_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Saved {OUT}/cv_results.json")


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    fold_nums = [r["fold"] for r in results]

    # Panel 1: Cos per fold with CI band
    ax = axes[0]
    ax.bar([x - 0.15 for x in fold_nums], cos_vals, width=0.3, color="#1565c0",
           alpha=0.8, label="Residual DiT")
    ax.bar([x + 0.15 for x in fold_nums], lin_cos_vals, width=0.3, color="#d32f2f",
           alpha=0.8, label="Linear baseline")
    ax.axhline(y=cos_m, color="#1565c0", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhspan(cos_m - cos_ci, cos_m + cos_ci, color="#1565c0", alpha=0.1)
    ax.axhline(y=baseline_cos, color="gray", linestyle=":", linewidth=1,
               label=f"Random ({baseline_cos:.3f})")
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("Test Cosine Similarity", fontsize=11)
    ax.set_title("Cosine Similarity per Fold", fontsize=12)
    ax.set_xticks(fold_nums)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Gap per fold
    ax = axes[1]
    colors = ["#d32f2f" if g < -0.01 else "#f57c00" if g < 0 else "#388e3c" for g in gap_cos_vals]
    ax.bar(fold_nums, gap_cos_vals, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.axhline(y=gap_cos_m, color="#1565c0", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhspan(gap_cos_m - gap_cos_ci, gap_cos_m + gap_cos_ci, color="#1565c0", alpha=0.1,
               label=f"Mean ± 95% CI: {gap_cos_m:+.3f} ± {gap_cos_ci:.3f}")
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("DiT cos − Linear cos", fontsize=11)
    ax.set_title("Gap vs Linear Baseline", fontsize=12)
    ax.set_xticks(fold_nums)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: SSIM per fold
    ax = axes[2]
    ax.bar([x - 0.15 for x in fold_nums], ssim_vals, width=0.3, color="#388e3c",
           alpha=0.8, label="Residual DiT")
    ax.bar([x + 0.15 for x in fold_nums], lin_ssim_vals, width=0.3, color="#d32f2f",
           alpha=0.8, label="Linear baseline")
    ax.axhline(y=ssim_m, color="#388e3c", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhspan(ssim_m - ssim_ci, ssim_m + ssim_ci, color="#388e3c", alpha=0.1)
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("Test SSIM", fontsize=11)
    ax.set_title("Structural Similarity per Fold", fontsize=12)
    ax.set_xticks(fold_nums)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("5-Fold Cross-Validation — Image Reconstruction", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cv_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT}/cv_summary.png")

except ImportError:
    print("  matplotlib not available, skipping visualization")
