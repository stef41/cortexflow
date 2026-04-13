"""Ablation study: measure each component's contribution to image reconstruction.

Runs the image pipeline with different configurations to isolate the effect of:
1. Brain encoder pre-training (warm-start)
2. Noise augmentation (regularization)
3. EMA (Exponential Moving Average)
4. Residual training (DiT learns what linear misses)

Uses 200 samples (160 train / 40 test) for speed. Each config takes ~3 min.
Total: ~15 min for 5 configurations.

Results saved to train_outputs/ablation_results.json and printed as a table.
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
N_TOTAL = 200
N_TRAIN = 160
N_TEST = 40
N_VOXELS = 512
IMG_SIZE = 32
N_STEPS = 10000  # reduced for speed
N_AVG = 8


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
# GENERATE DATA (once, shared across all configs)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("ABLATION STUDY: Component Contributions to Image Reconstruction")
print("=" * 70)
print(f"  {N_TOTAL} samples ({N_TRAIN} train / {N_TEST} test)")
print(f"  Each config: {N_STEPS} training steps, n_avg={N_AVG}")
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


# ═══════════════════════════════════════════════════════════════
# ABLATION CONFIGS
# ═══════════════════════════════════════════════════════════════
configs = [
    {
        "name": "Base DiT (no tricks)",
        "brain_pretrain": False,
        "noise_augment": 0.0,
        "use_ema": False,
        "residual": False,
    },
    {
        "name": "+ Brain pre-train",
        "brain_pretrain": True,
        "noise_augment": 0.0,
        "use_ema": False,
        "residual": False,
    },
    {
        "name": "+ Noise augment",
        "brain_pretrain": True,
        "noise_augment": 0.1,
        "use_ema": False,
        "residual": False,
    },
    {
        "name": "+ EMA",
        "brain_pretrain": True,
        "noise_augment": 0.1,
        "use_ema": True,
        "residual": False,
    },
    {
        "name": "+ Residual (full)",
        "brain_pretrain": True,
        "noise_augment": 0.1,
        "use_ema": True,
        "residual": True,
    },
]


def run_config(cfg):
    """Train and evaluate one ablation configuration."""
    torch.manual_seed(42)
    t0 = time.time()

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
        idx = torch.randint(0, N_TRAIN, (32,))
        batch = target_images[idx]
        recon, mu, logvar = model.vae(batch)
        loss, _ = model.vae.loss(batch, recon, mu, logvar)
        vae_opt.zero_grad()
        loss.backward()
        vae_opt.step()
    model.vae.eval()

    # Cache latents
    with torch.no_grad():
        latents, _, _ = model.vae.encode(target_images)

    # Compute linear mapping
    train_z = latents[:N_TRAIN].flatten(1)
    X_b = torch.cat([brain_patterns[:N_TRAIN], torch.ones(N_TRAIN, 1)], dim=1)
    W_lin = torch.linalg.lstsq(X_b, train_z).solution
    X_all = torch.cat([brain_patterns, torch.ones(N_TOTAL, 1)], dim=1)
    lin_preds = (X_all @ W_lin).view(N_TOTAL, *latents.shape[1:])
    residuals = latents - lin_preds

    # Compute linear baseline
    X_test = torch.cat([brain_patterns[N_TRAIN:], torch.ones(N_TEST, 1)], dim=1)
    pred_z_test = (X_test @ W_lin).view(N_TEST, *latents.shape[1:])
    with torch.no_grad():
        lin_decoded = model.vae.decode(pred_z_test).clamp(0, 1)
    lin_cos = [F.cosine_similarity(lin_decoded[i].flatten().unsqueeze(0),
                                    target_images[N_TRAIN + i].flatten().unsqueeze(0)).item()
               for i in range(N_TEST)]
    linear_baseline = sum(lin_cos) / len(lin_cos)

    # Optional brain encoder pre-training
    if cfg["brain_pretrain"]:
        enc_opt = torch.optim.Adam(model.brain_encoder.parameters(), lr=3e-3)
        for step in range(2000):
            idx = torch.randint(0, N_TRAIN, (32,))
            brain = BrainData(voxels=brain_patterns[idx])
            bg, _ = model.encode_brain(brain)
            target_flat = latents[idx].flatten(1)
            if not hasattr(model, '_warmup_proj'):
                model._warmup_proj = torch.nn.Linear(bg.shape[-1], target_flat.shape[-1])
            pred = model._warmup_proj(bg)
            loss = F.mse_loss(pred, target_flat)
            enc_opt.zero_grad()
            loss.backward()
            enc_opt.step()
        if hasattr(model, '_warmup_proj'):
            del model._warmup_proj

    # Train flow matching
    targets = residuals if cfg["residual"] else latents
    params = [p for n, p in model.named_parameters() if not n.startswith("vae.")]
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS, eta_min=3e-5)
    model.train()
    ema = EMAModel(model, decay=0.999) if cfg["use_ema"] else None

    for step in range(N_STEPS):
        idx = torch.randint(0, N_TRAIN, (32,))
        brain = BrainData(voxels=brain_patterns[idx])
        if cfg["noise_augment"] > 0:
            brain = BrainData(
                voxels=brain.voxels + cfg["noise_augment"] * torch.randn_like(brain.voxels)
            )
        z = targets[idx]
        bg, bt = model.encode_brain(brain)
        loss = model.flow_matcher.compute_loss(model.dit, z, bg, bt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

    if ema is not None:
        ema.apply_to(model)
    final_loss = loss.item()

    # Evaluate
    model.eval()
    latent_shape = tuple(latents.shape[1:])

    def eval_set(indices):
        results = {}
        for i in indices:
            brain = BrainData(voxels=brain_patterns[i:i + 1])
            bg, bt = model.encode_brain(brain)
            shape_1 = (1,) + latent_shape
            latent_sum = None
            for s in range(N_AVG):
                torch.manual_seed(s)
                z = model.flow_matcher.sample(
                    model.dit, shape_1, bg, bt, num_steps=50, cfg_scale=1.0,
                )
                if cfg["residual"]:
                    z = z + lin_preds[i:i + 1]
                if latent_sum is None:
                    latent_sum = z
                else:
                    latent_sum = latent_sum + z
            avg_z = latent_sum / N_AVG
            with torch.no_grad():
                recon = model.vae.decode(avg_z)[0].detach().clamp(0, 1)
            target = target_images[i]
            cos = F.cosine_similarity(recon.flatten().unsqueeze(0),
                                       target.flatten().unsqueeze(0)).item()
            ssim = compute_ssim(recon, target)
            results[i] = {"cos": cos, "ssim": ssim}
        return results

    train_res = eval_set(train_idx)
    test_res = eval_set(test_idx)

    train_cos = sum(r["cos"] for r in train_res.values()) / N_TRAIN
    train_ssim = sum(r["ssim"] for r in train_res.values()) / N_TRAIN
    test_cos = sum(r["cos"] for r in test_res.values()) / N_TEST
    test_ssim = sum(r["ssim"] for r in test_res.values()) / N_TEST

    elapsed = time.time() - t0
    return {
        "name": cfg["name"],
        "train_cos": round(train_cos, 3),
        "train_ssim": round(train_ssim, 3),
        "test_cos": round(test_cos, 3),
        "test_ssim": round(test_ssim, 3),
        "gap_cos": round(train_cos - test_cos, 3),
        "vs_linear": round(test_cos - linear_baseline, 3),
        "linear_baseline": round(linear_baseline, 3),
        "final_loss": round(final_loss, 4),
        "time_s": round(elapsed),
    }


# ═══════════════════════════════════════════════════════════════
# RUN ABLATIONS
# ═══════════════════════════════════════════════════════════════
results = []
for i, cfg in enumerate(configs):
    print(f"\n{'─' * 70}")
    print(f"  [{i+1}/{len(configs)}] {cfg['name']}")
    print(f"{'─' * 70}")
    res = run_config(cfg)
    results.append(res)
    print(f"  → test cos={res['test_cos']:.3f} SSIM={res['test_ssim']:.3f} "
          f"vs_linear={res['vs_linear']:+.3f} ({res['time_s']}s)")

# ═══════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ABLATION RESULTS")
print(f"{'=' * 70}")
print(f"  Data: {N_TOTAL} samples ({N_TRAIN} train / {N_TEST} test)")
print(f"  Random baseline cos: {baseline_cos:.3f}")
print(f"  Linear baseline cos: {results[0]['linear_baseline']:.3f}")
print()

header = f"  {'Configuration':<25s} {'Test cos':>8s} {'SSIM':>6s} {'Gap':>6s} {'vs Lin':>7s} {'Time':>5s}"
print(header)
print(f"  {'─' * 60}")
for r in results:
    line = (f"  {r['name']:<25s} {r['test_cos']:>8.3f} {r['test_ssim']:>6.3f} "
            f"{r['gap_cos']:>6.3f} {r['vs_linear']:>+7.3f} {r['time_s']:>4d}s")
    print(line)
print(f"  {'─' * 60}")
print(f"  {'Linear baseline':<25s} {results[0]['linear_baseline']:>8.3f}")

# Best vs worst
best = max(results, key=lambda r: r["test_cos"])
worst = min(results, key=lambda r: r["test_cos"])
print(f"\n  Improvement from ablation: {worst['test_cos']:.3f} → {best['test_cos']:.3f} "
      f"(+{best['test_cos'] - worst['test_cos']:.3f})")

# Save
with open(f"{OUT}/ablation_results.json", "w") as f:
    json.dump({"configs": results, "baseline_cos": baseline_cos}, f, indent=2)
print(f"\n  Saved {OUT}/ablation_results.json")

# Visualization
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = [r["name"].replace("+ ", "+\n") for r in results]
    test_cos = [r["test_cos"] for r in results]
    test_ssim = [r["test_ssim"] for r in results]
    vs_linear = [r["vs_linear"] for r in results]

    colors = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1565c0"]

    ax1.bar(range(len(results)), test_cos, color=colors, alpha=0.8)
    ax1.axhline(y=results[0]["linear_baseline"], color="black", linestyle="--",
                linewidth=1.5, label=f"Linear baseline ({results[0]['linear_baseline']:.3f})")
    ax1.axhline(y=baseline_cos, color="gray", linestyle=":", linewidth=1,
                label=f"Random baseline ({baseline_cos:.3f})")
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("Test Cosine Similarity")
    ax1.set_title("Ablation: Test cos by Configuration")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(range(len(results)), vs_linear, color=colors, alpha=0.8)
    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("DiT cos - Linear cos")
    ax2.set_title("Ablation: Gap vs Linear Baseline")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Ablation Study — Component Contributions ({N_TRAIN} train / {N_TEST} test)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ablation_study.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT}/ablation_study.png")

except ImportError:
    print("  matplotlib not available, skipping visualization")
