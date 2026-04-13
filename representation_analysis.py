"""Representation analysis: what did the brain encoder learn?

Trains the full image pipeline once, then analyzes:
1. Per-sample reconstruction quality distribution (histogram)
2. PCA of brain embeddings colored by reconstruction quality
3. Best/worst reconstruction grids
4. Correlation of image properties with reconstruction quality

Produces train_outputs/representation_analysis.png

~12 min total (10k steps + evaluation + analysis).
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
print("REPRESENTATION ANALYSIS")
print("=" * 70)

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
    return img.clamp(0, 1), shape_type


print("  Generating data...")
image_brains, image_targets, shape_types = [], [], []
for i in range(N_TOTAL):
    clean_img, stype = make_random_image(i * 7 + 13, IMG_SIZE)
    video = clean_img.unsqueeze(0)
    with torch.no_grad():
        bold = vision_forward.predict(video)
        brain_vec = bold.mean(dim=0)
    image_brains.append(brain_vec)
    image_targets.append(clean_img)
    shape_types.append(stype)

brain_patterns = torch.stack(image_brains)
target_images = torch.stack(image_targets)
shape_types = shape_types

train_idx = list(range(N_TRAIN))
test_idx = list(range(N_TRAIN, N_TOTAL))

shape_names = ["circle", "rectangle", "h-stripe", "v-stripe", "split"]

# Image properties for correlation analysis
img_brightness = [target_images[i].mean().item() for i in range(N_TOTAL)]
img_contrast = [target_images[i].std().item() for i in range(N_TOTAL)]
img_color_range = [(target_images[i].max() - target_images[i].min()).item() for i in range(N_TOTAL)]


# ═══════════════════════════════════════════════════════════════
# TRAIN MODEL
# ═══════════════════════════════════════════════════════════════
print("  Training model...")
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

X_all = torch.cat([brain_patterns[all_idx], torch.ones(len(all_idx), 1)], dim=1)
lin_preds_all = (X_all @ W_lin).view(len(all_idx), *latent_shape)
lin_preds_train = lin_preds_all[:N_TRAIN]
lin_preds_test = lin_preds_all[N_TRAIN:]

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

# Train flow matching
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
# EXTRACT BRAIN EMBEDDINGS + EVALUATE ALL SAMPLES
# ═══════════════════════════════════════════════════════════════
print("  Extracting embeddings and evaluating...")

# Get brain embeddings for ALL samples
all_embeddings = []
with torch.no_grad():
    for i in range(N_TOTAL):
        brain = BrainData(voxels=brain_patterns[i:i + 1])
        bg, _ = model.encode_brain(brain)
        all_embeddings.append(bg.squeeze(0))
embeddings = torch.stack(all_embeddings)  # (N_TOTAL, cond_dim)

# Evaluate ALL test samples
test_cos_list, test_ssim_list = [], []
test_recons = []
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
    test_recons.append(recon)

# Also evaluate linear baseline
lin_cos_list, lin_ssim_list = [], []
with torch.no_grad():
    lin_decoded = model.vae.decode(lin_preds_test).clamp(0, 1)
for i in range(N_TEST):
    cos = F.cosine_similarity(lin_decoded[i].flatten().unsqueeze(0),
                               target_images[test_idx[i]].flatten().unsqueeze(0)).item()
    ssim = compute_ssim(lin_decoded[i], target_images[test_idx[i]])
    lin_cos_list.append(cos)
    lin_ssim_list.append(ssim)

mean_cos = sum(test_cos_list) / N_TEST
mean_ssim = sum(test_ssim_list) / N_TEST
print(f"  Test cos={mean_cos:.3f}, SSIM={mean_ssim:.3f}")

# ═══════════════════════════════════════════════════════════════
# PCA ON BRAIN EMBEDDINGS
# ═══════════════════════════════════════════════════════════════
print("  Computing PCA...")
emb = embeddings.detach()
emb_centered = emb - emb.mean(dim=0, keepdim=True)
U, S, V = torch.svd(emb_centered)
pca_2d = (emb_centered @ V[:, :2]).numpy()

# Variance explained
var_explained = (S[:2] ** 2 / (S ** 2).sum()).numpy()
print(f"  PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")

# ═══════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("  Computing correlations...")

# Per-shape-type metrics
shape_metrics = {}
for stype in range(5):
    test_mask = [i for i in range(N_TEST) if shape_types[test_idx[i]] == stype]
    if test_mask:
        cos_vals = [test_cos_list[i] for i in test_mask]
        ssim_vals = [test_ssim_list[i] for i in test_mask]
        shape_metrics[shape_names[stype]] = {
            "n": len(test_mask),
            "cos_mean": round(sum(cos_vals) / len(cos_vals), 3),
            "ssim_mean": round(sum(ssim_vals) / len(ssim_vals), 3),
        }
        print(f"    {shape_names[stype]:>10s}: n={len(test_mask):3d}, "
              f"cos={shape_metrics[shape_names[stype]]['cos_mean']:.3f}, "
              f"ssim={shape_metrics[shape_names[stype]]['ssim_mean']:.3f}")

# Correlation of image properties with quality
def pearson_r(x, y):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = ((x - mx) ** 2).sum().sqrt() * ((y - my) ** 2).sum().sqrt()
    return (num / den).item() if den > 0 else 0.0

test_brightness = [img_brightness[test_idx[i]] for i in range(N_TEST)]
test_contrast = [img_contrast[test_idx[i]] for i in range(N_TEST)]

r_bright_cos = pearson_r(test_brightness, test_cos_list)
r_contrast_cos = pearson_r(test_contrast, test_cos_list)
print(f"    Brightness ↔ cos: r={r_bright_cos:.3f}")
print(f"    Contrast ↔ cos:   r={r_contrast_cos:.3f}")


# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
analysis = {
    "test_cos_mean": round(mean_cos, 4),
    "test_ssim_mean": round(mean_ssim, 4),
    "pca_variance_explained": [round(float(v), 4) for v in var_explained],
    "per_shape": shape_metrics,
    "correlations": {
        "brightness_cos": round(r_bright_cos, 4),
        "contrast_cos": round(r_contrast_cos, 4),
    },
    "per_sample_cos": [round(c, 4) for c in test_cos_list],
    "per_sample_ssim": [round(s, 4) for s in test_ssim_list],
}
with open(f"{OUT}/representation_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
print(f"\n  Saved {OUT}/representation_analysis.json")


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION (6-panel figure)
# ═══════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(20, 13))

    # ─── Panel 1: Per-sample cosine histogram ───
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(test_cos_list, bins=20, color="#1565c0", alpha=0.7, edgecolor="white",
             label="Residual DiT")
    ax1.hist(lin_cos_list, bins=20, color="#d32f2f", alpha=0.5, edgecolor="white",
             label="Linear baseline")
    ax1.axvline(x=mean_cos, color="#1565c0", linestyle="--", linewidth=2)
    ax1.set_xlabel("Cosine Similarity", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Per-Sample Cosine Distribution (Test)", fontsize=12)
    ax1.legend(fontsize=9)

    # ─── Panel 2: PCA of brain embeddings colored by shape type ───
    ax2 = fig.add_subplot(2, 3, 2)
    colors_map = {0: "#e53935", 1: "#1e88e5", 2: "#43a047", 3: "#fb8c00", 4: "#8e24aa"}
    for stype in range(5):
        mask = [i for i in range(N_TOTAL) if shape_types[i] == stype]
        if mask:
            train_mask = [i for i in mask if i < N_TRAIN]
            test_mask = [i for i in mask if i >= N_TRAIN]
            if train_mask:
                ax2.scatter(pca_2d[train_mask, 0], pca_2d[train_mask, 1],
                           c=colors_map[stype], alpha=0.3, s=15, marker="o")
            if test_mask:
                ax2.scatter(pca_2d[test_mask, 0], pca_2d[test_mask, 1],
                           c=colors_map[stype], alpha=0.8, s=40, marker="*",
                           label=shape_names[stype])
    ax2.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=11)
    ax2.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=11)
    ax2.set_title("Brain Embeddings (PCA) by Shape Type", fontsize=12)
    ax2.legend(fontsize=8, title="Test samples", markerscale=0.8)

    # ─── Panel 3: PCA colored by reconstruction quality ───
    ax3 = fig.add_subplot(2, 3, 3)
    # Only test samples, colored by cos
    test_pca = pca_2d[N_TRAIN:]
    sc = ax3.scatter(test_pca[:, 0], test_pca[:, 1], c=test_cos_list,
                      cmap="RdYlGn", s=40, alpha=0.8, vmin=0.6, vmax=1.0)
    plt.colorbar(sc, ax=ax3, label="Cosine Similarity")
    ax3.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=11)
    ax3.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=11)
    ax3.set_title("Test Embeddings Colored by Quality", fontsize=12)

    # ─── Panel 4: Per-shape-type bar chart ───
    ax4 = fig.add_subplot(2, 3, 4)
    names = list(shape_metrics.keys())
    cos_means = [shape_metrics[n]["cos_mean"] for n in names]
    ssim_means = [shape_metrics[n]["ssim_mean"] for n in names]
    x = range(len(names))
    ax4.bar([i - 0.15 for i in x], cos_means, width=0.3, color="#1565c0", alpha=0.8, label="Cos")
    ax4.bar([i + 0.15 for i in x], ssim_means, width=0.3, color="#388e3c", alpha=0.8, label="SSIM")
    ax4.set_xticks(list(x))
    ax4.set_xticklabels(names, fontsize=9)
    ax4.set_ylabel("Score", fontsize=11)
    ax4.set_title("Reconstruction Quality by Shape Type", fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    # ─── Panel 5: Best 8 test reconstructions ───
    ax5 = fig.add_subplot(2, 3, 5)
    sorted_idx = sorted(range(N_TEST), key=lambda i: test_cos_list[i], reverse=True)
    best_8 = sorted_idx[:8]
    grid_imgs = []
    for i in best_8:
        target = target_images[test_idx[i]].permute(1, 2, 0).numpy()
        recon = test_recons[i].permute(1, 2, 0).numpy()
        grid_imgs.append(np.concatenate([target, recon], axis=1))
    grid_row1 = np.concatenate(grid_imgs[:4], axis=1)
    grid_row2 = np.concatenate(grid_imgs[4:], axis=1)
    grid = np.concatenate([grid_row1, grid_row2], axis=0)
    ax5.imshow(grid)
    ax5.set_title(f"Best 8 Test Reconstructions (cos {test_cos_list[best_8[0]]:.3f}–{test_cos_list[best_8[7]]:.3f})",
                  fontsize=11)
    ax5.axis("off")
    # Add labels
    ax5.text(0.0, -0.02, "Target | Recon  ×4    repeated for both rows", fontsize=8,
             transform=ax5.transAxes, color="gray")

    # ─── Panel 6: Worst 8 test reconstructions ───
    ax6 = fig.add_subplot(2, 3, 6)
    worst_8 = sorted_idx[-8:][::-1]  # worst → less worst
    grid_imgs = []
    for i in worst_8:
        target = target_images[test_idx[i]].permute(1, 2, 0).numpy()
        recon = test_recons[i].permute(1, 2, 0).numpy()
        grid_imgs.append(np.concatenate([target, recon], axis=1))
    grid_row1 = np.concatenate(grid_imgs[:4], axis=1)
    grid_row2 = np.concatenate(grid_imgs[4:], axis=1)
    grid = np.concatenate([grid_row1, grid_row2], axis=0)
    ax6.imshow(grid)
    ax6.set_title(f"Worst 8 Test Reconstructions (cos {test_cos_list[worst_8[-1]]:.3f}–{test_cos_list[worst_8[0]]:.3f})",
                  fontsize=11)
    ax6.axis("off")

    fig.suptitle("Representation Analysis — What Did the Brain Encoder Learn?", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/representation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT}/representation_analysis.png")

except ImportError:
    print("  matplotlib not available, skipping visualization")
