"""Brain space interpolation and manipulation study.

Demonstrates that cortexflow's brain latent space is smooth, structured,
and supports semantic arithmetic — the brain decoding equivalent of
"latent space walks" in GANs/VAEs.

Analyses:
1. Cross-category interpolation (e.g., cat → dog, ship → airplane)
   — reconstruct at 9 evenly spaced steps in brain space
2. Brain arithmetic: category_A - mean(group_A) + mean(group_B) → ?
   e.g., cat - animal_mean + vehicle_mean → vehicle-like cat?
3. Targeted dimension manipulation: boost/suppress top PCA components
   — reveals what each brain dimension encodes
4. Smoothness analysis: quantify interpolation path continuity
5. Extrapolation: what happens beyond the training distribution?

Uses saved model from train_natural.py and cached brain patterns from
category_analysis.py — no retraining or feature extraction needed.
"""

import os
import time
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import STL10
from torchvision import transforms

from cortexflow import BrainData
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from cortexflow.brain2img import Brain2Image

OUT = "train_outputs"
DEVICE = "cuda:0"
IMG_SIZE = 128
N_VOXELS = 384
N_AVG = 4
IMG_CFG_SCALE = 1.5
N_INTERP_STEPS = 9  # interpolation points between endpoints

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# Category groupings for arithmetic
ANIMALS = ["bird", "cat", "deer", "dog", "horse", "monkey"]
VEHICLES = ["airplane", "car", "ship", "truck"]

# Interpolation pairs (chosen for scientific interest)
INTERP_PAIRS = [
    ("cat", "dog"),       # most similar brain reps (0.108)
    ("ship", "airplane"),  # both vehicles, different shape
    ("monkey", "ship"),    # most dissimilar brain reps (-0.108)
    ("horse", "car"),      # animal → vehicle
    ("bird", "airplane"),  # biological flyer → mechanical flyer
]


def compute_ssim(img1, img2):
    """SSIM between two (C,H,W) tensors in [0,1]."""
    x, y = img1.flatten(), img2.flatten()
    mu_x, mu_y = x.mean(), y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * cov + c2)) /
            ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))).item()


def reconstruct_from_brain(model, brain_vec, W_ridge, IMG_LATENT_SHAPE):
    """Reconstruct image from a brain pattern vector."""
    brain_data = BrainData(voxels=brain_vec.unsqueeze(0).to(DEVICE))
    bg, bt = model.encode_brain(brain_data)
    shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

    # Linear baseline
    lin_pred = (brain_vec.unsqueeze(0).cpu() @ W_ridge).view(1, *IMG_LATENT_SHAPE)

    # DiT multi-sample average
    latent_sum = None
    for s in range(N_AVG):
        torch.manual_seed(s)
        z = model.flow_matcher.sample(
            model.dit, shape_1, bg, bt, num_steps=50, cfg_scale=IMG_CFG_SCALE,
        )
        z = z + lin_pred.to(DEVICE)
        latent_sum = z if latent_sum is None else latent_sum + z
    avg_z = latent_sum / N_AVG

    with torch.no_grad():
        recon = model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()
    return recon


# ═══════════════════════════════════════════════════════════════
# 1. LOAD MODEL + DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("BRAIN SPACE INTERPOLATION & MANIPULATION STUDY")
print("=" * 70)

print("\nLoading saved model...")
saved_data = torch.load(
    os.path.join(OUT, "natural_brain_data.pt"), map_location="cpu"
)
brain_mean = saved_data["brain_mean"]
pca_V = saved_data["pca_V"]
brain_std = saved_data["brain_std"]
train_vjepa_feats = saved_data["vjepa_features"]

model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=256, depth=8, num_heads=8, cond_dim=256),
    vae_config=VAEConfig(hidden_dims=[64, 128, 256], latent_channels=8),
    flow_config=FlowConfig(),
).to(DEVICE)
state_dict = torch.load(
    os.path.join(OUT, "brain2img_natural.pt"), map_location="cpu"
)
model.load_state_dict(state_dict)
model.eval()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Model: {n_params:.1f}M params")

# Load TRIBE v2 weights for brain projection
print("Loading TRIBE v2 brain mapping head...")
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("facebook/tribev2", "best.ckpt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
tribe_state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
del ckpt

video_proj_w = tribe_state["projectors.video.weight"]
video_proj_b = tribe_state["projectors.video.bias"]
low_rank_w = tribe_state["low_rank_head.weight"]
pred_w = tribe_state["predictor.weights"]
pred_b = tribe_state["predictor.bias"]
combiner_w = tribe_state.get("combiner.weight")
combiner_b = tribe_state.get("combiner.bias")
combiner_norm_w = tribe_state.get("combiner.norm.weight")
combiner_norm_b = tribe_state.get("combiner.norm.bias")


def vjepa_to_brain(vjepa_features):
    """Project V-JEPA2 features → TRIBE v2 cortical activity → PCA."""
    with torch.no_grad():
        video_projected = F.linear(vjepa_features, video_proj_w, video_proj_b)
        B = video_projected.shape[0]
        combined = torch.cat([
            torch.zeros(B, 384), torch.zeros(B, 384), video_projected
        ], dim=1)
        if combiner_w is not None:
            combined = F.linear(combined, combiner_w, combiner_b)
            combined = F.gelu(combined)
            if combiner_norm_w is not None:
                combined = F.layer_norm(
                    combined, [1152], combiner_norm_w, combiner_norm_b
                )
        brain_2048 = F.linear(combined, low_rank_w)
        full_brain = torch.einsum("bh,shv->bv", brain_2048, pred_w) + pred_b.squeeze(0)
        centered = full_brain - brain_mean
        brain_pca = centered @ pca_V
        brain_patterns = brain_pca / brain_std
    return brain_patterns


# Re-derive ridge regression from training data
print("Re-deriving ridge regression...")
train_brain = vjepa_to_brain(train_vjepa_feats)
N_TRAIN_ORIG = 8000
train_brain_train = train_brain[:N_TRAIN_ORIG]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
])
unlabeled = STL10("/tmp/stl10", split="unlabeled", download=False, transform=transform)
torch.manual_seed(42)
orig_indices = torch.randperm(len(unlabeled))[:10000].tolist()
print("  Encoding training images through VAE...")
train_images = torch.stack([unlabeled[i][0] for i in orig_indices[:N_TRAIN_ORIG]])
train_images_gpu = train_images.to(DEVICE)
all_latents_train = []
with torch.no_grad():
    for i in range(0, N_TRAIN_ORIG, 64):
        z, _, _ = model.vae.encode(train_images_gpu[i:i + 64])
        all_latents_train.append(z.cpu())
all_latents_train = torch.cat(all_latents_train)
IMG_LATENT_SHAPE = all_latents_train.shape[1:]
print(f"  Latent shape: {list(IMG_LATENT_SHAPE)}")

X = train_brain_train.cpu()
Y = all_latents_train.flatten(1)
lam = 5.0
XtX = X.T @ X + lam * torch.eye(X.shape[1])
XtY = X.T @ Y
W_ridge = torch.linalg.solve(XtX, XtY)
print(f"  Ridge regression fitted (lambda={lam})")

del train_images, train_images_gpu, all_latents_train, train_brain
del tribe_state
torch.cuda.empty_cache()

# Load labeled test set + cached brain patterns
print("\nLoading labeled test set...")
labeled_test = STL10("/tmp/stl10", split="test", download=False, transform=transform)
N_TEST = len(labeled_test)
test_images = torch.stack([labeled_test[i][0] for i in range(N_TEST)])
test_labels = torch.tensor([labeled_test[i][1] for i in range(N_TEST)])

CACHE_PATH = os.path.join(OUT, "category_test_brain.pt")
cache = torch.load(CACHE_PATH, map_location="cpu")
test_brain = cache["test_brain"]
print(f"  Test brain patterns: {test_brain.shape} (cached)")

# Compute category centroids + select exemplars
print("\nComputing category centroids and selecting exemplars...")
cat_centroids = {}
cat_exemplars = {}  # best exemplar per category (closest to centroid)

for c in range(10):
    cat_name = STL10_CLASSES[c]
    mask = (test_labels == c)
    cat_brain = test_brain[mask]
    centroid = cat_brain.mean(dim=0)
    cat_centroids[cat_name] = centroid

    # Find image closest to centroid
    dists = torch.cdist(centroid.unsqueeze(0), cat_brain).squeeze(0)
    best_rel_idx = dists.argmin().item()
    # Map back to absolute index
    abs_indices = mask.nonzero(as_tuple=True)[0]
    best_abs_idx = abs_indices[best_rel_idx].item()
    cat_exemplars[cat_name] = {
        "idx": best_abs_idx,
        "brain": test_brain[best_abs_idx],
        "image": test_images[best_abs_idx],
    }
    print(f"  {cat_name:>10s}: centroid norm={centroid.norm():.3f}, "
          f"exemplar idx={best_abs_idx}")

# Group centroids
animal_mean = torch.stack([cat_centroids[c] for c in ANIMALS]).mean(dim=0)
vehicle_mean = torch.stack([cat_centroids[c] for c in VEHICLES]).mean(dim=0)
print(f"\n  Animal mean norm: {animal_mean.norm():.3f}")
print(f"  Vehicle mean norm: {vehicle_mean.norm():.3f}")
print(f"  Animal↔Vehicle cosine: {F.cosine_similarity(animal_mean.unsqueeze(0), vehicle_mean.unsqueeze(0)).item():.3f}")


# ═══════════════════════════════════════════════════════════════
# 2. CROSS-CATEGORY INTERPOLATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"CROSS-CATEGORY INTERPOLATION ({len(INTERP_PAIRS)} pairs × {N_INTERP_STEPS + 2} steps)")
print("=" * 70)

interp_results = {}
interp_recons = {}
t0 = time.time()

for pair_idx, (cat_a, cat_b) in enumerate(INTERP_PAIRS):
    pair_key = f"{cat_a}→{cat_b}"
    print(f"\n  [{pair_idx + 1}/{len(INTERP_PAIRS)}] {pair_key}")

    brain_a = cat_exemplars[cat_a]["brain"]
    brain_b = cat_exemplars[cat_b]["brain"]

    # Cosine similarity between endpoints
    endpoint_cos = F.cosine_similarity(
        brain_a.unsqueeze(0), brain_b.unsqueeze(0)
    ).item()
    print(f"    Brain distance: cos={endpoint_cos:.3f}")

    # Interpolation: alpha from 0 to 1 inclusive
    alphas = torch.linspace(0, 1, N_INTERP_STEPS + 2)
    recons = []
    frame_cos_scores = []

    for step, alpha in enumerate(alphas):
        brain_interp = (1 - alpha) * brain_a + alpha * brain_b
        recon = reconstruct_from_brain(model, brain_interp, W_ridge, IMG_LATENT_SHAPE)
        recons.append(recon)

        if step > 0:
            # Frame-to-frame smoothness
            prev = recons[-2]
            frame_cos = F.cosine_similarity(
                recon.flatten().unsqueeze(0), prev.flatten().unsqueeze(0)
            ).item()
            frame_cos_scores.append(frame_cos)

    # Smoothness metrics
    mean_frame_cos = np.mean(frame_cos_scores)
    min_frame_cos = np.min(frame_cos_scores)

    # Endpoint fidelity: recon at alpha=0 vs original image A, alpha=1 vs B
    fidelity_a = F.cosine_similarity(
        recons[0].flatten().unsqueeze(0),
        cat_exemplars[cat_a]["image"].flatten().unsqueeze(0),
    ).item()
    fidelity_b = F.cosine_similarity(
        recons[-1].flatten().unsqueeze(0),
        cat_exemplars[cat_b]["image"].flatten().unsqueeze(0),
    ).item()

    elapsed = time.time() - t0
    print(f"    Smoothness: mean_cos={mean_frame_cos:.3f}, "
          f"min_cos={min_frame_cos:.3f}")
    print(f"    Fidelity: A={fidelity_a:.3f}, B={fidelity_b:.3f}  ({elapsed:.0f}s)")

    interp_results[pair_key] = {
        "cat_a": cat_a,
        "cat_b": cat_b,
        "endpoint_brain_cos": round(endpoint_cos, 4),
        "mean_frame_cos": round(mean_frame_cos, 4),
        "min_frame_cos": round(min_frame_cos, 4),
        "fidelity_a": round(fidelity_a, 4),
        "fidelity_b": round(fidelity_b, 4),
        "frame_cos_scores": [round(s, 4) for s in frame_cos_scores],
    }
    interp_recons[pair_key] = {
        "recons": recons,
        "img_a": cat_exemplars[cat_a]["image"],
        "img_b": cat_exemplars[cat_b]["image"],
    }


# ═══════════════════════════════════════════════════════════════
# 3. BRAIN ARITHMETIC
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("BRAIN ARITHMETIC")
print("=" * 70)

# Operations:
# 1. cat - animal_mean + vehicle_mean → vehicle-like?
# 2. ship - vehicle_mean + animal_mean → animal-like?
# 3. horse - animal_mean + vehicle_mean → vehicle-like horse?
# 4. airplane - vehicle_mean + animal_mean → animal-like?
# 5. dog + (ship_centroid - car_centroid) → water-dog?

arithmetic_ops = [
    ("cat − animal + vehicle",
     lambda: cat_centroids["cat"] - animal_mean + vehicle_mean),
    ("ship − vehicle + animal",
     lambda: cat_centroids["ship"] - vehicle_mean + animal_mean),
    ("horse − animal + vehicle",
     lambda: cat_centroids["horse"] - vehicle_mean + vehicle_mean),
    ("airplane − vehicle + animal",
     lambda: cat_centroids["airplane"] - vehicle_mean + animal_mean),
    ("dog + (ship − car)",
     lambda: cat_centroids["dog"] + cat_centroids["ship"] - cat_centroids["car"]),
    ("bird + (airplane − bird)",  # should converge to airplane
     lambda: cat_centroids["bird"] + 0.5 * (cat_centroids["airplane"] - cat_centroids["bird"])),
]

arith_results = {}
arith_recons = {}
t0 = time.time()

for op_idx, (op_name, op_fn) in enumerate(arithmetic_ops):
    print(f"\n  [{op_idx + 1}/{len(arithmetic_ops)}] {op_name}")
    result_brain = op_fn()

    # What category is the result closest to?
    sims = {}
    for cat_name, centroid in cat_centroids.items():
        sim = F.cosine_similarity(
            result_brain.unsqueeze(0), centroid.unsqueeze(0)
        ).item()
        sims[cat_name] = sim

    nearest_cat = max(sims, key=sims.get)
    nearest_sim = sims[nearest_cat]

    # Reconstruct
    recon = reconstruct_from_brain(model, result_brain, W_ridge, IMG_LATENT_SHAPE)

    elapsed = time.time() - t0
    print(f"    Nearest category: {nearest_cat} (cos={nearest_sim:.3f})")
    print(f"    Top-3: {', '.join(f'{k}={v:.3f}' for k, v in sorted(sims.items(), key=lambda x: -x[1])[:3])}")
    print(f"    ({elapsed:.0f}s)")

    arith_results[op_name] = {
        "nearest_category": nearest_cat,
        "nearest_similarity": round(nearest_sim, 4),
        "all_similarities": {k: round(v, 4) for k, v in sims.items()},
    }
    arith_recons[op_name] = recon


# ═══════════════════════════════════════════════════════════════
# 4. DIMENSION MANIPULATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("TARGETED DIMENSION MANIPULATION")
print("=" * 70)

# Take a canonical image (ship exemplar — best reconstructed category)
# and manipulate top PCA dimensions to see what changes
base_cat = "ship"
base_brain = cat_exemplars[base_cat]["brain"].clone()
base_image = cat_exemplars[base_cat]["image"]

# Reconstruct baseline
base_recon = reconstruct_from_brain(model, base_brain, W_ridge, IMG_LATENT_SHAPE)

# Manipulate top 5 PCA dimensions: scale by ±2σ
N_DIMS_TO_MANIPULATE = 5
MANIPULATION_SCALES = [-2.0, -1.0, 0.0, 1.0, 2.0]

dim_results = {}
dim_recons = {}
t0 = time.time()

for d in range(N_DIMS_TO_MANIPULATE):
    print(f"\n  Dimension PC{d + 1} (base={base_brain[d].item():.3f})")
    dim_recons[d] = {}

    for scale in MANIPULATION_SCALES:
        manipulated = base_brain.clone()
        manipulated[d] = base_brain[d] + scale  # shift by scale * 1σ (already normalized)

        recon = reconstruct_from_brain(
            model, manipulated, W_ridge, IMG_LATENT_SHAPE
        )
        dim_recons[d][scale] = recon

        # Measure change from baseline
        change_cos = F.cosine_similarity(
            recon.flatten().unsqueeze(0),
            base_recon.flatten().unsqueeze(0),
        ).item()
        change_ssim = compute_ssim(recon, base_recon)

    elapsed = time.time() - t0
    # Compute total sensitivity: max change across scales
    changes = []
    for scale in [-2.0, 2.0]:
        r = dim_recons[d][scale]
        ch = 1.0 - F.cosine_similarity(
            r.flatten().unsqueeze(0), base_recon.flatten().unsqueeze(0)
        ).item()
        changes.append(ch)
    sensitivity = max(changes)
    dim_results[d] = {"sensitivity": round(sensitivity, 4)}
    print(f"    Sensitivity (1-cos at ±2σ): {sensitivity:.4f}  ({elapsed:.0f}s)")


# ═══════════════════════════════════════════════════════════════
# 5. EXTRAPOLATION STUDY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("EXTRAPOLATION — BEYOND TRAINING DISTRIBUTION")
print("=" * 70)

# Extend interpolation past alpha=1 to see what happens
cat_a, cat_b = "cat", "ship"  # most dissimilar pair
brain_a = cat_exemplars[cat_a]["brain"]
brain_b = cat_exemplars[cat_b]["brain"]

# alpha from -0.5 to 1.5 (extrapolation on both sides)
extrap_alphas = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
extrap_recons = []
extrap_norms = []

for alpha in extrap_alphas:
    brain_ext = (1 - alpha) * brain_a + alpha * brain_b
    norm = brain_ext.norm().item()
    extrap_norms.append(norm)
    recon = reconstruct_from_brain(model, brain_ext, W_ridge, IMG_LATENT_SHAPE)
    extrap_recons.append(recon)
    print(f"  α={alpha:+.2f}  brain_norm={norm:.2f}")

# Quality metrics for extrapolated points
extrap_results = {
    "pair": [cat_a, cat_b],
    "alphas": extrap_alphas,
    "brain_norms": [round(n, 3) for n in extrap_norms],
}


# ═══════════════════════════════════════════════════════════════
# 6. PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("GENERATING FIGURES")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Figure 1: Interpolation grid ──
fig1, axes1 = plt.subplots(
    len(INTERP_PAIRS), N_INTERP_STEPS + 4,
    figsize=(28, 3.2 * len(INTERP_PAIRS)),
)

for row, (cat_a, cat_b) in enumerate(INTERP_PAIRS):
    pair_key = f"{cat_a}→{cat_b}"
    data = interp_recons[pair_key]

    # Column 0: original A
    ax = axes1[row, 0]
    ax.imshow(data["img_a"].permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_title(f"orig {cat_a}", fontsize=8, fontweight="bold")
    ax.axis("off")

    # Columns 1 to N_INTERP_STEPS+2: interpolated reconstructions
    for step in range(N_INTERP_STEPS + 2):
        ax = axes1[row, step + 1]
        alpha = step / (N_INTERP_STEPS + 1)
        img = data["recons"][step].permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(f"α={alpha:.2f}", fontsize=7)
        ax.axis("off")

    # Last column: original B
    ax = axes1[row, -1]
    ax.imshow(data["img_b"].permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_title(f"orig {cat_b}", fontsize=8, fontweight="bold")
    ax.axis("off")

    # Row label
    res = interp_results[pair_key]
    axes1[row, 0].set_ylabel(
        f"{pair_key}\nsmooth={res['mean_frame_cos']:.3f}",
        fontsize=9, rotation=0, ha="right", va="center", labelpad=60,
    )

fig1.suptitle(
    "Brain Space Interpolation — Smooth Transitions Between Categories\n"
    "cortexflow × TRIBE v2 × STL-10   |   Linear interpolation in 384-dim brain space",
    fontsize=14, fontweight="bold", y=1.02,
)
fig1.tight_layout()
fig1.savefig(
    os.path.join(OUT, "brain_interpolation.png"),
    dpi=150, bbox_inches="tight",
)
plt.close(fig1)
print(f"  Saved: {OUT}/brain_interpolation.png")

# ── Figure 2: Brain arithmetic ──
n_ops = len(arithmetic_ops)
fig2, axes2 = plt.subplots(2, n_ops, figsize=(4 * n_ops, 8))

for col, (op_name, _) in enumerate(arithmetic_ops):
    recon = arith_recons[op_name]
    res = arith_results[op_name]

    # Row 0: reconstruction
    ax = axes2[0, col]
    ax.imshow(recon.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_title(f"{op_name}\n→ {res['nearest_category']}", fontsize=9,
                 fontweight="bold")
    ax.axis("off")

    # Row 1: similarity bar chart
    ax2 = axes2[1, col]
    cats = sorted(res["all_similarities"].keys())
    sims = [res["all_similarities"][c] for c in cats]
    colors = ["#e74c3c" if c == res["nearest_category"] else "#3498db" for c in cats]
    ax2.barh(range(len(cats)), sims, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels(cats, fontsize=7)
    ax2.set_xlabel("Cosine sim", fontsize=8)
    ax2.set_xlim(-0.3, 0.5)
    ax2.axvline(x=0, color="gray", linewidth=0.5)

fig2.suptitle(
    "Brain Arithmetic — Semantic Manipulation via Vector Operations\n"
    "category_centroid ± group_mean → reconstructed image",
    fontsize=14, fontweight="bold",
)
fig2.tight_layout()
fig2.savefig(
    os.path.join(OUT, "brain_arithmetic.png"),
    dpi=150, bbox_inches="tight",
)
plt.close(fig2)
print(f"  Saved: {OUT}/brain_arithmetic.png")

# ── Figure 3: Dimension manipulation ──
fig3, axes3 = plt.subplots(
    N_DIMS_TO_MANIPULATE + 1, len(MANIPULATION_SCALES) + 1,
    figsize=(3 * (len(MANIPULATION_SCALES) + 1), 3 * (N_DIMS_TO_MANIPULATE + 1)),
)

# Row 0: original + baseline recon
axes3[0, 0].imshow(base_image.permute(1, 2, 0).clamp(0, 1).numpy())
axes3[0, 0].set_title("Original", fontsize=10, fontweight="bold")
axes3[0, 0].axis("off")
for col in range(1, len(MANIPULATION_SCALES) + 1):
    axes3[0, col].axis("off")
axes3[0, len(MANIPULATION_SCALES) // 2 + 1].imshow(
    base_recon.permute(1, 2, 0).clamp(0, 1).numpy()
)
axes3[0, len(MANIPULATION_SCALES) // 2 + 1].set_title(
    "Baseline recon", fontsize=10, fontweight="bold"
)

# Rows 1+: dimension manipulations
for d in range(N_DIMS_TO_MANIPULATE):
    row = d + 1
    # Label column
    axes3[row, 0].text(
        0.5, 0.5, f"PC{d + 1}\nsens={dim_results[d]['sensitivity']:.3f}",
        ha="center", va="center", fontsize=10, fontweight="bold",
        transform=axes3[row, 0].transAxes,
    )
    axes3[row, 0].axis("off")

    for col, scale in enumerate(MANIPULATION_SCALES):
        ax = axes3[row, col + 1]
        img = dim_recons[d][scale].permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(f"{'−' if scale < 0 else '+'}{abs(scale):.0f}σ" if scale != 0 else "0",
                     fontsize=9)
        ax.axis("off")

fig3.suptitle(
    f"Targeted Brain Dimension Manipulation — {base_cat} exemplar\n"
    f"Shifting individual PCA components by ±1σ, ±2σ",
    fontsize=14, fontweight="bold",
)
fig3.tight_layout()
fig3.savefig(
    os.path.join(OUT, "brain_dimensions.png"),
    dpi=150, bbox_inches="tight",
)
plt.close(fig3)
print(f"  Saved: {OUT}/brain_dimensions.png")

# ── Figure 4: Extrapolation ──
fig4, axes4 = plt.subplots(1, len(extrap_alphas) + 2, figsize=(3 * (len(extrap_alphas) + 2), 3.5))

# Original images at ends
axes4[0].imshow(cat_exemplars[cat_a]["image"].permute(1, 2, 0).clamp(0, 1).numpy())
axes4[0].set_title(f"orig {cat_a}", fontsize=9, fontweight="bold")
axes4[0].axis("off")

for i, (alpha, recon) in enumerate(zip(extrap_alphas, extrap_recons)):
    ax = axes4[i + 1]
    ax.imshow(recon.permute(1, 2, 0).clamp(0, 1).numpy())
    in_range = 0 <= alpha <= 1
    color = "green" if in_range else "red"
    label = f"α={alpha:+.2f}" + ("" if in_range else " ⚠")
    ax.set_title(label, fontsize=9, color=color, fontweight="bold" if not in_range else "normal")
    ax.axis("off")

axes4[-1].imshow(cat_exemplars[cat_b]["image"].permute(1, 2, 0).clamp(0, 1).numpy())
axes4[-1].set_title(f"orig {cat_b}", fontsize=9, fontweight="bold")
axes4[-1].axis("off")

fig4.suptitle(
    f"Extrapolation Beyond Training Distribution ({cat_a} → {cat_b})\n"
    f"α < 0 and α > 1 go outside the interpolation range",
    fontsize=13, fontweight="bold",
)
fig4.tight_layout()
fig4.savefig(
    os.path.join(OUT, "brain_extrapolation.png"),
    dpi=150, bbox_inches="tight",
)
plt.close(fig4)
print(f"  Saved: {OUT}/brain_extrapolation.png")

# ── Figure 5: Smoothness analysis ──
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Per-pair smoothness curves
for pair_key, res in interp_results.items():
    scores = res["frame_cos_scores"]
    ax5a.plot(range(1, len(scores) + 1), scores, "o-", label=pair_key,
              linewidth=2, markersize=5)
ax5a.set_xlabel("Step", fontsize=11)
ax5a.set_ylabel("Frame-to-frame cosine similarity", fontsize=11)
ax5a.set_title("Interpolation Smoothness", fontsize=13, fontweight="bold")
ax5a.legend(fontsize=9)
ax5a.set_ylim(0.7, 1.0)
ax5a.grid(True, alpha=0.3)

# Panel B: Smoothness vs brain distance
brain_dists = [interp_results[k]["endpoint_brain_cos"] for k in interp_results]
mean_smooths = [interp_results[k]["mean_frame_cos"] for k in interp_results]
labels = list(interp_results.keys())

ax5b.scatter(brain_dists, mean_smooths, s=100, c="#e74c3c", zorder=5)
for i, lbl in enumerate(labels):
    ax5b.annotate(lbl, (brain_dists[i], mean_smooths[i]),
                  textcoords="offset points", xytext=(5, 5), fontsize=8)
ax5b.set_xlabel("Endpoint brain cosine similarity", fontsize=11)
ax5b.set_ylabel("Mean interpolation smoothness", fontsize=11)
ax5b.set_title("Smoothness vs Brain Distance", fontsize=13, fontweight="bold")
ax5b.grid(True, alpha=0.3)

fig5.tight_layout()
fig5.savefig(
    os.path.join(OUT, "brain_smoothness.png"),
    dpi=150, bbox_inches="tight",
)
plt.close(fig5)
print(f"  Saved: {OUT}/brain_smoothness.png")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SUMMARY — BRAIN SPACE INTERPOLATION & MANIPULATION STUDY")
print("=" * 70)

print("\n  Interpolation (cross-category):")
for pair_key, res in interp_results.items():
    print(f"    {pair_key:>25s}: smoothness={res['mean_frame_cos']:.3f}  "
          f"brain_cos={res['endpoint_brain_cos']:.3f}")

overall_smoothness = np.mean([r["mean_frame_cos"] for r in interp_results.values()])
print(f"    {'OVERALL':>25s}: {overall_smoothness:.3f}")

print("\n  Brain Arithmetic:")
for op_name, res in arith_results.items():
    print(f"    {op_name:>35s} → {res['nearest_category']:>10s} "
          f"(cos={res['nearest_similarity']:.3f})")

print("\n  Dimension Sensitivity (ship exemplar):")
for d in range(N_DIMS_TO_MANIPULATE):
    print(f"    PC{d + 1}: sensitivity={dim_results[d]['sensitivity']:.4f}")

print(f"\n  Extrapolation ({cat_a}→{cat_b}):")
for alpha, norm in zip(extrap_alphas, extrap_norms):
    in_range = "✓" if 0 <= alpha <= 1 else "⚠"
    print(f"    α={alpha:+.2f}: brain_norm={norm:.2f} {in_range}")

# Save all results
all_results = {
    "experiment": "brain_interpolation",
    "model": "cortexflow × TRIBE v2",
    "dataset": "STL-10 labeled test",
    "n_interp_steps": N_INTERP_STEPS,
    "overall_smoothness": round(overall_smoothness, 4),
    "interpolation": interp_results,
    "arithmetic": arith_results,
    "dimension_sensitivity": {f"PC{d + 1}": dim_results[d] for d in range(N_DIMS_TO_MANIPULATE)},
    "extrapolation": extrap_results,
}

with open(os.path.join(OUT, "interpolation_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Saved: {OUT}/interpolation_results.json")

print(f"\n  Figures:")
print(f"    {OUT}/brain_interpolation.png  — cross-category interpolation grid")
print(f"    {OUT}/brain_arithmetic.png     — semantic arithmetic results")
print(f"    {OUT}/brain_dimensions.png     — per-dimension manipulation")
print(f"    {OUT}/brain_extrapolation.png  — extrapolation beyond training distribution")
print(f"    {OUT}/brain_smoothness.png     — interpolation smoothness analysis")

print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)
