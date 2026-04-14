"""Category-level analysis of natural image brain decoding.

Loads the trained cortexflow model from train_natural.py and evaluates
reconstruction quality per STL-10 category (10 classes: airplane, bird,
car, cat, deer, dog, horse, monkey, ship, truck).

Analyses:
1. Per-category reconstruction quality (cos, SSIM) + DiT vs linear gap
2. Brain representation structure (t-SNE colored by category)
3. Category separability (kNN classification from brain patterns)
4. Category similarity matrix (which categories share brain representations)
5. PCA dimension scaling (how many brain dims are needed?)
6. Best/worst reconstructions per category

Uses saved model from train_natural.py — no retraining needed.
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
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, silhouette_score

from cortexflow import BrainData
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from cortexflow.brain2img import Brain2Image

OUT = "train_outputs"
DEVICE = "cuda:0"
IMG_SIZE = 128
N_VOXELS = 384
VJEPA_LAYERS = [20, 30]
N_AVG = 4
IMG_CFG_SCALE = 1.5
N_PER_CAT = 50          # images to reconstruct per category
N_TRAIN_ORIG = 8000      # original training split size

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
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


# ═══════════════════════════════════════════════════════════════
# 1. LOAD TRAINED MODEL + SAVED PCA DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("CATEGORY-LEVEL ANALYSIS — CORTEXFLOW × TRIBE v2 × STL-10")
print("=" * 70)

print("\nLoading saved model and PCA projection...")
saved_data = torch.load(
    os.path.join(OUT, "natural_brain_data.pt"), map_location="cpu"
)
brain_mean = saved_data["brain_mean"]
pca_V = saved_data["pca_V"]
brain_std = saved_data["brain_std"]
train_vjepa_feats = saved_data["vjepa_features"]  # (10000, 2816)

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
print(f"  Model loaded: {n_params:.1f}M params")

# ═══════════════════════════════════════════════════════════════
# 2. LOAD TRIBE v2 WEIGHTS (for projecting new images)
# ═══════════════════════════════════════════════════════════════
print("\nLoading TRIBE v2 brain mapping head...")
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
                combined = F.layer_norm(combined, [1152], combiner_norm_w, combiner_norm_b)
        brain_2048 = F.linear(combined, low_rank_w)
        full_brain = torch.einsum("bh,shv->bv", brain_2048, pred_w) + pred_b.squeeze(0)
        centered = full_brain - brain_mean
        brain_pca = centered @ pca_V
        brain_patterns = brain_pca / brain_std
    return brain_patterns


# ═══════════════════════════════════════════════════════════════
# 3. RE-DERIVE TRAINING DATA → RIDGE REGRESSION
# ═══════════════════════════════════════════════════════════════
print("\nRe-deriving training ridge regression from saved V-JEPA2 features...")
train_brain = vjepa_to_brain(train_vjepa_feats)
train_brain_train_split = train_brain[:N_TRAIN_ORIG]  # first 8000

# Re-load training images (same seed/indices as train_natural.py)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
])
unlabeled = STL10("/tmp/stl10", split="unlabeled", download=False, transform=transform)
torch.manual_seed(42)
orig_indices = torch.randperm(len(unlabeled))[:10000].tolist()
print(f"  Loading {N_TRAIN_ORIG} training images (same indices as train_natural.py)...")
train_images = torch.stack([unlabeled[i][0] for i in orig_indices[:N_TRAIN_ORIG]])
train_images_gpu = train_images.to(DEVICE)

# Encode through VAE to get latents
print("  Encoding training images through VAE...")
all_latents_train = []
with torch.no_grad():
    for i in range(0, N_TRAIN_ORIG, 64):
        z, _, _ = model.vae.encode(train_images_gpu[i:i + 64])
        all_latents_train.append(z.cpu())
all_latents_train = torch.cat(all_latents_train)
IMG_LATENT_SHAPE = all_latents_train.shape[1:]
print(f"  Latent shape: {list(IMG_LATENT_SHAPE)}")

# Ridge regression
print("  Fitting ridge regression...")
X = train_brain_train_split.cpu()
Y = all_latents_train.flatten(1)
lam = 5.0
XtX = X.T @ X + lam * torch.eye(X.shape[1])
XtY = X.T @ Y
W_ridge = torch.linalg.solve(XtX, XtY)

# Verify on training data
with torch.no_grad():
    train_lin_preds = (X @ W_ridge).view(-1, *IMG_LATENT_SHAPE)
    train_lin_cos = F.cosine_similarity(
        train_lin_preds.flatten(1), all_latents_train.flatten(1)
    ).mean().item()
print(f"  Ridge regression train cos: {train_lin_cos:.3f}")

del train_images, train_images_gpu, all_latents_train, train_brain
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# 4. LOAD LABELED TEST SET + EXTRACT FEATURES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("EXTRACTING BRAIN FEATURES FOR 8000 LABELED TEST IMAGES")
print("=" * 70)

labeled_test = STL10("/tmp/stl10", split="test", download=False, transform=transform)
N_TEST = len(labeled_test)
print(f"  Labeled test: {N_TEST} images, 10 categories")

test_images = torch.stack([labeled_test[i][0] for i in range(N_TEST)])
test_labels = torch.tensor([labeled_test[i][1] for i in range(N_TEST)])

CACHE_PATH = os.path.join(OUT, "category_test_brain.pt")

if os.path.exists(CACHE_PATH):
    print("  Loading cached brain features...")
    cache = torch.load(CACHE_PATH, map_location="cpu")
    test_brain = cache["test_brain"].to(DEVICE)
    print(f"  Test brain patterns: {test_brain.shape} (cached)")
else:
    # V-JEPA2 extraction
    print(f"\n  Loading V-JEPA2 (ViT-Giant)...")
    from transformers import AutoModel

    vjepa = AutoModel.from_pretrained(
        "facebook/vjepa2-vitg-fpc64-256", dtype=torch.float16,
    ).to(DEVICE).eval()

    print(f"  Extracting V-JEPA2 features for {N_TEST} images...")
    test_vjepa_feats = []
    t0 = time.time()
    VJEPA_BATCH = 16
    for batch_start in range(0, N_TEST, VJEPA_BATCH):
        batch_end = min(batch_start + VJEPA_BATCH, N_TEST)
        batch_imgs = F.interpolate(
            test_images[batch_start:batch_end],
            size=(256, 256), mode="bilinear", align_corners=False,
        )
        video_input = batch_imgs.unsqueeze(1).half().to(DEVICE)
        with torch.no_grad():
            out = vjepa(video_input, output_hidden_states=True)
            feats = []
            for layer_idx in VJEPA_LAYERS:
                h = out.hidden_states[layer_idx].mean(dim=1)
                feats.append(h.float().cpu())
            test_vjepa_feats.append(torch.cat(feats, dim=1))
        if (batch_start + VJEPA_BATCH) % 320 == 0 or batch_end == N_TEST:
            elapsed = time.time() - t0
            print(f"    {batch_end}/{N_TEST} ({elapsed:.1f}s)")

    test_vjepa_feats = torch.cat(test_vjepa_feats, dim=0)
    print(f"  V-JEPA2 features: {test_vjepa_feats.shape} ({time.time() - t0:.1f}s)")

    del vjepa
    torch.cuda.empty_cache()

    # Project through TRIBE v2 → PCA
    print("  Projecting through TRIBE v2 → PCA...")
    test_brain = vjepa_to_brain(test_vjepa_feats).to(DEVICE)
    print(f"  Test brain patterns: {test_brain.shape}")

    # Cache for re-runs
    torch.save({"test_brain": test_brain.cpu()}, CACHE_PATH)
    print(f"  Cached to {CACHE_PATH}")
    del test_vjepa_feats

del tribe_state
torch.cuda.empty_cache()

# Linear predictions for test
test_lin_preds = (test_brain.cpu() @ W_ridge).view(-1, *IMG_LATENT_SHAPE)

# ═══════════════════════════════════════════════════════════════
# 5. BRAIN REPRESENTATION ANALYSIS (fast — all 8000 images)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("BRAIN REPRESENTATION ANALYSIS")
print("=" * 70)

brain_cpu = test_brain.cpu().numpy()
labels_np = test_labels.numpy()

# 5a. t-SNE
print("\n  Computing t-SNE (8000 brain patterns)...")
t0 = time.time()
tsne = TSNE(n_components=2, perplexity=50, max_iter=1000, random_state=42)
tsne_coords = tsne.fit_transform(brain_cpu)
print(f"  t-SNE done ({time.time() - t0:.1f}s)")

# 5b. Silhouette score (category separability)
print("  Computing silhouette score...")
sil = silhouette_score(brain_cpu, labels_np, sample_size=2000, random_state=42)
print(f"  Silhouette score: {sil:.3f}")

# 5c. kNN classification from brain patterns
print("  kNN classification (5-NN)...")
n_half = N_TEST // 2
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(brain_cpu[:n_half], labels_np[:n_half])
knn_preds = knn.predict(brain_cpu[n_half:])
knn_acc = (knn_preds == labels_np[n_half:]).mean()
print(f"  Brain-space kNN accuracy: {knn_acc:.3f} (chance=0.100)")

# Confusion matrix
cm = confusion_matrix(labels_np[n_half:], knn_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
print(f"  Per-category accuracy:")
for i, name in enumerate(STL10_CLASSES):
    print(f"    {name:>10s}: {cm_norm[i, i]:.3f}")

# 5d. Category similarity matrix (brain space)
print("\n  Computing category similarity matrix...")
cat_centroids = []
for c in range(10):
    mask = labels_np == c
    centroid = brain_cpu[mask].mean(axis=0)
    cat_centroids.append(centroid)
cat_centroids = np.stack(cat_centroids)
cat_sim = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        cat_sim[i, j] = np.dot(cat_centroids[i], cat_centroids[j]) / (
            np.linalg.norm(cat_centroids[i]) * np.linalg.norm(cat_centroids[j]) + 1e-8
        )
print("  Top-5 most similar category pairs:")
pairs = []
for i in range(10):
    for j in range(i + 1, 10):
        pairs.append((cat_sim[i, j], STL10_CLASSES[i], STL10_CLASSES[j]))
pairs.sort(reverse=True)
for sim, a, b in pairs[:5]:
    print(f"    {a:>10s} ↔ {b:<10s}: {sim:.3f}")
print("  Top-5 most dissimilar:")
for sim, a, b in pairs[-5:]:
    print(f"    {a:>10s} ↔ {b:<10s}: {sim:.3f}")

# ═══════════════════════════════════════════════════════════════
# 6. PER-CATEGORY RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PER-CATEGORY RECONSTRUCTION ({N_PER_CAT} per category, {10 * N_PER_CAT} total)")
print("=" * 70)

category_results = {}
all_recons = {}  # store for best/worst visualization
t0 = time.time()

for cat_idx, cat_name in enumerate(STL10_CLASSES):
    cat_mask = (test_labels == cat_idx).nonzero(as_tuple=True)[0]
    selected = cat_mask[:N_PER_CAT]

    cos_scores, ssim_scores = [], []
    lin_cos_scores, lin_ssim_scores = [], []
    recons_for_cat = []

    for count, i in enumerate(selected):
        i = i.item()
        brain = BrainData(voxels=test_brain[i:i + 1])
        bg, bt = model.encode_brain(brain)
        shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

        latent_sum = None
        for s in range(N_AVG):
            torch.manual_seed(s)
            z = model.flow_matcher.sample(
                model.dit, shape_1, bg, bt, num_steps=50, cfg_scale=IMG_CFG_SCALE,
            )
            z = z + test_lin_preds[i:i + 1].to(DEVICE)
            latent_sum = z if latent_sum is None else latent_sum + z
        avg_z = latent_sum / N_AVG
        with torch.no_grad():
            recon = model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()

        target = test_images[i]
        cos = F.cosine_similarity(
            recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
        ).item()
        ssim = compute_ssim(recon, target)
        cos_scores.append(cos)
        ssim_scores.append(ssim)
        recons_for_cat.append({"idx": i, "cos": cos, "ssim": ssim, "recon": recon})

        # Linear baseline
        with torch.no_grad():
            lin_recon = model.vae.decode(test_lin_preds[i:i + 1].to(DEVICE))[0].clamp(0, 1).cpu()
        lin_cos = F.cosine_similarity(
            lin_recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
        ).item()
        lin_ssim = compute_ssim(lin_recon, target)
        lin_cos_scores.append(lin_cos)
        lin_ssim_scores.append(lin_ssim)

    elapsed = time.time() - t0
    mean_cos = np.mean(cos_scores)
    mean_ssim = np.mean(ssim_scores)
    mean_lin_ssim = np.mean(lin_ssim_scores)
    gap_ssim = mean_ssim - mean_lin_ssim
    print(f"  {cat_name:>10s}: cos={mean_cos:.3f}±{np.std(cos_scores):.3f}  "
          f"SSIM={mean_ssim:.3f}±{np.std(ssim_scores):.3f}  "
          f"gap={gap_ssim:+.3f}  ({elapsed:.0f}s)")

    category_results[cat_name] = {
        "cos": round(float(np.mean(cos_scores)), 4),
        "cos_std": round(float(np.std(cos_scores)), 4),
        "ssim": round(float(np.mean(ssim_scores)), 4),
        "ssim_std": round(float(np.std(ssim_scores)), 4),
        "lin_cos": round(float(np.mean(lin_cos_scores)), 4),
        "lin_ssim": round(float(np.mean(lin_ssim_scores)), 4),
        "gap_cos": round(float(np.mean(cos_scores) - np.mean(lin_cos_scores)), 4),
        "gap_ssim": round(float(np.mean(ssim_scores) - np.mean(lin_ssim_scores)), 4),
    }

    # Store best and worst
    recons_for_cat.sort(key=lambda x: x["ssim"])
    all_recons[cat_name] = {
        "worst": recons_for_cat[0],
        "best": recons_for_cat[-1],
    }

total_time = time.time() - t0
overall_cos = np.mean([r["cos"] for r in category_results.values()])
overall_ssim = np.mean([r["ssim"] for r in category_results.values()])
overall_lin_ssim = np.mean([r["lin_ssim"] for r in category_results.values()])
print(f"\n  Overall: cos={overall_cos:.3f}  SSIM={overall_ssim:.3f}  "
      f"gap={overall_ssim - overall_lin_ssim:+.3f}  ({total_time:.0f}s)")

# ═══════════════════════════════════════════════════════════════
# 7. PCA DIMENSION SCALING
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PCA DIMENSION SCALING")
print("=" * 70)

# For each k, restrict brain patterns to first k PCA dims, refit ridge, reconstruct
scaling_dims = [32, 64, 128, 192, 256, 384]
scaling_test_indices = []
for c in range(10):
    cat_mask = (test_labels == c).nonzero(as_tuple=True)[0]
    scaling_test_indices.extend(cat_mask[:10].tolist())

# Reload training brain full (all 384 dims already computed)
train_brain_full = vjepa_to_brain(train_vjepa_feats)[:N_TRAIN_ORIG]

# Reload training latents
print("  Re-encoding training images for scaling study...")
train_images_reload = torch.stack([unlabeled[i][0] for i in orig_indices[:N_TRAIN_ORIG]])
train_images_gpu = train_images_reload.to(DEVICE)
train_latents = []
with torch.no_grad():
    for i in range(0, N_TRAIN_ORIG, 64):
        z, _, _ = model.vae.encode(train_images_gpu[i:i + 64])
        train_latents.append(z.cpu())
train_latents = torch.cat(train_latents)
del train_images_reload, train_images_gpu
torch.cuda.empty_cache()

scaling_results = {}
t0 = time.time()

for k in scaling_dims:
    # Truncated brain patterns
    train_brain_k = train_brain_full[:, :k].cpu()
    test_brain_k = test_brain[:, :k].cpu()

    # Ridge regression on truncated
    Xk = train_brain_k
    Yk = train_latents.flatten(1)
    XtXk = Xk.T @ Xk + lam * torch.eye(k)
    XtYk = Xk.T @ Yk
    Wk = torch.linalg.solve(XtXk, XtYk)

    test_lin_k = (test_brain_k @ Wk).view(-1, *IMG_LATENT_SHAPE)

    # Reconstruct subset
    cos_scores, ssim_scores = [], []
    for i in scaling_test_indices:
        brain_k = BrainData(voxels=test_brain[i:i + 1, :k].to(DEVICE))

        # Rebuild brain encoder output at reduced dim — use linear only
        with torch.no_grad():
            lin_recon = model.vae.decode(test_lin_k[i:i + 1].to(DEVICE))[0].clamp(0, 1).cpu()
        target = test_images[i]
        cos = F.cosine_similarity(
            lin_recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
        ).item()
        ssim = compute_ssim(lin_recon, target)
        cos_scores.append(cos)
        ssim_scores.append(ssim)

    mean_cos = np.mean(cos_scores)
    mean_ssim = np.mean(ssim_scores)
    scaling_results[k] = {"cos": round(mean_cos, 4), "ssim": round(mean_ssim, 4)}
    elapsed = time.time() - t0
    print(f"  k={k:>3d} dims: cos={mean_cos:.3f}  SSIM={mean_ssim:.3f}  ({elapsed:.0f}s)")

del train_brain_full, train_latents
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# 8. PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("GENERATING FIGURES")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

fig = plt.figure(figsize=(24, 20))

# ── Panel A: Per-category bar chart ──
ax1 = fig.add_subplot(3, 3, (1, 2))
cats = STL10_CLASSES
x = np.arange(len(cats))
width = 0.25

cos_vals = [category_results[c]["cos"] for c in cats]
ssim_vals = [category_results[c]["ssim"] for c in cats]
gap_vals = [category_results[c]["gap_ssim"] for c in cats]
cos_err = [category_results[c]["cos_std"] for c in cats]
ssim_err = [category_results[c]["ssim_std"] for c in cats]

bars1 = ax1.bar(x - width, cos_vals, width, yerr=cos_err, capsize=3,
                label="Cosine", color="#3498db", alpha=0.85)
bars2 = ax1.bar(x, ssim_vals, width, yerr=ssim_err, capsize=3,
                label="SSIM", color="#2ecc71", alpha=0.85)
bars3 = ax1.bar(x + width, gap_vals, width,
                label="DiT gap (SSIM)", color="#e74c3c", alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(cats, rotation=30, ha="right", fontsize=10)
ax1.set_ylabel("Score", fontsize=11)
ax1.set_title("(a) Per-Category Reconstruction Quality", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.axhline(y=0, color="gray", linewidth=0.5)
ax1.set_ylim(-0.05, 1.05)

# ── Panel B: t-SNE ──
ax2 = fig.add_subplot(3, 3, 3)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for c in range(10):
    mask = labels_np == c
    ax2.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                c=[colors[c]], s=3, alpha=0.4, label=STL10_CLASSES[c])
ax2.set_title("(b) t-SNE of Brain Patterns", fontsize=13, fontweight="bold")
ax2.legend(fontsize=7, markerscale=3, loc="best", ncol=2)
ax2.set_xticks([])
ax2.set_yticks([])

# ── Panel C: Confusion matrix ──
ax3 = fig.add_subplot(3, 3, 4)
im = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax3.set_xticks(range(10))
ax3.set_yticks(range(10))
short_names = [n[:4] for n in STL10_CLASSES]
ax3.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax3.set_yticklabels(short_names, fontsize=8)
ax3.set_xlabel("Predicted", fontsize=10)
ax3.set_ylabel("True", fontsize=10)
ax3.set_title(f"(c) Brain-Space kNN (acc={knn_acc:.3f})", fontsize=13, fontweight="bold")
for i in range(10):
    for j in range(10):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax3.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                 fontsize=6, color=color)
plt.colorbar(im, ax=ax3, fraction=0.046)

# ── Panel D: Category similarity ──
ax4 = fig.add_subplot(3, 3, 5)
im2 = ax4.imshow(cat_sim, cmap="RdYlBu_r", vmin=-0.3, vmax=1.0)
ax4.set_xticks(range(10))
ax4.set_yticks(range(10))
ax4.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax4.set_yticklabels(short_names, fontsize=8)
ax4.set_title("(d) Category Similarity (Brain Space)", fontsize=13, fontweight="bold")
for i in range(10):
    for j in range(10):
        color = "white" if abs(cat_sim[i, j]) > 0.6 else "black"
        ax4.text(j, i, f"{cat_sim[i, j]:.2f}", ha="center", va="center",
                 fontsize=6, color=color)
plt.colorbar(im2, ax=ax4, fraction=0.046)

# ── Panel E: PCA scaling curve ──
ax5 = fig.add_subplot(3, 3, 6)
dims = sorted(scaling_results.keys())
sc_cos = [scaling_results[d]["cos"] for d in dims]
sc_ssim = [scaling_results[d]["ssim"] for d in dims]
ax5.plot(dims, sc_cos, "o-", color="#3498db", label="Cosine", linewidth=2, markersize=6)
ax5.plot(dims, sc_ssim, "s-", color="#2ecc71", label="SSIM", linewidth=2, markersize=6)
ax5.set_xlabel("PCA Dimensions", fontsize=11)
ax5.set_ylabel("Score", fontsize=11)
ax5.set_title("(e) Reconstruction vs Brain Dimensions", fontsize=13, fontweight="bold")
ax5.legend(fontsize=10)
ax5.set_xticks(dims)
ax5.grid(True, alpha=0.3)

# ── Panel F: Best and worst reconstructions per category ──
# Show 5 categories (best and worst overall)
cat_by_ssim = sorted(STL10_CLASSES, key=lambda c: category_results[c]["ssim"])
show_cats = cat_by_ssim[:2] + cat_by_ssim[4:6] + cat_by_ssim[-2:]  # worst2 + mid2 + best2
ax_recon = fig.add_subplot(3, 1, 3)
ax_recon.axis("off")

n_show = len(show_cats)
recon_axes = []
for row in range(n_show):
    for col in range(4):  # target, best, worst, linear-best
        ax = fig.add_axes([
            0.05 + col * 0.22,
            0.02 + (n_show - 1 - row) * (0.28 / n_show),
            0.18,
            0.24 / n_show,
        ])
        recon_axes.append((row, col, ax))

for row, cat_name in enumerate(show_cats):
    best = all_recons[cat_name]["best"]
    worst = all_recons[cat_name]["worst"]

    for col, (img_data, title) in enumerate([
        (test_images[best["idx"]], f"{cat_name} target"),
        (best["recon"], f"best SSIM={best['ssim']:.2f}"),
        (test_images[worst["idx"]], f"{cat_name} target"),
        (worst["recon"], f"worst SSIM={worst['ssim']:.2f}"),
    ]):
        ax = [a for r, c, a in recon_axes if r == row and c == col][0]
        img = img_data.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(title, fontsize=7, pad=2)
        ax.axis("off")

plt.suptitle(
    f"cortexflow × TRIBE v2 — Category-Level Brain Decoding Analysis\n"
    f"10 STL-10 categories | cos={overall_cos:.3f} SSIM={overall_ssim:.3f} "
    f"gap={overall_ssim - overall_lin_ssim:+.3f} | "
    f"kNN={knn_acc:.3f} silhouette={sil:.3f}",
    fontsize=14, fontweight="bold", y=0.98,
)
plt.savefig(os.path.join(OUT, "category_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT}/category_analysis.png")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SUMMARY — CATEGORY-LEVEL BRAIN DECODING ANALYSIS")
print("=" * 70)

print(f"\n  {'Category':>10s}  {'Cos':>8s}  {'SSIM':>8s}  {'Lin SSIM':>8s}  {'Gap':>8s}")
print(f"  {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
for cat in sorted(STL10_CLASSES, key=lambda c: category_results[c]["ssim"], reverse=True):
    r = category_results[cat]
    print(f"  {cat:>10s}  {r['cos']:>8.3f}  {r['ssim']:>8.3f}  {r['lin_ssim']:>8.3f}  {r['gap_ssim']:>+8.3f}")
print(f"  {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
print(f"  {'OVERALL':>10s}  {overall_cos:>8.3f}  {overall_ssim:>8.3f}  "
      f"{overall_lin_ssim:>8.3f}  {overall_ssim - overall_lin_ssim:>+8.3f}")

print(f"\n  Brain Representation Structure:")
print(f"    kNN accuracy (5-NN): {knn_acc:.3f} (chance=0.100)")
print(f"    Silhouette score: {sil:.3f}")

print(f"\n  PCA Scaling:")
for k in dims:
    print(f"    k={k:>3d}: cos={scaling_results[k]['cos']:.3f}  SSIM={scaling_results[k]['ssim']:.3f}")

# Save all results
results = {
    "experiment": "category_analysis",
    "model": "cortexflow × TRIBE v2",
    "dataset": "STL-10 labeled test",
    "n_test": N_TEST,
    "n_per_cat": N_PER_CAT,
    "overall_cos": round(overall_cos, 4),
    "overall_ssim": round(overall_ssim, 4),
    "overall_gap": round(overall_ssim - overall_lin_ssim, 4),
    "knn_accuracy": round(float(knn_acc), 4),
    "silhouette_score": round(float(sil), 4),
    "per_category": category_results,
    "scaling": scaling_results,
    "top_similar_pairs": [
        {"pair": [a, b], "sim": round(s, 4)} for s, a, b in pairs[:5]
    ],
    "top_dissimilar_pairs": [
        {"pair": [a, b], "sim": round(s, 4)} for s, a, b in pairs[-5:]
    ],
    "confusion_matrix": cm_norm.tolist(),
}

with open(os.path.join(OUT, "category_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: {OUT}/category_results.json")
print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)
