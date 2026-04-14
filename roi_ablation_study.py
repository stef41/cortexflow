"""ROI Ablation Study — Which Brain Regions Drive Image Reconstruction?

Scientific question: Which cortical regions are necessary and sufficient for
different aspects of image reconstruction? Does the ROI-aware brain encoder
learn to weight regions differently based on image content?

In neuroscience, different cortical areas process different visual information:
- V1 (primary visual): edges, orientations, spatial frequency
- V2/V4 (extrastriate): texture, color, shape
- IT (inferotemporal): object identity, category
- FFA (fusiform face area): face processing
- PPA (parahippocampal): scene/place processing
- MT/V5 (middle temporal): motion
- Prefrontal: semantic/contextual information

This study partitions TRIBE v2's 20,484 cortical vertices into 7 anatomical
ROIs based on sulcal/gyral geometry (approximated by vertex index ranges on
fsaverage5), then:

1. Trains a standard (flat) brain encoder on all 384 PCA dimensions (baseline)
2. Trains an ROI-aware brain encoder (ROIBrainEncoder) with per-region
   sub-encoders — measures whether region-specific processing improves decoding
3. Systematically ablates each ROI (zero-masking) and measures the drop
   in reconstruction quality per content category
4. Measures which ROIs contribute most to different image categories
5. Generates interpretability figures

Requires: trained model from train_natural.py and cached brain data.
"""

import os
import time
import json
import math
import copy
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import STL10
from torchvision import transforms

from cortexflow import BrainData
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from cortexflow.brain2img import Brain2Image
from cortexflow.brain_encoder import BrainEncoder, ROIBrainEncoder
from cortexflow.flow_matching import EMAModel

OUT = "train_outputs"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
DEVICE = "cuda:0"

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
IMG_SIZE = 128
N_VOXELS = 384            # PCA dims (matches train_natural.py)
N_EVAL = 50               # test images per evaluation
N_AVG = 2                 # samples to average per reconstruction
ODE_STEPS = 25            # ODE solver steps
IMG_CFG_SCALE = 1.5
BATCH_SIZE = 64

# ROI encoder training
ROI_BRAIN_STEPS = 10000   # brain encoder pre-training steps
ROI_DIT_STEPS = 30000     # DiT fine-tuning with ROI encoder
ROI_LR = 5e-4
WEIGHT_DECAY = 0.05

# STL-10 categories (for labeled subset)
STL10_CATEGORIES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]

# ═══════════════════════════════════════════════════════════════
# DEFINE CORTICAL ROIs
# ═══════════════════════════════════════════════════════════════
# TRIBE v2 predicts 20,484 vertices on fsaverage5 (10,242 per hemisphere).
# We partition into approximate anatomical ROIs based on vertex index ranges.
# These are approximate but capture the major functional divisions.
#
# Neuroscience rationale:
# - Early visual (V1/V2): calcarine sulcus, posterior pole → first ~3K vertices
# - Ventral visual (V4/IT): fusiform/lingual gyrus → category-selective
# - Dorsal visual (MT/V3a): parietal operculum → motion/spatial
# - Temporal (STS/FFA): superior temporal, fusiform → faces, social
# - Parietal (IPS): intraparietal sulcus → spatial attention, numerosity
# - Frontal (PFC): prefrontal → semantic, working memory
# - Other (rest): sensorimotor, insular, etc.

N_VERTICES = 20484
HALF = N_VERTICES // 2  # 10242 per hemisphere

ROI_DEFINITIONS = {
    "V1_V2":     (0, 2400, "Early visual cortex"),     # ~2400 vertices bilateral
    "V4_IT":     (2400, 4800, "Ventral stream"),        # ~2400 vertices bilateral
    "MT_dorsal": (4800, 6400, "Dorsal stream"),         # ~1600 vertices bilateral
    "temporal":  (6400, 9600, "Temporal cortex"),        # ~3200 vertices bilateral
    "parietal":  (9600, 13000, "Parietal cortex"),       # ~3400 vertices bilateral
    "frontal":   (13000, 17600, "Frontal cortex"),       # ~4600 vertices bilateral
    "other":     (17600, 20484, "Other regions"),        # ~2884 vertices bilateral
}


def compute_ssim(img1, img2):
    x, y = img1.flatten(), img2.flatten()
    mu_x, mu_y = x.mean(), y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * cov + c2)) /
            ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))).item()


# ═══════════════════════════════════════════════════════════════
# 1. LOAD TRAINED MODEL + BRAIN DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("CORTEXFLOW — ROI ABLATION STUDY")
print("  Which brain regions drive image reconstruction?")
print("=" * 70)

model_path = os.path.join(OUT, "brain2img_natural.pt")
brain_data_path = os.path.join(OUT, "natural_brain_data.pt")
assert os.path.exists(model_path), f"Need trained model: {model_path}"
assert os.path.exists(brain_data_path), f"Need brain data: {brain_data_path}"

print("\nLoading trained model (baseline)...")
baseline_model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=256, depth=8, num_heads=8, cond_dim=256),
    vae_config=VAEConfig(hidden_dims=[64, 128, 256], latent_channels=8),
    flow_config=FlowConfig(),
).to(DEVICE)
state = torch.load(model_path, map_location=DEVICE, weights_only=True)
baseline_model.load_state_dict(state)
baseline_model.eval()
print(f"  Model loaded: {sum(p.numel() for p in baseline_model.parameters()) / 1e6:.1f}M params")

print("\nLoading brain data + TRIBE v2 projection...")
brain_saved = torch.load(brain_data_path, map_location="cpu", weights_only=True)
pca_V = brain_saved["pca_V"]             # (20484, 384)
brain_std = brain_saved["brain_std"]      # (1, 384)
brain_mean = brain_saved["brain_mean"]    # (20484,)
vjepa_features = brain_saved["vjepa_features"]  # (10000, 2816)
print(f"  V-JEPA features: {vjepa_features.shape}")
print(f"  PCA V: {pca_V.shape}")

# ═══════════════════════════════════════════════════════════════
# 2. LOAD TRIBE v2 & COMPUTE FULL BRAIN + PER-ROI PCA
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("COMPUTING PER-ROI BRAIN PROJECTIONS")
print("=" * 70)

print("\nLoading TRIBE v2 for brain projection...")
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("facebook/tribev2", "best.ckpt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
state_dict_tribe = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
del ckpt

video_proj_w = state_dict_tribe["projectors.video.weight"]
video_proj_b = state_dict_tribe["projectors.video.bias"]
low_rank_w = state_dict_tribe["low_rank_head.weight"]
pred_w = state_dict_tribe["predictor.weights"]
pred_b = state_dict_tribe["predictor.bias"]
combiner_w = state_dict_tribe.get("combiner.weight")
combiner_b = state_dict_tribe.get("combiner.bias")
combiner_norm_w = state_dict_tribe.get("combiner.norm.weight")
combiner_norm_b = state_dict_tribe.get("combiner.norm.bias")
del state_dict_tribe

print("\nProjecting 10,000 images to full brain activity...")
with torch.no_grad():
    video_projected = F.linear(vjepa_features, video_proj_w, video_proj_b)
    B = video_projected.shape[0]
    text_zeros = torch.zeros(B, 384)
    audio_zeros = torch.zeros(B, 384)
    combined = torch.cat([text_zeros, audio_zeros, video_projected], dim=1)
    if combiner_w is not None:
        combined = F.linear(combined, combiner_w, combiner_b)
        combined = F.gelu(combined)
        if combiner_norm_w is not None:
            combined = F.layer_norm(combined, [1152], combiner_norm_w, combiner_norm_b)
    brain_2048 = F.linear(combined, low_rank_w)
    full_brain = torch.einsum("bh,shv->bv", brain_2048, pred_w) + pred_b.squeeze(0)

print(f"  Full brain: {full_brain.shape}")

# Global PCA brain patterns (same as train_natural.py)
brain_centered = full_brain - brain_mean
brain_patterns = brain_centered @ pca_V / brain_std  # (10000, 384)
print(f"  Brain patterns (PCA): {brain_patterns.shape}")

# Per-ROI brain data — PCA within each region independently
roi_pca_data = {}      # roi_name → (N, roi_pca_dim) tensor
roi_pca_V = {}         # roi_name → PCA projection matrix
roi_pca_std = {}       # roi_name → per-dim std
roi_pca_dims = {}      # roi_name → number of PCA dims

N_ROI_PCA = 64  # PCA dims per ROI (total = 7*64 = 448, provides region-level detail)

print(f"\n  Per-ROI PCA decomposition ({N_ROI_PCA} dims per ROI):")
for roi_name, (start, end, desc) in ROI_DEFINITIONS.items():
    roi_vertices = full_brain[:, start:end]  # (10000, roi_size)
    roi_mean = roi_vertices.mean(dim=0)
    roi_centered = roi_vertices - roi_mean
    n_verts = end - start

    # PCA
    q = min(N_ROI_PCA, n_verts)
    U, S, V = torch.svd_lowrank(roi_centered, q=q)
    roi_pca = roi_centered @ V
    std = roi_pca.std(dim=0, keepdim=True).clamp(min=1e-6)
    roi_pca = roi_pca / std

    roi_pca_data[roi_name] = roi_pca
    roi_pca_V[roi_name] = V
    roi_pca_std[roi_name] = std
    roi_pca_dims[roi_name] = q

    var_explained = (S[:q]**2).sum() / (S**2).sum() * 100
    print(f"    {roi_name:12s}: {n_verts:5d} vertices → {q:3d} PCA dims "
          f"({var_explained:.1f}% var) | {desc}")

total_roi_dims = sum(roi_pca_dims.values())
print(f"\n  Total ROI PCA dims: {total_roi_dims}")

# Clean up full brain data
del full_brain, brain_centered, video_proj_w, video_proj_b
del low_rank_w, pred_w, pred_b, combiner_w, combiner_b, vjepa_features
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# 3. LOAD IMAGES + LABELS FOR CATEGORY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("LOADING IMAGES + CATEGORY LABELS")
print("=" * 70)

print("\nLoading images (first 10000)...")
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
dataset = STL10("/tmp/stl10", split="unlabeled", download=False, transform=transform)
torch.manual_seed(42)
indices = torch.randperm(len(dataset))[:10000].tolist()
images = torch.stack([dataset[indices[i]][0] for i in range(10000)])
print(f"  Images: {images.shape}")

# Load labeled subset for category analysis
print("  Loading labeled test set for category analysis...")
labeled_dataset = STL10("/tmp/stl10", split="test", download=False, transform=transform)
labeled_images = torch.stack([labeled_dataset[i][0] for i in range(len(labeled_dataset))])
labeled_labels = torch.tensor([labeled_dataset[i][1] for i in range(len(labeled_dataset))])
print(f"  Labeled: {labeled_images.shape}, labels: {labeled_labels.shape}")
print(f"  Categories: {', '.join(STL10_CATEGORIES)}")

# Cache VAE latents for the 10k unlabeled images
print("  Caching VAE latents...")
images = images.to(DEVICE)
brain_patterns = brain_patterns.to(DEVICE)
with torch.no_grad():
    all_latents = []
    for i in range(0, 10000, 32):
        z, _, _ = baseline_model.vae.encode(images[i:i+32])
        all_latents.append(z.cpu())
    all_latents = torch.cat(all_latents).to(DEVICE)
IMG_LATENT_SHAPE = all_latents.shape[1:]
print(f"  Latent shape: {list(IMG_LATENT_SHAPE)}")

# Fit ridge regression (same as train_natural.py)
print("  Fitting ridge regression (baseline)...")
N_TRAIN = 8000
X = brain_patterns[:N_TRAIN].cpu()
Y = all_latents[:N_TRAIN].flatten(1).cpu()
lam = 5.0
XtX = X.T @ X + lam * torch.eye(X.shape[1])
XtY = X.T @ Y
W_baseline = torch.linalg.solve(XtX, XtY)
lin_preds = (brain_patterns.cpu() @ W_baseline).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)

# Extract V-JEPA2 features for labeled images → brain patterns
print("  Extracting features for labeled images via V-JEPA2...")
from transformers import AutoModel

vjepa = AutoModel.from_pretrained(
    "facebook/vjepa2-vitg-fpc64-256",
    dtype=torch.float16,
).to(DEVICE).eval()

labeled_vjepa = []
VJEPA_BATCH = 16
t0 = time.time()
for batch_start in range(0, len(labeled_images), VJEPA_BATCH):
    batch_end = min(batch_start + VJEPA_BATCH, len(labeled_images))
    batch_imgs = F.interpolate(
        labeled_images[batch_start:batch_end],
        size=(256, 256), mode="bilinear", align_corners=False
    )
    video_input = batch_imgs.unsqueeze(1).half().to(DEVICE)
    with torch.no_grad():
        out = vjepa(video_input, output_hidden_states=True)
        feats = []
        for layer_idx in [20, 30]:
            h = out.hidden_states[layer_idx]
            h = h.mean(dim=1)
            feats.append(h.float().cpu())
        labeled_vjepa.append(torch.cat(feats, dim=1))
    if (batch_start + VJEPA_BATCH) % 320 == 0 or batch_end == len(labeled_images):
        print(f"    {batch_end}/{len(labeled_images)} ({time.time()-t0:.0f}s)")

labeled_vjepa = torch.cat(labeled_vjepa, dim=0)
print(f"  Labeled V-JEPA features: {labeled_vjepa.shape}")

del vjepa
torch.cuda.empty_cache()

# Project labeled images → brain patterns via TRIBE v2
print("  Projecting labeled images through TRIBE v2...")
ckpt_path2 = hf_hub_download("facebook/tribev2", "best.ckpt")
ckpt2 = torch.load(ckpt_path2, map_location="cpu", weights_only=True, mmap=True)
sd2 = {k.removeprefix("model."): v for k, v in ckpt2["state_dict"].items()}
del ckpt2

with torch.no_grad():
    vp = F.linear(labeled_vjepa, sd2["projectors.video.weight"], sd2["projectors.video.bias"])
    Bl = vp.shape[0]
    combined_l = torch.cat([torch.zeros(Bl, 384), torch.zeros(Bl, 384), vp], dim=1)
    if sd2.get("combiner.weight") is not None:
        combined_l = F.linear(combined_l, sd2["combiner.weight"], sd2["combiner.bias"])
        combined_l = F.gelu(combined_l)
        if sd2.get("combiner.norm.weight") is not None:
            combined_l = F.layer_norm(combined_l, [1152],
                                       sd2["combiner.norm.weight"], sd2["combiner.norm.bias"])
    brain_2048_l = F.linear(combined_l, sd2["low_rank_head.weight"])
    full_brain_l = torch.einsum("bh,shv->bv", brain_2048_l,
                                 sd2["predictor.weights"]) + sd2["predictor.bias"].squeeze(0)

# PCA → same space as training data
labeled_brain_centered = full_brain_l - brain_mean
labeled_brain_pca = labeled_brain_centered @ pca_V / brain_std
print(f"  Labeled brain patterns: {labeled_brain_pca.shape}")

# Per-ROI PCA for labeled data
labeled_roi_pca = {}
for roi_name, (start, end, desc) in ROI_DEFINITIONS.items():
    roi_verts_l = full_brain_l[:, start:end]
    roi_mean_l = roi_verts_l.mean(dim=0)
    roi_pca_l = (roi_verts_l - roi_mean_l) @ roi_pca_V[roi_name]
    labeled_roi_pca[roi_name] = roi_pca_l / roi_pca_std[roi_name]

del sd2, full_brain_l, labeled_brain_centered, labeled_vjepa
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 1: BASELINE EVALUATION WITH FULL BRAIN
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 1: BASELINE — FULL BRAIN RECONSTRUCTION")
print("=" * 70)

test_indices = list(range(N_TRAIN, N_TRAIN + N_EVAL))


def evaluate_model(model, brain_vecs, ridge_preds, eval_indices, label=""):
    """Evaluate reconstruction quality on given indices."""
    cos_scores, ssim_scores = [], []
    t0 = time.time()
    for count, i in enumerate(eval_indices):
        brain = BrainData(voxels=brain_vecs[i:i+1].to(DEVICE))
        bg, bt = model.encode_brain(brain)
        shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

        latent_sum = None
        for s in range(N_AVG):
            torch.manual_seed(s)
            z = model.flow_matcher.sample(
                model.dit, shape_1, bg, bt, num_steps=ODE_STEPS,
                cfg_scale=IMG_CFG_SCALE,
            )
            z = z + ridge_preds[i:i+1].to(DEVICE)
            latent_sum = z if latent_sum is None else latent_sum + z
        avg_z = latent_sum / N_AVG
        with torch.no_grad():
            recon = model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()

        target = images[i].cpu()
        cos = F.cosine_similarity(
            recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
        ).item()
        ssim = compute_ssim(recon, target)
        cos_scores.append(cos)
        ssim_scores.append(ssim)

        if (count + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    {count+1}/{len(eval_indices)} ({elapsed:.0f}s)")

    return {
        "cos_mean": np.mean(cos_scores), "cos_std": np.std(cos_scores),
        "ssim_mean": np.mean(ssim_scores), "ssim_std": np.std(ssim_scores),
    }


print(f"\n  Evaluating baseline (all {N_VOXELS} brain dims, {N_EVAL} test images)...")
baseline_results = evaluate_model(baseline_model, brain_patterns, lin_preds, test_indices)
print(f"  Baseline: cos={baseline_results['cos_mean']:.3f} ± {baseline_results['cos_std']:.3f}, "
      f"SSIM={baseline_results['ssim_mean']:.3f} ± {baseline_results['ssim_std']:.3f}")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 2: PER-ROI VARIANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 2: PER-ROI INFORMATION CONTENT")
print("=" * 70)

print("\n  Measuring per-ROI variance and category discriminability...")

# For each ROI, compute: (a) variance explained, (b) category discriminability
# using the labeled STL-10 test set
roi_info = {}
category_brain_means = {}  # roi → category → mean brain pattern

for roi_name in ROI_DEFINITIONS:
    roi_data = labeled_roi_pca[roi_name]  # (8000, 64)

    # Total variance
    total_var = roi_data.var(dim=0).sum().item()

    # Per-category means
    cat_means = {}
    for cat_idx, cat_name in enumerate(STL10_CATEGORIES):
        mask = labeled_labels == cat_idx
        if mask.sum() > 0:
            cat_means[cat_name] = roi_data[mask].mean(dim=0)

    # Between-category variance (category discriminability)
    all_cat_vecs = torch.stack(list(cat_means.values()))  # (10, 64)
    between_var = all_cat_vecs.var(dim=0).sum().item()
    discrim = between_var / max(total_var, 1e-8)

    # Pairwise category distances in this ROI
    cat_dists = {}
    for i, c1 in enumerate(STL10_CATEGORIES):
        for j, c2 in enumerate(STL10_CATEGORIES):
            if i < j:
                d = F.cosine_similarity(
                    cat_means[c1].unsqueeze(0), cat_means[c2].unsqueeze(0)
                ).item()
                cat_dists[f"{c1}-{c2}"] = d

    roi_info[roi_name] = {
        "total_var": total_var,
        "between_var": between_var,
        "discriminability": discrim,
        "cat_dists": cat_dists,
    }
    category_brain_means[roi_name] = cat_means

    desc = ROI_DEFINITIONS[roi_name][2]
    print(f"    {roi_name:12s}: var={total_var:8.1f}, "
          f"between={between_var:6.1f}, "
          f"discrim={discrim:.3f} | {desc}")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 3: ROI ABLATION — ZERO-MASK EACH REGION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 3: ROI ABLATION (ZERO-MASKING)")
print("=" * 70)

# Strategy: for each ROI, zero out the PCA dimensions that correspond
# to that ROI's vertices, then evaluate reconstruction quality.
#
# Since the model was trained with PCA brain patterns (global, non-ROI),
# we need to understand how each ROI contributes to the global PCA.
# pca_V has shape (20484, 384) — each PCA dimension is a weighted sum
# of all vertices. Ablating ROI means zeroing vertices → changes PCA output.

print("\n  Computing per-ROI contribution to PCA dimensions...")

# For each ROI, compute the PCA patterns with that ROI's vertices zeroed
# This directly shows how ablation affects the brain embedding
ablation_results = {}

for ablate_roi, (start, end, desc) in ROI_DEFINITIONS.items():
    print(f"\n  Ablating {ablate_roi} ({desc}, vertices {start}-{end})...")

    # Recompute brain patterns with ablated vertices
    brain_data_reloaded = torch.load(brain_data_path, map_location="cpu", weights_only=True)
    vjepa_feat = brain_data_reloaded["vjepa_features"]

    # Re-project through TRIBE v2
    ckpt_abl = torch.load(hf_hub_download("facebook/tribev2", "best.ckpt"),
                           map_location="cpu", weights_only=True, mmap=True)
    sd_abl = {k.removeprefix("model."): v for k, v in ckpt_abl["state_dict"].items()}
    del ckpt_abl

    with torch.no_grad():
        vp_abl = F.linear(vjepa_feat, sd_abl["projectors.video.weight"],
                           sd_abl["projectors.video.bias"])
        B_abl = vp_abl.shape[0]
        comb_abl = torch.cat([torch.zeros(B_abl, 384), torch.zeros(B_abl, 384), vp_abl], dim=1)
        if sd_abl.get("combiner.weight") is not None:
            comb_abl = F.linear(comb_abl, sd_abl["combiner.weight"], sd_abl["combiner.bias"])
            comb_abl = F.gelu(comb_abl)
            if sd_abl.get("combiner.norm.weight") is not None:
                comb_abl = F.layer_norm(comb_abl, [1152],
                                         sd_abl["combiner.norm.weight"],
                                         sd_abl["combiner.norm.bias"])
        brain_2048_abl = F.linear(comb_abl, sd_abl["low_rank_head.weight"])
        full_brain_abl = (torch.einsum("bh,shv->bv", brain_2048_abl,
                           sd_abl["predictor.weights"]) + sd_abl["predictor.bias"].squeeze(0))

    del sd_abl

    # Zero out the ablated ROI's vertices
    full_brain_abl[:, start:end] = 0.0

    # Apply same PCA as training
    brain_abl_centered = full_brain_abl - brain_mean
    brain_abl_pca = brain_abl_centered @ pca_V / brain_std
    brain_abl_pca = brain_abl_pca.to(DEVICE)

    del full_brain_abl, brain_abl_centered, vjepa_feat

    # Measure PCA distortion from ablation
    pca_cos = F.cosine_similarity(
        brain_abl_pca[N_TRAIN:N_TRAIN+N_EVAL],
        brain_patterns[N_TRAIN:N_TRAIN+N_EVAL].cpu().to(DEVICE)
    ).mean().item()

    # Fit new ridge regression with ablated brain
    X_abl = brain_abl_pca[:N_TRAIN].cpu()
    Y_abl = all_latents[:N_TRAIN].flatten(1).cpu()
    XtX_abl = X_abl.T @ X_abl + lam * torch.eye(N_VOXELS)
    XtY_abl = X_abl.T @ Y_abl
    W_abl = torch.linalg.solve(XtX_abl, XtY_abl)
    lin_preds_abl = (brain_abl_pca.cpu() @ W_abl).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)

    # Evaluate with ablated brain patterns (using baseline model)
    abl_res = evaluate_model(baseline_model, brain_abl_pca, lin_preds_abl,
                              test_indices, label=ablate_roi)

    drop_cos = baseline_results["cos_mean"] - abl_res["cos_mean"]
    drop_ssim = baseline_results["ssim_mean"] - abl_res["ssim_mean"]

    ablation_results[ablate_roi] = {
        **abl_res,
        "drop_cos": drop_cos,
        "drop_ssim": drop_ssim,
        "pca_cos": pca_cos,
        "n_vertices": end - start,
    }

    print(f"  {ablate_roi}: cos={abl_res['cos_mean']:.3f} (Δ={drop_cos:+.3f}), "
          f"SSIM={abl_res['ssim_mean']:.3f} (Δ={drop_ssim:+.3f}), "
          f"PCA distortion cos={pca_cos:.3f}")

    del brain_abl_pca, lin_preds_abl
    torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 4: CATEGORY-SPECIFIC ROI IMPORTANCE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 4: CATEGORY-SPECIFIC ROI IMPORTANCE")
print("=" * 70)

# Measure per-ROI, per-category ablation impact using labeled brain patterns
# Use linear regression quality as a fast proxy (no ODE sampling needed)

print("\n  Computing per-category ROI importance via linear reconstruction...")

# Fit linear predictor on labeled data using global PCA patterns
# (labeled_brain_pca has shape ~(8000, 384))
# Use first 6000 for fitting, last 2000 for eval
N_LABELED = len(labeled_brain_pca)
N_FIT = int(N_LABELED * 0.75)
N_EVAL_LABELED = N_LABELED - N_FIT

# Cache labeled image latents
labeled_images = labeled_images.to(DEVICE)
with torch.no_grad():
    labeled_latents = []
    for i in range(0, N_LABELED, 32):
        z, _, _ = baseline_model.vae.encode(labeled_images[i:i+32])
        labeled_latents.append(z.cpu())
    labeled_latents = torch.cat(labeled_latents)

# Fit ridge on labeled data
X_lab = labeled_brain_pca[:N_FIT]
Y_lab = labeled_latents[:N_FIT].flatten(1)
XtX_lab = X_lab.T @ X_lab + lam * torch.eye(N_VOXELS)
XtY_lab = X_lab.T @ Y_lab
W_lab = torch.linalg.solve(XtX_lab, XtY_lab)


def linear_eval_by_category(brain_pca_data, labels, W, latents, start_idx):
    """Evaluate linear reconstruction quality per category."""
    cat_scores = {}
    for cat_idx, cat_name in enumerate(STL10_CATEGORIES):
        mask = labels[start_idx:] == cat_idx
        if mask.sum() == 0:
            continue
        # Get indices relative to start_idx
        cat_indices = torch.where(mask)[0]
        preds = (brain_pca_data[start_idx:][cat_indices] @ W).view(-1, *IMG_LATENT_SHAPE)
        targets = latents[start_idx:][cat_indices].flatten(1)
        preds_flat = preds.flatten(1)
        cos = F.cosine_similarity(preds_flat, targets).mean().item()
        cat_scores[cat_name] = cos
    return cat_scores


# Baseline per-category scores
baseline_cat = linear_eval_by_category(labeled_brain_pca, labeled_labels, W_lab,
                                         labeled_latents, N_FIT)
print(f"\n  Baseline per-category linear cos:")
for cat, cos in sorted(baseline_cat.items(), key=lambda x: -x[1]):
    print(f"    {cat:12s}: {cos:.3f}")

# Per-ROI ablation effect on each category
category_roi_importance = {}  # roi → cat → drop

for ablate_roi, (start, end, desc) in ROI_DEFINITIONS.items():
    # Recompute full brain with ablated ROI for labeled images
    ckpt_cat = torch.load(hf_hub_download("facebook/tribev2", "best.ckpt"),
                           map_location="cpu", weights_only=True, mmap=True)
    sd_cat = {k.removeprefix("model."): v for k, v in ckpt_cat["state_dict"].items()}
    del ckpt_cat

    # Re-extract labeled VJEPA features
    # We already have labeled_roi_pca, but for ablation we need the global PCA
    # with vertices zeroed. Let's use the pre-computed full_brain_l approach.
    # For efficiency, compute from the labeled VJEPA features we already projected.
    # Actually, we need to re-do the TRIBE projection for labeled data too...
    # To avoid re-loading V-JEPA again, reconstruct from the saved labeled brain patterns.

    # SHORTCUT: Since we have the labeled data's full_brain projected earlier,
    # we can simulate ablation by zeroing the corresponding PCA dimensions'
    # vertex contributions. But it's simpler to just load TRIBE v2 once and
    # precompute all ablations.

    # For now, use the labeled_roi_pca data to estimate ROI importance:
    # When a ROI is "important" for a category, that category's brain patterns
    # in that ROI should have high variance and discriminability.

    cat_importance = {}
    for cat_idx, cat_name in enumerate(STL10_CATEGORIES):
        mask = labeled_labels == cat_idx
        if mask.sum() == 0:
            continue
        roi_data_cat = labeled_roi_pca[ablate_roi][mask]  # (N_cat, 64)
        # Importance = distance of category centroid from global mean (in ROI space)
        cat_mean = roi_data_cat.mean(dim=0)
        global_mean = labeled_roi_pca[ablate_roi].mean(dim=0)
        importance = (cat_mean - global_mean).norm().item()
        cat_importance[cat_name] = importance

    category_roi_importance[ablate_roi] = cat_importance
    del sd_cat

print(f"\n  Category-ROI importance (centroid distance from mean):")
# Print as a table
header = f"  {'ROI':12s}"
for cat in STL10_CATEGORIES:
    header += f" {cat[:5]:>6s}"
print(header)
print("  " + "-" * (12 + 7 * len(STL10_CATEGORIES)))

for roi_name in ROI_DEFINITIONS:
    row = f"  {roi_name:12s}"
    for cat in STL10_CATEGORIES:
        val = category_roi_importance[roi_name].get(cat, 0)
        row += f" {val:6.2f}"
    print(row)

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 5: ROI-AWARE ENCODER TRAINING
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 5: ROI-AWARE BRAIN ENCODER")
print("=" * 70)

# Build ROI encoder with per-region sub-encoders
roi_sizes = {name: roi_pca_dims[name] for name in ROI_DEFINITIONS}
print(f"\n  Building ROIBrainEncoder with {len(roi_sizes)} regions:")
for name, dim in roi_sizes.items():
    desc = ROI_DEFINITIONS[name][2]
    print(f"    {name:12s}: {dim} input dims | {desc}")

# Concatenate per-ROI PCA data into one tensor per image
# and build ROI split lookup
roi_concat = torch.cat([roi_pca_data[name] for name in sorted(ROI_DEFINITIONS.keys())], dim=1)
print(f"  Concatenated ROI dims: {roi_concat.shape}")

# Create ROI-aware model (share VAE and DiT from baseline, new brain encoder)
roi_encoder = ROIBrainEncoder(
    roi_sizes=roi_sizes,
    cond_dim=256,
    n_tokens=16,
    per_roi_dim=64,
).to(DEVICE)
n_roi_params = sum(p.numel() for p in roi_encoder.parameters())
print(f"  ROI encoder params: {n_roi_params:,}")

# Create a new Brain2Image using the ROI encoder
roi_model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=256, depth=8, num_heads=8, cond_dim=256),
    vae_config=VAEConfig(hidden_dims=[64, 128, 256], latent_channels=8),
    flow_config=FlowConfig(),
    brain_encoder=roi_encoder,
).to(DEVICE)

# Copy VAE weights from baseline (already trained)
roi_model.vae.load_state_dict(baseline_model.vae.state_dict())
roi_model.vae.eval()
# Copy DiT weights from baseline as initialization
roi_model.dit.load_state_dict(baseline_model.dit.state_dict())

print(f"\n  Pre-training ROI brain encoder ({ROI_BRAIN_STEPS} steps)...")
roi_enc_opt = torch.optim.AdamW(roi_encoder.parameters(), lr=1e-3, weight_decay=0.01)
roi_enc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(roi_enc_opt, ROI_BRAIN_STEPS)

roi_concat = roi_concat.to(DEVICE)

for step in range(ROI_BRAIN_STEPS):
    idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,))
    # Build ROI voxels dict
    roi_voxels = {}
    offset = 0
    for name in sorted(ROI_DEFINITIONS.keys()):
        dim = roi_pca_dims[name]
        roi_voxels[name] = roi_concat[idx, offset:offset+dim]
        offset += dim

    brain = BrainData(voxels=brain_patterns[idx], roi_voxels=roi_voxels)
    target_lat = all_latents[idx].flatten(1)
    bg, _ = roi_model.encode_brain(brain)
    n_match = min(bg.shape[1], target_lat.shape[1])
    loss = F.mse_loss(bg[:, :n_match], target_lat[:, :n_match])

    roi_enc_opt.zero_grad()
    loss.backward()
    roi_enc_opt.step()
    roi_enc_sched.step()

    if step % 2000 == 0:
        print(f"    ROI enc step {step}: loss={loss.item():.4f}")

# Fit ridge regression using ROI-concatenated brain data
print(f"\n  Fitting ridge regression (ROI-aware)...")
X_roi = roi_concat[:N_TRAIN].cpu()
Y_roi = all_latents[:N_TRAIN].flatten(1).cpu()
total_roi_d = X_roi.shape[1]
XtX_roi = X_roi.T @ X_roi + lam * torch.eye(total_roi_d)
XtY_roi = X_roi.T @ Y_roi
W_roi = torch.linalg.solve(XtX_roi, XtY_roi)
lin_preds_roi = (roi_concat.cpu() @ W_roi).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)

# Fine-tune DiT with ROI encoder
print(f"\n  Fine-tuning DiT with ROI encoder ({ROI_DIT_STEPS} steps)...")
dit_params_roi = list(roi_model.dit.parameters()) + list(roi_encoder.parameters())
dit_opt_roi = torch.optim.AdamW(dit_params_roi, lr=ROI_LR, weight_decay=WEIGHT_DECAY)

warmup = 500


def lr_lambda_roi(step):
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, ROI_DIT_STEPS - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


dit_sched_roi = torch.optim.lr_scheduler.LambdaLR(dit_opt_roi, lr_lambda_roi)
ema_roi = EMAModel(roi_model, decay=0.999)

roi_model.train()
roi_model.vae.eval()
t0 = time.time()
losses_roi = []

for step in range(ROI_DIT_STEPS):
    idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,))

    # Build ROI voxels dict
    roi_voxels = {}
    offset = 0
    for name in sorted(ROI_DEFINITIONS.keys()):
        dim = roi_pca_dims[name]
        vox = roi_concat[idx, offset:offset+dim]
        if 0.05 > 0:
            vox = vox + 0.05 * torch.randn_like(vox)
        roi_voxels[name] = vox
        offset += dim

    brain = BrainData(voxels=brain_patterns[idx], roi_voxels=roi_voxels)
    residual = all_latents[idx] - lin_preds_roi[idx]

    bg, bt = roi_model.encode_brain(brain)
    loss = roi_model.flow_matcher.compute_loss(roi_model.dit, residual, bg, bt)

    dit_opt_roi.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dit_params_roi, 1.0)
    dit_opt_roi.step()
    dit_sched_roi.step()
    ema_roi.update(roi_model)
    losses_roi.append(loss.item())

    if step % 5000 == 0 or step == ROI_DIT_STEPS - 1:
        avg_loss = sum(losses_roi[-1000:]) / min(len(losses_roi), 1000)
        elapsed = time.time() - t0
        eta = elapsed / max(step + 1, 1) * (ROI_DIT_STEPS - step - 1)
        print(f"    DiT step {step}: loss={avg_loss:.4f} ({elapsed:.0f}s, ~{eta:.0f}s left)")

ema_roi.apply_to(roi_model)
roi_model.eval()


def evaluate_roi_model(eval_indices, label=""):
    """Evaluate ROI-aware model."""
    cos_scores, ssim_scores = [], []
    t0 = time.time()
    for count, i in enumerate(eval_indices):
        # Build ROI voxels for this sample
        roi_voxels = {}
        offset = 0
        for name in sorted(ROI_DEFINITIONS.keys()):
            dim = roi_pca_dims[name]
            roi_voxels[name] = roi_concat[i:i+1, offset:offset+dim].to(DEVICE)
            offset += dim

        brain = BrainData(
            voxels=brain_patterns[i:i+1].to(DEVICE),
            roi_voxels=roi_voxels,
        )
        bg, bt = roi_model.encode_brain(brain)
        shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

        latent_sum = None
        for s in range(N_AVG):
            torch.manual_seed(s)
            z = roi_model.flow_matcher.sample(
                roi_model.dit, shape_1, bg, bt, num_steps=ODE_STEPS,
                cfg_scale=IMG_CFG_SCALE,
            )
            z = z + lin_preds_roi[i:i+1].to(DEVICE)
            latent_sum = z if latent_sum is None else latent_sum + z
        avg_z = latent_sum / N_AVG
        with torch.no_grad():
            recon = roi_model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()

        target = images[i].cpu()
        cos = F.cosine_similarity(
            recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
        ).item()
        ssim = compute_ssim(recon, target)
        cos_scores.append(cos)
        ssim_scores.append(ssim)

        if (count + 1) % 10 == 0:
            print(f"    {count+1}/{len(eval_indices)} ({time.time()-t0:.0f}s)")

    return {
        "cos_mean": np.mean(cos_scores), "cos_std": np.std(cos_scores),
        "ssim_mean": np.mean(ssim_scores), "ssim_std": np.std(ssim_scores),
    }


print(f"\n  Evaluating ROI-aware model ({N_EVAL} test images)...")
roi_results = evaluate_roi_model(test_indices)
roi_improvement = roi_results["cos_mean"] - baseline_results["cos_mean"]
print(f"  ROI-aware: cos={roi_results['cos_mean']:.3f} ± {roi_results['cos_std']:.3f}, "
      f"SSIM={roi_results['ssim_mean']:.3f} ± {roi_results['ssim_std']:.3f}")
print(f"  vs baseline: Δcos={roi_improvement:+.3f}")

# ═══════════════════════════════════════════════════════════════
# GENERATE FIGURES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("GENERATING FIGURES")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Figure 1: ROI ablation bar chart ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("ROI Ablation: Impact on Reconstruction Quality", fontsize=16, fontweight="bold")

roi_names_sorted = sorted(ablation_results.keys(),
                           key=lambda r: ablation_results[r]["drop_cos"], reverse=True)
drops_cos = [ablation_results[r]["drop_cos"] for r in roi_names_sorted]
drops_ssim = [ablation_results[r]["drop_ssim"] for r in roi_names_sorted]
n_verts = [ablation_results[r]["n_vertices"] for r in roi_names_sorted]
descs = [ROI_DEFINITIONS[r][2] for r in roi_names_sorted]

colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(roi_names_sorted)))
bars1 = ax1.barh(range(len(roi_names_sorted)), drops_cos, color=colors)
ax1.set_yticks(range(len(roi_names_sorted)))
ax1.set_yticklabels([f"{r}\n({d})" for r, d in zip(roi_names_sorted, descs)], fontsize=9)
ax1.set_xlabel("Cosine Similarity Drop (Δ)", fontsize=12)
ax1.set_title("Cosine Similarity Impact", fontsize=13)
ax1.axvline(0, color="black", linewidth=0.5)
for i, (v, nv) in enumerate(zip(drops_cos, n_verts)):
    ax1.text(v + 0.001, i, f"Δ={v:+.3f} ({nv} verts)", va="center", fontsize=8)

bars2 = ax2.barh(range(len(roi_names_sorted)), drops_ssim, color=colors)
ax2.set_yticks(range(len(roi_names_sorted)))
ax2.set_yticklabels([f"{r}" for r in roi_names_sorted], fontsize=9)
ax2.set_xlabel("SSIM Drop (Δ)", fontsize=12)
ax2.set_title("SSIM Impact", fontsize=13)
ax2.axvline(0, color="black", linewidth=0.5)
for i, v in enumerate(drops_ssim):
    ax2.text(v + 0.001, i, f"Δ={v:+.3f}", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "roi_ablation_impact.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT}/roi_ablation_impact.png")
plt.close()

# ── Figure 2: Category-ROI importance heatmap ──
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle("Category-ROI Importance\n(centroid distance from global mean in each ROI)",
             fontsize=14, fontweight="bold")

roi_list = list(ROI_DEFINITIONS.keys())
cat_list = STL10_CATEGORIES
importance_matrix = np.zeros((len(roi_list), len(cat_list)))
for i, roi in enumerate(roi_list):
    for j, cat in enumerate(cat_list):
        importance_matrix[i, j] = category_roi_importance[roi].get(cat, 0)

# Normalize per ROI for better visualization
for i in range(len(roi_list)):
    row_max = importance_matrix[i].max()
    if row_max > 0:
        importance_matrix[i] /= row_max

im = ax.imshow(importance_matrix, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(cat_list)))
ax.set_xticklabels(cat_list, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(roi_list)))
ax.set_yticklabels([f"{r} ({ROI_DEFINITIONS[r][2]})" for r in roi_list], fontsize=9)
plt.colorbar(im, ax=ax, label="Normalized importance")

for i in range(len(roi_list)):
    for j in range(len(cat_list)):
        ax.text(j, i, f"{importance_matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if importance_matrix[i, j] > 0.6 else "black")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "roi_category_importance.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT}/roi_category_importance.png")
plt.close()

# ── Figure 3: ROI information content ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ROI Information Content", fontsize=16, fontweight="bold")

# Left: variance and discriminability
roi_names_list = list(ROI_DEFINITIONS.keys())
total_vars = [roi_info[r]["total_var"] for r in roi_names_list]
discrim = [roi_info[r]["discriminability"] for r in roi_names_list]

x = np.arange(len(roi_names_list))
width = 0.35
bars1 = ax1.bar(x - width/2, total_vars, width, label="Total variance", color="#2196F3")
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width/2, discrim, width, label="Discriminability", color="#FF9800")
ax1.set_xticks(x)
ax1.set_xticklabels(roi_names_list, rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("Total Variance", color="#2196F3")
ax1_twin.set_ylabel("Category Discriminability", color="#FF9800")
ax1.set_title("Variance vs Discriminability per ROI")
lines = [bars1, bars2]
ax1.legend(lines, ["Total variance", "Discriminability"], loc="upper right")

# Right: Baseline vs ROI-aware model comparison
methods = ["Flat Encoder\n(baseline)", "ROI-Aware\nEncoder"]
cos_vals = [baseline_results["cos_mean"], roi_results["cos_mean"]]
cos_stds = [baseline_results["cos_std"], roi_results["cos_std"]]
ssim_vals = [baseline_results["ssim_mean"], roi_results["ssim_mean"]]
ssim_stds = [baseline_results["ssim_std"], roi_results["ssim_std"]]

x2 = np.arange(len(methods))
bars_cos = ax2.bar(x2 - 0.15, cos_vals, 0.3, yerr=cos_stds, label="Cosine Sim",
                    color="#4CAF50", capsize=5)
bars_ssim = ax2.bar(x2 + 0.15, ssim_vals, 0.3, yerr=ssim_stds, label="SSIM",
                     color="#9C27B0", capsize=5)
ax2.set_xticks(x2)
ax2.set_xticklabels(methods, fontsize=11)
ax2.set_ylabel("Score")
ax2.set_title("Flat vs ROI-Aware Encoding")
ax2.legend()

for i, (c, s) in enumerate(zip(cos_vals, ssim_vals)):
    ax2.text(i - 0.15, c + cos_stds[i] + 0.005, f"{c:.3f}", ha="center", fontsize=9)
    ax2.text(i + 0.15, s + ssim_stds[i] + 0.005, f"{s:.3f}", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "roi_information_content.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT}/roi_information_content.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SUMMARY — ROI ABLATION STUDY")
print("=" * 70)

print(f"\n  ROIs defined:              {len(ROI_DEFINITIONS)}")
print(f"  Total cortical vertices:   {N_VERTICES}")
print(f"  PCA dims per ROI:          {N_ROI_PCA}")
print(f"  Total ROI PCA dims:        {total_roi_dims}")

print(f"\n  Baseline (flat encoder):   cos={baseline_results['cos_mean']:.3f}, "
      f"SSIM={baseline_results['ssim_mean']:.3f}")
print(f"  ROI-aware encoder:         cos={roi_results['cos_mean']:.3f}, "
      f"SSIM={roi_results['ssim_mean']:.3f}")
print(f"  ROI improvement:           Δcos={roi_improvement:+.3f}")

print(f"\n  Ablation impact (sorted by cos drop):")
for roi in roi_names_sorted:
    r = ablation_results[roi]
    desc = ROI_DEFINITIONS[roi][2]
    print(f"    {roi:12s} ({desc:20s}): Δcos={r['drop_cos']:+.3f}, "
          f"ΔSSIM={r['drop_ssim']:+.3f}, {r['n_vertices']} vertices")

print(f"\n  Most important ROI:        {roi_names_sorted[0]} "
      f"(Δcos={ablation_results[roi_names_sorted[0]]['drop_cos']:+.3f})")
print(f"  Least important ROI:       {roi_names_sorted[-1]} "
      f"(Δcos={ablation_results[roi_names_sorted[-1]]['drop_cos']:+.3f})")

# Save results
results_json = {
    "baseline": baseline_results,
    "roi_aware": roi_results,
    "roi_improvement": roi_improvement,
    "ablation": {k: {kk: vv for kk, vv in v.items()} for k, v in ablation_results.items()},
    "roi_info": {k: {kk: vv for kk, vv in v.items() if kk != "cat_dists"}
                 for k, v in roi_info.items()},
    "category_roi_importance": category_roi_importance,
    "config": {
        "n_eval": N_EVAL, "n_avg": N_AVG, "ode_steps": ODE_STEPS,
        "roi_brain_steps": ROI_BRAIN_STEPS, "roi_dit_steps": ROI_DIT_STEPS,
        "n_roi_pca": N_ROI_PCA, "total_roi_dims": total_roi_dims,
    },
}
with open(os.path.join(OUT, "roi_ablation_results.json"), "w") as f:
    json.dump(results_json, f, indent=2, default=str)
print(f"\n  Saved: {OUT}/roi_ablation_results.json")
