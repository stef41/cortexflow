"""Sampling Dynamics Study — How do CFG, ODE steps, and brain noise
affect reconstruction quality and diversity?

This experiment systematically characterizes the sampling parameter space
of cortexflow's flow matching backbone for brain-to-image reconstruction.
It answers three previously unstudied questions:

1. **CFG scale**: What guidance strength is optimal for brain decoding?
   (Prior work uses arbitrary values without justification.)
2. **ODE step efficiency**: How few sampling steps are needed before
   quality saturates? (Important for real-time BCI applications.)
3. **Brain noise diversity**: Can injecting noise into brain embeddings
   produce diverse BUT faithful reconstructions? (Biologically motivated:
   fMRI signals are inherently noisy.)
4. **Per-category sensitivity**: Do different visual categories require
   different sampling parameters? (Animals vs. vehicles vs. objects.)

Uses the trained model from train_natural.py (brain2img_natural.pt).
"""

import os
import sys
import time
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.datasets import STL10
from torchvision import transforms

from cortexflow import BrainData
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from cortexflow.brain2img import Brain2Image

OUT = "train_outputs"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
DEVICE = "cuda:0"

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
N_TOTAL = 10000
N_TRAIN = 8000
N_TEST = 2000
IMG_SIZE = 128
N_VOXELS = 384
VJEPA_LAYERS = [20, 30]

# Evaluation subset sizes
N_EVAL = 50           # images per sweep point (quality measurement)
N_DIV_BRAINS = 20     # brains for diversity measurement
N_DIV_SAMPLES = 8     # samples per brain for diversity
ODE_STEPS_DEFAULT = 25

# Sweep ranges
CFG_SCALES = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
ODE_STEPS_RANGE = [5, 10, 15, 20, 25, 50, 75, 100]
BRAIN_NOISE_RANGE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

# STL-10 categories
STL10_CATEGORIES = [
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
# 1. LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("SAMPLING DYNAMICS STUDY")
print("How do CFG, ODE steps, and brain noise affect brain decoding?")
print("=" * 70)

# ── Load STL-10 ──
print(f"\nLoading STL-10 natural images...")
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = STL10("/tmp/stl10", split="unlabeled", download=False, transform=transform)
torch.manual_seed(42)
indices = torch.randperm(len(dataset))[:N_TOTAL].tolist()
target_images = torch.stack([dataset[i][0] for i in indices])
print(f"  Images: {target_images.shape}")

# ── Load labeled test set for per-category analysis ──
print("  Loading labeled test set...")
labeled_dataset = STL10("/tmp/stl10", split="test", download=False, transform=transform)
labeled_images = torch.stack([labeled_dataset[i][0] for i in range(len(labeled_dataset))])
labeled_labels = torch.tensor([labeled_dataset[i][1] for i in range(len(labeled_dataset))])
print(f"  Labeled: {labeled_images.shape}, labels: {labeled_labels.shape}")

# ── V-JEPA2 feature extraction ──
print(f"\nLoading V-JEPA2 (ViT-Giant)...")
from transformers import AutoModel

vjepa = AutoModel.from_pretrained(
    "facebook/vjepa2-vitg-fpc64-256", dtype=torch.float16,
).to(DEVICE).eval()

print(f"  Extracting features for {N_TOTAL} unlabeled images...")
vjepa_features = []
t0 = time.time()
VJEPA_BATCH = 16
for batch_start in range(0, N_TOTAL, VJEPA_BATCH):
    batch_end = min(batch_start + VJEPA_BATCH, N_TOTAL)
    batch_imgs = F.interpolate(
        target_images[batch_start:batch_end],
        size=(256, 256), mode="bilinear", align_corners=False,
    )
    video_input = batch_imgs.unsqueeze(1).half().to(DEVICE)
    with torch.no_grad():
        out = vjepa(video_input, output_hidden_states=True)
        feats = []
        for layer_idx in VJEPA_LAYERS:
            h = out.hidden_states[layer_idx].mean(dim=1)
            feats.append(h.float().cpu())
        vjepa_features.append(torch.cat(feats, dim=1))
    if (batch_start + VJEPA_BATCH) % 320 == 0 or batch_end == N_TOTAL:
        elapsed = time.time() - t0
        print(f"    {batch_end}/{N_TOTAL} ({elapsed:.1f}s)")

vjepa_features = torch.cat(vjepa_features, dim=0)
print(f"  V-JEPA2 features: {vjepa_features.shape}")

# Labeled images through V-JEPA2
N_LABELED = len(labeled_dataset)
print(f"\n  Extracting features for {N_LABELED} labeled images...")
labeled_vjepa = []
t0 = time.time()
for batch_start in range(0, N_LABELED, VJEPA_BATCH):
    batch_end = min(batch_start + VJEPA_BATCH, N_LABELED)
    batch_imgs = F.interpolate(
        labeled_images[batch_start:batch_end],
        size=(256, 256), mode="bilinear", align_corners=False,
    )
    video_input = batch_imgs.unsqueeze(1).half().to(DEVICE)
    with torch.no_grad():
        out = vjepa(video_input, output_hidden_states=True)
        feats = []
        for layer_idx in VJEPA_LAYERS:
            h = out.hidden_states[layer_idx].mean(dim=1)
            feats.append(h.float().cpu())
        labeled_vjepa.append(torch.cat(feats, dim=1))
    if (batch_start + VJEPA_BATCH) % 320 == 0 or batch_end == N_LABELED:
        elapsed = time.time() - t0
        print(f"    {batch_end}/{N_LABELED} ({elapsed:.1f}s)")

labeled_vjepa = torch.cat(labeled_vjepa, dim=0)
print(f"  Labeled V-JEPA2 features: {labeled_vjepa.shape}")

del vjepa
torch.cuda.empty_cache()

# ── TRIBE v2 brain mapping ──
print(f"\nLoading TRIBE v2 brain mapping head...")
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("facebook/tribev2", "best.ckpt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
del ckpt

video_proj_w = state_dict["projectors.video.weight"]
video_proj_b = state_dict["projectors.video.bias"]
low_rank_w = state_dict["low_rank_head.weight"]
pred_w = state_dict["predictor.weights"]
pred_b = state_dict["predictor.bias"]
combiner_w = state_dict.get("combiner.weight")
combiner_b = state_dict.get("combiner.bias")


def tribe_forward(feats):
    with torch.no_grad():
        projected = F.linear(feats, video_proj_w, video_proj_b)
        B = projected.shape[0]
        combined = torch.cat([
            torch.zeros(B, 384), torch.zeros(B, 384), projected
        ], dim=1)
        if combiner_w is not None:
            combined = F.linear(combined, combiner_w, combiner_b)
            combined = F.gelu(combined)
            if "combiner.norm.weight" in state_dict:
                combined = F.layer_norm(
                    combined, [1152],
                    state_dict["combiner.norm.weight"],
                    state_dict["combiner.norm.bias"],
                )
        brain_2048 = F.linear(combined, low_rank_w)
        brain = torch.einsum("bh,shv->bv", brain_2048, pred_w) + pred_b.squeeze(0)
    return brain


print("  Mapping unlabeled images through TRIBE v2...")
full_brain = tribe_forward(vjepa_features)
print(f"  Full brain: {full_brain.shape}")

# PCA
brain_mean = full_brain.mean(dim=0)
brain_centered = full_brain - brain_mean
U, S, V_pca = torch.svd_lowrank(brain_centered, q=N_VOXELS)
brain_patterns = (brain_centered @ V_pca)
brain_std = brain_patterns.std(dim=0, keepdim=True).clamp(min=1e-6)
brain_patterns = brain_patterns / brain_std
print(f"  Brain patterns: {brain_patterns.shape}")

# Labeled brain patterns
labeled_brain_full = tribe_forward(labeled_vjepa)
labeled_brain = ((labeled_brain_full - brain_mean) @ V_pca) / brain_std
print(f"  Labeled brain: {labeled_brain.shape}")

del full_brain, brain_centered, labeled_brain_full, vjepa_features, labeled_vjepa
del state_dict, video_proj_w, video_proj_b, low_rank_w, pred_w, pred_b
torch.cuda.empty_cache()

train_idx = list(range(N_TRAIN))
test_idx = list(range(N_TRAIN, N_TOTAL))

# ── Load trained model ──
print(f"\nLoading trained cortexflow model...")
img_model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=256, depth=8, num_heads=8, cond_dim=256),
    vae_config=VAEConfig(hidden_dims=[64, 128, 256], latent_channels=8),
    flow_config=FlowConfig(),
).to(DEVICE)

ckpt = torch.load(os.path.join(OUT, "brain2img_natural.pt"), map_location=DEVICE)
img_model.load_state_dict(ckpt)
del ckpt
img_model.eval()
print(f"  Model loaded: {sum(p.numel() for p in img_model.parameters()) / 1e6:.1f}M params")

# Cache latents + linear predictions
target_images = target_images.to(DEVICE)
brain_patterns = brain_patterns.to(DEVICE)

with torch.no_grad():
    all_latents = []
    for i in range(0, N_TOTAL, 32):
        z, _, _ = img_model.vae.encode(target_images[i:i + 32])
        all_latents.append(z.cpu())
    all_latents = torch.cat(all_latents).to(DEVICE)
    IMG_LATENT_SHAPE = all_latents.shape[1:]

X = brain_patterns[:N_TRAIN].cpu()
Y = all_latents[:N_TRAIN].flatten(1).cpu()
lam = 5.0
XtX = X.T @ X + lam * torch.eye(X.shape[1])
XtY = X.T @ Y
W_ridge = torch.linalg.solve(XtX, XtY)
img_lin_preds = (brain_patterns.cpu() @ W_ridge).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)

# Labeled linear predictions
labeled_brain = labeled_brain.to(DEVICE)
labeled_images = labeled_images.to(DEVICE)
labeled_lin_preds = (labeled_brain.cpu() @ W_ridge).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)

# Labeled latents
with torch.no_grad():
    labeled_latents = []
    for i in range(0, N_LABELED, 32):
        z, _, _ = img_model.vae.encode(labeled_images[i:i + 32])
        labeled_latents.append(z.cpu())
    labeled_latents = torch.cat(labeled_latents).to(DEVICE)

print(f"  Latent shape: {list(IMG_LATENT_SHAPE)}")
print(f"  Ready for sampling analysis\n")


# ── Evaluation helpers ──
def evaluate_single(brain_idx, brain_voxels, target_img, lin_pred,
                    cfg_scale=1.5, num_steps=25, n_avg=2, brain_noise=0.0):
    """Reconstruct one image and measure quality."""
    brain = BrainData(voxels=brain_voxels.unsqueeze(0))
    bg, bt = img_model.encode_brain(brain)
    shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

    latent_sum = None
    for s in range(n_avg):
        torch.manual_seed(s * 1000 + brain_idx)

        # Inject brain noise if requested
        bg_s, bt_s = bg, bt
        if brain_noise > 0:
            g_scale = bg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            bg_s = bg + brain_noise * g_scale * torch.randn_like(bg)
            t_scale = bt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            bt_s = bt + brain_noise * t_scale * torch.randn_like(bt)

        # Unconditional embeddings for CFG
        uncond_g = img_model.uncond_global.expand(1, -1)
        uncond_t = img_model.uncond_tokens.expand(1, -1, -1)

        z = img_model.flow_matcher.sample(
            img_model.dit, shape_1, bg_s, bt_s, num_steps=num_steps,
            cfg_scale=cfg_scale,
            brain_global_uncond=uncond_g,
            brain_tokens_uncond=uncond_t,
        )
        z = z + lin_pred.unsqueeze(0)
        latent_sum = z if latent_sum is None else latent_sum + z

    avg_z = latent_sum / n_avg
    with torch.no_grad():
        recon = img_model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()

    target = target_img.cpu()
    cos = F.cosine_similarity(
        recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
    ).item()
    ssim = compute_ssim(recon, target)
    return cos, ssim


def evaluate_diversity(brain_idx, brain_voxels, lin_pred,
                       cfg_scale=1.5, num_steps=25, brain_noise=0.0,
                       n_samples=N_DIV_SAMPLES):
    """Generate multiple samples and measure intra-brain diversity."""
    brain = BrainData(voxels=brain_voxels.unsqueeze(0))
    bg, bt = img_model.encode_brain(brain)
    shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

    samples = []
    for s in range(n_samples):
        torch.manual_seed(s * 7919 + brain_idx)

        bg_s, bt_s = bg, bt
        if brain_noise > 0:
            g_scale = bg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            bg_s = bg + brain_noise * g_scale * torch.randn_like(bg)
            t_scale = bt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            bt_s = bt + brain_noise * t_scale * torch.randn_like(bt)

        uncond_g = img_model.uncond_global.expand(1, -1)
        uncond_t = img_model.uncond_tokens.expand(1, -1, -1)

        z = img_model.flow_matcher.sample(
            img_model.dit, shape_1, bg_s, bt_s, num_steps=num_steps,
            cfg_scale=cfg_scale,
            brain_global_uncond=uncond_g,
            brain_tokens_uncond=uncond_t,
        )
        z = z + lin_pred.unsqueeze(0)
        with torch.no_grad():
            recon = img_model.vae.decode(z)[0].detach().clamp(0, 1).cpu()
        samples.append(recon)

    # Intra-brain diversity = 1 - mean pairwise cosine similarity
    flat = torch.stack([s.flatten() for s in samples])
    pair_sims = []
    for a in range(n_samples):
        for b in range(a + 1, n_samples):
            sim = F.cosine_similarity(flat[a:a+1], flat[b:b+1]).item()
            pair_sims.append(sim)
    diversity = 1.0 - np.mean(pair_sims)
    return diversity


# ═══════════════════════════════════════════════════════════════
# 2. ANALYSIS 1: CFG SCALE SWEEP
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("ANALYSIS 1: CFG SCALE SENSITIVITY")
print("=" * 70)

eval_indices = test_idx[:N_EVAL]
cfg_results = {}

for cfg in CFG_SCALES:
    t0 = time.time()
    cos_vals, ssim_vals = [], []
    for i in eval_indices:
        cos, ssim = evaluate_single(
            i, brain_patterns[i], target_images[i], img_lin_preds[i],
            cfg_scale=cfg, num_steps=ODE_STEPS_DEFAULT, n_avg=2,
        )
        cos_vals.append(cos)
        ssim_vals.append(ssim)

    cfg_results[cfg] = {
        "cos_mean": float(np.mean(cos_vals)),
        "cos_std": float(np.std(cos_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
    }
    elapsed = time.time() - t0
    print(f"  CFG={cfg:.2f}: cos={np.mean(cos_vals):.4f}±{np.std(cos_vals):.4f}, "
          f"SSIM={np.mean(ssim_vals):.4f}±{np.std(ssim_vals):.4f} ({elapsed:.1f}s)")

# Find optimal CFG
best_cfg = max(cfg_results, key=lambda c: cfg_results[c]["cos_mean"])
print(f"\n  → Optimal CFG scale: {best_cfg} "
      f"(cos={cfg_results[best_cfg]['cos_mean']:.4f})")
OPTIMAL_CFG = best_cfg

# ═══════════════════════════════════════════════════════════════
# 3. ANALYSIS 2: ODE STEP EFFICIENCY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 2: ODE STEP EFFICIENCY")
print("=" * 70)

step_results = {}

for steps in ODE_STEPS_RANGE:
    t0 = time.time()
    cos_vals, ssim_vals = [], []
    for i in eval_indices:
        cos, ssim = evaluate_single(
            i, brain_patterns[i], target_images[i], img_lin_preds[i],
            cfg_scale=OPTIMAL_CFG, num_steps=steps, n_avg=2,
        )
        cos_vals.append(cos)
        ssim_vals.append(ssim)

    elapsed = time.time() - t0
    time_per_sample = elapsed / N_EVAL

    step_results[steps] = {
        "cos_mean": float(np.mean(cos_vals)),
        "cos_std": float(np.std(cos_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "time_per_sample": float(time_per_sample),
        "total_time": float(elapsed),
    }
    print(f"  Steps={steps:3d}: cos={np.mean(cos_vals):.4f}±{np.std(cos_vals):.4f}, "
          f"SSIM={np.mean(ssim_vals):.4f}±{np.std(ssim_vals):.4f}, "
          f"{time_per_sample:.3f}s/sample ({elapsed:.1f}s)")

# Find diminishing returns point
best_cos_100 = step_results[100]["cos_mean"]
for s in sorted(step_results.keys()):
    if step_results[s]["cos_mean"] >= 0.99 * best_cos_100:
        min_steps = s
        break
print(f"\n  → 99% of max quality at {min_steps} steps "
      f"(speedup: {step_results[100]['time_per_sample'] / step_results[min_steps]['time_per_sample']:.1f}×)")

# ═══════════════════════════════════════════════════════════════
# 4. ANALYSIS 3: BRAIN NOISE → QUALITY VS DIVERSITY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 3: BRAIN NOISE — QUALITY VS DIVERSITY TRADEOFF")
print("=" * 70)

div_test_indices = test_idx[:N_DIV_BRAINS]
noise_results = {}

for noise in BRAIN_NOISE_RANGE:
    t0 = time.time()

    # Quality (cosine / SSIM to ground truth)
    cos_vals, ssim_vals = [], []
    for i in eval_indices:
        cos, ssim = evaluate_single(
            i, brain_patterns[i], target_images[i], img_lin_preds[i],
            cfg_scale=OPTIMAL_CFG, num_steps=ODE_STEPS_DEFAULT,
            n_avg=2, brain_noise=noise,
        )
        cos_vals.append(cos)
        ssim_vals.append(ssim)

    # Diversity (intra-brain sample variation)
    div_vals = []
    for i in div_test_indices:
        div = evaluate_diversity(
            i, brain_patterns[i], img_lin_preds[i],
            cfg_scale=OPTIMAL_CFG, num_steps=ODE_STEPS_DEFAULT,
            brain_noise=noise, n_samples=N_DIV_SAMPLES,
        )
        div_vals.append(div)

    elapsed = time.time() - t0
    noise_results[noise] = {
        "cos_mean": float(np.mean(cos_vals)),
        "cos_std": float(np.std(cos_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "diversity_mean": float(np.mean(div_vals)),
        "diversity_std": float(np.std(div_vals)),
    }
    print(f"  noise={noise:.2f}: cos={np.mean(cos_vals):.4f}, "
          f"SSIM={np.mean(ssim_vals):.4f}, "
          f"diversity={np.mean(div_vals):.4f}±{np.std(div_vals):.4f} "
          f"({elapsed:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# 5. ANALYSIS 4: PER-CATEGORY CFG SENSITIVITY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 4: PER-CATEGORY CFG SENSITIVITY")
print("=" * 70)

# Use 5 samples per category across a few CFG scales
N_PER_CAT = 10
CAT_CFG_SCALES = [1.0, 1.5, 2.0, 3.0, 5.0]
category_cfg_results = {}

for cat_idx, cat_name in enumerate(STL10_CATEGORIES):
    cat_mask = (labeled_labels == cat_idx)
    cat_indices = torch.where(cat_mask)[0][:N_PER_CAT].tolist()

    category_cfg_results[cat_name] = {}
    for cfg in CAT_CFG_SCALES:
        cos_vals, ssim_vals = [], []
        for idx in cat_indices:
            cos, ssim = evaluate_single(
                idx + N_TOTAL,  # offset seed to avoid collision
                labeled_brain[idx],
                labeled_images[idx],
                labeled_lin_preds[idx],
                cfg_scale=cfg, num_steps=ODE_STEPS_DEFAULT, n_avg=2,
            )
            cos_vals.append(cos)
            ssim_vals.append(ssim)

        category_cfg_results[cat_name][cfg] = {
            "cos_mean": float(np.mean(cos_vals)),
            "ssim_mean": float(np.mean(ssim_vals)),
        }

    # Print per-category optimal
    best_cat_cfg = max(
        CAT_CFG_SCALES,
        key=lambda c: category_cfg_results[cat_name][c]["cos_mean"]
    )
    print(f"  {cat_name:>8s}: best CFG={best_cat_cfg:.1f} "
          f"(cos={category_cfg_results[cat_name][best_cat_cfg]['cos_mean']:.4f})")


# ═══════════════════════════════════════════════════════════════
# 6. SUMMARY & SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)

all_results = {
    "cfg_sweep": {str(k): v for k, v in cfg_results.items()},
    "step_sweep": {str(k): v for k, v in step_results.items()},
    "noise_sweep": {str(k): v for k, v in noise_results.items()},
    "category_cfg": category_cfg_results,
    "optimal_cfg": float(OPTIMAL_CFG),
    "min_steps_99pct": int(min_steps),
}

# Summary table
print(f"\n  Optimal CFG scale: {OPTIMAL_CFG}")
print(f"  Cos at optimal CFG: {cfg_results[OPTIMAL_CFG]['cos_mean']:.4f}")
print(f"  Cos at CFG=1.0 (no guidance): {cfg_results[1.0]['cos_mean']:.4f}")
print(f"  CFG improvement: {cfg_results[OPTIMAL_CFG]['cos_mean'] - cfg_results[1.0]['cos_mean']:+.4f}")

print(f"\n  Min steps for 99% quality: {min_steps}")
print(f"  Speed at {min_steps} steps: {step_results[min_steps]['time_per_sample']:.3f}s/sample")
print(f"  Speed at 100 steps: {step_results[100]['time_per_sample']:.3f}s/sample")

print(f"\n  Brain noise diversity:")
print(f"    noise=0.0: quality={noise_results[0.0]['cos_mean']:.4f}, "
      f"diversity={noise_results[0.0]['diversity_mean']:.4f}")
best_div_noise = max(BRAIN_NOISE_RANGE,
                     key=lambda n: noise_results[n]["diversity_mean"])
print(f"    noise={best_div_noise}: quality={noise_results[best_div_noise]['cos_mean']:.4f}, "
      f"diversity={noise_results[best_div_noise]['diversity_mean']:.4f}")

with open(os.path.join(OUT, "sampling_dynamics_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Results saved to {OUT}/sampling_dynamics_results.json")


# ═══════════════════════════════════════════════════════════════
# 7. FIGURES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("GENERATING FIGURES")
print("=" * 70)

# ── Figure 1: CFG Scale Sensitivity ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

cfgs = sorted(cfg_results.keys())
cos_means = [cfg_results[c]["cos_mean"] for c in cfgs]
cos_stds = [cfg_results[c]["cos_std"] for c in cfgs]
ssim_means = [cfg_results[c]["ssim_mean"] for c in cfgs]
ssim_stds = [cfg_results[c]["ssim_std"] for c in cfgs]

ax1.errorbar(cfgs, cos_means, yerr=cos_stds, marker="o", linewidth=2,
             capsize=4, color="#2196F3", label="Cosine Similarity")
ax1.axhline(y=cfg_results[1.0]["cos_mean"], color="gray", linestyle="--",
            alpha=0.5, label="No guidance (CFG=1.0)")
ax1.axvline(x=OPTIMAL_CFG, color="red", linestyle=":", alpha=0.6,
            label=f"Optimal ({OPTIMAL_CFG})")
ax1.set_xlabel("CFG Scale", fontsize=12)
ax1.set_ylabel("Cosine Similarity", fontsize=12)
ax1.set_title("CFG Scale → Semantic Fidelity", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.errorbar(cfgs, ssim_means, yerr=ssim_stds, marker="s", linewidth=2,
             capsize=4, color="#FF9800", label="SSIM")
ax2.axhline(y=cfg_results[1.0]["ssim_mean"], color="gray", linestyle="--",
            alpha=0.5, label="No guidance (CFG=1.0)")
ax2.axvline(x=OPTIMAL_CFG, color="red", linestyle=":", alpha=0.6,
            label=f"Optimal ({OPTIMAL_CFG})")
ax2.set_xlabel("CFG Scale", fontsize=12)
ax2.set_ylabel("SSIM", fontsize=12)
ax2.set_title("CFG Scale → Structural Fidelity", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "cfg_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → cfg_sensitivity.png")

# ── Figure 2: ODE Step Efficiency ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

steps_list = sorted(step_results.keys())
cos_means = [step_results[s]["cos_mean"] for s in steps_list]
cos_stds = [step_results[s]["cos_std"] for s in steps_list]
times = [step_results[s]["time_per_sample"] for s in steps_list]

ax1.errorbar(steps_list, cos_means, yerr=cos_stds, marker="o", linewidth=2,
             capsize=4, color="#4CAF50")
ax1.axhline(y=0.99 * step_results[100]["cos_mean"], color="red", linestyle="--",
            alpha=0.5, label=f"99% of max ({0.99 * step_results[100]['cos_mean']:.4f})")
ax1.axvline(x=min_steps, color="red", linestyle=":", alpha=0.6,
            label=f"Min steps: {min_steps}")
ax1.set_xlabel("ODE Steps", fontsize=12)
ax1.set_ylabel("Cosine Similarity", fontsize=12)
ax1.set_title("Step Count → Quality", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(steps_list, times, marker="s", linewidth=2, color="#9C27B0")
ax2.axvline(x=min_steps, color="red", linestyle=":", alpha=0.6,
            label=f"Min steps: {min_steps}")
ax2.set_xlabel("ODE Steps", fontsize=12)
ax2.set_ylabel("Time per Sample (s)", fontsize=12)
ax2.set_title("Step Count → Latency", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "step_efficiency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → step_efficiency.png")

# ── Figure 3: Quality-Diversity Tradeoff (Pareto front) ──
fig, ax = plt.subplots(figsize=(8, 6))

noises = sorted(noise_results.keys())
cos_vals = [noise_results[n]["cos_mean"] for n in noises]
div_vals = [noise_results[n]["diversity_mean"] for n in noises]

# Color by noise level
colors = plt.cm.viridis(np.linspace(0, 1, len(noises)))

for ni, (n, cos_v, div_v, c) in enumerate(zip(noises, cos_vals, div_vals, colors)):
    ax.scatter(div_v, cos_v, color=c, s=120, zorder=5, edgecolors="black", linewidth=0.5)
    ax.annotate(f"σ={n}", (div_v, cos_v), textcoords="offset points",
                xytext=(8, 4), fontsize=9)

# Draw Pareto curve
ax.plot(div_vals, cos_vals, color="gray", linestyle="--", alpha=0.5, linewidth=1)

ax.set_xlabel("Diversity (1 − mean pairwise cosine)", fontsize=12)
ax.set_ylabel("Quality (cosine similarity to target)", fontsize=12)
ax.set_title("Brain Noise: Quality vs. Diversity Tradeoff", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap="viridis",
                           norm=plt.Normalize(min(noises), max(noises)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Brain Noise σ")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "diversity_tradeoff.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → diversity_tradeoff.png")

# ── Figure 4: Per-Category CFG Heatmap ──
fig, ax = plt.subplots(figsize=(10, 7))

cat_names = STL10_CATEGORIES
cat_cfgs = CAT_CFG_SCALES
heatmap_data = np.zeros((len(cat_names), len(cat_cfgs)))

for ci, cat in enumerate(cat_names):
    for cfi, cfg in enumerate(cat_cfgs):
        heatmap_data[ci, cfi] = category_cfg_results[cat][cfg]["cos_mean"]

im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(cat_cfgs)))
ax.set_xticklabels([f"{c:.1f}" for c in cat_cfgs], fontsize=11)
ax.set_yticks(range(len(cat_names)))
ax.set_yticklabels(cat_names, fontsize=11)
ax.set_xlabel("CFG Scale", fontsize=12)
ax.set_ylabel("Category", fontsize=12)
ax.set_title("Per-Category CFG Sensitivity (Cosine Similarity)", fontsize=13, fontweight="bold")

# Annotate cells
for ci in range(len(cat_names)):
    for cfi in range(len(cat_cfgs)):
        val = heatmap_data[ci, cfi]
        ax.text(cfi, ci, f"{val:.3f}", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if val < np.median(heatmap_data) else "black")

# Mark best CFG per category
for ci in range(len(cat_names)):
    best_cfi = np.argmax(heatmap_data[ci, :])
    ax.add_patch(plt.Rectangle((best_cfi - 0.5, ci - 0.5), 1, 1,
                                fill=False, edgecolor="red", linewidth=2.5))

plt.colorbar(im, ax=ax, label="Cosine Similarity")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "category_cfg_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → category_cfg_sensitivity.png")

print(f"\n{'=' * 70}")
print("SAMPLING DYNAMICS STUDY COMPLETE")
print("=" * 70)
print(f"\nKey findings:")
print(f"  1. Optimal CFG: {OPTIMAL_CFG} (improvement over no guidance: "
      f"{cfg_results[OPTIMAL_CFG]['cos_mean'] - cfg_results[1.0]['cos_mean']:+.4f} cos)")
print(f"  2. Min ODE steps for 99% quality: {min_steps} "
      f"({step_results[min_steps]['time_per_sample']:.3f}s vs "
      f"{step_results[100]['time_per_sample']:.3f}s at 100 steps)")
print(f"  3. Brain noise σ=0.0: div={noise_results[0.0]['diversity_mean']:.4f}, "
      f"cos={noise_results[0.0]['cos_mean']:.4f}")
print(f"     Brain noise σ={best_div_noise}: "
      f"div={noise_results[best_div_noise]['diversity_mean']:.4f}, "
      f"cos={noise_results[best_div_noise]['cos_mean']:.4f}")
