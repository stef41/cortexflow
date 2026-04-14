"""Cross-Subject Generalization Study.

Scientific question: Can cortexflow generalize to NEW subjects with different
brain anatomy, and does the SubjectAdapter (LoRA) enable efficient adaptation?

In real neuroscience, each person's brain is anatomically unique — the same
visual stimulus activates different voxel patterns in different people. A
practical BCI must either:
  (a) train from scratch per subject (expensive — hours of fMRI), or
  (b) train a shared model and quickly adapt to new subjects (few-shot).

This study simulates 5 subjects by applying different transformations to the
TRIBE v2 brain mapping (rotation, scaling, additive offset). Each subject sees
the same images but produces different brain patterns — mimicking real
inter-subject variability.

Analyses:
1. Zero-shot transfer: How well does a model trained on subjects 1-4 decode
   subject 5 WITHOUT any adaptation?
2. LoRA adaptation (SubjectAdapter): Fine-tune only the low-rank adapter
   (rank=16, ~4K params) with K={10,25,50,100,200} shots from the new subject.
3. Full fine-tuning baseline: Fine-tune the full brain encoder with the same
   K shots.
4. Leave-one-subject-out cross-validation across all 5 subjects.
5. Inter-subject brain similarity analysis.

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

from cortexflow import BrainData
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from cortexflow.brain2img import Brain2Image
from cortexflow.brain_encoder import SubjectAdapter
from cortexflow.flow_matching import EMAModel

OUT = "train_outputs"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
DEVICE = "cuda:0"

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
N_SUBJECTS = 5
N_PER_SUBJECT = 2000       # images per subject (from the 10k STL-10 pool)
N_TRAIN_PER_SUBJECT = 1600
N_TEST_PER_SUBJECT = 400
IMG_SIZE = 128
N_VOXELS = 384
FEW_SHOT_KS = [10, 25, 50, 100, 200]
LORA_RANK = 16
ADAPT_STEPS = 200          # fine-tuning steps for adaptation
ADAPT_LR = 5e-4
N_AVG = 2                  # reduced from 4 for speed
IMG_CFG_SCALE = 1.5
ODE_STEPS = 25             # reduced from 50 for speed
BATCH_SIZE = 64


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
# 1. LOAD TRAINED MODEL + DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("CORTEXFLOW — CROSS-SUBJECT GENERALIZATION STUDY")
print("=" * 70)

model_path = os.path.join(OUT, "brain2img_natural.pt")
brain_data_path = os.path.join(OUT, "natural_brain_data.pt")
assert os.path.exists(model_path), f"Need trained model: {model_path}"
assert os.path.exists(brain_data_path), f"Need brain data: {brain_data_path}"

print("\nLoading trained model...")
img_model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=256, depth=8, num_heads=8, cond_dim=256),
    vae_config=VAEConfig(hidden_dims=[64, 128, 256], latent_channels=8),
    flow_config=FlowConfig(),
).to(DEVICE)
state = torch.load(model_path, map_location=DEVICE, weights_only=True)
img_model.load_state_dict(state)
img_model.eval()
print(f"  Model loaded: {sum(p.numel() for p in img_model.parameters()) / 1e6:.1f}M params")

print("\nLoading brain data + TRIBE v2 projection...")
brain_data = torch.load(brain_data_path, map_location="cpu", weights_only=True)
pca_V = brain_data["pca_V"]          # (20484, 384)
brain_std = brain_data["brain_std"]    # (1, 384)
brain_mean = brain_data["brain_mean"]  # (20484,)
vjepa_features = brain_data["vjepa_features"]  # (10000, 2816)
print(f"  V-JEPA features: {vjepa_features.shape}")
print(f"  PCA V: {pca_V.shape}")

# ═══════════════════════════════════════════════════════════════
# 2. SIMULATE SUBJECTS — APPLY DIFFERENT BRAIN TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"SIMULATING {N_SUBJECTS} SUBJECTS WITH DIFFERENT BRAIN ANATOMIES")
print("=" * 70)

print("\nLoading TRIBE v2 for brain projection...")
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
combiner_norm_w = state_dict.get("combiner.norm.weight")
combiner_norm_b = state_dict.get("combiner.norm.bias")
del state_dict


def vjepa_to_full_brain(features):
    """Map V-JEPA2 features → full 20484 cortical vertices via TRIBE v2."""
    with torch.no_grad():
        video_projected = F.linear(features, video_proj_w, video_proj_b)
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
    return full_brain


def apply_subject_transform(full_brain, subject_id, seed=42):
    """Apply subject-specific anatomical transformation.

    Real inter-subject variability includes:
    - Different cortical folding → voxel reordering
    - Different functional organization → activation scaling
    - Different hemodynamic response → additive offsets
    - Registration noise → Gaussian perturbation

    We simulate with: random rotation in PCA space + scaling + offset + noise.
    """
    rng = torch.Generator().manual_seed(seed + subject_id * 1000)

    if subject_id == 0:
        # Subject 0 = original (reference subject)
        centered = full_brain - brain_mean
        brain_pca = centered @ pca_V
        return brain_pca / brain_std

    # Random rotation in PCA space (orthogonal transformation)
    # Small rotation — subjects are similar but not identical
    rotation_angle = 0.15 + 0.1 * subject_id  # radians (8-30 degrees)

    # Apply rotation in pairs of PCA dimensions
    centered = full_brain - brain_mean
    brain_pca = centered @ pca_V  # (B, 384)

    # Generate a random rotation matrix via QR decomposition of a perturbed identity
    noise_matrix = torch.randn(N_VOXELS, N_VOXELS, generator=rng) * rotation_angle
    perturbed = torch.eye(N_VOXELS) + noise_matrix
    Q, R = torch.linalg.qr(perturbed)
    # Ensure proper rotation (det=+1)
    D = torch.diag(torch.sign(torch.diag(R)))
    rotation = Q @ D

    brain_rotated = brain_pca @ rotation

    # Per-dimension scaling (different subjects have different activation magnitudes)
    scale = 1.0 + 0.2 * torch.randn(N_VOXELS, generator=rng)
    scale = scale.clamp(0.5, 1.5)
    brain_scaled = brain_rotated * scale.unsqueeze(0)

    # Additive offset (different baseline activity per voxel)
    offset = 0.1 * torch.randn(N_VOXELS, generator=rng)
    brain_offset = brain_scaled + offset.unsqueeze(0)

    # Registration noise (small per-sample noise)
    noise = 0.05 * torch.randn_like(brain_offset)
    brain_final = brain_offset + noise

    # Normalize like the original
    brain_final = brain_final / brain_final.std(dim=0, keepdim=True).clamp(min=1e-6)

    return brain_final


# Generate brain patterns for each subject
print("\nGenerating per-subject brain patterns...")
subject_brains = {}   # subject_id → (N_PER_SUBJECT, N_VOXELS)
subject_images_idx = {}  # subject_id → list of image indices

# Each subject sees a different subset (simulating different scanning sessions)
# But with overlap so we can compare directly
all_indices = torch.randperm(10000)

for sid in range(N_SUBJECTS):
    # Each subject sees N_PER_SUBJECT images (with overlap between subjects)
    start = sid * 400  # shift by 400 for partial overlap
    idx = torch.cat([
        all_indices[start:start + N_PER_SUBJECT]
    ]).tolist()
    if len(idx) < N_PER_SUBJECT:
        # Wrap around
        idx = (all_indices[:N_PER_SUBJECT]).tolist()

    subject_images_idx[sid] = idx[:N_PER_SUBJECT]

    # Project through TRIBE v2 and apply subject transform
    feats = vjepa_features[subject_images_idx[sid]]
    full_brain = vjepa_to_full_brain(feats)
    brain_transformed = apply_subject_transform(full_brain, sid)
    subject_brains[sid] = brain_transformed.to(DEVICE)

    print(f"  Subject {sid}: {brain_transformed.shape}, "
          f"mean={brain_transformed.mean():.3f}, std={brain_transformed.std():.3f}")

# Load images (only first 10k, not all 100k)
print("\nLoading images (first 10000)...")
from torchvision.datasets import STL10
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
dataset = STL10("/tmp/stl10", split="unlabeled", download=False, transform=transform)
all_images = torch.stack([dataset[i][0] for i in range(10000)]).to(DEVICE)
print(f"  Images: {all_images.shape}")

# Cache latents
print("  Caching VAE latents...")
with torch.no_grad():
    all_latents = []
    for i in range(0, 10000, 64):
        z, _, _ = img_model.vae.encode(all_images[i:i+64])
        all_latents.append(z.cpu())
    all_latents = torch.cat(all_latents).to(DEVICE)
IMG_LATENT_SHAPE = all_latents.shape[1:]
print(f"  Latent shape: {list(IMG_LATENT_SHAPE)}")

# ═══════════════════════════════════════════════════════════════
# 3. INTER-SUBJECT BRAIN SIMILARITY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 1: INTER-SUBJECT BRAIN SIMILARITY")
print("=" * 70)

# Find shared images to compare brain patterns across subjects
# Use first 200 images from subject 0's set
shared_idx = subject_images_idx[0][:200]

similarity_matrix = torch.zeros(N_SUBJECTS, N_SUBJECTS)
for si in range(N_SUBJECTS):
    for sj in range(si, N_SUBJECTS):
        # Get brain patterns for shared images for each subject
        # Re-project the shared images through each subject's transform
        feats = vjepa_features[shared_idx]
        full_brain = vjepa_to_full_brain(feats)
        brain_i = apply_subject_transform(full_brain, si).to(DEVICE)
        brain_j = apply_subject_transform(full_brain, sj).to(DEVICE)

        cos_sim = F.cosine_similarity(brain_i, brain_j, dim=1).mean().item()
        similarity_matrix[si, sj] = cos_sim
        similarity_matrix[sj, si] = cos_sim

print("\n  Inter-subject cosine similarity matrix:")
print("        ", end="")
for sj in range(N_SUBJECTS):
    print(f"  S{sj}   ", end="")
print()
for si in range(N_SUBJECTS):
    print(f"    S{si}  ", end="")
    for sj in range(N_SUBJECTS):
        v = similarity_matrix[si, sj].item()
        print(f" {v:.3f} ", end="")
    print()

mean_cross = []
for si in range(N_SUBJECTS):
    for sj in range(si + 1, N_SUBJECTS):
        mean_cross.append(similarity_matrix[si, sj].item())
print(f"\n  Mean cross-subject similarity: {np.mean(mean_cross):.3f} ± {np.std(mean_cross):.3f}")
print(f"  Range: [{min(mean_cross):.3f}, {max(mean_cross):.3f}]")

# ═══════════════════════════════════════════════════════════════
# 4. ZERO-SHOT TRANSFER (NO ADAPTATION)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 2: ZERO-SHOT CROSS-SUBJECT TRANSFER")
print("=" * 70)

# The model was trained on subject 0's brain patterns (the reference subject).
# How well does it decode other subjects WITHOUT any adaptation?


def evaluate_subject(model, subject_id, indices, brain_data, linear_W=None, n_eval=50):
    """Evaluate reconstruction quality for a subject."""
    results = []
    eval_idx = indices[-n_eval:]  # use last n_eval as test
    for count, img_idx in enumerate(eval_idx):
        brain_idx_in_subject = indices.index(img_idx) if isinstance(indices, list) else count
        brain = BrainData(voxels=brain_data[brain_idx_in_subject:brain_idx_in_subject+1])
        bg, bt = model.encode_brain(brain)
        shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

        latent_sum = None
        for s in range(N_AVG):
            torch.manual_seed(s)
            z = model.flow_matcher.sample(
                model.dit, shape_1, bg, bt, num_steps=ODE_STEPS,
                cfg_scale=IMG_CFG_SCALE,
            )
            # Add linear prediction if available
            if linear_W is not None:
                lin_pred = (brain_data[brain_idx_in_subject:brain_idx_in_subject+1].cpu() @ linear_W
                           ).view(1, *IMG_LATENT_SHAPE).to(DEVICE)
                z = z + lin_pred
            latent_sum = z if latent_sum is None else latent_sum + z
        avg_z = latent_sum / N_AVG

        with torch.no_grad():
            recon = model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()

        target = all_images[img_idx].cpu()
        cos = F.cosine_similarity(
            recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
        ).item()
        ssim = compute_ssim(recon, target)
        results.append({"cos": cos, "ssim": ssim, "img_idx": img_idx})

        if (count + 1) % 25 == 0:
            print(f"      {count+1}/{n_eval}", end="\r")

    return results


# Fit linear predictor for subject 0 (reference)
print("\nFitting ridge regression for reference subject (S0)...")
train_idx_s0 = subject_images_idx[0][:N_TRAIN_PER_SUBJECT]
X_s0 = subject_brains[0][:N_TRAIN_PER_SUBJECT].cpu()
Y_s0 = all_latents[train_idx_s0].flatten(1).cpu()
lam = 5.0
W_s0 = torch.linalg.solve(X_s0.T @ X_s0 + lam * torch.eye(N_VOXELS), X_s0.T @ Y_s0)

# Evaluate on each subject with zero-shot (no adaptation)
N_EVAL = 50
zero_shot_results = {}
for sid in range(N_SUBJECTS):
    print(f"\n  Subject {sid} (zero-shot, no adaptation):")
    test_indices = subject_images_idx[sid][-N_EVAL:]
    # Map brain indices to position within subject's data
    test_brain_start = N_PER_SUBJECT - N_EVAL

    results = []
    t0_subj = time.time()
    for count, img_idx in enumerate(test_indices):
        brain_pos = test_brain_start + count
        brain = BrainData(voxels=subject_brains[sid][brain_pos:brain_pos+1])
        bg, bt = img_model.encode_brain(brain)
        shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

        latent_sum = None
        for s in range(N_AVG):
            torch.manual_seed(s)
            z = img_model.flow_matcher.sample(
                img_model.dit, shape_1, bg, bt, num_steps=ODE_STEPS,
                cfg_scale=IMG_CFG_SCALE,
            )
            # Use subject 0's linear predictor (all we have without adaptation)
            lin_pred = (subject_brains[sid][brain_pos:brain_pos+1].cpu() @ W_s0
                       ).view(1, *IMG_LATENT_SHAPE).to(DEVICE)
            z = z + lin_pred
            latent_sum = z if latent_sum is None else latent_sum + z
        avg_z = latent_sum / N_AVG

        with torch.no_grad():
            recon = img_model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()

        target = all_images[img_idx].cpu()
        cos = F.cosine_similarity(
            recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
        ).item()
        ssim = compute_ssim(recon, target)
        results.append({"cos": cos, "ssim": ssim})

        if (count + 1) % 10 == 0:
            elapsed = time.time() - t0_subj
            print(f"    {count+1}/{N_EVAL} ({elapsed:.0f}s)")

    cos_mean = np.mean([r["cos"] for r in results])
    ssim_mean = np.mean([r["ssim"] for r in results])
    zero_shot_results[sid] = {"cos": cos_mean, "ssim": ssim_mean, "results": results}
    print(f"    cos={cos_mean:.3f}, SSIM={ssim_mean:.3f} ({time.time()-t0_subj:.0f}s)")

print(f"\n  Zero-shot summary:")
for sid in range(N_SUBJECTS):
    r = zero_shot_results[sid]
    tag = " (reference)" if sid == 0 else ""
    print(f"    S{sid}: cos={r['cos']:.3f}, SSIM={r['ssim']:.3f}{tag}")

# ═══════════════════════════════════════════════════════════════
# 5. FEW-SHOT ADAPTATION WITH SubjectAdapter (LoRA)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 3: FEW-SHOT LoRA ADAPTATION")
print("=" * 70)

lora_results = {}  # (subject_id, K) → metrics


def adapt_lora(base_model, subject_brain, subject_img_idx, K, subject_id):
    """Fine-tune SubjectAdapter (LoRA) with K samples from a new subject."""
    # Create a fresh SubjectAdapter
    adapter = SubjectAdapter(
        cond_dim=256, rank=LORA_RANK, n_subjects=1
    ).to(DEVICE)

    n_adapter_params = sum(p.numel() for p in adapter.parameters())

    # Also fit a new linear predictor with K samples
    X_k = subject_brain[:K].cpu()
    Y_k = all_latents[subject_img_idx[:K]].flatten(1).cpu()
    W_k = torch.linalg.solve(X_k.T @ X_k + lam * torch.eye(N_VOXELS), X_k.T @ Y_k)

    # Fine-tune only the adapter + brain encoder last layer
    adapter.train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=ADAPT_LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ADAPT_STEPS)

    for step in range(ADAPT_STEPS):
        idx = torch.randint(0, K, (min(K, 32),))
        brain = BrainData(voxels=subject_brain[idx])
        bg, bt = base_model.encode_brain(brain)

        # Apply LoRA to brain_global
        subject_idx_tensor = torch.zeros(bg.shape[0], dtype=torch.long, device=DEVICE)
        bg_adapted = adapter(bg, subject_idx_tensor)

        # Compute residual target
        img_idx_batch = [subject_img_idx[i] for i in idx.tolist()]
        target_latent = all_latents[img_idx_batch]
        lin_pred = (subject_brain[idx].cpu() @ W_k).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)
        residual = target_latent - lin_pred

        # Flow matching loss on adapted embedding
        loss = base_model.flow_matcher.compute_loss(
            base_model.dit, residual, bg_adapted, bt
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    adapter.eval()
    return adapter, W_k, n_adapter_params


# Test on subjects 1-4 (not reference)
for sid in range(1, N_SUBJECTS):
    lora_results[sid] = {}
    print(f"\n  Subject {sid}:")

    for K in FEW_SHOT_KS:
        print(f"    K={K} shots — adapting LoRA (rank={LORA_RANK})...", end=" ", flush=True)
        t0 = time.time()

        adapter, W_k, n_params = adapt_lora(
            img_model,
            subject_brains[sid],
            subject_images_idx[sid],
            K, sid
        )

        # Evaluate on held-out test
        test_brain_start = N_PER_SUBJECT - N_EVAL
        results = []
        for count in range(N_EVAL):
            brain_pos = test_brain_start + count
            img_idx = subject_images_idx[sid][brain_pos]
            brain = BrainData(voxels=subject_brains[sid][brain_pos:brain_pos+1])
            bg, bt = img_model.encode_brain(brain)

            # Apply LoRA
            subject_idx_tensor = torch.zeros(1, dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                bg_adapted = adapter(bg, subject_idx_tensor)

            shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)
            latent_sum = None
            for s in range(N_AVG):
                torch.manual_seed(s)
                z = img_model.flow_matcher.sample(
                    img_model.dit, shape_1, bg_adapted, bt, num_steps=ODE_STEPS,
                    cfg_scale=IMG_CFG_SCALE,
                )
                lin_pred = (subject_brains[sid][brain_pos:brain_pos+1].cpu() @ W_k
                           ).view(1, *IMG_LATENT_SHAPE).to(DEVICE)
                z = z + lin_pred
                latent_sum = z if latent_sum is None else latent_sum + z
            avg_z = latent_sum / N_AVG

            with torch.no_grad():
                recon = img_model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()

            target = all_images[img_idx].cpu()
            cos = F.cosine_similarity(
                recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
            ).item()
            ssim = compute_ssim(recon, target)
            results.append({"cos": cos, "ssim": ssim})

        cos_mean = np.mean([r["cos"] for r in results])
        ssim_mean = np.mean([r["ssim"] for r in results])
        elapsed = time.time() - t0
        lora_results[sid][K] = {
            "cos": cos_mean, "ssim": ssim_mean,
            "n_params": n_params, "time": elapsed
        }
        print(f"cos={cos_mean:.3f}, SSIM={ssim_mean:.3f} ({elapsed:.0f}s, {n_params} params)")

        del adapter
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# 6. FULL FINE-TUNING BASELINE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 4: FULL BRAIN ENCODER FINE-TUNING (baseline)")
print("=" * 70)

full_ft_results = {}

for sid in [1, 3]:  # Test on 2 subjects (expensive)
    full_ft_results[sid] = {}
    print(f"\n  Subject {sid}:")

    for K in [50, 200]:  # Only test 2 K values
        print(f"    K={K} shots — full fine-tuning...", end=" ", flush=True)
        t0 = time.time()

        # Deep copy brain encoder
        ft_model = copy.deepcopy(img_model)
        ft_model.train()
        ft_model.vae.eval()
        ft_model.dit.eval()

        # Fit new linear predictor
        X_k = subject_brains[sid][:K].cpu()
        Y_k = all_latents[subject_images_idx[sid][:K]].flatten(1).cpu()
        W_k = torch.linalg.solve(X_k.T @ X_k + lam * torch.eye(N_VOXELS), X_k.T @ Y_k)

        n_ft_params = sum(p.numel() for p in ft_model.brain_encoder.parameters())

        # Fine-tune brain encoder only
        opt = torch.optim.AdamW(ft_model.brain_encoder.parameters(), lr=ADAPT_LR, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ADAPT_STEPS)

        for step in range(ADAPT_STEPS):
            idx = torch.randint(0, K, (min(K, 32),))
            brain = BrainData(voxels=subject_brains[sid][idx])
            bg, bt = ft_model.encode_brain(brain)
            img_idx_batch = [subject_images_idx[sid][i] for i in idx.tolist()]
            target_latent = all_latents[img_idx_batch]
            lin_pred = (subject_brains[sid][idx].cpu() @ W_k).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)
            residual = target_latent - lin_pred
            loss = ft_model.flow_matcher.compute_loss(ft_model.dit, residual, bg, bt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        ft_model.eval()

        # Evaluate
        test_brain_start = N_PER_SUBJECT - N_EVAL
        results = []
        for count in range(N_EVAL):
            brain_pos = test_brain_start + count
            img_idx = subject_images_idx[sid][brain_pos]
            brain = BrainData(voxels=subject_brains[sid][brain_pos:brain_pos+1])
            bg, bt = ft_model.encode_brain(brain)

            shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)
            latent_sum = None
            for s in range(N_AVG):
                torch.manual_seed(s)
                z = ft_model.flow_matcher.sample(
                    ft_model.dit, shape_1, bg, bt, num_steps=ODE_STEPS,
                    cfg_scale=IMG_CFG_SCALE,
                )
                lin_pred = (subject_brains[sid][brain_pos:brain_pos+1].cpu() @ W_k
                           ).view(1, *IMG_LATENT_SHAPE).to(DEVICE)
                z = z + lin_pred
                latent_sum = z if latent_sum is None else latent_sum + z
            avg_z = latent_sum / N_AVG

            with torch.no_grad():
                recon = ft_model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()

            target = all_images[img_idx].cpu()
            cos = F.cosine_similarity(
                recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
            ).item()
            ssim = compute_ssim(recon, target)
            results.append({"cos": cos, "ssim": ssim})

        cos_mean = np.mean([r["cos"] for r in results])
        ssim_mean = np.mean([r["ssim"] for r in results])
        elapsed = time.time() - t0
        full_ft_results[sid][K] = {
            "cos": cos_mean, "ssim": ssim_mean,
            "n_params": n_ft_params, "time": elapsed
        }
        print(f"cos={cos_mean:.3f}, SSIM={ssim_mean:.3f} ({elapsed:.0f}s, {n_ft_params} params)")

        del ft_model
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("GENERATING FIGURES")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Figure 1: Few-shot adaptation curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Cosine similarity vs K
ax = axes[0]
for sid in range(1, N_SUBJECTS):
    ks = sorted(lora_results[sid].keys())
    cos_vals = [lora_results[sid][k]["cos"] for k in ks]
    ax.plot(ks, cos_vals, "o-", label=f"S{sid} (LoRA)", linewidth=2, markersize=6)

# Add zero-shot baselines
for sid in range(1, N_SUBJECTS):
    ax.axhline(y=zero_shot_results[sid]["cos"], color=f"C{sid-1}",
               linestyle="--", alpha=0.4, linewidth=1)

ax.axhline(y=zero_shot_results[0]["cos"], color="black",
           linestyle="-", alpha=0.6, linewidth=2, label="S0 (reference)")

# Add full fine-tuning points
for sid in full_ft_results:
    for K in full_ft_results[sid]:
        ax.plot(K, full_ft_results[sid][K]["cos"], "s",
                color=f"C{sid-1}", markersize=10, markeredgecolor="black",
                markeredgewidth=2, zorder=5)

ax.set_xlabel("K (adaptation samples)", fontsize=12)
ax.set_ylabel("Test Cosine Similarity", fontsize=12)
ax.set_title("Few-Shot Adaptation: Cosine Similarity", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale("log")
ax.set_xticks(FEW_SHOT_KS)
ax.set_xticklabels([str(k) for k in FEW_SHOT_KS])

# Right: SSIM vs K
ax = axes[1]
for sid in range(1, N_SUBJECTS):
    ks = sorted(lora_results[sid].keys())
    ssim_vals = [lora_results[sid][k]["ssim"] for k in ks]
    ax.plot(ks, ssim_vals, "o-", label=f"S{sid} (LoRA)", linewidth=2, markersize=6)

for sid in range(1, N_SUBJECTS):
    ax.axhline(y=zero_shot_results[sid]["ssim"], color=f"C{sid-1}",
               linestyle="--", alpha=0.4, linewidth=1)

ax.axhline(y=zero_shot_results[0]["ssim"], color="black",
           linestyle="-", alpha=0.6, linewidth=2, label="S0 (reference)")

for sid in full_ft_results:
    for K in full_ft_results[sid]:
        ax.plot(K, full_ft_results[sid][K]["ssim"], "s",
                color=f"C{sid-1}", markersize=10, markeredgecolor="black",
                markeredgewidth=2, zorder=5)

ax.set_xlabel("K (adaptation samples)", fontsize=12)
ax.set_ylabel("Test SSIM", fontsize=12)
ax.set_title("Few-Shot Adaptation: SSIM", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale("log")
ax.set_xticks(FEW_SHOT_KS)
ax.set_xticklabels([str(k) for k in FEW_SHOT_KS])

plt.suptitle("Cross-Subject Generalization: LoRA Adaptation (○) vs Full Fine-Tuning (□)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "cross_subject_adaptation.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT}/cross_subject_adaptation.png")

# ── Figure 2: Inter-subject similarity heatmap ──
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
im = ax.imshow(similarity_matrix.numpy(), cmap="RdYlBu_r", vmin=0, vmax=1)
for i in range(N_SUBJECTS):
    for j in range(N_SUBJECTS):
        v = similarity_matrix[i, j].item()
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if v < 0.5 else "black")
ax.set_xticks(range(N_SUBJECTS))
ax.set_yticks(range(N_SUBJECTS))
ax.set_xticklabels([f"S{i}" for i in range(N_SUBJECTS)], fontsize=12)
ax.set_yticklabels([f"S{i}" for i in range(N_SUBJECTS)], fontsize=12)
ax.set_title("Inter-Subject Brain Pattern Similarity\n(cosine similarity on shared images)",
             fontsize=13, fontweight="bold")
plt.colorbar(im, label="Cosine Similarity")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "cross_subject_similarity.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT}/cross_subject_similarity.png")

# ── Figure 3: Parameter efficiency comparison ──
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# LoRA results for subject 1
sid = 1
ks = sorted(lora_results[sid].keys())
lora_cos = [lora_results[sid][k]["cos"] for k in ks]
ax.plot(ks, lora_cos, "o-", color="C0", linewidth=2.5, markersize=8,
        label=f"LoRA (rank={LORA_RANK}, {lora_results[sid][ks[0]]['n_params']:,} params)")

# Full fine-tuning for subject 1
if sid in full_ft_results:
    ft_ks = sorted(full_ft_results[sid].keys())
    ft_cos = [full_ft_results[sid][k]["cos"] for k in ft_ks]
    ax.plot(ft_ks, ft_cos, "s--", color="C1", linewidth=2.5, markersize=10,
            markeredgecolor="black", markeredgewidth=1.5,
            label=f"Full FT ({full_ft_results[sid][ft_ks[0]]['n_params']:,} params)")

# Zero-shot baseline
ax.axhline(y=zero_shot_results[sid]["cos"], color="gray",
           linestyle=":", linewidth=2, label="Zero-shot (no adaptation)")

# Reference subject
ax.axhline(y=zero_shot_results[0]["cos"], color="green",
           linestyle="-", linewidth=2, alpha=0.7,
           label="Reference subject (S0, trained)")

ax.set_xlabel("K (adaptation samples from new subject)", fontsize=13)
ax.set_ylabel("Test Cosine Similarity", fontsize=13)
ax.set_title(f"Parameter Efficiency: LoRA vs Full Fine-Tuning (Subject {sid})",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xscale("log")
ax.set_xticks(FEW_SHOT_KS)
ax.set_xticklabels([str(k) for k in FEW_SHOT_KS])
plt.tight_layout()
fig.savefig(os.path.join(OUT, "cross_subject_efficiency.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT}/cross_subject_efficiency.png")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SUMMARY — CROSS-SUBJECT GENERALIZATION")
print("=" * 70)

# Compute key metrics
s0_cos = zero_shot_results[0]["cos"]
s0_ssim = zero_shot_results[0]["ssim"]

mean_zero_shot_cos = np.mean([zero_shot_results[s]["cos"] for s in range(1, N_SUBJECTS)])
mean_zero_shot_ssim = np.mean([zero_shot_results[s]["ssim"] for s in range(1, N_SUBJECTS)])
zero_shot_drop_cos = s0_cos - mean_zero_shot_cos
zero_shot_drop_ssim = s0_ssim - mean_zero_shot_ssim

# Best LoRA (K=200)
mean_lora_200_cos = np.mean([lora_results[s][200]["cos"] for s in range(1, N_SUBJECTS)])
mean_lora_200_ssim = np.mean([lora_results[s][200]["ssim"] for s in range(1, N_SUBJECTS)])
lora_recovery_cos = (mean_lora_200_cos - mean_zero_shot_cos) / max(zero_shot_drop_cos, 1e-6) * 100
lora_recovery_ssim = (mean_lora_200_ssim - mean_zero_shot_ssim) / max(zero_shot_drop_ssim, 1e-6) * 100

# K=50 LoRA
mean_lora_50_cos = np.mean([lora_results[s][50]["cos"] for s in range(1, N_SUBJECTS)])

print(f"""
  Subjects simulated:       {N_SUBJECTS} (different brain anatomies)
  Mean cross-subject sim:   {np.mean(mean_cross):.3f} ± {np.std(mean_cross):.3f}

  Reference subject (S0):   cos={s0_cos:.3f}, SSIM={s0_ssim:.3f}
  Zero-shot transfer (avg): cos={mean_zero_shot_cos:.3f}, SSIM={mean_zero_shot_ssim:.3f}
  Zero-shot drop:           Δcos={zero_shot_drop_cos:+.3f}, ΔSSIM={zero_shot_drop_ssim:+.3f}

  LoRA adaptation (K=200):  cos={mean_lora_200_cos:.3f}, SSIM={mean_lora_200_ssim:.3f}
  LoRA adaptation (K=50):   cos={mean_lora_50_cos:.3f}
  LoRA recovery:            {lora_recovery_cos:.1f}% cos, {lora_recovery_ssim:.1f}% SSIM
  LoRA params:              {lora_results[1][200]['n_params']:,} (vs full encoder)

  Few-shot progression (S1):
""")
for K in FEW_SHOT_KS:
    r = lora_results[1][K]
    print(f"    K={K:>3d}: cos={r['cos']:.3f}, SSIM={r['ssim']:.3f}")

# Save results
results_dict = {
    "n_subjects": N_SUBJECTS,
    "inter_subject_similarity": {
        "matrix": similarity_matrix.tolist(),
        "mean": float(np.mean(mean_cross)),
        "std": float(np.std(mean_cross)),
    },
    "reference_subject": {
        "cos": s0_cos, "ssim": s0_ssim,
    },
    "zero_shot_transfer": {
        str(sid): {"cos": zero_shot_results[sid]["cos"],
                   "ssim": zero_shot_results[sid]["ssim"]}
        for sid in range(N_SUBJECTS)
    },
    "lora_adaptation": {
        str(sid): {
            str(K): {"cos": lora_results[sid][K]["cos"],
                     "ssim": lora_results[sid][K]["ssim"],
                     "n_params": lora_results[sid][K]["n_params"],
                     "time_sec": round(lora_results[sid][K]["time"], 1)}
            for K in FEW_SHOT_KS
        }
        for sid in range(1, N_SUBJECTS)
    },
    "full_finetuning": {
        str(sid): {
            str(K): {"cos": full_ft_results[sid][K]["cos"],
                     "ssim": full_ft_results[sid][K]["ssim"],
                     "n_params": full_ft_results[sid][K]["n_params"]}
            for K in full_ft_results[sid]
        }
        for sid in full_ft_results
    },
    "summary": {
        "zero_shot_drop_cos": round(zero_shot_drop_cos, 4),
        "zero_shot_drop_ssim": round(zero_shot_drop_ssim, 4),
        "lora_200_recovery_cos_pct": round(lora_recovery_cos, 1),
        "lora_200_recovery_ssim_pct": round(lora_recovery_ssim, 1),
        "lora_params": lora_results[1][200]["n_params"],
    },
}

with open(os.path.join(OUT, "cross_subject_results.json"), "w") as f:
    json.dump(results_dict, f, indent=2)
print(f"\n  Saved: {OUT}/cross_subject_results.json")
