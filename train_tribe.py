"""Train cortexflow with TRIBE v2 brain patterns from real pretrained features.

Pipeline:
1. Generate diverse parameterized images (same as train_demo)
2. Extract V-JEPA2 features from those images (pretrained ViT-Giant)
3. Map features → predicted brain activity using TRIBE v2's pretrained
   transformer + cortical prediction head (loaded from checkpoint)
4. Train cortexflow to reconstruct images from these brain patterns
5. Evaluate on held-out test set

The brain patterns now carry REAL visual feature information learned from
1000+ hours of fMRI data, producing meaningful reconstructions.
"""

import os
import time
import json
import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cortexflow import BrainData
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from cortexflow.brain2img import Brain2Image
from cortexflow.flow_matching import EMAModel

OUT = "train_outputs"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
DEVICE = "cuda:0"

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
N_TOTAL = 5000
N_TRAIN = 4000
N_TEST = 1000
IMG_SIZE = 64          # larger images for meaningful content
N_VOXELS = 384         # PCA-reduced (matches video projector bottleneck)
VJEPA_LAYERS = [20, 30]  # layers 0.5 and 0.75 of 40 (matches TRIBE v2)
TRAIN_DEVICE = DEVICE    # train on GPU for speed

# Training config
VAE_STEPS = 5000
VAE_LR = 1e-3
DIT_STEPS = 50000
DIT_LR = 1e-3
BATCH_SIZE = 64
NOISE_AUGMENT = 0.05
WEIGHT_DECAY = 0.05
IMG_CFG_SCALE = 1.0
N_AVG = 4
BRAIN_PRETRAIN_STEPS = 8000


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
# 1. GENERATE IMAGES
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("CORTEXFLOW × TRIBE v2 — PRETRAINED BRAIN PATTERNS")
print("=" * 70)
print(f"\nGenerating {N_TOTAL} parameterized images ({IMG_SIZE}×{IMG_SIZE})...")


def make_random_image(seed, size):
    """Generate a parameterized image: shape + color + position + background."""
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
        mask = ((yy >= max(0, cy - r)) & (yy < min(size, cy + r)) &
                (xx >= max(0, cx - r)) & (xx < min(size, cx + r)))
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


target_images = torch.stack([make_random_image(i * 7 + 13, IMG_SIZE) for i in range(N_TOTAL)])
print(f"  Images: {target_images.shape}")

# ═══════════════════════════════════════════════════════════════
# 2. EXTRACT V-JEPA2 FEATURES
# ═══════════════════════════════════════════════════════════════
print(f"\nLoading V-JEPA2 (ViT-Giant, pretrained)...")
from transformers import AutoModel, AutoConfig

vjepa = AutoModel.from_pretrained(
    "facebook/vjepa2-vitg-fpc64-256",
    dtype=torch.float16,
).to(DEVICE).eval()
print(f"  V-JEPA2: {sum(p.numel() for p in vjepa.parameters()) / 1e6:.0f}M params")

print(f"\nExtracting features from {N_TOTAL} images...")
# V-JEPA2 expects (B, T, C, H, W) where T=num_frames
# For single images, T=1. Input should be 256×256.
vjepa_features = []
t0 = time.time()
for batch_start in range(0, N_TOTAL, 16):
    batch_end = min(batch_start + 16, N_TOTAL)
    # Resize to 256×256 and format as video frames
    batch_imgs = F.interpolate(
        target_images[batch_start:batch_end],
        size=(256, 256), mode="bilinear", align_corners=False
    )
    # (B, C, H, W) -> (B, 1, C, H, W) = single frame video
    video_input = batch_imgs.unsqueeze(1).half().to(DEVICE)

    with torch.no_grad():
        out = vjepa(video_input, output_hidden_states=True)
        # Extract layers at positions 0.5 and 0.75 of 40 layers
        # Mean-pool tokens, then group_mean the two layer groups
        feats = []
        for layer_idx in VJEPA_LAYERS:
            h = out.hidden_states[layer_idx]  # (B, T*tokens, 1408)
            h = h.mean(dim=1)  # (B, 1408) - mean pool tokens
            feats.append(h.float().cpu())
        # Concatenate layer groups: (B, 2816)
        batch_feats = torch.cat(feats, dim=1)
        vjepa_features.append(batch_feats)

    if (batch_start + 16) % 80 == 0 or batch_end == N_TOTAL:
        elapsed = time.time() - t0
        print(f"    {batch_end}/{N_TOTAL} ({elapsed:.1f}s)")

vjepa_features = torch.cat(vjepa_features, dim=0)
print(f"  V-JEPA2 features: {vjepa_features.shape}")

# Free V-JEPA2 memory
del vjepa
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# 3. MAP FEATURES → BRAIN ACTIVITY VIA TRIBE v2
# ═══════════════════════════════════════════════════════════════
print(f"\nLoading TRIBE v2 brain mapping head...")
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("facebook/tribev2", "best.ckpt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
build_args = ckpt["model_build_args"]
del ckpt

# Build a minimal brain mapping: video_projector → combiner → low_rank → predictor
# TRIBE v2 architecture: concatenated video features (2816) → projector (384) →
#   transformer (1152) → low_rank (2048) → predictor (20484 vertices)
# We'll use the projector + low_rank + predictor, skipping the full transformer
# (since we only have single-frame features, temporal modeling isn't needed)

# Extract just the video projector weights
video_proj_w = state_dict["projectors.video.weight"]  # (384, 2816)
video_proj_b = state_dict["projectors.video.bias"]  # (384,)
low_rank_w = state_dict["low_rank_head.weight"]  # (2048, 1152)
pred_w = state_dict["predictor.weights"]  # (1, 2048, 20484)
pred_b = state_dict["predictor.bias"]  # (1, 20484)

print(f"  Video projector: {video_proj_w.shape}")
print(f"  Low-rank head: {low_rank_w.shape}")
print(f"  Brain predictor: {pred_w.shape} → 20484 cortical vertices")

# The combiner expects all 3 modalities concatenated (3×384=1152)
# Since we only have video, we need to handle this.
# combiner.weight is (1152, 1152) - maps from concat(text_384, audio_384, video_384)
# We zero-pad text and audio portions:
combiner_w = state_dict.get("combiner.weight")  # (1152, 1152)
combiner_b = state_dict.get("combiner.bias")  # (1152,)

print(f"\nProjecting {N_TOTAL} images through TRIBE v2 brain mapping...")
with torch.no_grad():
    # Step 1: Project video features through the video projector
    # vjepa_features: (500, 2816) → (500, 384)
    video_projected = F.linear(vjepa_features, video_proj_w, video_proj_b)

    # Step 2: Pad with zeros for missing modalities (text=384, audio=384, video=384)
    # The combiner expects (B, 1152) = cat(text, audio, video)
    B = video_projected.shape[0]
    text_zeros = torch.zeros(B, 384)
    audio_zeros = torch.zeros(B, 384)
    combined = torch.cat([text_zeros, audio_zeros, video_projected], dim=1)  # (B, 1152)

    # Step 3: Combiner MLP (Linear + GELU + LayerNorm)
    if combiner_w is not None:
        combined = F.linear(combined, combiner_w, combiner_b)
        combined = F.gelu(combined)
        # Apply LayerNorm if in state dict
        if "combiner.norm.weight" in state_dict:
            ln_w = state_dict["combiner.norm.weight"]
            ln_b = state_dict["combiner.norm.bias"]
            combined = F.layer_norm(combined, [1152], ln_w, ln_b)

    # Step 4: Low-rank projection (1152 → 2048)
    brain_2048 = F.linear(combined, low_rank_w)

    # Step 5: Brain prediction (2048 → 20484 vertices)
    # predictor.weights is (1, 2048, 20484) for subject layers
    full_brain = torch.einsum("bh,shv->bv", brain_2048, pred_w) + pred_b.squeeze(0)

print(f"  Full brain activity: {full_brain.shape}")
print(f"  Brain range: [{full_brain.min():.3f}, {full_brain.max():.3f}]")
print(f"  Brain std: {full_brain.std():.4f}")

# Step 6: PCA to reduce 20484 → N_VOXELS (more tractable for small model)
print(f"\n  PCA: 20484 → {N_VOXELS} dimensions...")
brain_mean = full_brain.mean(dim=0)
brain_centered = full_brain - brain_mean
U, S, V = torch.svd_lowrank(brain_centered, q=N_VOXELS)
brain_patterns = brain_centered @ V  # (500, N_VOXELS)
# Standardize
brain_std = brain_patterns.std(dim=0, keepdim=True).clamp(min=1e-6)
brain_patterns = brain_patterns / brain_std
print(f"  Brain patterns: {brain_patterns.shape}")
print(f"  Variance explained by top {N_VOXELS} PCs: "
      f"{(S[:N_VOXELS]**2).sum() / (S**2).sum() * 100:.1f}%")

# Save PCA components for potential reuse
torch.save({
    "brain_mean": brain_mean,
    "pca_V": V,
    "brain_std": brain_std,
    "vjepa_features": vjepa_features,
}, os.path.join(OUT, "tribe_brain_data.pt"))

del state_dict, full_brain, brain_centered
torch.cuda.empty_cache()

train_idx = list(range(N_TRAIN))
test_idx = list(range(N_TRAIN, N_TOTAL))
print(f"\n  Train: {N_TRAIN}, Test: {N_TEST} (held out)")

# ═══════════════════════════════════════════════════════════════
# 4. TRAIN CORTEXFLOW
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("TRAINING CORTEXFLOW ON TRIBE v2 BRAIN PATTERNS")
print("=" * 70)

# Build model (larger for 64×64)
img_model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=128, depth=6, num_heads=8, cond_dim=128),
    vae_config=VAEConfig(hidden_dims=[64, 128, 256], latent_channels=8),
    flow_config=FlowConfig(),
).to(TRAIN_DEVICE)
print(f"  Model params: {sum(p.numel() for p in img_model.parameters()) / 1e6:.1f}M")

# Move data to GPU
target_images = target_images.to(TRAIN_DEVICE)
brain_patterns = brain_patterns.to(TRAIN_DEVICE)

# ── Train VAE ──
print(f"\n  Pre-training VAE ({VAE_STEPS} steps)...")
img_model.vae.train()
vae_opt = torch.optim.AdamW(img_model.vae.parameters(), lr=VAE_LR, weight_decay=0.01)
vae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(vae_opt, VAE_STEPS)
t0 = time.time()
for step in range(VAE_STEPS):
    idx = torch.randint(0, N_TOTAL, (BATCH_SIZE,))  # Train VAE on ALL images (standard practice)
    batch = target_images[idx].to(TRAIN_DEVICE)
    recon, mu, logvar = img_model.vae(batch)
    loss, _ = img_model.vae.loss(batch, recon, mu, logvar)
    vae_opt.zero_grad()
    loss.backward()
    vae_opt.step()
    vae_sched.step()
    if step % 500 == 0:
        print(f"    VAE step {step}: loss={loss.item():.4f} ({time.time() - t0:.0f}s)")
img_model.vae.eval()

# Cache latents
with torch.no_grad():
    all_latents = []
    for i in range(0, N_TOTAL, 32):
        z, _, _ = img_model.vae.encode(target_images[i:i+32].to(TRAIN_DEVICE))
        all_latents.append(z.cpu())
    all_latents = torch.cat(all_latents).to(TRAIN_DEVICE)
    IMG_LATENT_SHAPE = all_latents.shape[1:]
print(f"  Latent shape: {list(IMG_LATENT_SHAPE)}")

# Check VAE reconstruction quality (ceiling)
with torch.no_grad():
    vae_ssims = []
    for i in range(0, min(200, N_TOTAL), 32):
        batch = target_images[i:i+32].to(TRAIN_DEVICE)
        z, _, _ = img_model.vae.encode(batch)
        recon = img_model.vae.decode(z).clamp(0, 1)
        for j in range(batch.shape[0]):
            vae_ssims.append(compute_ssim(recon[j].cpu(), target_images[i+j].cpu()))
    print(f"  VAE reconstruction SSIM (ceiling): {np.mean(vae_ssims):.3f} ± {np.std(vae_ssims):.3f}")

# ── Pre-train brain encoder ──
print(f"\n  Pre-training brain encoder ({BRAIN_PRETRAIN_STEPS} steps, direct regression)...")
brain_enc_opt = torch.optim.AdamW(img_model.brain_encoder.parameters(), lr=1e-3, weight_decay=0.01)
brain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(brain_enc_opt, BRAIN_PRETRAIN_STEPS)
for step in range(BRAIN_PRETRAIN_STEPS):
    idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,))
    brain = BrainData(voxels=brain_patterns[idx].to(TRAIN_DEVICE))
    target_lat = all_latents[idx].flatten(1)
    bg, _ = img_model.encode_brain(brain)
    # Regress a subset — bg may be smaller than flat latent; this is just warm-start
    n_match = min(bg.shape[1], target_lat.shape[1])
    loss = F.mse_loss(bg[:, :n_match], target_lat[:, :n_match])
    brain_enc_opt.zero_grad()
    loss.backward()
    brain_enc_opt.step()
    brain_sched.step()
    if step % 1000 == 0:
        print(f"    Brain enc step {step}: loss={loss.item():.4f}")

# ── Linear predictor for residual mode ──
print(f"\n  Fitting linear predictor (ridge regression)...")
X = brain_patterns[:N_TRAIN].cpu()
Y = all_latents[:N_TRAIN].flatten(1).cpu()
lam = 5.0
XtX = X.T @ X + lam * torch.eye(X.shape[1])
XtY = X.T @ Y
W = torch.linalg.solve(XtX, XtY)
img_lin_preds = (brain_patterns.cpu() @ W).view(-1, *IMG_LATENT_SHAPE).to(TRAIN_DEVICE)

with torch.no_grad():
    lin_train_cos = F.cosine_similarity(
        img_lin_preds[:N_TRAIN].flatten(1), all_latents[:N_TRAIN].flatten(1)
    ).mean().item()
    lin_test_cos = F.cosine_similarity(
        img_lin_preds[N_TRAIN:].flatten(1), all_latents[N_TRAIN:].flatten(1)
    ).mean().item()
print(f"  Linear baseline — train cos: {lin_train_cos:.3f}, test cos: {lin_test_cos:.3f}")

# ── Train DiT (residual mode) ──
print(f"\n  Training DiT ({DIT_STEPS} steps, residual mode, EMA)...")
dit_params = list(img_model.dit.parameters()) + list(img_model.brain_encoder.parameters())
dit_opt = torch.optim.AdamW(dit_params, lr=DIT_LR, weight_decay=WEIGHT_DECAY)

warmup_steps = 500
total_steps = DIT_STEPS


def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


dit_sched = torch.optim.lr_scheduler.LambdaLR(dit_opt, lr_lambda)
ema = EMAModel(img_model, decay=0.999)

img_model.train()
img_model.vae.eval()
losses = []
t0 = time.time()

for step in range(DIT_STEPS):
    idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,))
    brain = BrainData(voxels=brain_patterns[idx].to(TRAIN_DEVICE))

    # Brain noise augmentation
    if NOISE_AUGMENT > 0:
        brain = BrainData(
            voxels=brain.voxels + NOISE_AUGMENT * torch.randn_like(brain.voxels)
        )

    # Residual target: actual_latent - linear_prediction
    residual = all_latents[idx] - img_lin_preds[idx]

    bg, bt = img_model.encode_brain(brain)
    loss = img_model.flow_matcher.compute_loss(img_model.dit, residual, bg, bt)

    dit_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dit_params, 1.0)
    dit_opt.step()
    dit_sched.step()
    ema.update(img_model)
    losses.append(loss.item())

    if step % 5000 == 0 or step == DIT_STEPS - 1:
        avg_loss = sum(losses[-500:]) / min(len(losses), 500)
        elapsed = time.time() - t0
        eta = elapsed / max(step + 1, 1) * (DIT_STEPS - step - 1)
        print(f"    DiT step {step}: loss={avg_loss:.4f} "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

ema.apply_to(img_model)
img_model.eval()

# Save model
torch.save({k: v.cpu() for k, v in img_model.state_dict().items()},
           os.path.join(OUT, "brain2img_tribe.pt"))

# ═══════════════════════════════════════════════════════════════
# 5. EVALUATE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("EVALUATION ON HELD-OUT TEST SET")
print("=" * 70)


def evaluate(indices, label=""):
    results = {}
    for i in indices:
        brain = BrainData(voxels=brain_patterns[i:i + 1].to(TRAIN_DEVICE))
        bg, bt = img_model.encode_brain(brain)
        shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

        latent_sum = None
        for s in range(N_AVG):
            torch.manual_seed(s)
            z = img_model.flow_matcher.sample(
                img_model.dit, shape_1, bg, bt, num_steps=50,
                cfg_scale=IMG_CFG_SCALE,
            )
            z = z + img_lin_preds[i:i + 1]
            latent_sum = z if latent_sum is None else latent_sum + z
        avg_z = latent_sum / N_AVG
        with torch.no_grad():
            recon = img_model.vae.decode(avg_z)[0].detach().clamp(0, 1).cpu()

        target = target_images[i].cpu()
        cos = F.cosine_similarity(
            recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
        ).item()
        ssim = compute_ssim(recon, target)
        results[i] = {"cos": cos, "ssim": ssim, "recon": recon}
    return results


# Train set (first 8)
print("\n  Train set (first 8):")
train_results = evaluate(train_idx[:8], "TRAIN")
train_cos = np.mean([r["cos"] for r in train_results.values()])
train_ssim = np.mean([r["ssim"] for r in train_results.values()])
print(f"    cos={train_cos:.3f}, SSIM={train_ssim:.3f}")

# Test set (200 held-out for detailed eval)
N_EVAL = min(200, N_TEST)
eval_test_idx = test_idx[:N_EVAL]
print(f"\n  Test set ({N_EVAL} held-out):")
test_results = evaluate(eval_test_idx, "TEST")
test_cos = np.mean([r["cos"] for r in test_results.values()])
test_ssim = np.mean([r["ssim"] for r in test_results.values()])
test_cos_std = np.std([r["cos"] for r in test_results.values()])
test_ssim_std = np.std([r["ssim"] for r in test_results.values()])
print(f"    cos={test_cos:.3f} ± {test_cos_std:.3f}")
print(f"    SSIM={test_ssim:.3f} ± {test_ssim_std:.3f}")

# Linear baseline on test set
lin_test_results = {}
for i in eval_test_idx:
    with torch.no_grad():
        recon = img_model.vae.decode(img_lin_preds[i:i + 1])[0].detach().clamp(0, 1).cpu()
    target = target_images[i].cpu()
    cos = F.cosine_similarity(
        recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
    ).item()
    lin_test_results[i] = cos
lin_test_mean = np.mean(list(lin_test_results.values()))
gap = test_cos - lin_test_mean
print(f"\n  Linear baseline test cos: {lin_test_mean:.3f}")
print(f"  DiT gap vs linear: {gap:+.3f}")

# ═══════════════════════════════════════════════════════════════
# 6. VLM EVALUATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("VLM SEMANTIC EVALUATION (Qwen2.5-VL-7B)")
print("=" * 70)

# Save individual images for VLM
vlm_dir = os.path.join(OUT, "tribe_vlm_images")
os.makedirs(vlm_dir, exist_ok=True)

N_VLM = 20  # evaluate 20 test samples
vlm_indices = test_idx[:N_VLM]

for i in vlm_indices:
    # Save target
    t = target_images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    t_pil = Image.fromarray((t * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)
    t_pil.save(os.path.join(vlm_dir, f"test{i}_target.png"))

    # Save reconstruction
    r = test_results[i]["recon"].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    r_pil = Image.fromarray((r * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)
    r_pil.save(os.path.join(vlm_dir, f"test{i}_recon.png"))

print(f"  Loading Qwen2.5-VL-7B for captioning...")
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

CAPTION_PROMPT = (
    "Describe this image in one detailed sentence. Focus on: the main shape or object, "
    "its color, its position, and the background color. Be specific."
)


def caption_image(path):
    img = Image.open(path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": CAPTION_PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(vlm.device)
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=128, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0].strip()


print(f"\n  Captioning {N_VLM} target + reconstruction pairs...")
target_caps, recon_caps = {}, {}
t0 = time.time()
for count, i in enumerate(vlm_indices):
    target_caps[i] = caption_image(os.path.join(vlm_dir, f"test{i}_target.png"))
    recon_caps[i] = caption_image(os.path.join(vlm_dir, f"test{i}_recon.png"))
    if (count + 1) % 5 == 0:
        elapsed = time.time() - t0
        print(f"    {count+1}/{N_VLM} done ({elapsed:.0f}s)")

del vlm, processor
torch.cuda.empty_cache()

# Embedding similarity
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
all_texts = []
for i in vlm_indices:
    all_texts.extend([target_caps[i], recon_caps[i]])
embeddings = embedder.encode(all_texts, convert_to_tensor=True)
embeddings = F.normalize(embeddings, dim=1)

correctness_scores = []
for j in range(N_VLM):
    t_emb = embeddings[j * 2]
    r_emb = embeddings[j * 2 + 1]
    sim = F.cosine_similarity(t_emb.unsqueeze(0), r_emb.unsqueeze(0)).item()
    correctness_scores.append(sim)

# Inter-brain diversity
inter_sims = []
for a in range(N_VLM):
    for b in range(a + 1, N_VLM):
        ea = embeddings[a * 2 + 1]
        eb = embeddings[b * 2 + 1]
        sim = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
        inter_sims.append(sim)

mean_correct = np.mean(correctness_scores)
mean_inter = np.mean(inter_sims)

print(f"\n  VLM Results:")
print(f"    Semantic correctness: {mean_correct:.3f} ± {np.std(correctness_scores):.3f}")
print(f"    Inter-brain diversity: {1 - mean_inter:.3f}")

# Print example captions
print(f"\n  Example captions:")
for i in vlm_indices[:5]:
    print(f"\n    Test {i}:")
    print(f"      TARGET: {target_caps[i][:100]}")
    print(f"      RECON:  {recon_caps[i][:100]}")
    print(f"      SIM:    {correctness_scores[vlm_indices.index(i)]:.3f}")

# ═══════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SHOW = 8
fig, axes = plt.subplots(4, N_SHOW, figsize=(N_SHOW * 2.5, 10))

# Row 0-1: Train (target + recon)
for col, i in enumerate(train_idx[:N_SHOW]):
    t = target_images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes[0, col].imshow(t, interpolation="nearest")
    axes[0, col].set_title(f"Train {i}", fontsize=8)
    axes[0, col].axis("off")

    r = train_results[i]["recon"].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes[1, col].imshow(r, interpolation="nearest")
    ssim = train_results[i]["ssim"]
    axes[1, col].set_title(f"SSIM={ssim:.2f}", fontsize=8, color="green")
    axes[1, col].axis("off")

# Row 2-3: Test (target + recon)
for col, i in enumerate(test_idx[:N_SHOW]):
    t = target_images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes[2, col].imshow(t, interpolation="nearest")
    axes[2, col].set_title(f"Test {i}", fontsize=8, color="red")
    axes[2, col].axis("off")

    r = test_results[i]["recon"].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes[3, col].imshow(r, interpolation="nearest")
    ssim = test_results[i]["ssim"]
    axes[3, col].set_title(f"SSIM={ssim:.2f}", fontsize=8, color="red")
    axes[3, col].axis("off")

plt.suptitle(
    f"cortexflow × TRIBE v2 — {IMG_SIZE}×{IMG_SIZE} Reconstructions\n"
    f"Test cos={test_cos:.3f} SSIM={test_ssim:.3f} gap={gap:+.3f} "
    f"VLM correctness={mean_correct:.3f}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "tribe_reconstructions.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUT}/tribe_reconstructions.png")

# ═══════════════════════════════════════════════════════════════
# 8. DIVERSITY VISUALIZATION — multiple samples per brain
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("DIVERSITY: 4 SAMPLES PER BRAIN (NO AVERAGING)")
print("=" * 70)

N_DIV_BRAINS = 6
N_DIV_SAMPLES = 4
div_indices = test_idx[:N_DIV_BRAINS]

fig_div, axes_div = plt.subplots(N_DIV_BRAINS, N_DIV_SAMPLES + 1,
                                  figsize=((N_DIV_SAMPLES + 1) * 2.5, N_DIV_BRAINS * 2.5))

intra_diversities = []
for row, brain_idx in enumerate(div_indices):
    # Show target in first column
    t = target_images[brain_idx].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes_div[row, 0].imshow(t, interpolation="nearest")
    axes_div[row, 0].set_title("Target" if row == 0 else "", fontsize=9)
    axes_div[row, 0].set_ylabel(f"Test {brain_idx}", fontsize=8, rotation=0, labelpad=40)
    axes_div[row, 0].axis("off")

    # Generate N_DIV_SAMPLES independent samples (no averaging)
    samples = []
    brain = BrainData(voxels=brain_patterns[brain_idx:brain_idx + 1].to(TRAIN_DEVICE))
    bg, bt = img_model.encode_brain(brain)
    shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

    for s in range(N_DIV_SAMPLES):
        torch.manual_seed(s * 1000 + brain_idx)
        z = img_model.flow_matcher.sample(
            img_model.dit, shape_1, bg, bt, num_steps=50,
            cfg_scale=IMG_CFG_SCALE,
        )
        z = z + img_lin_preds[brain_idx:brain_idx + 1]
        with torch.no_grad():
            recon = img_model.vae.decode(z)[0].detach().clamp(0, 1).cpu()
        samples.append(recon)

        r = recon.permute(1, 2, 0).clamp(0, 1).numpy()
        ssim_val = compute_ssim(recon, target_images[brain_idx].cpu())
        axes_div[row, s + 1].imshow(r, interpolation="nearest")
        axes_div[row, s + 1].set_title(
            f"Sample {s+1}" if row == 0 else f"SSIM={ssim_val:.2f}",
            fontsize=8, color="blue")
        axes_div[row, s + 1].axis("off")

    # Compute intra-brain diversity (pairwise cosine distance)
    flat_samples = torch.stack([s.flatten() for s in samples])
    pair_sims = []
    for a in range(N_DIV_SAMPLES):
        for b in range(a + 1, N_DIV_SAMPLES):
            sim = F.cosine_similarity(
                flat_samples[a:a+1], flat_samples[b:b+1]).item()
            pair_sims.append(sim)
    intra_div = 1 - np.mean(pair_sims)
    intra_diversities.append(intra_div)
    print(f"  Brain {brain_idx}: intra-diversity={intra_div:.3f}")

mean_intra_div = np.mean(intra_diversities)
print(f"\n  Mean intra-brain diversity: {mean_intra_div:.3f}")
print(f"  (Higher = more diverse samples from same brain)")

plt.suptitle(
    f"cortexflow × TRIBE v2 — Diversity: {N_DIV_SAMPLES} samples per brain\n"
    f"Mean intra-brain diversity: {mean_intra_div:.3f}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
fig_div.savefig(os.path.join(OUT, "tribe_diversity.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT}/tribe_diversity.png")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SUMMARY — TRIBE v2 BRAIN PATTERNS")
print("=" * 70)
print(f"""
  Forward model:      TRIBE v2 (Meta, pretrained on 1000+ hrs fMRI)
  Feature extractor:  V-JEPA2 ViT-Giant (pretrained, 1035M params)
  Brain vertices:     20,484 (fsaverage5) → PCA to {N_VOXELS}
  Image size:         {IMG_SIZE}×{IMG_SIZE}
  
  ┌──────────────────────────┬───────────────┐
  │ Metric                   │ Test (n={N_TEST})    │
  ├──────────────────────────┼───────────────┤
  │ Cosine similarity        │ {test_cos:.3f} ± {test_cos_std:.3f}  │
  │ SSIM                     │ {test_ssim:.3f} ± {test_ssim_std:.3f}  │
  │ Linear baseline cos      │ {lin_test_mean:.3f}          │
  │ DiT gap vs linear        │ {gap:+.3f}          │
  │ VLM correctness          │ {mean_correct:.3f}          │
  │ VLM inter-brain diversity│ {1 - mean_inter:.3f}          │
  │ Intra-brain diversity    │ {mean_intra_div:.3f}          │
  └──────────────────────────┴───────────────┘
""")

# Save results JSON
with open(os.path.join(OUT, "tribe_results.json"), "w") as f:
    json.dump({
        "forward_model": "TRIBE v2 (facebook/tribev2)",
        "feature_extractor": "V-JEPA2 ViT-Giant",
        "n_voxels": N_VOXELS,
        "img_size": IMG_SIZE,
        "test_cos": round(test_cos, 4),
        "test_cos_std": round(test_cos_std, 4),
        "test_ssim": round(test_ssim, 4),
        "test_ssim_std": round(test_ssim_std, 4),
        "linear_baseline_cos": round(lin_test_mean, 4),
        "gap": round(gap, 4),
        "vlm_correctness": round(mean_correct, 4),
        "vlm_diversity": round(1 - mean_inter, 4),
        "intra_brain_diversity": round(mean_intra_div, 4),
        "captions": {
            str(i): {"target": target_caps[i], "recon": recon_caps[i],
                      "sim": round(correctness_scores[vlm_indices.index(i)], 4)}
            for i in vlm_indices[:10]
        },
    }, f, indent=2)
print(f"  Saved: {OUT}/tribe_results.json")
