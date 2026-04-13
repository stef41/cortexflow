"""Train cortexflow on NATURAL IMAGES with TRIBE v2 brain patterns.

Key difference from train_tribe.py: uses real photographs (STL-10) instead of
synthetic parameterized shapes. This tests whether cortexflow + TRIBE v2 can
reconstruct complex natural scenes from predicted brain activity.

Pipeline:
1. Load natural images from STL-10 (96×96 photographs, resized to 128×128)
2. Extract V-JEPA2 features (pretrained ViT-Giant, 1035M params)
3. Map features → predicted fMRI via TRIBE v2 (20,484 cortical vertices)
4. PCA reduce to N_VOXELS dimensions
5. Train cortexflow (bigger model for 128×128)
6. Evaluate on held-out test set + VLM captioning + diversity

Scientific question: Does the DiT's nonlinear capacity matter MORE for
natural images than for simple shapes? (Hypothesis: yes — natural images
have richer structure that linear regression can't fully capture.)
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
from torchvision.datasets import STL10
from torchvision import transforms

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
N_TOTAL = 10000
N_TRAIN = 8000
N_TEST = 2000
IMG_SIZE = 128
N_VOXELS = 384         # PCA dims (full rank of TRIBE v2 video projector)
VJEPA_LAYERS = [20, 30]

# Training config — bigger model for 128×128 natural images
VAE_STEPS = 8000
VAE_LR = 1e-3
DIT_STEPS = 60000
DIT_LR = 1e-3
BATCH_SIZE = 64
NOISE_AUGMENT = 0.05
WEIGHT_DECAY = 0.05
IMG_CFG_SCALE = 1.5    # slight CFG for sharper outputs
N_AVG = 4
BRAIN_PRETRAIN_STEPS = 10000


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
# 1. LOAD NATURAL IMAGES (STL-10)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("CORTEXFLOW × TRIBE v2 — NATURAL IMAGE RECONSTRUCTION")
print("=" * 70)

print(f"\nLoading STL-10 natural images...")
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = STL10("/tmp/stl10", split="unlabeled", download=False, transform=transform)
print(f"  STL-10 unlabeled: {len(dataset)} images")

# Sample N_TOTAL random images
indices = torch.randperm(len(dataset))[:N_TOTAL].tolist()
target_images = torch.stack([dataset[i][0] for i in indices])
print(f"  Selected {N_TOTAL} images: {target_images.shape}")
print(f"  Value range: [{target_images.min():.3f}, {target_images.max():.3f}]")

# ═══════════════════════════════════════════════════════════════
# 2. EXTRACT V-JEPA2 FEATURES
# ═══════════════════════════════════════════════════════════════
print(f"\nLoading V-JEPA2 (ViT-Giant, pretrained)...")
from transformers import AutoModel

vjepa = AutoModel.from_pretrained(
    "facebook/vjepa2-vitg-fpc64-256",
    dtype=torch.float16,
).to(DEVICE).eval()
print(f"  V-JEPA2: {sum(p.numel() for p in vjepa.parameters()) / 1e6:.0f}M params")

print(f"\nExtracting features from {N_TOTAL} images...")
vjepa_features = []
t0 = time.time()
VJEPA_BATCH = 16
for batch_start in range(0, N_TOTAL, VJEPA_BATCH):
    batch_end = min(batch_start + VJEPA_BATCH, N_TOTAL)
    batch_imgs = F.interpolate(
        target_images[batch_start:batch_end],
        size=(256, 256), mode="bilinear", align_corners=False
    )
    video_input = batch_imgs.unsqueeze(1).half().to(DEVICE)

    with torch.no_grad():
        out = vjepa(video_input, output_hidden_states=True)
        feats = []
        for layer_idx in VJEPA_LAYERS:
            h = out.hidden_states[layer_idx]
            h = h.mean(dim=1)
            feats.append(h.float().cpu())
        batch_feats = torch.cat(feats, dim=1)
        vjepa_features.append(batch_feats)

    if (batch_start + VJEPA_BATCH) % 160 == 0 or batch_end == N_TOTAL:
        elapsed = time.time() - t0
        print(f"    {batch_end}/{N_TOTAL} ({elapsed:.1f}s)")

vjepa_features = torch.cat(vjepa_features, dim=0)
print(f"  V-JEPA2 features: {vjepa_features.shape}")

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
del ckpt

video_proj_w = state_dict["projectors.video.weight"]
video_proj_b = state_dict["projectors.video.bias"]
low_rank_w = state_dict["low_rank_head.weight"]
pred_w = state_dict["predictor.weights"]
pred_b = state_dict["predictor.bias"]
combiner_w = state_dict.get("combiner.weight")
combiner_b = state_dict.get("combiner.bias")

print(f"  Video projector: {video_proj_w.shape}")
print(f"  Brain predictor: {pred_w.shape} → 20484 cortical vertices")

print(f"\nProjecting {N_TOTAL} images through TRIBE v2 brain mapping...")
with torch.no_grad():
    video_projected = F.linear(vjepa_features, video_proj_w, video_proj_b)
    B = video_projected.shape[0]
    text_zeros = torch.zeros(B, 384)
    audio_zeros = torch.zeros(B, 384)
    combined = torch.cat([text_zeros, audio_zeros, video_projected], dim=1)

    if combiner_w is not None:
        combined = F.linear(combined, combiner_w, combiner_b)
        combined = F.gelu(combined)
        if "combiner.norm.weight" in state_dict:
            ln_w = state_dict["combiner.norm.weight"]
            ln_b = state_dict["combiner.norm.bias"]
            combined = F.layer_norm(combined, [1152], ln_w, ln_b)

    brain_2048 = F.linear(combined, low_rank_w)
    full_brain = torch.einsum("bh,shv->bv", brain_2048, pred_w) + pred_b.squeeze(0)

print(f"  Full brain activity: {full_brain.shape}")
print(f"  Brain range: [{full_brain.min():.3f}, {full_brain.max():.3f}]")
print(f"  Brain std: {full_brain.std():.4f}")

# PCA
print(f"\n  PCA: 20484 → {N_VOXELS} dimensions...")
brain_mean = full_brain.mean(dim=0)
brain_centered = full_brain - brain_mean
U, S, V = torch.svd_lowrank(brain_centered, q=N_VOXELS)
brain_patterns = brain_centered @ V
brain_std = brain_patterns.std(dim=0, keepdim=True).clamp(min=1e-6)
brain_patterns = brain_patterns / brain_std
print(f"  Brain patterns: {brain_patterns.shape}")
print(f"  Variance explained: {(S[:N_VOXELS]**2).sum() / (S**2).sum() * 100:.1f}%")

torch.save({
    "brain_mean": brain_mean, "pca_V": V, "brain_std": brain_std,
    "vjepa_features": vjepa_features,
}, os.path.join(OUT, "natural_brain_data.pt"))

del state_dict, full_brain, brain_centered, vjepa_features
torch.cuda.empty_cache()

train_idx = list(range(N_TRAIN))
test_idx = list(range(N_TRAIN, N_TOTAL))
print(f"\n  Train: {N_TRAIN}, Test: {N_TEST}")

# ═══════════════════════════════════════════════════════════════
# 4. TRAIN CORTEXFLOW
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("TRAINING CORTEXFLOW ON NATURAL IMAGES")
print("=" * 70)

# Bigger model for 128×128
# VAE: [64,128,256] + 8 latent → latent (8, 16, 16) = 2048 dims, 64 DiT tokens
# DiT: hidden=256, depth=8, heads=8 → ~30M params
img_model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=256, depth=8, num_heads=8, cond_dim=256),
    vae_config=VAEConfig(hidden_dims=[64, 128, 256], latent_channels=8),
    flow_config=FlowConfig(),
).to(DEVICE)
print(f"  Model params: {sum(p.numel() for p in img_model.parameters()) / 1e6:.1f}M")

target_images = target_images.to(DEVICE)
brain_patterns = brain_patterns.to(DEVICE)

# ── Train VAE ──
print(f"\n  Pre-training VAE ({VAE_STEPS} steps)...")
img_model.vae.train()
vae_opt = torch.optim.AdamW(img_model.vae.parameters(), lr=VAE_LR, weight_decay=0.01)
vae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(vae_opt, VAE_STEPS)
t0 = time.time()
for step in range(VAE_STEPS):
    idx = torch.randint(0, N_TOTAL, (BATCH_SIZE,))
    batch = target_images[idx]
    recon, mu, logvar = img_model.vae(batch)
    loss, _ = img_model.vae.loss(batch, recon, mu, logvar)
    vae_opt.zero_grad()
    loss.backward()
    vae_opt.step()
    vae_sched.step()
    if step % 1000 == 0:
        print(f"    VAE step {step}: loss={loss.item():.4f} ({time.time() - t0:.0f}s)")
img_model.vae.eval()

# Cache latents
with torch.no_grad():
    all_latents = []
    for i in range(0, N_TOTAL, 32):
        z, _, _ = img_model.vae.encode(target_images[i:i+32])
        all_latents.append(z.cpu())
    all_latents = torch.cat(all_latents).to(DEVICE)
    IMG_LATENT_SHAPE = all_latents.shape[1:]
print(f"  Latent shape: {list(IMG_LATENT_SHAPE)}")

# VAE ceiling
with torch.no_grad():
    vae_ssims = []
    for i in range(0, min(200, N_TOTAL), 32):
        batch = target_images[i:i+32]
        z, _, _ = img_model.vae.encode(batch)
        recon = img_model.vae.decode(z).clamp(0, 1)
        for j in range(batch.shape[0]):
            vae_ssims.append(compute_ssim(recon[j].cpu(), target_images[i+j].cpu()))
    print(f"  VAE reconstruction SSIM (ceiling): {np.mean(vae_ssims):.3f} ± {np.std(vae_ssims):.3f}")

# ── Pre-train brain encoder ──
print(f"\n  Pre-training brain encoder ({BRAIN_PRETRAIN_STEPS} steps)...")
brain_enc_opt = torch.optim.AdamW(img_model.brain_encoder.parameters(), lr=1e-3, weight_decay=0.01)
brain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(brain_enc_opt, BRAIN_PRETRAIN_STEPS)
for step in range(BRAIN_PRETRAIN_STEPS):
    idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,))
    brain = BrainData(voxels=brain_patterns[idx])
    target_lat = all_latents[idx].flatten(1)
    bg, _ = img_model.encode_brain(brain)
    n_match = min(bg.shape[1], target_lat.shape[1])
    loss = F.mse_loss(bg[:, :n_match], target_lat[:, :n_match])
    brain_enc_opt.zero_grad()
    loss.backward()
    brain_enc_opt.step()
    brain_sched.step()
    if step % 2000 == 0:
        print(f"    Brain enc step {step}: loss={loss.item():.4f}")

# ── Linear predictor for residual mode ──
print(f"\n  Fitting linear predictor (ridge regression)...")
X = brain_patterns[:N_TRAIN].cpu()
Y = all_latents[:N_TRAIN].flatten(1).cpu()
lam = 5.0
XtX = X.T @ X + lam * torch.eye(X.shape[1])
XtY = X.T @ Y
W = torch.linalg.solve(XtX, XtY)
img_lin_preds = (brain_patterns.cpu() @ W).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)

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

warmup_steps = 1000
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
    brain = BrainData(voxels=brain_patterns[idx])

    if NOISE_AUGMENT > 0:
        brain = BrainData(
            voxels=brain.voxels + NOISE_AUGMENT * torch.randn_like(brain.voxels)
        )

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

    if step % 10000 == 0 or step == DIT_STEPS - 1:
        avg_loss = sum(losses[-1000:]) / min(len(losses), 1000)
        elapsed = time.time() - t0
        eta = elapsed / max(step + 1, 1) * (DIT_STEPS - step - 1)
        print(f"    DiT step {step}: loss={avg_loss:.4f} "
              f"({elapsed:.0f}s, ~{eta:.0f}s left)")

ema.apply_to(img_model)
img_model.eval()

torch.save({k: v.cpu() for k, v in img_model.state_dict().items()},
           os.path.join(OUT, "brain2img_natural.pt"))

# ═══════════════════════════════════════════════════════════════
# 5. EVALUATE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("EVALUATION ON HELD-OUT NATURAL IMAGES")
print("=" * 70)


def evaluate(indices, label=""):
    results = {}
    for i in indices:
        brain = BrainData(voxels=brain_patterns[i:i + 1])
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

# Test set (200 for detailed eval)
N_EVAL = min(200, N_TEST)
eval_test_idx = test_idx[:N_EVAL]
print(f"\n  Test set ({N_EVAL} held-out natural images):")
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
    ssim_val = compute_ssim(recon, target)
    lin_test_results[i] = {"cos": cos, "ssim": ssim_val}
lin_test_cos_mean = np.mean([r["cos"] for r in lin_test_results.values()])
lin_test_ssim_mean = np.mean([r["ssim"] for r in lin_test_results.values()])
gap_cos = test_cos - lin_test_cos_mean
gap_ssim = test_ssim - lin_test_ssim_mean
print(f"\n  Linear baseline test cos: {lin_test_cos_mean:.3f}, SSIM: {lin_test_ssim_mean:.3f}")
print(f"  DiT gap — cos: {gap_cos:+.3f}, SSIM: {gap_ssim:+.3f}")

# ═══════════════════════════════════════════════════════════════
# 6. VLM EVALUATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("VLM SEMANTIC EVALUATION (Qwen2.5-VL-7B)")
print("=" * 70)

vlm_dir = os.path.join(OUT, "natural_vlm_images")
os.makedirs(vlm_dir, exist_ok=True)

N_VLM = 20
vlm_indices = test_idx[:N_VLM]

for i in vlm_indices:
    t = target_images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    t_pil = Image.fromarray((t * 255).astype(np.uint8)).resize((256, 256), Image.LANCZOS)
    t_pil.save(os.path.join(vlm_dir, f"test{i}_target.png"))

    r = test_results[i]["recon"].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    r_pil = Image.fromarray((r * 255).astype(np.uint8)).resize((256, 256), Image.LANCZOS)
    r_pil.save(os.path.join(vlm_dir, f"test{i}_recon.png"))

print(f"  Loading Qwen2.5-VL-7B for captioning...")
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

CAPTION_PROMPT = (
    "Describe this image in one detailed sentence. Focus on: the main subject, "
    "colors, textures, composition, and overall scene. Be specific."
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

print(f"\n  Example captions:")
for i in vlm_indices[:5]:
    print(f"\n    Test {i}:")
    print(f"      TARGET: {target_caps[i][:120]}")
    print(f"      RECON:  {recon_caps[i][:120]}")
    print(f"      SIM:    {correctness_scores[vlm_indices.index(i)]:.3f}")

# ═══════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SHOW = 8
fig, axes = plt.subplots(4, N_SHOW, figsize=(N_SHOW * 2.5, 10))

for col, i in enumerate(train_idx[:N_SHOW]):
    t = target_images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes[0, col].imshow(t)
    axes[0, col].set_title(f"Train {i}", fontsize=8)
    axes[0, col].axis("off")

    r = train_results[i]["recon"].permute(1, 2, 0).clamp(0, 1).numpy()
    axes[1, col].imshow(r)
    ssim = train_results[i]["ssim"]
    axes[1, col].set_title(f"SSIM={ssim:.2f}", fontsize=8, color="green")
    axes[1, col].axis("off")

for col, i in enumerate(test_idx[:N_SHOW]):
    t = target_images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes[2, col].imshow(t)
    axes[2, col].set_title(f"Test {i}", fontsize=8, color="red")
    axes[2, col].axis("off")

    r = test_results[i]["recon"].permute(1, 2, 0).clamp(0, 1).numpy()
    axes[3, col].imshow(r)
    ssim = test_results[i]["ssim"]
    axes[3, col].set_title(f"SSIM={ssim:.2f}", fontsize=8, color="red")
    axes[3, col].axis("off")

plt.suptitle(
    f"cortexflow × TRIBE v2 — {IMG_SIZE}×{IMG_SIZE} Natural Image Reconstruction\n"
    f"Test cos={test_cos:.3f} SSIM={test_ssim:.3f} gap={gap_cos:+.3f} "
    f"VLM={mean_correct:.3f}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "natural_reconstructions.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUT}/natural_reconstructions.png")

# ═══════════════════════════════════════════════════════════════
# 8. DIVERSITY — 4 samples per brain
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("DIVERSITY: 4 SAMPLES PER BRAIN")
print("=" * 70)

N_DIV_BRAINS = 6
N_DIV_SAMPLES = 4
div_indices = test_idx[:N_DIV_BRAINS]

fig_div, axes_div = plt.subplots(N_DIV_BRAINS, N_DIV_SAMPLES + 1,
                                  figsize=((N_DIV_SAMPLES + 1) * 2.5, N_DIV_BRAINS * 2.5))

intra_diversities = []
for row, brain_idx in enumerate(div_indices):
    t = target_images[brain_idx].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes_div[row, 0].imshow(t)
    axes_div[row, 0].set_title("Target" if row == 0 else "", fontsize=9)
    axes_div[row, 0].set_ylabel(f"Test {brain_idx}", fontsize=8, rotation=0, labelpad=40)
    axes_div[row, 0].axis("off")

    samples = []
    brain = BrainData(voxels=brain_patterns[brain_idx:brain_idx + 1])
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
        axes_div[row, s + 1].imshow(r)
        axes_div[row, s + 1].set_title(
            f"Sample {s+1}" if row == 0 else f"SSIM={ssim_val:.2f}",
            fontsize=8, color="blue")
        axes_div[row, s + 1].axis("off")

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

plt.suptitle(
    f"cortexflow × TRIBE v2 — Natural Image Diversity: {N_DIV_SAMPLES} per brain\n"
    f"Intra-brain diversity: {mean_intra_div:.3f}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
fig_div.savefig(os.path.join(OUT, "natural_diversity.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT}/natural_diversity.png")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SUMMARY — NATURAL IMAGE RECONSTRUCTION")
print("=" * 70)
print(f"""
  Dataset:            STL-10 (real photographs)
  Forward model:      TRIBE v2 (Meta, pretrained on 1000+ hrs fMRI)
  Feature extractor:  V-JEPA2 ViT-Giant (pretrained, 1035M params)
  Brain vertices:     20,484 (fsaverage5) → PCA to {N_VOXELS}
  Image size:         {IMG_SIZE}×{IMG_SIZE}
  Samples:            {N_TRAIN} train / {N_TEST} test
  
  ┌──────────────────────────┬───────────────────┐
  │ Metric                   │ Value              │
  ├──────────────────────────┼───────────────────┤
  │ Test cosine similarity   │ {test_cos:.3f} ± {test_cos_std:.3f}       │
  │ Test SSIM                │ {test_ssim:.3f} ± {test_ssim_std:.3f}       │
  │ Linear baseline cos      │ {lin_test_cos_mean:.3f}              │
  │ Linear baseline SSIM     │ {lin_test_ssim_mean:.3f}              │
  │ DiT gap (cos)            │ {gap_cos:+.3f}              │
  │ DiT gap (SSIM)           │ {gap_ssim:+.3f}              │
  │ VLM correctness          │ {mean_correct:.3f}              │
  │ VLM inter-brain diversity│ {1 - mean_inter:.3f}              │
  │ Intra-brain diversity    │ {mean_intra_div:.3f}              │
  └──────────────────────────┴───────────────────┘
""")

with open(os.path.join(OUT, "natural_results.json"), "w") as f:
    json.dump({
        "dataset": "STL-10 (natural photographs)",
        "forward_model": "TRIBE v2 (facebook/tribev2)",
        "feature_extractor": "V-JEPA2 ViT-Giant",
        "n_voxels": N_VOXELS,
        "img_size": IMG_SIZE,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "test_cos": round(test_cos, 4),
        "test_cos_std": round(test_cos_std, 4),
        "test_ssim": round(test_ssim, 4),
        "test_ssim_std": round(test_ssim_std, 4),
        "linear_baseline_cos": round(lin_test_cos_mean, 4),
        "linear_baseline_ssim": round(lin_test_ssim_mean, 4),
        "gap_cos": round(gap_cos, 4),
        "gap_ssim": round(gap_ssim, 4),
        "vlm_correctness": round(mean_correct, 4),
        "vlm_diversity": round(1 - mean_inter, 4),
        "intra_brain_diversity": round(mean_intra_div, 4),
        "captions": {
            str(i): {"target": target_caps[i], "recon": recon_caps[i],
                      "sim": round(correctness_scores[vlm_indices.index(i)], 4)}
            for i in vlm_indices[:10]
        },
    }, f, indent=2)
print(f"  Saved: {OUT}/natural_results.json")
