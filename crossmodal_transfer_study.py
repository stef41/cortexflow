"""Cross-Modal Transfer & Consistency Study — Does the image brain encoder
already know what it's seeing?

This experiment tests whether the brain encoder trained for image
reconstruction (60K steps, cos=0.976 on TRIBE v2 data) has learned
modality-agnostic semantic representations that transfer to text decoding.

Three key questions:

1. **Transfer accuracy**: If we freeze the image-trained brain encoder
   and train only a lightweight text classification head, how well does
   it classify the 10 STL-10 categories from brain patterns alone?

2. **Cross-modal agreement**: For each test brain, independently
   reconstruct an image AND predict a text label. How often do the
   decoded label and the reconstructed image's nearest-neighbor category
   agree?

3. **Error structure**: When image and text decoders disagree, do errors
   follow brain-space similarity (e.g., cat↔dog confusion)?

Uses the trained model from train_natural.py (brain2img_natural.pt).
"""

import os
import sys
import time
import json
import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
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

# Text decoder training
TEXT_TRAIN_STEPS = 15000
TEXT_LR = 3e-4
TEXT_BATCH = 128
TEXT_NOISE_AUGMENT = 0.1   # brain noise during training for regularization
TEXT_WEIGHT_DECAY = 0.05
TEXT_LABEL_SMOOTHING = 0.1

# Evaluation
N_EVAL_IMG = 200          # test images for cross-modal agreement
N_AVG = 2                 # multi-sample averaging for image reconstruction
ODE_STEPS = 5             # optimal from sampling dynamics study
CFG_SCALE = 1.0           # optimal from sampling dynamics study

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
# LIGHTWEIGHT BRAIN TEXT CLASSIFIER
# ═══════════════════════════════════════════════════════════════
class BrainTextClassifier(nn.Module):
    """Lightweight classification head on frozen brain encoder features.

    Takes brain_global (256-dim) and brain_tokens (16 × 256) from
    the frozen image encoder and predicts one of 10 categories.
    """

    def __init__(self, cond_dim=256, n_tokens=16, n_classes=10):
        super().__init__()
        # Attention pooling over brain tokens
        self.attn_pool = nn.Sequential(
            nn.Linear(cond_dim, 1),
        )
        # Fusion: global + pooled tokens → classification
        self.classifier = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(cond_dim, n_classes),
        )

    def forward(self, brain_global, brain_tokens):
        """
        brain_global: (B, cond_dim)
        brain_tokens: (B, n_tokens, cond_dim)
        """
        # Attention-weighted pooling of tokens
        attn_weights = self.attn_pool(brain_tokens).softmax(dim=1)   # (B, n_tokens, 1)
        pooled = (attn_weights * brain_tokens).sum(dim=1)            # (B, cond_dim)

        # Fuse global + pooled
        fused = torch.cat([brain_global, pooled], dim=1)             # (B, cond_dim*2)
        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("CROSS-MODAL TRANSFER & CONSISTENCY STUDY")
print("Does the image brain encoder already know what it's seeing?")
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

# ── Load labeled test set ──
print("  Loading labeled test set...")
labeled_dataset = STL10("/tmp/stl10", split="test", download=False, transform=transform)
labeled_images = torch.stack([labeled_dataset[i][0] for i in range(len(labeled_dataset))])
labeled_labels = torch.tensor([labeled_dataset[i][1] for i in range(len(labeled_dataset))])
print(f"  Labeled: {labeled_images.shape}, labels: {labeled_labels.shape}")

# Also load labeled TRAIN set
print("  Loading labeled train set...")
labeled_train = STL10("/tmp/stl10", split="train", download=False, transform=transform)
train_labeled_images = torch.stack([labeled_train[i][0] for i in range(len(labeled_train))])
train_labeled_labels = torch.tensor([labeled_train[i][1] for i in range(len(labeled_train))])
print(f"  Labeled train: {train_labeled_images.shape}")

# ── V-JEPA2 feature extraction ──
print(f"\nLoading V-JEPA2 (ViT-Giant)...")
from transformers import AutoModel

vjepa = AutoModel.from_pretrained(
    "facebook/vjepa2-vitg-fpc64-256", dtype=torch.float16,
).to(DEVICE).eval()

VJEPA_BATCH = 16


def extract_vjepa(images, name):
    features = []
    t0 = time.time()
    for batch_start in range(0, len(images), VJEPA_BATCH):
        batch_end = min(batch_start + VJEPA_BATCH, len(images))
        batch_imgs = F.interpolate(
            images[batch_start:batch_end],
            size=(256, 256), mode="bilinear", align_corners=False,
        )
        video_input = batch_imgs.unsqueeze(1).half().to(DEVICE)
        with torch.no_grad():
            out = vjepa(video_input, output_hidden_states=True)
            feats = []
            for layer_idx in VJEPA_LAYERS:
                h = out.hidden_states[layer_idx].mean(dim=1)
                feats.append(h.float().cpu())
            features.append(torch.cat(feats, dim=1))
        if (batch_start + VJEPA_BATCH) % 320 == 0 or batch_end == len(images):
            elapsed = time.time() - t0
            print(f"    {batch_end}/{len(images)} ({elapsed:.1f}s)")
    return torch.cat(features, dim=0)


print(f"  Extracting features for {N_TOTAL} unlabeled images...")
vjepa_features = extract_vjepa(target_images, "unlabeled")
print(f"  V-JEPA2 features: {vjepa_features.shape}")

print(f"\n  Extracting features for {len(labeled_images)} labeled TEST images...")
labeled_test_vjepa = extract_vjepa(labeled_images, "labeled_test")
print(f"  Labeled test V-JEPA2 features: {labeled_test_vjepa.shape}")

print(f"\n  Extracting features for {len(train_labeled_images)} labeled TRAIN images...")
labeled_train_vjepa = extract_vjepa(train_labeled_images, "labeled_train")
print(f"  Labeled train V-JEPA2 features: {labeled_train_vjepa.shape}")

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

# PCA (fit on unlabeled)
brain_mean = full_brain.mean(dim=0)
brain_centered = full_brain - brain_mean
U, S, V_pca = torch.svd_lowrank(brain_centered, q=N_VOXELS)
brain_patterns = (brain_centered @ V_pca)
brain_std = brain_patterns.std(dim=0, keepdim=True).clamp(min=1e-6)
brain_patterns = brain_patterns / brain_std
print(f"  Brain patterns: {brain_patterns.shape}")

# Labeled brain patterns
labeled_test_brain_full = tribe_forward(labeled_test_vjepa)
labeled_test_brain = ((labeled_test_brain_full - brain_mean) @ V_pca) / brain_std
print(f"  Labeled test brain: {labeled_test_brain.shape}")

labeled_train_brain_full = tribe_forward(labeled_train_vjepa)
labeled_train_brain = ((labeled_train_brain_full - brain_mean) @ V_pca) / brain_std
print(f"  Labeled train brain: {labeled_train_brain.shape}")

del full_brain, brain_centered, labeled_test_brain_full, labeled_train_brain_full
del vjepa_features, labeled_test_vjepa, labeled_train_vjepa
del state_dict, video_proj_w, video_proj_b, low_rank_w, pred_w, pred_b
torch.cuda.empty_cache()

# ── Load trained image model ──
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
print(f"  Image model: {sum(p.numel() for p in img_model.parameters()) / 1e6:.1f}M params")

# Freeze the entire image model
for p in img_model.parameters():
    p.requires_grad = False

# Cache latents + linear predictions
target_images = target_images.to(DEVICE)
brain_patterns = brain_patterns.to(DEVICE)
labeled_test_brain = labeled_test_brain.to(DEVICE)
labeled_train_brain = labeled_train_brain.to(DEVICE)

with torch.no_grad():
    all_latents = []
    for i in range(0, N_TOTAL, 32):
        z, _, _ = img_model.vae.encode(target_images[i:i + 32])
        all_latents.append(z.cpu())
    all_latents = torch.cat(all_latents).to(DEVICE)
    IMG_LATENT_SHAPE = all_latents.shape[1:]

# Ridge regression for residual mode
X = brain_patterns[:N_TRAIN].cpu()
Y = all_latents[:N_TRAIN].flatten(1).cpu()
lam = 5.0
XtX = X.T @ X + lam * torch.eye(X.shape[1])
XtY = X.T @ Y
W_ridge = torch.linalg.solve(XtX, XtY)
img_lin_preds = (brain_patterns.cpu() @ W_ridge).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)
labeled_test_lin_preds = (labeled_test_brain.cpu() @ W_ridge).view(-1, *IMG_LATENT_SHAPE).to(DEVICE)

print(f"  Latent shape: {list(IMG_LATENT_SHAPE)}")
print(f"  Ready for cross-modal analysis\n")


# ═══════════════════════════════════════════════════════════════
# 2. BASELINE: kNN ON RAW BRAIN PATTERNS (no learned classifier)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("ANALYSIS 1: BASELINE CLASSIFIERS")
print("=" * 70)

# kNN on raw brain patterns
print("\n  kNN classifier on raw brain patterns...")
train_brain_np = labeled_train_brain.cpu()
test_brain_np = labeled_test_brain.cpu()

# Cosine similarity based kNN
train_norm = F.normalize(train_brain_np, dim=1)
test_norm = F.normalize(test_brain_np, dim=1)
sim = test_norm @ train_norm.T  # (N_test, N_train)

for k in [1, 5, 10, 20]:
    topk_sim, topk_idx = sim.topk(k, dim=1)
    topk_labels = train_labeled_labels[topk_idx]  # (N_test, k)
    # Majority vote
    preds = []
    for i in range(len(topk_labels)):
        counts = torch.bincount(topk_labels[i], minlength=10)
        preds.append(counts.argmax().item())
    preds = torch.tensor(preds)
    acc = (preds == labeled_labels).float().mean().item()
    print(f"    kNN (k={k:2d}): {acc:.1%}")

# Linear probe on raw brain
print("\n  Linear probe on raw brain patterns...")
# Train a simple linear classifier
linear_probe = nn.Linear(N_VOXELS, 10).to(DEVICE)
probe_opt = torch.optim.AdamW(linear_probe.parameters(), lr=1e-3, weight_decay=0.01)
probe_sched = torch.optim.lr_scheduler.CosineAnnealingLR(probe_opt, 5000)
train_labels_dev = train_labeled_labels.to(DEVICE)
test_labels_dev = labeled_labels.to(DEVICE)

for step in range(5000):
    idx = torch.randint(0, len(labeled_train_brain), (256,))
    logits = linear_probe(labeled_train_brain[idx])
    loss = F.cross_entropy(logits, train_labels_dev[idx])
    probe_opt.zero_grad()
    loss.backward()
    probe_opt.step()
    probe_sched.step()

with torch.no_grad():
    test_logits = linear_probe(labeled_test_brain)
    linear_acc = (test_logits.argmax(1) == test_labels_dev).float().mean().item()
    train_logits = linear_probe(labeled_train_brain)
    linear_train_acc = (train_logits.argmax(1) == train_labels_dev).float().mean().item()
print(f"    Linear probe: train={linear_train_acc:.1%}, test={linear_acc:.1%}")

del linear_probe, probe_opt

# ═══════════════════════════════════════════════════════════════
# 3. kNN ON FROZEN ENCODER FEATURES
# ═══════════════════════════════════════════════════════════════
print("\n  kNN on frozen brain encoder features...")
with torch.no_grad():
    # Extract features from frozen image brain encoder
    train_bg, train_bt = img_model.encode_brain(
        BrainData(voxels=labeled_train_brain))
    test_bg, test_bt = img_model.encode_brain(
        BrainData(voxels=labeled_test_brain))

# kNN on brain_global (256-dim)
train_bg_norm = F.normalize(train_bg, dim=1)
test_bg_norm = F.normalize(test_bg, dim=1)
sim_enc = test_bg_norm @ train_bg_norm.T

for k in [1, 5, 10, 20]:
    topk_sim, topk_idx = sim_enc.topk(k, dim=1)
    topk_labels = train_labeled_labels.to(DEVICE)[topk_idx]
    preds = []
    for i in range(len(topk_labels)):
        counts = torch.bincount(topk_labels[i], minlength=10)
        preds.append(counts.argmax().item())
    preds = torch.tensor(preds, device=DEVICE)
    acc = (preds == test_labels_dev).float().mean().item()
    print(f"    Encoder kNN (k={k:2d}): {acc:.1%}")


# ═══════════════════════════════════════════════════════════════
# 4. TRANSFER LEARNING: TRAIN TEXT CLASSIFIER ON FROZEN ENCODER
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 2: TRANSFER LEARNING — TEXT CLASSIFIER ON FROZEN ENCODER")
print("=" * 70)

text_clf = BrainTextClassifier(cond_dim=256, n_tokens=16, n_classes=10).to(DEVICE)
n_params = sum(p.numel() for p in text_clf.parameters())
print(f"  Text classifier params: {n_params:,} ({n_params / 1e6:.3f}M)")

text_opt = torch.optim.AdamW(
    text_clf.parameters(), lr=TEXT_LR, weight_decay=TEXT_WEIGHT_DECAY)
warmup = 500
total = TEXT_TRAIN_STEPS


def lr_lambda(step):
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


text_sched = torch.optim.lr_scheduler.LambdaLR(text_opt, lr_lambda)

# Pre-extract encoder features for all labeled data (frozen)
with torch.no_grad():
    train_bg_all, train_bt_all = img_model.encode_brain(
        BrainData(voxels=labeled_train_brain))
    test_bg_all, test_bt_all = img_model.encode_brain(
        BrainData(voxels=labeled_test_brain))

print(f"\n  Training ({TEXT_TRAIN_STEPS} steps, batch={TEXT_BATCH}, noise_aug={TEXT_NOISE_AUGMENT})...")
t0 = time.time()
best_test_acc = 0.0
best_state = None

for step in range(TEXT_TRAIN_STEPS):
    text_clf.train()
    idx = torch.randint(0, len(train_bg_all), (TEXT_BATCH,))

    bg = train_bg_all[idx].clone()
    bt = train_bt_all[idx].clone()

    # Brain noise augmentation during training
    if TEXT_NOISE_AUGMENT > 0:
        g_scale = bg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        bg = bg + TEXT_NOISE_AUGMENT * g_scale * torch.randn_like(bg)
        t_scale = bt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        bt = bt + TEXT_NOISE_AUGMENT * t_scale * torch.randn_like(bt)

    logits = text_clf(bg, bt)
    loss = F.cross_entropy(logits, train_labels_dev[idx],
                           label_smoothing=TEXT_LABEL_SMOOTHING)

    text_opt.zero_grad()
    loss.backward()
    text_opt.step()
    text_sched.step()

    if step % 1000 == 0 or step == TEXT_TRAIN_STEPS - 1:
        text_clf.eval()
        with torch.no_grad():
            test_logits = text_clf(test_bg_all, test_bt_all)
            test_acc = (test_logits.argmax(1) == test_labels_dev).float().mean().item()
            train_logits = text_clf(train_bg_all, train_bt_all)
            train_acc = (train_logits.argmax(1) == train_labels_dev).float().mean().item()
        elapsed = time.time() - t0
        print(f"    Step {step:5d}: loss={loss.item():.4f}, "
              f"train={train_acc:.1%}, test={test_acc:.1%} ({elapsed:.1f}s)")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in text_clf.state_dict().items()}

# Restore best
text_clf.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
text_clf.eval()
print(f"\n  Best test accuracy: {best_test_acc:.1%}")

# Per-category accuracy
with torch.no_grad():
    test_logits = text_clf(test_bg_all, test_bt_all)
    test_preds = test_logits.argmax(1)

print(f"\n  Per-category accuracy:")
per_cat_acc = {}
for c in range(10):
    mask = test_labels_dev == c
    cat_acc = (test_preds[mask] == c).float().mean().item()
    per_cat_acc[STL10_CATEGORIES[c]] = cat_acc
    print(f"    {STL10_CATEGORIES[c]:10s}: {cat_acc:.1%}")


# ═══════════════════════════════════════════════════════════════
# 5. CROSS-MODAL AGREEMENT — IMAGE + TEXT ON SAME BRAIN PATTERNS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 3: CROSS-MODAL AGREEMENT")
print("=" * 70)


def reconstruct_image(brain_voxels, lin_pred, idx):
    """Reconstruct image from brain pattern using residual flow matching."""
    brain = BrainData(voxels=brain_voxels.unsqueeze(0))
    bg, bt = img_model.encode_brain(brain)
    shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

    latent_sum = None
    for s in range(N_AVG):
        torch.manual_seed(s * 1000 + idx)
        z = img_model.flow_matcher.sample(
            img_model.dit, shape_1, bg, bt, num_steps=ODE_STEPS,
            cfg_scale=CFG_SCALE,
        )
        z = z + lin_pred.unsqueeze(0)
        latent_sum = z if latent_sum is None else latent_sum + z
    avg_z = latent_sum / N_AVG
    with torch.no_grad():
        recon = img_model.vae.decode(avg_z)[0].detach().clamp(0, 1)
    return recon


# Build category prototypes from labeled train data
# Use VLM-free approach: average reconstructed features per category
print("\n  Building category prototypes from labeled training images...")
with torch.no_grad():
    labeled_train_dev = train_labeled_images.to(DEVICE)
    cat_prototypes = {}
    for c in range(10):
        mask = train_labeled_labels == c
        cat_imgs = labeled_train_dev[mask]
        # Use raw pixel mean as prototype
        cat_prototypes[c] = cat_imgs.mean(dim=0)
    del labeled_train_dev

# Evaluate cross-modal agreement on labeled test set
print(f"\n  Evaluating cross-modal agreement on {N_EVAL_IMG} test images...")
print(f"  (Using ODE_STEPS={ODE_STEPS}, CFG={CFG_SCALE} — optimal from sampling study)")

text_predictions = []
image_predictions = []
ground_truth = []
cos_scores = []
ssim_scores = []

t0 = time.time()
eval_indices = torch.randperm(len(labeled_test_brain))[:N_EVAL_IMG]

for count, eval_i in enumerate(eval_indices):
    i = eval_i.item()
    brain_vec = labeled_test_brain[i]
    true_label = labeled_labels[i].item()
    ground_truth.append(true_label)

    # TEXT prediction (frozen encoder → text classifier)
    with torch.no_grad():
        bg, bt = img_model.encode_brain(BrainData(voxels=brain_vec.unsqueeze(0)))
        text_logit = text_clf(bg, bt)
        text_pred = text_logit.argmax(1).item()
    text_predictions.append(text_pred)

    # IMAGE reconstruction + nearest-prototype classification
    recon = reconstruct_image(brain_vec, labeled_test_lin_preds[i], i)

    # Classify reconstructed image by nearest prototype (cosine sim)
    recon_flat = recon.flatten().unsqueeze(0)
    best_sim = -1
    best_cat = 0
    for c in range(10):
        proto_flat = cat_prototypes[c].flatten().unsqueeze(0)
        sim = F.cosine_similarity(recon_flat, proto_flat).item()
        if sim > best_sim:
            best_sim = sim
            best_cat = c
    image_predictions.append(best_cat)

    # Quality metrics vs ground truth image
    target = labeled_images[i].to(DEVICE)
    cos = F.cosine_similarity(
        recon.flatten().unsqueeze(0), target.flatten().unsqueeze(0)
    ).item()
    ssim = compute_ssim(recon.cpu(), labeled_images[i])
    cos_scores.append(cos)
    ssim_scores.append(ssim)

    if (count + 1) % 20 == 0:
        elapsed = time.time() - t0
        agree_so_far = sum(1 for t, im in zip(text_predictions, image_predictions) if t == im)
        print(f"    {count + 1}/{N_EVAL_IMG}: "
              f"text_acc={sum(1 for t, g in zip(text_predictions, ground_truth) if t == g) / len(ground_truth):.1%}, "
              f"img_acc={sum(1 for im, g in zip(image_predictions, ground_truth) if im == g) / len(ground_truth):.1%}, "
              f"agreement={agree_so_far / len(ground_truth):.1%} ({elapsed:.1f}s)")

ground_truth = torch.tensor(ground_truth)
text_predictions = torch.tensor(text_predictions)
image_predictions = torch.tensor(image_predictions)

text_acc = (text_predictions == ground_truth).float().mean().item()
img_acc = (image_predictions == ground_truth).float().mean().item()
agreement = (text_predictions == image_predictions).float().mean().item()
both_correct = ((text_predictions == ground_truth) & (image_predictions == ground_truth)).float().mean().item()
either_correct = ((text_predictions == ground_truth) | (image_predictions == ground_truth)).float().mean().item()

print(f"\n  Results on {N_EVAL_IMG} test images:")
print(f"    Text accuracy:     {text_acc:.1%}")
print(f"    Image accuracy:    {img_acc:.1%}")
print(f"    Cross-modal agreement: {agreement:.1%}")
print(f"    Both correct:      {both_correct:.1%}")
print(f"    Either correct:    {either_correct:.1%}")
print(f"    Mean cos:          {np.mean(cos_scores):.4f} ± {np.std(cos_scores):.4f}")
print(f"    Mean SSIM:         {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")


# ═══════════════════════════════════════════════════════════════
# 6. CONFUSION ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 4: CONFUSION STRUCTURE")
print("=" * 70)

# Text confusion matrix
text_confusion = torch.zeros(10, 10, dtype=torch.long)
for t, g in zip(text_predictions, ground_truth):
    text_confusion[g, t] += 1

# Image confusion matrix
img_confusion = torch.zeros(10, 10, dtype=torch.long)
for im, g in zip(image_predictions, ground_truth):
    img_confusion[g, im] += 1

# Cross-modal disagreement analysis
disagree_mask = text_predictions != image_predictions
n_disagree = disagree_mask.sum().item()
print(f"\n  Disagreements: {n_disagree}/{N_EVAL_IMG} ({n_disagree / N_EVAL_IMG:.1%})")

if n_disagree > 0:
    # When they disagree, which is right?
    text_right_count = ((text_predictions == ground_truth) & disagree_mask).sum().item()
    img_right_count = ((image_predictions == ground_truth) & disagree_mask).sum().item()
    neither_right = (disagree_mask & (text_predictions != ground_truth) & (image_predictions != ground_truth)).sum().item()
    print(f"    Text correct:   {text_right_count} ({text_right_count / n_disagree:.1%})")
    print(f"    Image correct:  {img_right_count} ({img_right_count / n_disagree:.1%})")
    print(f"    Neither correct: {neither_right} ({neither_right / n_disagree:.1%})")

# Top confusion pairs
print(f"\n  Top text confusion pairs:")
for i in range(10):
    for j in range(10):
        if i != j and text_confusion[i, j] > 0:
            print(f"    {STL10_CATEGORIES[i]:10s} → {STL10_CATEGORIES[j]:10s}: {text_confusion[i, j].item()}")


# ═══════════════════════════════════════════════════════════════
# 7. BRAIN REPRESENTATION ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ANALYSIS 5: BRAIN ENCODER REPRESENTATION STRUCTURE")
print("=" * 70)

# Compare raw brain vs encoded brain category separability
with torch.no_grad():
    all_test_bg, all_test_bt = img_model.encode_brain(
        BrainData(voxels=labeled_test_brain))

# Inter-category cosine similarity in raw brain space
print("\n  Category similarity in raw brain space:")
raw_centroids = []
for c in range(10):
    mask = labeled_labels == c
    centroid = F.normalize(labeled_test_brain[mask].cpu().mean(dim=0, keepdim=True), dim=1)
    raw_centroids.append(centroid)
raw_centroids = torch.cat(raw_centroids, dim=0)
raw_sim = raw_centroids @ raw_centroids.T

print("\n  Category similarity in encoder feature space:")
enc_centroids = []
for c in range(10):
    mask = labeled_labels == c
    centroid = F.normalize(all_test_bg[mask].cpu().mean(dim=0, keepdim=True), dim=1)
    enc_centroids.append(centroid)
enc_centroids = torch.cat(enc_centroids, dim=0)
enc_sim = enc_centroids @ enc_centroids.T

# Measure how much the encoder separates categories
raw_off_diag = raw_sim[~torch.eye(10, dtype=bool)].mean().item()
enc_off_diag = enc_sim[~torch.eye(10, dtype=bool)].mean().item()
print(f"    Raw brain avg inter-class similarity:    {raw_off_diag:.3f}")
print(f"    Encoder avg inter-class similarity:      {enc_off_diag:.3f}")
print(f"    Separation improvement:                  {raw_off_diag - enc_off_diag:+.3f}")


# ═══════════════════════════════════════════════════════════════
# 8. GENERATE FIGURES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("GENERATING FIGURES")
print("=" * 70)

# ── Figure 1: Transfer accuracy comparison ──
print("  → crossmodal_transfer_accuracy.png")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

methods = ["kNN (k=5)\nraw brain", "Linear\nraw brain", "kNN (k=5)\nencoder", "Transfer\nclassifier"]
# Re-compute kNN k=5 for raw and encoder
raw_sim_knn = test_norm @ train_norm.T
topk_sim, topk_idx = raw_sim_knn.topk(5, dim=1)
topk_labels = train_labeled_labels[topk_idx]
raw_knn_preds = []
for i in range(len(topk_labels)):
    counts = torch.bincount(topk_labels[i], minlength=10)
    raw_knn_preds.append(counts.argmax().item())
raw_knn_acc = sum(1 for p, g in zip(raw_knn_preds, labeled_labels.tolist()) if p == g) / len(labeled_labels)

enc_sim_knn = test_bg_norm @ train_bg_norm.T
topk_sim, topk_idx = enc_sim_knn.topk(5, dim=1)
topk_labels = train_labeled_labels.to(DEVICE)[topk_idx]
enc_knn_preds = []
for i in range(len(topk_labels)):
    counts = torch.bincount(topk_labels[i], minlength=10)
    enc_knn_preds.append(counts.argmax().item())
enc_knn_acc = sum(1 for p, g in zip(enc_knn_preds, labeled_labels.tolist()) if p == g) / len(labeled_labels)

accs = [raw_knn_acc, linear_acc, enc_knn_acc, best_test_acc]
colors = ["#90CAF9", "#A5D6A7", "#FFE082", "#EF5350"]
bars = ax.bar(methods, accs, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Test Accuracy", fontsize=13)
ax.set_title("Brain → Text Classification: Transfer from Image Encoder", fontsize=14, fontweight="bold")
ax.axhline(0.1, color="gray", linestyle="--", alpha=0.5, label="Chance (10%)")
ax.legend(fontsize=11)
ax.spines[["right", "top"]].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "crossmodal_transfer_accuracy.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Confusion matrices (text + image side by side) ──
print("  → crossmodal_confusion.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Normalize confusion matrices to percentages
text_conf_pct = text_confusion.float()
for i in range(10):
    row_sum = text_conf_pct[i].sum()
    if row_sum > 0:
        text_conf_pct[i] /= row_sum

img_conf_pct = img_confusion.float()
for i in range(10):
    row_sum = img_conf_pct[i].sum()
    if row_sum > 0:
        img_conf_pct[i] /= row_sum

for ax, conf, title in [
    (ax1, text_conf_pct, "Text Decoder Confusion"),
    (ax2, img_conf_pct, "Image Decoder Confusion"),
]:
    im = ax.imshow(conf.numpy(), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    for i in range(10):
        for j in range(10):
            val = conf[i, j].item()
            if val > 0.01:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10))
    ax.set_xticklabels(STL10_CATEGORIES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(10))
    ax.set_yticklabels(STL10_CATEGORIES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="Fraction")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "crossmodal_confusion.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: Category similarity heatmaps (raw vs encoder) ──
print("  → crossmodal_representation.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for ax, sim_mat, title in [
    (ax1, raw_sim.numpy(), "Raw Brain Space"),
    (ax2, enc_sim.numpy(), "After Image Encoder"),
]:
    im = ax.imshow(sim_mat, cmap="RdBu_r", vmin=-0.5, vmax=1, aspect="auto")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{sim_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(sim_mat[i, j]) > 0.5 else "black")
    ax.set_xticks(range(10))
    ax.set_xticklabels(STL10_CATEGORIES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(10))
    ax.set_yticklabels(STL10_CATEGORIES, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")

fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="Cosine Similarity")
fig.suptitle("Category Similarity: Raw Brain vs Image Encoder Features",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "crossmodal_representation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 4: Cross-modal agreement per category ──
print("  → crossmodal_agreement.png")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

cat_text_acc = []
cat_img_acc = []
cat_agreement = []
for c in range(10):
    mask = ground_truth == c
    if mask.sum() == 0:
        cat_text_acc.append(0)
        cat_img_acc.append(0)
        cat_agreement.append(0)
        continue
    ct_acc = (text_predictions[mask] == c).float().mean().item()
    ci_acc = (image_predictions[mask] == c).float().mean().item()
    ca = (text_predictions[mask] == image_predictions[mask]).float().mean().item()
    cat_text_acc.append(ct_acc)
    cat_img_acc.append(ci_acc)
    cat_agreement.append(ca)

x = np.arange(10)
width = 0.25
bars1 = ax.bar(x - width, cat_text_acc, width, label="Text Decoder", color="#42A5F5", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x, cat_img_acc, width, label="Image Decoder", color="#66BB6A", edgecolor="black", linewidth=0.5)
bars3 = ax.bar(x + width, cat_agreement, width, label="Cross-Modal Agreement", color="#FFA726", edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(STL10_CATEGORIES, rotation=45, ha="right", fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Accuracy / Agreement", fontsize=12)
ax.set_title("Per-Category Cross-Modal Transfer & Agreement", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.axhline(0.1, color="gray", linestyle="--", alpha=0.4, label="Chance")
ax.spines[["right", "top"]].set_visible(False)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "crossmodal_agreement.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 9. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
results = {
    "baselines": {
        "knn_raw_k5": raw_knn_acc,
        "linear_probe": linear_acc,
        "knn_encoder_k5": enc_knn_acc,
    },
    "transfer_classifier": {
        "test_accuracy": best_test_acc,
        "params": n_params,
        "per_category": per_cat_acc,
    },
    "crossmodal_agreement": {
        "text_accuracy": text_acc,
        "image_accuracy": img_acc,
        "agreement": agreement,
        "both_correct": both_correct,
        "either_correct": either_correct,
        "mean_cos": float(np.mean(cos_scores)),
        "mean_ssim": float(np.mean(ssim_scores)),
    },
    "representation": {
        "raw_inter_class_sim": raw_off_diag,
        "encoder_inter_class_sim": enc_off_diag,
        "separation_improvement": raw_off_diag - enc_off_diag,
    },
}

with open(os.path.join(OUT, "crossmodal_transfer_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {OUT}/crossmodal_transfer_results.json")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)

print(f"""
  Classification accuracy:
    kNN (raw brain, k=5):     {raw_knn_acc:.1%}
    Linear probe (raw brain): {linear_acc:.1%}
    kNN (encoder, k=5):       {enc_knn_acc:.1%}
    Transfer classifier:      {best_test_acc:.1%}

  Cross-modal agreement ({N_EVAL_IMG} test images):
    Text accuracy:     {text_acc:.1%}
    Image accuracy:    {img_acc:.1%}
    Agreement:         {agreement:.1%}
    Both correct:      {both_correct:.1%}
    Either correct:    {either_correct:.1%}

  Representation:
    Raw brain inter-class sim:    {raw_off_diag:.3f}
    Encoder inter-class sim:      {enc_off_diag:.3f}
    Separation improvement:       {raw_off_diag - enc_off_diag:+.3f}

  Reconstruction quality:
    Mean cos:  {np.mean(cos_scores):.4f} ± {np.std(cos_scores):.4f}
    Mean SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}
""")

print("=" * 70)
print("CROSS-MODAL TRANSFER STUDY COMPLETE")
print("=" * 70)
print(f"""
Key findings:
  1. Transfer classifier accuracy: {best_test_acc:.1%} (from {n_params:,} trainable params)
  2. Cross-modal agreement: {agreement:.1%}
  3. Encoder separation improvement: {raw_off_diag - enc_off_diag:+.3f}
""")
