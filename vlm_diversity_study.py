"""VLM-based semantic diversity evaluation for cortexflow reconstructions.

Uses Qwen2.5-VL-7B-Instruct to caption target and reconstructed images,
then measures:
  1. Semantic correctness: do reconstruction captions match target captions?
  2. Inter-brain diversity: are captions for different brain inputs different?
  3. Intra-brain consistency: are multiple samples from the same brain similar?

Uses sentence-transformers for embedding-based similarity of captions.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ── Config ──
N_TOTAL = 500
N_TRAIN = 400
N_TEST = 100
N_VOXELS = 512
IMG_SIZE = 32
N_TEST_SAMPLES = 20       # number of test brain patterns to evaluate
N_DIVERSE = 4             # samples per brain pattern (for intra-brain)
IMG_CFG_SCALE = 1.0
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"

OUT = "train_outputs"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)


def make_random_image(seed, size):
    """Generate a parameterized image: random shape, position, color, background."""
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


# ═══════════════════════════════════════════════════════════════
# 1. REGENERATE DATA + LOAD TRAINED MODEL
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("VLM SEMANTIC DIVERSITY STUDY")
print("=" * 70)

from cortexflow import BrainData
from cortexflow._types import DiTConfig, VAEConfig, FlowConfig
from cortexflow.brain2img import Brain2Image
from neuroprobe.media import build_brain_model

# Rebuild forward model + data (deterministic)
vision_forward = build_brain_model(
    modality="video", feature_dim=256, n_vertices=N_VOXELS,
    hidden_dim=128, seed=42,
)

print(f"  Generating {N_TOTAL} images + brain patterns...")
image_brains, image_targets = [], []
for i in range(N_TOTAL):
    clean_img = make_random_image(i * 7 + 13, IMG_SIZE)
    video = clean_img.unsqueeze(0)
    with torch.no_grad():
        bold = vision_forward.predict(video)
        brain_vec = bold.mean(dim=0)
    image_brains.append(brain_vec)
    image_targets.append(clean_img)

brain_patterns_img = torch.stack(image_brains)
target_images = torch.stack(image_targets)
test_idx = list(range(N_TRAIN, N_TRAIN + N_TEST))

# Rebuild model + load weights
print("  Loading trained model from train_outputs/brain2img.pt...")
img_model = Brain2Image(
    n_voxels=N_VOXELS, img_size=IMG_SIZE,
    dit_config=DiTConfig(hidden_dim=64, depth=4, num_heads=4, cond_dim=64),
    vae_config=VAEConfig(hidden_dims=[32, 64, 128]),
    flow_config=FlowConfig(),
)
state = torch.load(os.path.join(OUT, "brain2img.pt"), map_location="cpu",
                   weights_only=True)
img_model.load_state_dict(state)
img_model.eval()

# Linear predictor for residual mode
print("  Fitting linear predictor (residual mode)...")
X_train = brain_patterns_img[:N_TRAIN]
img_model.vae.eval()
with torch.no_grad():
    all_latents = []
    for i in range(0, N_TOTAL, 32):
        batch = target_images[i:i+32]
        z, _, _ = img_model.vae.encode(batch)
        all_latents.append(z)
    all_latents = torch.cat(all_latents)
    IMG_LATENT_SHAPE = all_latents.shape[1:]

Y_train = all_latents[:N_TRAIN].flatten(1)
X_all = brain_patterns_img
# Ridge regression
lam = 1.0
XtX = X_train.T @ X_train + lam * torch.eye(X_train.shape[1])
XtY = X_train.T @ Y_train
W = torch.linalg.solve(XtX, XtY)
img_lin_preds = (X_all @ W).view(-1, *IMG_LATENT_SHAPE)

print(f"  Model loaded. Latent shape: {list(IMG_LATENT_SHAPE)}")

# ═══════════════════════════════════════════════════════════════
# 2. GENERATE INDIVIDUAL IMAGES
# ═══════════════════════════════════════════════════════════════
print(f"\n  Generating reconstructions for {N_TEST_SAMPLES} test samples...")
print(f"  ({N_DIVERSE} diverse samples each + 1 averaged reconstruction)")

individual_dir = os.path.join(OUT, "vlm_images")
os.makedirs(individual_dir, exist_ok=True)

all_images = {}  # {idx: {"target": path, "recon_avg": path, "samples": [paths]}}

for count, i in enumerate(test_idx[:N_TEST_SAMPLES]):
    brain = BrainData(voxels=brain_patterns_img[i:i+1])
    bg, bt = img_model.encode_brain(brain)
    shape_1 = (1,) + tuple(IMG_LATENT_SHAPE)

    entry = {"target": None, "recon_avg": None, "samples": []}

    # Save target
    target = target_images[i].permute(1, 2, 0).clamp(0, 1).numpy()
    target_pil = Image.fromarray((target * 255).astype(np.uint8)).resize(
        (256, 256), Image.NEAREST)
    tpath = os.path.join(individual_dir, f"test{i}_target.png")
    target_pil.save(tpath)
    entry["target"] = tpath

    # Generate N_DIVERSE individual samples
    latent_sum = None
    for s in range(N_DIVERSE):
        torch.manual_seed(s * 17 + 3)
        z = img_model.flow_matcher.sample(
            img_model.dit, shape_1, bg, bt, num_steps=50, cfg_scale=IMG_CFG_SCALE,
        )
        z_res = z + img_lin_preds[i:i+1]
        if latent_sum is None:
            latent_sum = z_res.clone()
        else:
            latent_sum = latent_sum + z_res

        with torch.no_grad():
            recon = img_model.vae.decode(z_res)[0].detach().clamp(0, 1)
        r = recon.permute(1, 2, 0).numpy()
        rpil = Image.fromarray((r * 255).astype(np.uint8)).resize(
            (256, 256), Image.NEAREST)
        spath = os.path.join(individual_dir, f"test{i}_sample{s}.png")
        rpil.save(spath)
        entry["samples"].append(spath)

    # Averaged reconstruction
    avg_z = latent_sum / N_DIVERSE
    with torch.no_grad():
        avg_recon = img_model.vae.decode(avg_z)[0].detach().clamp(0, 1)
    ar = avg_recon.permute(1, 2, 0).numpy()
    avg_pil = Image.fromarray((ar * 255).astype(np.uint8)).resize(
        (256, 256), Image.NEAREST)
    apath = os.path.join(individual_dir, f"test{i}_avg.png")
    avg_pil.save(apath)
    entry["recon_avg"] = apath

    all_images[i] = entry
    if (count + 1) % 5 == 0:
        print(f"    {count+1}/{N_TEST_SAMPLES} done")

print(f"  Generated {N_TEST_SAMPLES * (N_DIVERSE + 2)} images total")

# ═══════════════════════════════════════════════════════════════
# 3. VLM CAPTIONING
# ═══════════════════════════════════════════════════════════════
print(f"\n  Loading VLM: {VLM_MODEL}")

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(VLM_MODEL)

CAPTION_PROMPT = (
    "Describe this image in one detailed sentence. Focus on: the main shape or object, "
    "its color, its position, and the background color. Be specific about colors and geometry."
)


def caption_image(image_path):
    """Get a VLM caption for one image."""
    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": CAPTION_PROMPT},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[img], padding=True, return_tensors="pt",
    ).to(vlm.device)
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=128, do_sample=False)
    decoded = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)[0].strip()
    return decoded


print(f"\n  Captioning {N_TEST_SAMPLES} targets...")
target_captions = {}
t0 = time.time()
for count, i in enumerate(test_idx[:N_TEST_SAMPLES]):
    cap = caption_image(all_images[i]["target"])
    target_captions[i] = cap
    if (count + 1) % 5 == 0:
        elapsed = time.time() - t0
        rate = (count + 1) / elapsed
        remaining = (N_TEST_SAMPLES - count - 1) / rate
        print(f"    {count+1}/{N_TEST_SAMPLES} targets captioned "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

print(f"\n  Captioning {N_TEST_SAMPLES} averaged reconstructions...")
recon_captions = {}
t0 = time.time()
for count, i in enumerate(test_idx[:N_TEST_SAMPLES]):
    cap = caption_image(all_images[i]["recon_avg"])
    recon_captions[i] = cap
    if (count + 1) % 5 == 0:
        elapsed = time.time() - t0
        rate = (count + 1) / elapsed
        remaining = (N_TEST_SAMPLES - count - 1) / rate
        print(f"    {count+1}/{N_TEST_SAMPLES} reconstructions captioned "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

print(f"\n  Captioning {N_TEST_SAMPLES * N_DIVERSE} diverse samples...")
sample_captions = {}  # {i: [cap0, cap1, cap2, cap3]}
t0 = time.time()
total_samples = N_TEST_SAMPLES * N_DIVERSE
done = 0
for i in test_idx[:N_TEST_SAMPLES]:
    caps = []
    for s in range(N_DIVERSE):
        cap = caption_image(all_images[i]["samples"][s])
        caps.append(cap)
        done += 1
        if done % 10 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (total_samples - done) / rate
            print(f"    {done}/{total_samples} samples captioned "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
    sample_captions[i] = caps

# Free VLM memory
del vlm, processor
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# 4. EMBEDDING-BASED SIMILARITY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n  Loading sentence embedder: {EMBED_MODEL}")
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(EMBED_MODEL)

# Collect all captions for batch embedding
all_texts = []
text_map = {}  # maps index in all_texts → (type, brain_idx, sample_idx)
for i in test_idx[:N_TEST_SAMPLES]:
    idx = len(all_texts)
    all_texts.append(target_captions[i])
    text_map[idx] = ("target", i, None)

    idx = len(all_texts)
    all_texts.append(recon_captions[i])
    text_map[idx] = ("recon_avg", i, None)

    for s in range(N_DIVERSE):
        idx = len(all_texts)
        all_texts.append(sample_captions[i][s])
        text_map[idx] = ("sample", i, s)

print(f"  Embedding {len(all_texts)} captions...")
all_embeddings = embedder.encode(all_texts, convert_to_tensor=True,
                                  show_progress_bar=False)
all_embeddings = F.normalize(all_embeddings, dim=1)

# Build lookup: (type, brain_idx, sample_idx) → embedding index
lookup = {}
for idx, key in text_map.items():
    lookup[key] = idx

# ── Metric 1: Semantic correctness (target ↔ reconstruction similarity) ──
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

correctness_scores = []
for i in test_idx[:N_TEST_SAMPLES]:
    t_emb = all_embeddings[lookup[("target", i, None)]]
    r_emb = all_embeddings[lookup[("recon_avg", i, None)]]
    sim = F.cosine_similarity(t_emb.unsqueeze(0), r_emb.unsqueeze(0)).item()
    correctness_scores.append(sim)

mean_correctness = np.mean(correctness_scores)
std_correctness = np.std(correctness_scores)
print(f"\n  1. SEMANTIC CORRECTNESS (target ↔ reconstruction caption similarity)")
print(f"     Mean cosine similarity: {mean_correctness:.3f} ± {std_correctness:.3f}")
print(f"     Range: [{min(correctness_scores):.3f}, {max(correctness_scores):.3f}]")

# ── Metric 2: Inter-brain diversity (different brains → different captions) ──
inter_sims = []
indices_list = test_idx[:N_TEST_SAMPLES]
for a_pos in range(len(indices_list)):
    for b_pos in range(a_pos + 1, len(indices_list)):
        ia, ib = indices_list[a_pos], indices_list[b_pos]
        ea = all_embeddings[lookup[("recon_avg", ia, None)]]
        eb = all_embeddings[lookup[("recon_avg", ib, None)]]
        sim = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
        inter_sims.append(sim)

mean_inter = np.mean(inter_sims)
std_inter = np.std(inter_sims)
print(f"\n  2. INTER-BRAIN DIVERSITY (different brain inputs → different captions)")
print(f"     Mean pairwise similarity: {mean_inter:.3f} ± {std_inter:.3f}")
print(f"     (Lower = more diverse across different brain inputs)")

# ── Metric 3: Intra-brain consistency (same brain → similar captions) ──
intra_sims = []
for i in test_idx[:N_TEST_SAMPLES]:
    for a in range(N_DIVERSE):
        for b in range(a + 1, N_DIVERSE):
            ea = all_embeddings[lookup[("sample", i, a)]]
            eb = all_embeddings[lookup[("sample", i, b)]]
            sim = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
            intra_sims.append(sim)

mean_intra = np.mean(intra_sims)
std_intra = np.std(intra_sims)
print(f"\n  3. INTRA-BRAIN CONSISTENCY (same brain → similar captions)")
print(f"     Mean pairwise similarity: {mean_intra:.3f} ± {std_intra:.3f}")
print(f"     (Higher = more consistent for same brain input)")

# ── Metric 4: Diversity ratio ──
diversity_ratio = (1 - mean_inter) / max(1 - mean_intra, 0.001)
print(f"\n  4. DIVERSITY RATIO: {diversity_ratio:.2f}")
print(f"     (Inter-brain variation / Intra-brain variation; >1 = brain-conditioned)")

# ── Metric 5: Target inter-diversity (baseline — how diverse are the stimuli themselves?) ──
target_inter_sims = []
for a_pos in range(len(indices_list)):
    for b_pos in range(a_pos + 1, len(indices_list)):
        ia, ib = indices_list[a_pos], indices_list[b_pos]
        ea = all_embeddings[lookup[("target", ia, None)]]
        eb = all_embeddings[lookup[("target", ib, None)]]
        sim = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
        target_inter_sims.append(sim)

mean_target_inter = np.mean(target_inter_sims)
print(f"\n  5. TARGET DIVERSITY (baseline)")
print(f"     Mean pairwise target caption similarity: {mean_target_inter:.3f}")
print(f"     Reconstruction diversity / Target diversity: "
      f"{(1 - mean_inter) / max(1 - mean_target_inter, 0.001):.2f}")

# ═══════════════════════════════════════════════════════════════
# 5. EXAMPLE CAPTIONS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("EXAMPLE CAPTIONS (5 test samples)")
print("=" * 70)

for count, i in enumerate(test_idx[:5]):
    print(f"\n  ── Test sample {i} ──")
    print(f"  TARGET:  {target_captions[i]}")
    print(f"  RECON:   {recon_captions[i]}")
    sim = correctness_scores[count]
    print(f"  SIMILARITY: {sim:.3f}")
    print(f"  DIVERSE SAMPLES:")
    for s in range(N_DIVERSE):
        print(f"    Sample {s}: {sample_captions[i][s]}")

# ═══════════════════════════════════════════════════════════════
# 6. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
results = {
    "vlm_model": VLM_MODEL,
    "embed_model": EMBED_MODEL,
    "n_test_samples": N_TEST_SAMPLES,
    "n_diverse": N_DIVERSE,
    "metrics": {
        "semantic_correctness": {
            "mean": round(mean_correctness, 4),
            "std": round(std_correctness, 4),
            "min": round(min(correctness_scores), 4),
            "max": round(max(correctness_scores), 4),
        },
        "inter_brain_diversity": {
            "mean_similarity": round(mean_inter, 4),
            "std": round(std_inter, 4),
            "diversity_score": round(1 - mean_inter, 4),
        },
        "intra_brain_consistency": {
            "mean_similarity": round(mean_intra, 4),
            "std": round(std_intra, 4),
        },
        "diversity_ratio": round(diversity_ratio, 4),
        "target_diversity": {
            "mean_similarity": round(mean_target_inter, 4),
            "diversity_preservation": round(
                (1 - mean_inter) / max(1 - mean_target_inter, 0.001), 4),
        },
    },
    "captions": {
        str(i): {
            "target": target_captions[i],
            "recon_avg": recon_captions[i],
            "samples": sample_captions[i],
            "correctness": round(correctness_scores[idx], 4),
        }
        for idx, i in enumerate(test_idx[:N_TEST_SAMPLES])
    },
}

with open(os.path.join(OUT, "vlm_diversity_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ═══════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Panel 1: Correctness distribution
ax = axes[0, 0]
ax.hist(correctness_scores, bins=10, color="#2196F3", edgecolor="white", alpha=0.8)
ax.axvline(mean_correctness, color="red", linestyle="--", linewidth=2,
           label=f"Mean={mean_correctness:.3f}")
ax.set_xlabel("Caption Similarity (target ↔ recon)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Semantic Correctness", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

# Panel 2: Inter vs Intra similarity
ax = axes[0, 1]
categories = ["Inter-brain\n(different inputs)", "Intra-brain\n(same input)"]
means = [mean_inter, mean_intra]
stds = [std_inter, std_intra]
colors = ["#FF7043", "#66BB6A"]
bars = ax.bar(categories, means, yerr=stds, color=colors, edgecolor="white",
              capsize=8, linewidth=1.5)
ax.set_ylabel("Caption Embedding Similarity", fontsize=11)
ax.set_title("Diversity Analysis", fontsize=13, fontweight="bold")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{m:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05)

# Panel 3: Correctness per sample (sorted)
ax = axes[0, 2]
sorted_scores = sorted(correctness_scores, reverse=True)
ax.bar(range(len(sorted_scores)), sorted_scores, color="#AB47BC", edgecolor="white")
ax.axhline(mean_correctness, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Test Sample (sorted)", fontsize=11)
ax.set_ylabel("Caption Similarity", fontsize=11)
ax.set_title("Per-Sample Correctness", fontsize=13, fontweight="bold")

# Panel 4: Example images with captions (3 samples)
show_indices = test_idx[:3]
for col, i in enumerate(show_indices):
    ax = axes[1, col]
    # Show target + reconstruction side by side
    target_pil = Image.open(all_images[i]["target"])
    recon_pil = Image.open(all_images[i]["recon_avg"])
    combined = Image.new("RGB", (512 + 10, 256), (40, 40, 40))
    combined.paste(target_pil, (0, 0))
    combined.paste(recon_pil, (266, 0))
    ax.imshow(combined)
    ax.set_xticks([128, 128 + 266])
    ax.set_xticklabels(["Target", "Reconstruction"], fontsize=9)
    ax.set_yticks([])

    # Truncate captions for display
    tc = target_captions[i][:80] + ("..." if len(target_captions[i]) > 80 else "")
    rc = recon_captions[i][:80] + ("..." if len(recon_captions[i]) > 80 else "")
    sim = correctness_scores[test_idx[:N_TEST_SAMPLES].index(i)]
    ax.set_title(f"Test {i} (sim={sim:.2f})", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.05, f"T: {tc}", transform=ax.transAxes,
            fontsize=7, ha="center", va="top", color="#2196F3",
            wrap=True)
    ax.text(0.5, -0.15, f"R: {rc}", transform=ax.transAxes,
            fontsize=7, ha="center", va="top", color="#FF7043",
            wrap=True)

plt.suptitle(
    f"VLM Semantic Diversity Analysis (Qwen2.5-VL-7B)\n"
    f"Correctness={mean_correctness:.3f}  "
    f"Inter-diversity={1 - mean_inter:.3f}  "
    f"Intra-consistency={mean_intra:.3f}  "
    f"Ratio={diversity_ratio:.2f}",
    fontsize=14, fontweight="bold", y=1.02,
)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "vlm_diversity.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("VLM DIVERSITY STUDY — SUMMARY")
print("=" * 70)
print(f"""
  VLM:                        {VLM_MODEL}
  Embedding model:            {EMBED_MODEL}
  Test samples evaluated:     {N_TEST_SAMPLES}
  Diverse samples per brain:  {N_DIVERSE}

  ┌─────────────────────────────┬──────────┐
  │ Metric                      │ Value    │
  ├─────────────────────────────┼──────────┤
  │ Semantic correctness        │ {mean_correctness:.3f}    │
  │ Inter-brain similarity      │ {mean_inter:.3f}    │
  │ Inter-brain diversity       │ {1 - mean_inter:.3f}    │
  │ Intra-brain consistency     │ {mean_intra:.3f}    │
  │ Diversity ratio             │ {diversity_ratio:.2f}     │
  │ Target baseline diversity   │ {1 - mean_target_inter:.3f}    │
  │ Diversity preservation      │ {(1 - mean_inter) / max(1 - mean_target_inter, 0.001):.2f}     │
  └─────────────────────────────┴──────────┘

  Saved: {OUT}/vlm_diversity.png
         {OUT}/vlm_diversity_results.json
""")
