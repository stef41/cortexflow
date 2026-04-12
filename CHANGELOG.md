# Changelog

All notable changes to CortexFlow will be documented in this file.

## [0.4.0] - 2026-04-12

### Added

- **ROI-aware pipeline support**: All pipelines (`Brain2Image`, `Brain2Audio`, `Brain2Text`) now accept a custom `brain_encoder` parameter. Plug in `ROIBrainEncoder` to process V1, FFA, A1, etc. independently and ablate specific brain regions.
- **`roi_voxels` field on `BrainData`**: Pass per-region tensors directly; pipelines auto-dispatch to `ROIBrainEncoder` when available.
- **`encode_brain()` method** on all pipelines: Unified brain encoding interface that handles both standard and ROI encoders.
- ROI ablation example + diversity measurement demo (`examples/diversity_demo.py`).

### Fixed

- **`brain_noise` now scales relative to embedding norm**: `brain_noise=0.3` means "perturb by 30% of the signal magnitude." Previously used absolute noise that was too weak after training.

## [0.3.0] - 2026-04-12

### Added

- **Brain-signal diversity** (`brain_noise`): All pipelines accept a `brain_noise` parameter that injects Gaussian perturbations into brain conditioning embeddings. Each sample explores a different interpretation of the brain signal — semantic diversity driven by the brain, not just generation noise. Typical range: 0.1–1.0.

## [0.2.0] - 2026-04-12

### Added

- **Semantic diversity**: All pipelines now support `num_samples` parameter to generate multiple diverse reconstructions per brain input. Each sample uses independent noise (image/audio) or independent random draws (text), producing semantically varied outputs.
- **Nucleus (top-p) sampling** for Brain2Text: `top_p` parameter enables nucleus filtering alongside top-k, giving finer control over text generation diversity.
- When `num_samples > 1`, output shapes become `(B, num_samples, ...)` for image/audio; text metadata returns grouped lists per brain input.

## [0.1.1] - 2025-06-26

### Fixed

- **DiT zero-init**: `_initialize_weights()` no longer overwrites critical zero-initialized gating parameters (AdaLN modulation, final layer projection). Output is now exactly zero at initialization for stable training.
- **Brain2Text BOS mismatch**: Training now prepends BOS token so the model learns to predict from BOS context, matching inference behavior.
- **Brain2Text empty output**: `reconstruct()` now correctly skips the BOS token when decoding generated sequences to text.

## [0.1.0] - 2025-06-25

### Added

- **DiT backbone**: Diffusion Transformer with AdaLN-Zero conditioning, QK-Norm, SwiGLU, and cross-attention to brain tokens
- **Rectified Flow Matching**: Linear interpolation paths, logit-normal timestep sampling (SD3-style), Euler and midpoint ODE solvers
- **Brain Encoder**: MLP projector producing global embeddings (for AdaLN) and token sequences (for cross-attention)
- **ROI-Aware Encoder**: Per-region sub-encoders with fusion for anatomically-informed encoding
- **Subject Adapter**: LoRA-style per-subject low-rank adaptation
- **Brain2Image**: Full fMRI → LatentVAE → DiT → image pipeline with classifier-free guidance
- **Brain2Audio**: fMRI → 1D DiT → mel spectrogram → Griffin-Lim waveform
- **Brain2Text**: fMRI → transformer decoder → autoregressive byte-level text generation
- **LatentVAE / AudioVAE**: Lightweight convolutional VAEs for image and audio latent spaces
- **Training utilities**: Trainer class, warmup-cosine LR scheduler, EMA, synthetic data generators
- **Comprehensive test suite**: 100+ tests covering all modules
