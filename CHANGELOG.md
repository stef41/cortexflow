# Changelog

All notable changes to CortexFlow will be documented in this file.

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
