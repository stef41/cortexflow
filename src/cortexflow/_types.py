"""Core data types for cortexflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class Modality(str, Enum):
    """Output modality for brain decoding."""

    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"


@dataclass
class BrainData:
    """Container for fMRI brain activity data.

    Attributes:
        voxels: Tensor of shape ``(batch, n_voxels)`` or ``(n_voxels,)``.
        subject_id: Optional subject identifier for subject-specific adapters.
        roi_mask: Optional boolean mask indicating which voxels belong to ROIs.
        tr: Repetition time in seconds (fMRI temporal resolution).
    """

    voxels: torch.Tensor
    subject_id: str | None = None
    roi_mask: torch.Tensor | None = None
    tr: float = 2.0

    @property
    def n_voxels(self) -> int:
        return self.voxels.shape[-1]

    @property
    def batch_size(self) -> int:
        if self.voxels.ndim == 1:
            return 1
        return self.voxels.shape[0]


@dataclass
class ReconstructionResult:
    """Result of a brain decoding reconstruction."""

    modality: Modality
    output: torch.Tensor  # decoded output (image / waveform / token ids)
    brain_condition: torch.Tensor  # the conditioning embeddings used
    n_steps: int = 50  # diffusion steps used
    cfg_scale: float = 1.0  # classifier-free guidance scale
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training a brain decoder."""

    learning_rate: float = 1e-4
    batch_size: int = 16
    n_epochs: int = 100
    warmup_steps: int = 500
    weight_decay: float = 0.01
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    mixed_precision: bool = False
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500


@dataclass
class DiTConfig:
    """Configuration for the Diffusion Transformer."""

    in_channels: int = 4  # latent channels
    hidden_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    patch_size: int = 2
    cond_dim: int = 768  # brain conditioning dimension
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    use_cross_attn: bool = True


@dataclass
class VAEConfig:
    """Configuration for the latent VAE."""

    in_channels: int = 3  # RGB
    latent_channels: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    kl_weight: float = 1e-6


@dataclass
class FlowConfig:
    """Configuration for flow matching."""

    num_steps: int = 50
    sigma_min: float = 1e-5
    logit_normal: bool = True  # SD3-style timestep sampling
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    solver: str = "euler"  # "euler" or "midpoint"
    cfg_scale: float = 4.0


class CortexFlowError(Exception):
    """Base exception for cortexflow."""
