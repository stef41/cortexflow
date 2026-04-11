"""Shared test fixtures for cortexflow."""

from __future__ import annotations

import pytest
import torch

from cortexflow._types import BrainData, DiTConfig, FlowConfig, VAEConfig


# ── Small configs for fast CPU testing ───────────────────────────────────

BATCH = 2
N_VOXELS = 64
IMG_SIZE = 16
HIDDEN_DIM = 32
DEPTH = 2
NUM_HEADS = 4
N_BRAIN_TOKENS = 4
N_MELS = 16
AUDIO_LEN = 32
VOCAB_SIZE = 256
MAX_TEXT_LEN = 16


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def brain_voxels():
    return torch.randn(BATCH, N_VOXELS)


@pytest.fixture
def brain_data(brain_voxels):
    return BrainData(voxels=brain_voxels)


@pytest.fixture
def dit_config():
    return DiTConfig(
        in_channels=4,
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        patch_size=2,
        cond_dim=HIDDEN_DIM,
        mlp_ratio=2.0,
        qk_norm=True,
        use_cross_attn=True,
    )


@pytest.fixture
def vae_config():
    return VAEConfig(
        in_channels=3,
        latent_channels=4,
        hidden_dims=[16, 32],
        kl_weight=1e-6,
    )


@pytest.fixture
def flow_config():
    return FlowConfig(
        num_steps=4,
        logit_normal=True,
        solver="euler",
        cfg_scale=2.0,
    )
