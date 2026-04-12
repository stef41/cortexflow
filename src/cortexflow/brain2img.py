"""Brain → Image reconstruction pipeline.

End-to-end pipeline: fMRI voxels → BrainEncoder → DiT (flow matching)
→ VAE decoder → RGB image.

This is the core brain decoding pipeline. Given measured fMRI activity
while a subject views an image, reconstruct what they saw.

Reference architecture inspired by:
- MindEye (Scotti et al. 2023) for brain-to-image conditioning
- SD3 / FLUX for the DiT + flow matching backbone
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cortexflow._types import (
    BrainData,
    DiTConfig,
    FlowConfig,
    Modality,
    ReconstructionResult,
    VAEConfig,
)
from cortexflow.brain_encoder import BrainEncoder
from cortexflow.dit import DiffusionTransformer
from cortexflow.flow_matching import RectifiedFlowMatcher
from cortexflow.vae import LatentVAE


class Brain2Image(nn.Module):
    """Reconstruct images from brain activity using DiT + Flow Matching.

    Full pipeline::

        fMRI voxels
            → BrainEncoder (global + tokens)
            → DiffusionTransformer (flow matching ODE)
            → LatentVAE.decode()
            → RGB image

    Args:
        n_voxels: Number of fMRI voxels.
        img_size: Output image spatial resolution.
        dit_config: DiT architecture configuration.
        vae_config: VAE configuration.
        flow_config: Flow matching configuration.
        n_brain_tokens: Number of brain conditioning tokens.
    """

    def __init__(
        self,
        n_voxels: int = 1024,
        img_size: int = 64,
        dit_config: DiTConfig | None = None,
        vae_config: VAEConfig | None = None,
        flow_config: FlowConfig | None = None,
        n_brain_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.img_size = img_size

        # Configs
        dit_cfg = dit_config or DiTConfig()
        vae_cfg = vae_config or VAEConfig()
        flow_cfg = flow_config or FlowConfig()

        # VAE: determines latent spatial size
        self.vae = LatentVAE(vae_cfg)
        n_downsample = len(vae_cfg.hidden_dims)
        latent_size = img_size // (2 ** n_downsample)

        # Brain encoder
        self.brain_encoder = BrainEncoder(
            n_voxels=n_voxels,
            cond_dim=dit_cfg.cond_dim,
            n_tokens=n_brain_tokens,
        )

        # Unconditional embeddings for classifier-free guidance
        self.uncond_global = nn.Parameter(torch.zeros(1, dit_cfg.cond_dim))
        self.uncond_tokens = nn.Parameter(torch.zeros(1, n_brain_tokens, dit_cfg.cond_dim))

        # DiT operates on VAE latent space
        dit_cfg_updated = DiTConfig(
            in_channels=vae_cfg.latent_channels,
            hidden_dim=dit_cfg.hidden_dim,
            depth=dit_cfg.depth,
            num_heads=dit_cfg.num_heads,
            patch_size=dit_cfg.patch_size,
            cond_dim=dit_cfg.cond_dim,
            mlp_ratio=dit_cfg.mlp_ratio,
            qk_norm=dit_cfg.qk_norm,
            use_cross_attn=dit_cfg.use_cross_attn,
        )
        self.dit = DiffusionTransformer(dit_cfg_updated, img_size=latent_size)

        # Flow matcher
        self.flow_matcher = RectifiedFlowMatcher(flow_cfg)

        self._latent_size = latent_size
        self._latent_channels = vae_cfg.latent_channels

    def encode_brain(
        self, brain_data: BrainData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode fMRI to conditioning signals."""
        return self.brain_encoder(brain_data.voxels)

    def training_loss(
        self,
        images: torch.Tensor,
        brain_data: BrainData,
        cfg_dropout: float = 0.1,
    ) -> torch.Tensor:
        """Compute training loss for brain-conditioned image generation.

        Args:
            images: Target images ``(B, 3, H, W)``.
            brain_data: Corresponding fMRI data.
            cfg_dropout: Probability of dropping brain condition for CFG training.

        Returns:
            Scalar loss.
        """
        B = images.shape[0]

        # Encode images to latent space (detach VAE — train separately or frozen)
        with torch.no_grad():
            z, _, _ = self.vae.encode(images)

        # Encode brain
        brain_global, brain_tokens = self.encode_brain(brain_data)

        # CFG training: randomly drop conditioning
        if cfg_dropout > 0 and self.training:
            mask = torch.rand(B, device=images.device) < cfg_dropout
            if mask.any():
                brain_global = brain_global.clone()
                brain_tokens = brain_tokens.clone()
                brain_global[mask] = self.uncond_global.expand(mask.sum(), -1)
                brain_tokens[mask] = self.uncond_tokens.expand(mask.sum(), -1, -1)

        # Flow matching loss on latent space
        return self.flow_matcher.compute_loss(
            self.dit, z, brain_global, brain_tokens
        )

    @torch.no_grad()
    def reconstruct(
        self,
        brain_data: BrainData,
        num_steps: int = 50,
        cfg_scale: float = 4.0,
        num_samples: int = 1,
        brain_noise: float = 0.0,
    ) -> ReconstructionResult:
        """Reconstruct an image from brain activity.

        Args:
            brain_data: fMRI data to decode.
            num_steps: Number of ODE solver steps.
            cfg_scale: Classifier-free guidance scale.
            num_samples: Number of diverse samples per brain input.
                Each sample uses independent generation noise **and**
                a different perturbation of the brain conditioning.
                Output shape becomes ``(B, num_samples, C, H, W)``
                when ``num_samples > 1``.
            brain_noise: Scale of Gaussian noise injected into brain
                embeddings. Each sample gets an independent perturbation,
                so different samples explore different interpretations
                of the brain signal. 0.0 = no perturbation (diversity
                comes only from generation noise). Typical range: 0.1–1.0.

        Returns:
            ReconstructionResult with the decoded image(s).
        """
        B = brain_data.batch_size
        device = brain_data.voxels.device

        # Encode brain
        brain_global, brain_tokens = self.encode_brain(brain_data)

        # Repeat conditioning for multiple samples per input
        if num_samples > 1:
            brain_global = brain_global.repeat_interleave(num_samples, dim=0)
            brain_tokens = brain_tokens.repeat_interleave(num_samples, dim=0)

        BN = B * num_samples

        # Perturb brain embeddings — each sample explores a different
        # interpretation of the brain signal
        if brain_noise > 0.0:
            brain_global = brain_global + brain_noise * torch.randn_like(brain_global)
            brain_tokens = brain_tokens + brain_noise * torch.randn_like(brain_tokens)

        # Unconditional embeddings for CFG
        uncond_global = self.uncond_global.expand(BN, -1)
        uncond_tokens = self.uncond_tokens.expand(BN, -1, -1)

        # Sample latents via flow matching (each gets independent noise)
        latent_shape = (BN, self._latent_channels, self._latent_size, self._latent_size)
        z = self.flow_matcher.sample(
            self.dit,
            shape=latent_shape,
            brain_global=brain_global,
            brain_tokens=brain_tokens,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            brain_global_uncond=uncond_global,
            brain_tokens_uncond=uncond_tokens,
        )

        # Decode latents to images
        images = self.vae.decode(z)
        images = images.clamp(0, 1)

        # Reshape to (B, num_samples, C, H, W) when generating multiple
        if num_samples > 1:
            C, H, W = images.shape[1:]
            images = images.view(B, num_samples, C, H, W)

        return ReconstructionResult(
            modality=Modality.IMAGE,
            output=images,
            brain_condition=brain_global[:B],
            n_steps=num_steps,
            cfg_scale=cfg_scale,
            metadata={"num_samples": num_samples, "brain_noise": brain_noise},
        )


def build_brain2img(
    n_voxels: int = 1024,
    img_size: int = 64,
    hidden_dim: int = 256,
    depth: int = 6,
    num_heads: int = 8,
) -> Brain2Image:
    """Build a Brain2Image model with sensible defaults.

    Args:
        n_voxels: Number of fMRI voxels.
        img_size: Target image size (square).
        hidden_dim: DiT hidden dimension.
        depth: Number of DiT blocks.
        num_heads: Attention heads.
    """
    dit_cfg = DiTConfig(
        hidden_dim=hidden_dim,
        depth=depth,
        num_heads=num_heads,
        cond_dim=hidden_dim,
    )
    vae_cfg = VAEConfig(hidden_dims=[32, 64])  # lightweight for small images
    flow_cfg = FlowConfig()
    return Brain2Image(
        n_voxels=n_voxels,
        img_size=img_size,
        dit_config=dit_cfg,
        vae_config=vae_cfg,
        flow_config=flow_cfg,
    )
