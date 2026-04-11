"""Diffusion Transformer (DiT) backbone.

Implements the DiT architecture from Peebles & Xie (2022) with modern
improvements from SD3/FLUX:

- **AdaLN-Zero** conditioning: timestep + brain embeddings modulate
  LayerNorm parameters with a learned zero-initialized gate.
- **QK-Norm**: normalizes query and key projections for training
  stability at scale (per Dehghani et al. 2023).
- **SwiGLU** activation in the feedforward network (Shazeer 2020).
- **Cross-attention** to brain conditioning tokens for rich
  fMRI→latent interaction.

Reference: "Scalable Diffusion Models with Transformers" (arXiv:2212.09748)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortexflow._types import DiTConfig


# ── Helpers ──────────────────────────────────────────────────────────────


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive LayerNorm modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLU(nn.Module):
    """SwiGLU feedforward network (Shazeer 2020).

    Projects to 2× intermediate dim, applies SiLU gate, projects back.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ── Timestep Embedding ──────────────────────────────────────────────────


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP projection.

    Maps scalar timestep ``t ∈ [0, 1]`` to a ``dim``-dimensional vector.
    """

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        # t: (batch,) or (batch, 1)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t_flat = t.view(-1, 1).float()
        args = t_flat * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


# ── Patch Embedding ─────────────────────────────────────────────────────


class PatchEmbed(nn.Module):
    """Convert spatial latent maps into a sequence of patch tokens."""

    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)  # (B, D, H', W')
        return x.flatten(2).transpose(1, 2)


# ── DiT Block ───────────────────────────────────────────────────────────


class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN-Zero conditioning.

    Each block contains:
    1. AdaLN-modulated self-attention (with optional QK-Norm)
    2. Optional cross-attention to brain conditioning tokens
    3. AdaLN-modulated SwiGLU feedforward
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        use_cross_attn: bool = True,
        cond_dim: int = 768,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cross_attn = use_cross_attn

        # AdaLN modulation: produces 6 vectors (shift/scale/gate × 2)
        n_adaln = 9 if use_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, n_adaln * hidden_dim),
        )

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )

        # QK-Norm: normalize Q and K for training stability
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(hidden_dim // num_heads, eps=1e-6)
            self.k_norm = nn.LayerNorm(hidden_dim // num_heads, eps=1e-6)
            self.num_heads = num_heads

        # Cross-attention to brain conditioning
        if use_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True, dropout=0.0
            )
            self.cond_proj = (
                nn.Linear(cond_dim, hidden_dim)
                if cond_dim != hidden_dim
                else nn.Identity()
            )

        # Feedforward
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden)

        # Zero-initialize the gating parameters
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _qk_norm_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Self-attention with per-head QK normalization."""
        B, N, D = q.shape
        head_dim = D // self.num_heads

        # Reshape to (B, heads, N, head_dim)
        q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, head_dim).transpose(1, 2)

        # Normalize Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        return out.transpose(1, 2).reshape(B, N, D)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        brain_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Patch tokens ``(B, N, D)``.
            c: Conditioning vector ``(B, D)`` — fused timestep + brain global.
            brain_tokens: Brain conditioning tokens ``(B, T, cond_dim)`` for
                cross-attention.
        """
        # Compute all modulation parameters at once
        mod = self.adaLN_modulation(c)
        if self.use_cross_attn:
            (
                shift_sa, scale_sa, gate_sa,
                shift_ca, scale_ca, gate_ca,
                shift_ff, scale_ff, gate_ff,
            ) = mod.chunk(9, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = mod.chunk(
                6, dim=-1
            )

        # 1. Self-attention with AdaLN
        h = modulate(self.norm1(x), shift_sa, scale_sa)
        if self.qk_norm:
            h = self._qk_norm_attn(h, h, h)
        else:
            h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_sa.unsqueeze(1) * h

        # 2. Cross-attention to brain tokens
        if self.use_cross_attn and brain_tokens is not None:
            h = modulate(self.norm_cross(x), shift_ca, scale_ca)
            kv = self.cond_proj(brain_tokens)
            h, _ = self.cross_attn(h, kv, kv, need_weights=False)
            x = x + gate_ca.unsqueeze(1) * h

        # 3. Feedforward with AdaLN
        h = modulate(self.norm2(x), shift_ff, scale_ff)
        h = self.mlp(h)
        x = x + gate_ff.unsqueeze(1) * h

        return x


# ── Final Layer ─────────────────────────────────────────────────────────


class FinalLayer(nn.Module):
    """DiT final layer: AdaLN → linear projection back to patch space."""

    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.proj = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)

        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.proj(x)


# ── Diffusion Transformer ──────────────────────────────────────────────


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer (DiT) for brain-conditioned generation.

    Operates on latent patches and predicts the velocity field for
    rectified flow matching.

    Args:
        config: DiT configuration.
        img_size: Spatial size of the latent input (H = W).
    """

    def __init__(self, config: DiTConfig | None = None, img_size: int = 32) -> None:
        super().__init__()
        cfg = config or DiTConfig()
        self.config = cfg
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = PatchEmbed(cfg.in_channels, cfg.hidden_dim, cfg.patch_size)
        n_patches = (img_size // cfg.patch_size) ** 2

        # Learned positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(cfg.hidden_dim)

        # Brain conditioning global projection
        self.cond_global_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qk_norm=cfg.qk_norm,
                    use_cross_attn=cfg.use_cross_attn,
                    cond_dim=cfg.cond_dim,
                )
                for _ in range(cfg.depth)
            ]
        )

        # Output projection
        self.final_layer = FinalLayer(cfg.hidden_dim, cfg.patch_size, cfg.in_channels)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights following DiT conventions.

        Global Xavier init first, then re-apply zero-init on gating
        parameters (AdaLN modulation, final layer) for stable training.
        """

        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)

        # Re-apply zero-init on gating parameters (overwritten by global init)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.proj.weight)
        nn.init.zeros_(self.final_layer.proj.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape patch tokens back to spatial latent maps.

        Args:
            x: ``(B, N, patch_size² × C)``

        Returns:
            ``(B, C, H, W)``
        """
        p = self.config.patch_size
        c = self.config.in_channels
        h = w = self.img_size // p
        x = x.view(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(-1, c, h * p, w * p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        brain_global: torch.Tensor,
        brain_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity field v(x_t, t | brain).

        Args:
            x: Noisy latent ``(B, C, H, W)``.
            t: Timestep ``(B,)`` in ``[0, 1]``.
            brain_global: Global brain embedding ``(B, cond_dim)`` — pooled
                fMRI representation used for AdaLN conditioning.
            brain_tokens: Sequence of brain tokens ``(B, T, cond_dim)`` for
                cross-attention. If None, cross-attention is skipped.

        Returns:
            Predicted velocity ``(B, C, H, W)``.
        """
        # Embed patches + add positional encoding
        x = self.patch_embed(x) + self.pos_embed

        # Condition = timestep + brain global embedding
        t_emb = self.time_embed(t)
        c_global = self.cond_global_proj(brain_global)
        c = t_emb + c_global

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, brain_tokens)

        # Final output
        x = self.final_layer(x, c)
        return self.unpatchify(x)
