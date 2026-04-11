"""Brain encoder: fMRI voxels → conditioning embeddings for DiT.

Supports two encoding modes:

1. **Global**: MLP projects full voxel vector to a single global embedding.
   Used for AdaLN conditioning in DiT blocks.
2. **Tokenized**: MLP projects voxels to a sequence of ``n_tokens`` vectors.
   Used for cross-attention in DiT blocks, giving the model rich spatial
   information about brain activity patterns.

Optionally supports **ROI-aware encoding**: separate sub-encoders for
different cortical regions (V1, FFA, A1, etc.) whose outputs are
concatenated and projected.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BrainEncoder(nn.Module):
    """Project fMRI voxels to DiT conditioning embeddings.

    Produces both a global embedding (for AdaLN) and a token sequence
    (for cross-attention).

    Args:
        n_voxels: Number of input fMRI voxels.
        cond_dim: Dimension of each conditioning vector.
        n_tokens: Number of brain tokens for cross-attention.
        hidden_dim: Intermediate MLP dimension.
        dropout: Dropout rate for regularization.
    """

    def __init__(
        self,
        n_voxels: int,
        cond_dim: int = 768,
        n_tokens: int = 16,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_voxels = n_voxels
        self.cond_dim = cond_dim
        self.n_tokens = n_tokens
        h = hidden_dim or cond_dim * 2

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_voxels, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Global projection (for AdaLN conditioning)
        self.global_proj = nn.Sequential(
            nn.Linear(h, cond_dim),
            nn.LayerNorm(cond_dim),
        )

        # Token projection (for cross-attention)
        self.token_proj = nn.Sequential(
            nn.Linear(h, n_tokens * cond_dim),
        )

    def forward(self, voxels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode fMRI voxels to conditioning signals.

        Args:
            voxels: ``(B, n_voxels)`` BOLD activity.

        Returns:
            (brain_global, brain_tokens):
                - brain_global: ``(B, cond_dim)`` for AdaLN.
                - brain_tokens: ``(B, n_tokens, cond_dim)`` for cross-attention.
        """
        if voxels.ndim == 1:
            voxels = voxels.unsqueeze(0)

        h = self.backbone(voxels)
        brain_global = self.global_proj(h)
        brain_tokens = self.token_proj(h).view(-1, self.n_tokens, self.cond_dim)
        return brain_global, brain_tokens


class ROIBrainEncoder(nn.Module):
    """ROI-aware brain encoder with per-region sub-encoders.

    Different brain regions encode different information (V1 → low-level
    visual, FFA → faces, A1 → audio, etc.). This encoder processes each
    ROI independently and then fuses them.

    Args:
        roi_sizes: Dict mapping ROI names to voxel counts.
        cond_dim: Output conditioning dimension.
        n_tokens: Number of brain tokens for cross-attention.
        per_roi_dim: Hidden dim for each ROI sub-encoder.
    """

    def __init__(
        self,
        roi_sizes: dict[str, int],
        cond_dim: int = 768,
        n_tokens: int = 16,
        per_roi_dim: int = 128,
    ) -> None:
        super().__init__()
        self.roi_names = sorted(roi_sizes.keys())
        self.roi_sizes = roi_sizes
        self.cond_dim = cond_dim
        self.n_tokens = n_tokens

        # Per-ROI sub-encoders
        self.roi_encoders = nn.ModuleDict()
        for name in self.roi_names:
            n = roi_sizes[name]
            self.roi_encoders[name] = nn.Sequential(
                nn.Linear(n, per_roi_dim),
                nn.LayerNorm(per_roi_dim),
                nn.GELU(),
                nn.Linear(per_roi_dim, per_roi_dim),
            )

        total_roi_dim = per_roi_dim * len(self.roi_names)

        # Fusion
        self.global_proj = nn.Sequential(
            nn.Linear(total_roi_dim, cond_dim),
            nn.LayerNorm(cond_dim),
        )
        self.token_proj = nn.Linear(total_roi_dim, n_tokens * cond_dim)

    def forward(
        self, roi_voxels: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode per-ROI voxels.

        Args:
            roi_voxels: Dict mapping ROI names to tensors ``(B, n_voxels_roi)``.

        Returns:
            (brain_global, brain_tokens)
        """
        encoded = []
        for name in self.roi_names:
            x = roi_voxels[name]
            if x.ndim == 1:
                x = x.unsqueeze(0)
            encoded.append(self.roi_encoders[name](x))

        h = torch.cat(encoded, dim=-1)
        brain_global = self.global_proj(h)
        brain_tokens = self.token_proj(h).view(-1, self.n_tokens, self.cond_dim)
        return brain_global, brain_tokens


class SubjectAdapter(nn.Module):
    """Lightweight per-subject adapter (LoRA-style).

    Different subjects have different brain anatomy and functional
    organization. This adapter learns a low-rank residual per subject
    that adjusts the shared brain encoder output.

    Args:
        cond_dim: Conditioning dimension to adapt.
        rank: Low-rank dimension.
        n_subjects: Number of subjects.
    """

    def __init__(self, cond_dim: int = 768, rank: int = 16, n_subjects: int = 10) -> None:
        super().__init__()
        self.subject_embed = nn.Embedding(n_subjects, rank)
        self.down = nn.Linear(cond_dim, rank, bias=False)
        self.up = nn.Linear(rank, cond_dim, bias=False)

        nn.init.zeros_(self.up.weight)

    def forward(
        self, brain_global: torch.Tensor, subject_idx: torch.Tensor
    ) -> torch.Tensor:
        """Apply subject-specific adaptation.

        Args:
            brain_global: ``(B, cond_dim)``
            subject_idx: ``(B,)`` integer subject indices.

        Returns:
            Adapted ``(B, cond_dim)``
        """
        s = self.subject_embed(subject_idx)  # (B, rank)
        residual = self.up(self.down(brain_global) * s)
        return brain_global + residual


def make_synthetic_brain_data(
    batch_size: int = 4,
    n_voxels: int = 1024,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate synthetic fMRI-like data for testing.

    Produces smooth, spatially-correlated activations that roughly
    mimic the statistics of real BOLD signal.
    """
    # Base signal with spatial correlation
    raw = torch.randn(batch_size, n_voxels, device=device)
    # Smooth with a 1D Gaussian kernel
    kernel_size = min(31, n_voxels // 2 * 2 + 1)
    if kernel_size >= 3:
        sigma = kernel_size / 6
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)
        raw_3d = raw.unsqueeze(1)  # (B, 1, V)
        padding = kernel_size // 2
        smoothed = torch.nn.functional.conv1d(raw_3d, kernel, padding=padding)
        raw = smoothed.squeeze(1)
    # Normalize to zero mean, unit variance
    raw = (raw - raw.mean(dim=-1, keepdim=True)) / (raw.std(dim=-1, keepdim=True) + 1e-8)
    return raw
