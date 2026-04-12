"""Brain → Audio reconstruction pipeline.

Reconstruct what someone heard from their fMRI activity.

Architecture: fMRI → BrainEncoder → DiT (flow matching on mel spectrograms)
→ Griffin-Lim or learned vocoder → waveform.

The DiT operates on 1D latent sequences (compressed mel spectrograms)
rather than 2D spatial maps.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortexflow._types import (
    BrainData,
    DiTConfig,
    FlowConfig,
    Modality,
    ReconstructionResult,
)
from cortexflow.brain_encoder import BrainEncoder, ROIBrainEncoder
from cortexflow.flow_matching import RectifiedFlowMatcher


# ── 1D DiT for audio ────────────────────────────────────────────────────


class DiTBlock1D(nn.Module):
    """DiT block adapted for 1D sequences (audio spectrograms)."""

    def __init__(self, hidden_dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim != hidden_dim else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_h = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_h), nn.SiLU(),
            nn.Linear(mlp_h, hidden_dim),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, brain_tokens: torch.Tensor | None = None):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)

        h = self.norm1(x) * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * h

        if brain_tokens is not None:
            h = self.norm_cross(x)
            kv = self.cond_proj(brain_tokens)
            h, _ = self.cross_attn(h, kv, kv, need_weights=False)
            x = x + h

        h = self.norm2(x) * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1)
        h = self.mlp(h)
        x = x + g2.unsqueeze(1) * h
        return x


class AudioDiT(nn.Module):
    """1D Diffusion Transformer for mel spectrogram generation.

    Operates on a sequence of mel frames (or compressed latents).
    """

    def __init__(
        self,
        n_mels: int = 80,
        seq_len: int = 128,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.seq_len = seq_len
        self.input_proj = nn.Linear(n_mels, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.time_dim = hidden_dim
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

        self.blocks = nn.ModuleList([
            DiTBlock1D(hidden_dim, num_heads, cond_dim) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, n_mels)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.view(-1, 1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        brain_global: torch.Tensor,
        brain_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity for mel spectrogram flow matching.

        Args:
            x: ``(B, n_mels, T)`` noisy mel spectrogram.
            t: ``(B,)`` timestep.
            brain_global: ``(B, cond_dim)``
            brain_tokens: ``(B, K, cond_dim)``

        Returns:
            ``(B, n_mels, T)`` predicted velocity.
        """
        B = x.shape[0]
        # (B, n_mels, T) → (B, T, n_mels) → (B, T, hidden)
        x = x.transpose(1, 2)
        x = self.input_proj(x) + self.pos_embed[:, :x.shape[1]]

        t_emb = self.time_embed(self._sinusoidal_embed(t))
        c = t_emb + self.cond_proj(brain_global)

        for block in self.blocks:
            x = block(x, c, brain_tokens)

        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, T, n_mels)
        return x.transpose(1, 2)  # (B, n_mels, T)


# ── Brain2Audio Pipeline ────────────────────────────────────────────────


class Brain2Audio(nn.Module):
    """Reconstruct audio from brain activity using 1D DiT + flow matching.

    Pipeline::

        fMRI → BrainEncoder → AudioDiT (flow matching) → mel spectrogram
        → Griffin-Lim → waveform

    Args:
        n_voxels: Number of fMRI voxels.
        n_mels: Number of mel frequency bins.
        audio_len: Number of mel spectrogram frames.
        sample_rate: Audio sample rate (for mel spectrogram computation).
    """

    def __init__(
        self,
        n_voxels: int = 1024,
        n_mels: int = 80,
        audio_len: int = 128,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        sample_rate: int = 16000,
        brain_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.audio_len = audio_len
        self.sample_rate = sample_rate

        cond_dim = hidden_dim
        # Brain encoder — custom or default
        if brain_encoder is not None:
            self.brain_encoder = brain_encoder
        else:
            self.brain_encoder = BrainEncoder(
                n_voxels=n_voxels, cond_dim=cond_dim, n_tokens=16,
            )
        self.dit = AudioDiT(
            n_mels=n_mels, seq_len=audio_len, hidden_dim=hidden_dim,
            depth=depth, num_heads=num_heads, cond_dim=cond_dim,
        )
        self.flow_matcher = RectifiedFlowMatcher(FlowConfig())

        # Unconditional embeddings for CFG
        self.uncond_global = nn.Parameter(torch.zeros(1, cond_dim))
        self.uncond_tokens = nn.Parameter(torch.zeros(1, 16, cond_dim))

    def encode_brain(
        self, brain_data: BrainData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode fMRI to conditioning signals."""
        if isinstance(self.brain_encoder, ROIBrainEncoder) and brain_data.roi_voxels is not None:
            return self.brain_encoder(brain_data.roi_voxels)
        return self.brain_encoder(brain_data.voxels)

    def training_loss(self, mel: torch.Tensor, brain_data: BrainData) -> torch.Tensor:
        """Flow matching loss on mel spectrograms.

        Args:
            mel: Target mel spectrogram ``(B, n_mels, T)``.
            brain_data: Corresponding fMRI.
        """
        brain_global, brain_tokens = self.encode_brain(brain_data)
        return self.flow_matcher.compute_loss(self.dit, mel, brain_global, brain_tokens)

    @torch.no_grad()
    def reconstruct(
        self,
        brain_data: BrainData,
        num_steps: int = 50,
        cfg_scale: float = 3.0,
        num_samples: int = 1,
        brain_noise: float = 0.0,
    ) -> ReconstructionResult:
        """Reconstruct audio mel spectrogram from brain activity.

        Args:
            brain_data: fMRI data to decode.
            num_steps: Number of ODE solver steps.
            cfg_scale: Classifier-free guidance scale.
            num_samples: Number of diverse samples per brain input.
                Each sample uses independent generation noise **and**
                a different perturbation of the brain conditioning.
                Output shape becomes ``(B, num_samples, n_mels, T)``
                when ``num_samples > 1``.
            brain_noise: Scale of Gaussian noise injected into brain
                embeddings. Each sample gets an independent perturbation,
                so different samples explore different interpretations
                of the brain signal. 0.0 = no perturbation. Typical
                range: 0.1–1.0.

        Returns:
            ReconstructionResult with the decoded mel spectrogram(s).
        """
        B = brain_data.batch_size
        device = brain_data.voxels.device
        brain_global, brain_tokens = self.encode_brain(brain_data)

        # Repeat conditioning for multiple samples per input
        if num_samples > 1:
            brain_global = brain_global.repeat_interleave(num_samples, dim=0)
            brain_tokens = brain_tokens.repeat_interleave(num_samples, dim=0)

        BN = B * num_samples

        # Perturb brain embeddings — each sample explores a different
        # interpretation of the brain signal (scaled relative to embedding norm)
        if brain_noise > 0.0:
            g_scale = brain_global.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            brain_global = brain_global + brain_noise * g_scale * torch.randn_like(brain_global)
            t_scale = brain_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            brain_tokens = brain_tokens + brain_noise * t_scale * torch.randn_like(brain_tokens)

        mel_shape = (BN, self.n_mels, self.audio_len)
        mel = self.flow_matcher.sample(
            self.dit, mel_shape, brain_global, brain_tokens,
            num_steps=num_steps, cfg_scale=cfg_scale,
            brain_global_uncond=self.uncond_global.expand(BN, -1),
            brain_tokens_uncond=self.uncond_tokens.expand(BN, -1, -1),
        )

        # Reshape to (B, num_samples, n_mels, T) when generating multiple
        if num_samples > 1:
            mel = mel.view(B, num_samples, self.n_mels, self.audio_len)

        return ReconstructionResult(
            modality=Modality.AUDIO,
            output=mel,
            brain_condition=brain_global[:B],
            n_steps=num_steps,
            cfg_scale=cfg_scale,
            metadata={"num_samples": num_samples, "brain_noise": brain_noise},
        )

    @staticmethod
    def mel_to_waveform(mel: torch.Tensor, n_fft: int = 1024, hop_length: int = 256) -> torch.Tensor:
        """Convert mel spectrogram to waveform via Griffin-Lim.

        Args:
            mel: ``(B, n_mels, T)`` mel spectrogram (linear scale).
            n_fft: FFT size.
            hop_length: Hop length.

        Returns:
            ``(B, samples)`` waveform.
        """
        B, n_mels, T = mel.shape
        # Create simple mel filterbank inverse (pseudo-inverse approach)
        n_freqs = n_fft // 2 + 1
        # Approximate: project mel back to linear spectrogram
        mel_basis = torch.linspace(0, 1, n_mels, device=mel.device).unsqueeze(1)
        freq_basis = torch.linspace(0, 1, n_freqs, device=mel.device).unsqueeze(0)
        filterbank = torch.exp(-0.5 * ((mel_basis - freq_basis) / 0.05) ** 2)
        filterbank = filterbank / (filterbank.sum(dim=0, keepdim=True) + 1e-8)

        # Mel → linear spectrogram (pseudo-inverse)
        fb_pinv = filterbank.T  # (n_freqs, n_mels)
        magnitude = torch.matmul(fb_pinv.unsqueeze(0), mel.clamp(min=0))  # (B, n_freqs, T)

        # Griffin-Lim: iterative phase estimation
        window = torch.hann_window(n_fft, device=mel.device)
        out_length = T * hop_length
        phase = torch.randn(B, n_freqs, T, device=mel.device) * 2 * math.pi
        for _ in range(32):
            stft = magnitude * torch.exp(1j * phase)
            waveform = torch.istft(
                stft, n_fft=n_fft, hop_length=hop_length,
                window=window, length=out_length,
            )
            new_stft = torch.stft(
                waveform, n_fft=n_fft, hop_length=hop_length,
                window=window, return_complex=True,
            )
            # Trim or pad to match magnitude shape
            nt = min(new_stft.shape[-1], T)
            phase = torch.zeros_like(magnitude)
            phase[:, :, :nt] = new_stft[:, :, :nt].angle()

        stft = magnitude * torch.exp(1j * phase)
        waveform = torch.istft(
            stft, n_fft=n_fft, hop_length=hop_length,
            window=window, length=out_length,
        )
        return waveform


def build_brain2audio(
    n_voxels: int = 1024,
    n_mels: int = 80,
    audio_len: int = 128,
    hidden_dim: int = 256,
    depth: int = 6,
) -> Brain2Audio:
    """Build a Brain2Audio model with sensible defaults."""
    return Brain2Audio(
        n_voxels=n_voxels, n_mels=n_mels, audio_len=audio_len,
        hidden_dim=hidden_dim, depth=depth,
    )
