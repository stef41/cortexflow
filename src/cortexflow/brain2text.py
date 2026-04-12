"""Brain → Text reconstruction pipeline.

Reconstruct what someone read or thought in linguistic form from fMRI.

Architecture: fMRI → BrainEncoder → Transformer Decoder with
cross-attention to brain tokens → autoregressive text generation.

Unlike image/audio which use flow matching on continuous latents,
text uses autoregressive decoding since language is inherently discrete.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortexflow._types import BrainData, Modality, ReconstructionResult
from cortexflow.brain_encoder import BrainEncoder, ROIBrainEncoder


class TextDecoderBlock(nn.Module):
    """Transformer decoder block with causal self-attention + brain cross-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim != hidden_dim else nn.Identity()
        self.norm3 = nn.LayerNorm(hidden_dim, eps=1e-6)
        mlp_h = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_h), nn.SiLU(),
            nn.Linear(mlp_h, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        brain_tokens: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Causal self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h

        # Cross-attention to brain tokens
        h = self.norm2(x)
        kv = self.cond_proj(brain_tokens)
        h, _ = self.cross_attn(h, kv, kv, need_weights=False)
        x = x + h

        # Feedforward
        h = self.norm3(x)
        x = x + self.mlp(h)
        return x


class BrainTextDecoder(nn.Module):
    """Transformer decoder for brain → text.

    Uses a simple character/word-piece vocabulary for self-contained
    operation without external tokenizers.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        max_len: int = 128,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        self.blocks = nn.ModuleList([
            TextDecoderBlock(hidden_dim, num_heads, cond_dim) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embed.weight

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(
        self,
        token_ids: torch.Tensor,
        brain_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training (teacher forcing).

        Args:
            token_ids: ``(B, T)`` input token IDs.
            brain_tokens: ``(B, K, cond_dim)`` brain conditioning tokens.

        Returns:
            ``(B, T, vocab_size)`` logits.
        """
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(positions)

        mask = self._causal_mask(T, token_ids.device)
        for block in self.blocks:
            x = block(x, brain_tokens, attn_mask=mask)

        x = self.final_norm(x)
        return self.lm_head(x)


class Brain2Text(nn.Module):
    """Reconstruct text from brain activity.

    Pipeline::

        fMRI → BrainEncoder → BrainTextDecoder (autoregressive) → tokens → text

    Uses byte-level encoding (vocab_size=256) for simplicity. Each byte
    is one token, so this handles any text without a tokenizer.

    Args:
        n_voxels: Number of fMRI voxels.
        max_len: Maximum output text length.
        hidden_dim: Model dimension.
        depth: Number of transformer layers.
        num_heads: Attention heads.
    """

    def __init__(
        self,
        n_voxels: int = 1024,
        max_len: int = 128,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        brain_encoder: nn.Module | None = None,
    ):
        super().__init__()
        cond_dim = hidden_dim
        # Brain encoder — custom or default
        if brain_encoder is not None:
            self.brain_encoder = brain_encoder
        else:
            self.brain_encoder = BrainEncoder(
                n_voxels=n_voxels, cond_dim=cond_dim, n_tokens=16,
            )
        self.decoder = BrainTextDecoder(
            vocab_size=256,  # byte-level
            max_len=max_len,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            cond_dim=cond_dim,
        )
        self.max_len = max_len
        self.bos_token = 0  # null byte as BOS

    def encode_brain(
        self, brain_data: BrainData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode fMRI to conditioning signals."""
        if isinstance(self.brain_encoder, ROIBrainEncoder) and brain_data.roi_voxels is not None:
            return self.brain_encoder(brain_data.roi_voxels)
        return self.brain_encoder(brain_data.voxels)

    @staticmethod
    def text_to_tokens(text: str) -> torch.Tensor:
        """Encode text to byte-level token IDs."""
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

    @staticmethod
    def tokens_to_text(tokens: torch.Tensor) -> str:
        """Decode byte-level token IDs to text."""
        byte_list = tokens.cpu().tolist()
        # Stop at first null byte
        if 0 in byte_list:
            byte_list = byte_list[: byte_list.index(0)]
        return bytes(byte_list).decode("utf-8", errors="replace")

    def training_loss(
        self,
        text_tokens: torch.Tensor,
        brain_data: BrainData,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for text generation.

        Args:
            text_tokens: ``(B, T)`` target token IDs (byte-level).
                Raw text bytes — BOS is prepended automatically.
            brain_data: Corresponding fMRI data.

        Returns:
            Scalar cross-entropy loss.
        """
        B, T = text_tokens.shape
        _, brain_tokens = self.encode_brain(brain_data)

        # Prepend BOS so the model learns to predict from BOS context
        # Input: [BOS, t1, t2, ..., t_{T-1}], Target: [t1, t2, ..., t_T]
        bos = torch.full(
            (B, 1), self.bos_token, dtype=torch.long, device=text_tokens.device
        )
        input_tokens = torch.cat([bos, text_tokens[:, :-1]], dim=1)
        target_tokens = text_tokens

        logits = self.decoder(input_tokens, brain_tokens)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_tokens.reshape(-1),
            ignore_index=0,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        brain_data: BrainData,
        max_len: int | None = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.0,
        num_samples: int = 1,
        brain_noise: float = 0.0,
    ) -> ReconstructionResult:
        """Reconstruct text from brain activity via autoregressive decoding.

        Args:
            brain_data: fMRI data to decode.
            max_len: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-k filtering (0 to disable).
            top_p: Nucleus sampling threshold (0.0 to disable). When set,
                only the smallest set of tokens with cumulative probability
                >= ``top_p`` are kept. Promotes semantic diversity.
            num_samples: Number of diverse samples per brain input.
                Each sample decodes independently with different random
                draws **and** a different perturbation of the brain
                conditioning. When ``num_samples > 1``,
                ``metadata["texts"]`` is a list of lists: ``texts[i]``
                contains ``num_samples`` strings for brain input *i*.
            brain_noise: Scale of Gaussian noise injected into brain
                embeddings. Each sample gets an independent perturbation,
                so different samples explore different interpretations
                of the brain signal. 0.0 = no perturbation. Typical
                range: 0.1–1.0.

        Returns:
            ReconstructionResult with generated text as metadata.
        """
        B = brain_data.batch_size
        device = brain_data.voxels.device
        gen_len = max_len or self.max_len

        _, brain_tokens = self.encode_brain(brain_data)

        # Repeat conditioning for multiple samples per input
        if num_samples > 1:
            brain_tokens = brain_tokens.repeat_interleave(num_samples, dim=0)

        BN = B * num_samples

        # Perturb brain embeddings — each sample explores a different
        # interpretation of the brain signal (scaled relative to embedding norm)
        if brain_noise > 0.0:
            t_scale = brain_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            brain_tokens = brain_tokens + brain_noise * t_scale * torch.randn_like(brain_tokens)

        # Start with BOS token
        generated = torch.full((BN, 1), self.bos_token, dtype=torch.long, device=device)

        for _ in range(gen_len - 1):
            logits = self.decoder(generated, brain_tokens)
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = next_logits.topk(min(top_k, next_logits.size(-1)), dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            # Nucleus (top-p) filtering
            if top_p > 0.0:
                sorted_logits, sorted_idx = next_logits.sort(dim=-1, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens with cumulative probability above top_p
                mask = cum_probs - sorted_logits.softmax(dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                # Scatter back
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences produced a null byte
            if (next_token == 0).all():
                break

        # Decode to text (skip BOS token at position 0)
        if num_samples > 1:
            # Group samples: texts[i] = list of num_samples strings
            texts = []
            for i in range(B):
                group = []
                for s in range(num_samples):
                    group.append(self.tokens_to_text(generated[i * num_samples + s, 1:]))
                texts.append(group)
        else:
            texts = [self.tokens_to_text(generated[i, 1:]) for i in range(B)]

        return ReconstructionResult(
            modality=Modality.TEXT,
            output=generated,
            brain_condition=brain_tokens[:B].mean(dim=1),
            metadata={"texts": texts, "num_samples": num_samples, "brain_noise": brain_noise},
        )


def build_brain2text(
    n_voxels: int = 1024,
    max_len: int = 128,
    hidden_dim: int = 256,
    depth: int = 6,
) -> Brain2Text:
    """Build a Brain2Text model with sensible defaults."""
    return Brain2Text(
        n_voxels=n_voxels, max_len=max_len,
        hidden_dim=hidden_dim, depth=depth,
    )
