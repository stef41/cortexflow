"""Rectified Flow Matching for training and sampling.

Implements the rectified flow framework from Lipman et al. (2022) and
the improvements from Esser et al. (2024, Stable Diffusion 3):

- **Linear interpolation** paths: x_t = (1 - t) · x_0 + t · x_1
- **Velocity prediction**: v = x_1 - x_0  (the model learns the vector
  field that transports noise to data)
- **Logit-normal timestep sampling**: biases training toward perceptually
  relevant noise levels (SD3 improvement over uniform sampling)
- **Euler / Midpoint ODE solvers** for inference

The training loss is simply MSE between predicted and target velocity:
    L = E_{t, x_0, x_1} [ || v_θ(x_t, t, c) - (x_1 - x_0) ||² ]

Reference: "Flow Matching Guide and Code" (arXiv:2412.06264)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortexflow._types import FlowConfig


class RectifiedFlowMatcher:
    """Rectified flow matching training and sampling.

    Usage::

        fm = RectifiedFlowMatcher()

        # Training step
        loss = fm.compute_loss(model, x_clean, brain_global, brain_tokens)
        loss.backward()

        # Sampling
        x_gen = fm.sample(model, shape=(B, C, H, W), brain_global=bg, brain_tokens=bt)
    """

    def __init__(self, config: FlowConfig | None = None) -> None:
        self.config = config or FlowConfig()

    # ── Timestep Sampling ────────────────────────────────────────────

    def sample_timesteps(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Sample timesteps t ∈ (0, 1).

        If ``logit_normal`` is enabled (SD3-style), samples from a
        logit-normal distribution which biases toward intermediate noise
        levels where the perceptual signal is strongest.
        """
        cfg = self.config

        if cfg.logit_normal:
            # Logit-normal: sample u ~ N(mean, std²), then t = sigmoid(u)
            u = torch.randn(batch_size, device=device)
            u = u * cfg.logit_normal_std + cfg.logit_normal_mean
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device)

        # Clamp away from exact 0 and 1 for numerical stability
        return t.clamp(cfg.sigma_min, 1.0 - cfg.sigma_min)

    # ── Training ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        model: nn.Module,
        x_1: torch.Tensor,
        brain_global: torch.Tensor,
        brain_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute rectified flow matching loss.

        Args:
            model: DiT model that predicts velocity v(x_t, t, condition).
            x_1: Clean data (target) ``(B, C, H, W)``.
            brain_global: Global brain embedding ``(B, D)``.
            brain_tokens: Brain token sequence ``(B, T, D)`` for cross-attention.

        Returns:
            Scalar MSE loss: E[ ||v_pred - v_target||² ]
        """
        B = x_1.shape[0]
        device = x_1.device

        # Sample timesteps
        t = self.sample_timesteps(B, device)

        # Sample noise
        x_0 = torch.randn_like(x_1)

        # Linear interpolation: x_t = (1-t) * noise + t * data
        t_expand = t.view(B, *([1] * (x_1.ndim - 1)))
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1

        # Target velocity: points from noise to data
        v_target = x_1 - x_0

        # Model prediction
        v_pred = model(x_t, t, brain_global, brain_tokens)

        return F.mse_loss(v_pred, v_target)

    # ── Sampling ─────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        brain_global: torch.Tensor,
        brain_tokens: torch.Tensor | None = None,
        num_steps: int | None = None,
        cfg_scale: float | None = None,
        brain_global_uncond: torch.Tensor | None = None,
        brain_tokens_uncond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples by solving the ODE from noise to data.

        Args:
            model: Trained DiT model.
            shape: Output shape ``(B, C, H, W)``.
            brain_global: Global brain condition.
            brain_tokens: Brain token sequence.
            num_steps: Override number of ODE steps.
            cfg_scale: Classifier-free guidance scale. 1.0 = no guidance.
            brain_global_uncond: Unconditional brain embedding for CFG.
            brain_tokens_uncond: Unconditional brain tokens for CFG.

        Returns:
            Generated latent ``(B, C, H, W)``.
        """
        cfg = self.config
        steps = num_steps or cfg.num_steps
        guidance = cfg_scale if cfg_scale is not None else cfg.cfg_scale
        do_cfg = guidance > 1.0 and brain_global_uncond is not None
        device = brain_global.device

        # Start from pure noise
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t_val = i / steps
            t = torch.full((shape[0],), t_val, device=device)

            if cfg.solver == "midpoint":
                # Midpoint method: evaluate at t + dt/2
                v = self._get_velocity(
                    model, x, t, brain_global, brain_tokens,
                    guidance, do_cfg, brain_global_uncond, brain_tokens_uncond,
                )
                x_mid = x + v * (dt / 2)
                t_mid = torch.full((shape[0],), t_val + dt / 2, device=device)
                v_mid = self._get_velocity(
                    model, x_mid, t_mid, brain_global, brain_tokens,
                    guidance, do_cfg, brain_global_uncond, brain_tokens_uncond,
                )
                x = x + v_mid * dt
            else:
                # Euler method
                v = self._get_velocity(
                    model, x, t, brain_global, brain_tokens,
                    guidance, do_cfg, brain_global_uncond, brain_tokens_uncond,
                )
                x = x + v * dt

        return x

    def _get_velocity(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        brain_global: torch.Tensor,
        brain_tokens: torch.Tensor | None,
        guidance: float,
        do_cfg: bool,
        brain_global_uncond: torch.Tensor | None,
        brain_tokens_uncond: torch.Tensor | None,
    ) -> torch.Tensor:
        """Get velocity with optional classifier-free guidance."""
        if do_cfg and brain_global_uncond is not None:
            # Conditional prediction
            v_cond = model(x, t, brain_global, brain_tokens)
            # Unconditional prediction
            v_uncond = model(x, t, brain_global_uncond, brain_tokens_uncond)
            # CFG interpolation
            return v_uncond + guidance * (v_cond - v_uncond)
        else:
            return model(x, t, brain_global, brain_tokens)


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated as:
        θ_ema = decay · θ_ema + (1 - decay) · θ_model
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Replace model params with EMA params. Returns original params."""
        originals: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                originals[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return originals

    def restore(self, model: nn.Module, originals: dict[str, torch.Tensor]) -> None:
        """Restore original params after EMA evaluation."""
        for name, param in model.named_parameters():
            if name in originals:
                param.data.copy_(originals[name])
