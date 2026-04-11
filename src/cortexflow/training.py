"""Training utilities for cortexflow models.

Provides a generic training loop, learning rate schedulers, and data
utilities for training brain decoders on fMRI datasets.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn

from cortexflow._types import TrainingConfig
from cortexflow.flow_matching import EMAModel


class WarmupCosineScheduler:
    """Learning rate schedule: linear warmup → cosine decay.

    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Linear warmup duration.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as a fraction of peak.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self) -> None:
        self._step += 1
        lr_scale = self._get_scale(self._step)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * lr_scale

    def _get_scale(self, step: int) -> float:
        if step <= self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class SyntheticBrainDataset:
    """Synthetic dataset for testing and development.

    Yields (stimulus, fmri) pairs with configurable modality.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_voxels: int = 1024,
        modality: str = "image",
        img_size: int = 64,
        n_mels: int = 80,
        audio_len: int = 128,
        max_text_len: int = 64,
    ) -> None:
        self.n_samples = n_samples
        self.n_voxels = n_voxels
        self.modality = modality
        self.img_size = img_size
        self.n_mels = n_mels
        self.audio_len = audio_len
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate a single synthetic sample."""
        g = torch.Generator().manual_seed(idx)
        fmri = torch.randn(self.n_voxels, generator=g)

        if self.modality == "image":
            stimulus = torch.rand(3, self.img_size, self.img_size, generator=g)
        elif self.modality == "audio":
            stimulus = torch.randn(self.n_mels, self.audio_len, generator=g).abs()
        elif self.modality == "text":
            # Random byte-level tokens (printable ASCII range)
            tokens = torch.randint(32, 127, (self.max_text_len,), generator=g)
            stimulus = tokens
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        return {"fmri": fmri, "stimulus": stimulus}

    def to_loader(
        self, batch_size: int = 8, shuffle: bool = True
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Simple batch iterator (no DataLoader dependency)."""
        indices = list(range(self.n_samples))
        if shuffle:
            g = torch.Generator()
            perm = torch.randperm(self.n_samples, generator=g).tolist()
            indices = perm

        for start in range(0, self.n_samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            items = [self[i] for i in batch_indices]
            batch: dict[str, torch.Tensor] = {}
            for key in items[0]:
                batch[key] = torch.stack([item[key] for item in items])
            yield batch


class Trainer:
    """Generic training loop for cortexflow models.

    Handles optimizer setup, LR scheduling, EMA, gradient clipping,
    and logging.

    Args:
        model: A cortexflow model with a ``training_loss`` method.
        config: Training configuration.
        device: Device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config or TrainingConfig()
        self.device = torch.device(device)

        cfg = self.config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )

        self.ema: EMAModel | None = None
        if cfg.ema_decay > 0:
            self.ema = EMAModel(model, decay=cfg.ema_decay)

        self.global_step = 0
        self.scheduler: WarmupCosineScheduler | None = None
        self._log_fn: Callable[[dict[str, Any]], None] | None = None

    def set_logger(self, fn: Callable[[dict[str, Any]], None]) -> None:
        """Set a logging callback: fn({"step": ..., "loss": ..., ...})."""
        self._log_fn = fn

    def _log(self, metrics: dict[str, Any]) -> None:
        if self._log_fn is not None:
            self._log_fn(metrics)

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        loss_fn: Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor],
    ) -> float:
        """Execute a single training step.

        Args:
            batch: Dict of tensors (moved to device automatically).
            loss_fn: Computes loss from (model, batch) → scalar.

        Returns:
            Loss value as float.
        """
        cfg = self.config
        self.model.train()

        # Move data to device
        batch_dev = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        loss = loss_fn(self.model, batch_dev)
        loss.backward()

        # Gradient clipping
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # EMA update
        if self.ema is not None:
            self.ema.update(self.model)

        # LR schedule
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        # Logging
        if self.global_step % cfg.log_every == 0:
            lr = (
                self.scheduler.current_lr
                if self.scheduler
                else cfg.learning_rate
            )
            self._log({
                "step": self.global_step,
                "loss": loss.item(),
                "lr": lr,
            })

        return loss.item()

    def fit(
        self,
        dataset: SyntheticBrainDataset,
        loss_fn: Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor],
        n_epochs: int | None = None,
    ) -> list[float]:
        """Full training loop over a dataset.

        Args:
            dataset: Training data.
            loss_fn: Loss function (model, batch) → scalar.
            n_epochs: Override number of epochs.

        Returns:
            List of per-step losses.
        """
        cfg = self.config
        epochs = n_epochs or cfg.n_epochs
        steps_per_epoch = max(1, len(dataset) // cfg.batch_size)
        total_steps = epochs * steps_per_epoch

        self.scheduler = WarmupCosineScheduler(
            self.optimizer, cfg.warmup_steps, total_steps
        )

        losses: list[float] = []
        for epoch in range(epochs):
            for batch in dataset.to_loader(batch_size=cfg.batch_size, shuffle=True):
                loss = self.train_step(batch, loss_fn)
                losses.append(loss)

        return losses

    def save_checkpoint(self, path: str) -> None:
        """Save model + optimizer + EMA state."""
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if self.ema is not None:
            state["ema"] = self.ema.shadow
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model + optimizer + EMA state."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        if self.ema is not None and "ema" in state:
            self.ema.shadow = state["ema"]
