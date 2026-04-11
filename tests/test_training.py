"""Tests for training utilities."""

import torch
import pytest

from cortexflow._types import BrainData, TrainingConfig
from cortexflow.brain2img import build_brain2img
from cortexflow.training import SyntheticBrainDataset, Trainer, WarmupCosineScheduler
from conftest import BATCH, HIDDEN_DIM, N_VOXELS, NUM_HEADS


class TestWarmupCosineScheduler:
    def test_warmup_phase(self):
        opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(sched.current_lr)
        # LR should increase during warmup
        assert lrs[-1] > lrs[0]

    def test_cosine_decay(self):
        opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=0, total_steps=100)
        lrs = []
        for _ in range(100):
            sched.step()
            lrs.append(sched.current_lr)
        # LR should decrease after warmup
        assert lrs[-1] < lrs[0]

    def test_never_negative(self):
        opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=20)
        for _ in range(30):
            sched.step()
            assert sched.current_lr >= 0

    def test_min_lr_ratio(self):
        opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=0, total_steps=10, min_lr_ratio=0.1)
        for _ in range(20):
            sched.step()
        assert sched.current_lr >= 1e-3 * 0.1 - 1e-8


class TestSyntheticBrainDataset:
    def test_image_modality(self):
        ds = SyntheticBrainDataset(n_samples=10, n_voxels=32, modality="image", img_size=8)
        assert len(ds) == 10
        sample = ds[0]
        assert sample["fmri"].shape == (32,)
        assert sample["stimulus"].shape == (3, 8, 8)

    def test_audio_modality(self):
        ds = SyntheticBrainDataset(n_samples=10, n_voxels=32, modality="audio", n_mels=16, audio_len=32)
        sample = ds[0]
        assert sample["stimulus"].shape == (16, 32)

    def test_text_modality(self):
        ds = SyntheticBrainDataset(n_samples=10, n_voxels=32, modality="text", max_text_len=16)
        sample = ds[0]
        assert sample["stimulus"].shape == (16,)
        assert sample["stimulus"].dtype == torch.long

    def test_deterministic(self):
        ds = SyntheticBrainDataset(n_samples=5, n_voxels=16, modality="image", img_size=4)
        s1 = ds[0]
        s2 = ds[0]
        torch.testing.assert_close(s1["fmri"], s2["fmri"])

    def test_different_indices(self):
        ds = SyntheticBrainDataset(n_samples=5, n_voxels=16, modality="image", img_size=4)
        s0 = ds[0]
        s1 = ds[1]
        assert not torch.equal(s0["fmri"], s1["fmri"])

    def test_to_loader(self):
        ds = SyntheticBrainDataset(n_samples=10, n_voxels=16, modality="image", img_size=4)
        batches = list(ds.to_loader(batch_size=4, shuffle=False))
        assert len(batches) == 3  # 10/4 = 3 batches (4, 4, 2)
        assert batches[0]["fmri"].shape[0] == 4

    def test_invalid_modality(self):
        ds = SyntheticBrainDataset(modality="video")
        with pytest.raises(ValueError):
            ds[0]


class TestTrainer:
    @pytest.fixture
    def model(self):
        return build_brain2img(
            n_voxels=N_VOXELS, img_size=8, hidden_dim=16, depth=1, num_heads=4,
        )

    def test_init(self, model):
        trainer = Trainer(model, TrainingConfig(learning_rate=1e-4))
        assert trainer.global_step == 0
        assert trainer.ema is not None

    def test_train_step(self, model):
        trainer = Trainer(model, TrainingConfig())

        def loss_fn(m, batch):
            bd = BrainData(voxels=batch["fmri"])
            return m.training_loss(batch["stimulus"], bd)

        batch = {
            "fmri": torch.randn(2, N_VOXELS),
            "stimulus": torch.rand(2, 3, 8, 8),
        }
        loss = trainer.train_step(batch, loss_fn)
        assert loss > 0
        assert trainer.global_step == 1

    def test_multiple_steps(self, model):
        trainer = Trainer(model, TrainingConfig(log_every=2))
        logs = []
        trainer.set_logger(lambda m: logs.append(m))

        def loss_fn(m, batch):
            bd = BrainData(voxels=batch["fmri"])
            return m.training_loss(batch["stimulus"], bd)

        for _ in range(4):
            batch = {
                "fmri": torch.randn(2, N_VOXELS),
                "stimulus": torch.rand(2, 3, 8, 8),
            }
            trainer.train_step(batch, loss_fn)

        assert trainer.global_step == 4
        assert len(logs) == 2  # logged at step 2 and 4

    def test_ema_disabled(self, model):
        cfg = TrainingConfig(ema_decay=0)
        trainer = Trainer(model, cfg)
        assert trainer.ema is None

    def test_save_load_checkpoint(self, model, tmp_path):
        trainer = Trainer(model, TrainingConfig())
        trainer.global_step = 42
        path = str(tmp_path / "ckpt.pt")
        trainer.save_checkpoint(path)

        trainer2 = Trainer(model, TrainingConfig())
        trainer2.load_checkpoint(path)
        assert trainer2.global_step == 42

    def test_fit(self, model):
        trainer = Trainer(model, TrainingConfig(batch_size=2))
        ds = SyntheticBrainDataset(n_samples=4, n_voxels=N_VOXELS, modality="image", img_size=8)

        def loss_fn(m, batch):
            bd = BrainData(voxels=batch["fmri"])
            return m.training_loss(batch["stimulus"], bd)

        losses = trainer.fit(ds, loss_fn, n_epochs=1)
        assert len(losses) > 0
        assert all(l > 0 for l in losses)
