"""Tests for the public API (__init__.py imports)."""

import pytest


class TestImports:
    def test_import_cortexflow(self):
        import cortexflow
        assert hasattr(cortexflow, "__version__")
        assert cortexflow.__version__ == "0.4.0"

    def test_import_types(self):
        from cortexflow import BrainData, Modality, ReconstructionResult
        from cortexflow import DiTConfig, FlowConfig, VAEConfig, TrainingConfig
        from cortexflow import CortexFlowError

    def test_import_core(self):
        from cortexflow import DiffusionTransformer
        from cortexflow import RectifiedFlowMatcher, EMAModel
        from cortexflow import LatentVAE, AudioVAE
        from cortexflow import BrainEncoder, ROIBrainEncoder, SubjectAdapter

    def test_import_pipelines(self):
        from cortexflow import Brain2Image, Brain2Audio, Brain2Text
        from cortexflow import AudioDiT, BrainTextDecoder

    def test_import_factories(self):
        from cortexflow import build_brain2img, build_brain2audio, build_brain2text
        from cortexflow import make_synthetic_brain_data

    def test_import_training(self):
        from cortexflow import Trainer, WarmupCosineScheduler, SyntheticBrainDataset

    def test_all_exports(self):
        import cortexflow
        for name in cortexflow.__all__:
            assert hasattr(cortexflow, name), f"Missing export: {name}"
