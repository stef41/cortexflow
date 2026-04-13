"""CortexFlow — Brain decoding with Diffusion Transformers & Flow Matching.

Reconstruct what someone saw, heard, or thought from fMRI brain activity.

Architecture: fMRI → BrainEncoder → DiT (flow matching) → stimulus

Modules:
    dit             Diffusion Transformer backbone (AdaLN-Zero, QK-Norm, SwiGLU)
    flow_matching   Rectified flow training & ODE sampling
    vae             Latent space compression (2D images, 1D audio)
    brain_encoder   fMRI → conditioning embeddings (global + tokens)
    brain2img       Full brain → image pipeline
    brain2audio     Full brain → audio pipeline
    brain2text      Full brain → text pipeline (autoregressive)
    training        Training loops, schedulers, synthetic data
"""

from cortexflow._types import (
    BrainData,
    CortexFlowError,
    DiTConfig,
    FlowConfig,
    Modality,
    ReconstructionResult,
    TrainingConfig,
    VAEConfig,
)
from cortexflow.brain2audio import AudioDiT, Brain2Audio, build_brain2audio
from cortexflow.brain2img import Brain2Image, build_brain2img
from cortexflow.brain2text import Brain2Text, BrainTextDecoder, build_brain2text
from cortexflow.brain_encoder import (
    BrainEncoder,
    ROIBrainEncoder,
    SubjectAdapter,
    make_synthetic_brain_data,
)
from cortexflow.dit import DiffusionTransformer
from cortexflow.flow_matching import EMAModel, RectifiedFlowMatcher
from cortexflow.training import SyntheticBrainDataset, Trainer, WarmupCosineScheduler
from cortexflow.vae import AudioVAE, LatentVAE

__version__ = "0.5.0"

__all__ = [
    # Types
    "BrainData",
    "ReconstructionResult",
    "DiTConfig",
    "FlowConfig",
    "VAEConfig",
    "TrainingConfig",
    "Modality",
    "CortexFlowError",
    # Core
    "DiffusionTransformer",
    "RectifiedFlowMatcher",
    "EMAModel",
    "LatentVAE",
    "AudioVAE",
    "BrainEncoder",
    "ROIBrainEncoder",
    "SubjectAdapter",
    # Pipelines
    "Brain2Image",
    "Brain2Audio",
    "Brain2Text",
    "AudioDiT",
    "BrainTextDecoder",
    # Factories
    "build_brain2img",
    "build_brain2audio",
    "build_brain2text",
    "make_synthetic_brain_data",
    # Training
    "Trainer",
    "WarmupCosineScheduler",
    "SyntheticBrainDataset",
]
