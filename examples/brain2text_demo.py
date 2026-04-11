"""Example: Reconstruct text from synthetic fMRI data.

Demonstrates the Brain2Text pipeline with autoregressive
byte-level text generation.
"""

import torch

from cortexflow import BrainData, Brain2Text, build_brain2text, make_synthetic_brain_data

# Build model
model = build_brain2text(
    n_voxels=1024,
    max_len=128,
    hidden_dim=256,
    depth=6,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate synthetic fMRI
fmri = make_synthetic_brain_data(batch_size=2, n_voxels=1024)
brain = BrainData(voxels=fmri)

# Training step — encode text to byte tokens
texts = ["The cat sat on the mat.", "A dog ran through the park."]
max_len = 64
token_batch = []
for text in texts:
    tokens = Brain2Text.text_to_tokens(text)
    # Pad to max_len
    padded = torch.zeros(max_len, dtype=torch.long)
    padded[:len(tokens)] = tokens
    token_batch.append(padded)
token_batch = torch.stack(token_batch)

model.train()
loss = model.training_loss(token_batch, brain)
print(f"Training loss: {loss.item():.4f}")

# Reconstruct text
model.eval()
result = model.reconstruct(brain, max_len=64, temperature=0.8, top_k=50)
print("\nGenerated texts:")
for i, text in enumerate(result.metadata["texts"]):
    print(f"  Sample {i}: {repr(text)}")
