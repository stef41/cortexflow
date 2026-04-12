"""Tests for the Brain → Text pipeline."""

import torch
import pytest

from cortexflow._types import BrainData
from cortexflow.brain2text import Brain2Text, BrainTextDecoder, TextDecoderBlock, build_brain2text
from conftest import BATCH, HIDDEN_DIM, N_VOXELS, NUM_HEADS, MAX_TEXT_LEN


class TestTextDecoderBlock:
    def test_forward_shape(self):
        block = TextDecoderBlock(HIDDEN_DIM, NUM_HEADS, cond_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, 8, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = block(x, bt)
        assert out.shape == x.shape

    def test_with_causal_mask(self):
        block = TextDecoderBlock(HIDDEN_DIM, NUM_HEADS, cond_dim=HIDDEN_DIM)
        seq_len = 8
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        x = torch.randn(BATCH, seq_len, HIDDEN_DIM)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = block(x, bt, attn_mask=mask)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        block = TextDecoderBlock(HIDDEN_DIM, NUM_HEADS, cond_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, 8, HIDDEN_DIM, requires_grad=True)
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        out = block(x, bt)
        out.sum().backward()
        assert x.grad is not None


class TestBrainTextDecoder:
    @pytest.fixture
    def decoder(self):
        return BrainTextDecoder(
            vocab_size=256, max_len=MAX_TEXT_LEN, hidden_dim=HIDDEN_DIM,
            depth=2, num_heads=NUM_HEADS, cond_dim=HIDDEN_DIM,
        )

    def test_forward_shape(self, decoder):
        tokens = torch.randint(0, 256, (BATCH, MAX_TEXT_LEN - 1))
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        logits = decoder(tokens, bt)
        assert logits.shape == (BATCH, MAX_TEXT_LEN - 1, 256)

    def test_causal_mask(self, decoder):
        mask = decoder._causal_mask(8, torch.device("cpu"))
        assert mask.shape == (8, 8)
        assert mask[0, 1] == float("-inf")  # future is masked
        assert mask[1, 0] == 0  # past is visible

    def test_weight_tying(self, decoder):
        """lm_head and token_embed should share weights."""
        assert decoder.lm_head.weight is decoder.token_embed.weight

    def test_gradient_flow(self, decoder):
        tokens = torch.randint(0, 256, (BATCH, 8))
        bt = torch.randn(BATCH, 4, HIDDEN_DIM)
        logits = decoder(tokens, bt)
        logits.sum().backward()
        for p in decoder.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()
                break


class TestBrain2Text:
    @pytest.fixture
    def model(self):
        return build_brain2text(
            n_voxels=N_VOXELS, max_len=MAX_TEXT_LEN,
            hidden_dim=HIDDEN_DIM, depth=1,
        )

    @pytest.fixture
    def brain_data(self):
        return BrainData(voxels=torch.randn(BATCH, N_VOXELS))

    def test_text_to_tokens(self):
        tokens = Brain2Text.text_to_tokens("hello")
        assert tokens.tolist() == [104, 101, 108, 108, 111]

    def test_tokens_to_text(self):
        tokens = torch.tensor([104, 101, 108, 108, 111])
        assert Brain2Text.tokens_to_text(tokens) == "hello"

    def test_tokens_roundtrip(self):
        text = "Brain decoding is amazing!"
        tokens = Brain2Text.text_to_tokens(text)
        recovered = Brain2Text.tokens_to_text(tokens)
        assert recovered == text

    def test_tokens_to_text_stops_at_null(self):
        tokens = torch.tensor([72, 105, 0, 99, 99])
        assert Brain2Text.tokens_to_text(tokens) == "Hi"

    def test_training_loss(self, model, brain_data):
        # Create token sequences with BOS
        tokens = torch.randint(32, 127, (BATCH, MAX_TEXT_LEN))
        loss = model.training_loss(tokens, brain_data)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_training_loss_backward(self, model, brain_data):
        tokens = torch.randint(32, 127, (BATCH, MAX_TEXT_LEN))
        loss = model.training_loss(tokens, brain_data)
        loss.backward()
        for p in model.decoder.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_reconstruct(self, model, brain_data):
        model.eval()
        result = model.reconstruct(brain_data, max_len=8, temperature=1.0)
        assert result.modality.value == "text"
        assert result.output.ndim == 2
        assert result.output.shape[0] == BATCH
        assert "texts" in result.metadata
        assert len(result.metadata["texts"]) == BATCH

    def test_reconstruct_produces_text(self, model, brain_data):
        model.eval()
        result = model.reconstruct(brain_data, max_len=8)
        for text in result.metadata["texts"]:
            assert isinstance(text, str)

    def test_reconstruct_top_k(self, model, brain_data):
        model.eval()
        result = model.reconstruct(brain_data, max_len=8, top_k=10)
        assert result.output.shape[0] == BATCH

    def test_reconstruct_single_sample(self, model):
        bd = BrainData(voxels=torch.randn(1, N_VOXELS))
        model.eval()
        result = model.reconstruct(bd, max_len=8)
        assert len(result.metadata["texts"]) == 1


    def test_reconstruct_num_samples(self, model, brain_data):
        model.eval()
        result = model.reconstruct(brain_data, max_len=8, num_samples=3)
        texts = result.metadata["texts"]
        assert len(texts) == BATCH
        for group in texts:
            assert isinstance(group, list)
            assert len(group) == 3
            for t in group:
                assert isinstance(t, str)

    def test_diverse_text_samples_differ(self, model, brain_data):
        """Multiple text samples from same brain input should differ."""
        model.eval()
        # High temperature for max diversity
        result = model.reconstruct(
            brain_data, max_len=8, temperature=1.5, num_samples=4,
        )
        texts = result.metadata["texts"]
        # At least one brain input should produce non-identical samples
        any_differ = False
        for group in texts:
            if len(set(group)) > 1:
                any_differ = True
                break
        assert any_differ, "At high temperature, diverse samples should differ"

    def test_reconstruct_top_p(self, model, brain_data):
        model.eval()
        result = model.reconstruct(brain_data, max_len=8, top_p=0.9, top_k=0)
        assert result.output.shape[0] == BATCH
        assert len(result.metadata["texts"]) == BATCH

    def test_reconstruct_top_p_and_top_k(self, model, brain_data):
        model.eval()
        result = model.reconstruct(brain_data, max_len=8, top_p=0.9, top_k=20)
        assert result.output.shape[0] == BATCH

    def test_reconstruct_num_samples_1(self, model, brain_data):
        """num_samples=1 should return flat list of strings."""
        model.eval()
        result = model.reconstruct(brain_data, max_len=8, num_samples=1)
        texts = result.metadata["texts"]
        assert len(texts) == BATCH
        for t in texts:
            assert isinstance(t, str)


class TestBuildBrain2Text:
    def test_default_build(self):
        model = build_brain2text(n_voxels=64, max_len=16, hidden_dim=16, depth=1)
        assert isinstance(model, Brain2Text)

    def test_param_count(self):
        model = build_brain2text(n_voxels=64, max_len=16, hidden_dim=16, depth=1)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1000
