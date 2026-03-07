"""Tests for NeighborNodeTypeEncoder (precomputed GloVe embeddings)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from unittest.mock import patch
from encoders import NeighborNodeTypeEncoder


class _FakeGlove:
    """Fake GloveTextEmbedding that returns deterministic 300-d vectors."""
    def __init__(self, device="cpu"):
        pass

    def __call__(self, names):
        out = torch.zeros(len(names), 300)
        for i, name in enumerate(names):
            torch.manual_seed(hash(name) % 2**32)
            out[i] = torch.randn(300)
        return out


def _make_encoder(node_type_map, embedding_dim=64):
    """Create encoder with mocked GloVe to avoid downloading the model."""
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        encoder = NeighborNodeTypeEncoder(
            node_type_map=node_type_map,
            embedding_dim=embedding_dim,
        )
    return encoder


class TestNeighborNodeTypeEncoderInit:

    def test_buffer_shape(self):
        """Precomputed embeddings have shape [num_types+1, 300]."""
        enc = _make_encoder({"drivers": 0, "races": 1})
        assert enc.glove_embeddings.shape == (3, 300)  # 2 types + mask

    def test_buffer_shape_single_type(self):
        enc = _make_encoder({"drivers": 0})
        assert enc.glove_embeddings.shape == (2, 300)  # 1 type + mask

    def test_buffer_is_not_parameter(self):
        """GloVe embeddings are buffers, not trainable parameters."""
        enc = _make_encoder({"drivers": 0, "races": 1})
        buffer_names = {n for n, _ in enc.named_buffers()}
        param_names = {n for n, _ in enc.named_parameters()}
        assert "glove_embeddings" in buffer_names
        assert "glove_embeddings" not in param_names

    def test_no_sentence_transformer_submodule(self):
        """The sentence-transformers model should NOT be stored."""
        enc = _make_encoder({"drivers": 0})
        assert not hasattr(enc, "embedder")
        assert not hasattr(enc, "st_model")

    def test_proj_layer_exists(self):
        enc = _make_encoder({"drivers": 0}, embedding_dim=128)
        assert enc.proj.in_features == 300
        assert enc.proj.out_features == 128


class TestNeighborNodeTypeEncoderForward:

    def test_output_shape(self):
        enc = _make_encoder({"drivers": 0, "races": 1, "results": 2}, embedding_dim=64)
        # indices in {0,1,2}, mask_idx=3
        x = torch.randint(0, 3, (4, 100))
        out = enc(x)
        assert out.shape == (4, 100, 64)

    def test_mask_index_produces_nonzero(self):
        enc = _make_encoder({"drivers": 0, "races": 1})
        mask_idx = 2  # num_types=2, mask is at index 2
        x = torch.full((2, 10), mask_idx, dtype=torch.long)
        out = enc(x)
        # After projection, mask token should not be all zeros
        assert out.abs().sum() > 0

    def test_different_names_produce_different_embeddings(self):
        enc = _make_encoder({"drivers": 0, "races": 1})
        # embeddings for index 0, 1, 2 should all be different
        assert not torch.allclose(enc.glove_embeddings[0], enc.glove_embeddings[1])
        assert not torch.allclose(enc.glove_embeddings[0], enc.glove_embeddings[2])
        assert not torch.allclose(enc.glove_embeddings[1], enc.glove_embeddings[2])

    def test_gradients_flow_through_proj_only(self):
        enc = _make_encoder({"drivers": 0}, embedding_dim=64)
        x = torch.randint(0, 1, (2, 5))
        out = enc(x)
        loss = out.sum()
        loss.backward()

        assert enc.proj.weight.grad is not None
        assert enc.glove_embeddings.grad is None  # buffer, no grad

    def test_deterministic_output(self):
        """Same input should produce same output (no randomness in forward)."""
        enc = _make_encoder({"a": 0, "b": 1}, embedding_dim=32)
        x = torch.tensor([[0, 1, 2], [1, 0, 2]])
        out1 = enc(x)
        out2 = enc(x)
        assert torch.allclose(out1, out2)

    def test_batch_size_one(self):
        enc = _make_encoder({"drivers": 0}, embedding_dim=64)
        x = torch.tensor([[0, 1, 0]])
        out = enc(x)
        assert out.shape == (1, 3, 64)

    def test_reset_parameters(self):
        """reset_parameters should not crash."""
        enc = _make_encoder({"drivers": 0}, embedding_dim=64)
        enc.reset_parameters()
        x = torch.tensor([[0, 1]])
        out = enc(x)
        assert out.shape == (1, 2, 64)


class TestNeighborNodeTypeEncoderValidation:

    def test_non_contiguous_indices_raises(self):
        """Gap in indices (e.g., {0, 2} with no 1) should raise ValueError."""
        with pytest.raises(ValueError, match="contiguous"):
            _make_encoder({"a": 0, "b": 2})

    def test_non_zero_based_indices_raises(self):
        """Indices not starting from 0 should raise ValueError."""
        with pytest.raises(ValueError, match="contiguous"):
            _make_encoder({"a": 1, "b": 2})

    def test_contiguous_indices_ok(self):
        """Contiguous 0-based indices should not raise."""
        enc = _make_encoder({"a": 0, "b": 1, "c": 2})
        assert enc.glove_embeddings.shape == (4, 300)  # 3 types + mask
