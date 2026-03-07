"""Tests for SharedEmbeddingEncoder (dimension-aware projectors)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from encoders import SharedEmbeddingEncoder


class TestSharedEmbeddingEncoderInit:

    def test_creates_projectors_for_given_dims(self):
        enc = SharedEmbeddingEncoder(64, emb_dims={300, 768})
        assert "300" in enc.projectors
        assert "768" in enc.projectors
        assert len(enc.projectors) == 2

    def test_projector_shapes(self):
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        assert enc.projectors["300"].in_features == 300
        assert enc.projectors["300"].out_features == 64

    def test_empty_emb_dims(self):
        enc = SharedEmbeddingEncoder(64, emb_dims=set())
        assert len(enc.projectors) == 0

    def test_none_emb_dims(self):
        enc = SharedEmbeddingEncoder(64, emb_dims=None)
        assert len(enc.projectors) == 0

    def test_single_dim(self):
        enc = SharedEmbeddingEncoder(128, emb_dims={300})
        assert "300" in enc.projectors
        assert len(enc.projectors) == 1


class TestSharedEmbeddingEncoderForward3D:
    """Tests for already-shaped 3D input [B, num_cols, emb_dim]."""

    def test_3d_input_300(self):
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        x = torch.randn(8, 3, 300)
        out = enc(x)
        assert out.shape == (8, 3, 64)

    def test_3d_input_768(self):
        enc = SharedEmbeddingEncoder(64, emb_dims={768})
        x = torch.randn(4, 2, 768)
        out = enc(x)
        assert out.shape == (4, 2, 64)

    def test_3d_dispatches_to_correct_projector(self):
        enc = SharedEmbeddingEncoder(64, emb_dims={300, 768})
        x300 = torch.randn(4, 1, 300)
        x768 = torch.randn(4, 1, 768)
        # Should not raise — each dispatches to the right projector
        out300 = enc(x300)
        out768 = enc(x768)
        assert out300.shape == (4, 1, 64)
        assert out768.shape == (4, 1, 64)

    def test_3d_unknown_dim_raises_with_message(self):
        """3D path should give a descriptive error, not a bare KeyError."""
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        x = torch.randn(4, 1, 512)
        with pytest.raises(KeyError, match="No projector for embedding dim 512"):
            enc(x)


class TestSharedEmbeddingEncoderForward2D:
    """Tests for flattened 2D input [B, num_cols * emb_dim] — the bug fix."""

    def test_2d_single_column(self):
        """[B, 300] -> reshape to [B, 1, 300] -> project to [B, 1, 64]."""
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        x = torch.randn(8, 300)
        out = enc(x)
        assert out.shape == (8, 1, 64)

    def test_2d_multi_column(self):
        """[B, 1200] with emb_dim=300 -> reshape to [B, 4, 300] -> [B, 4, 64]."""
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        x = torch.randn(8, 1200)
        out = enc(x)
        assert out.shape == (8, 4, 64)

    def test_2d_multi_column_768(self):
        """[B, 2304] with emb_dim=768 -> reshape to [B, 3, 768] -> [B, 3, 64]."""
        enc = SharedEmbeddingEncoder(64, emb_dims={768})
        x = torch.randn(4, 2304)
        out = enc(x)
        assert out.shape == (4, 3, 64)

    def test_2d_unknown_dim_raises(self):
        """No projector dim divides 1200 evenly when only 768 is known."""
        enc = SharedEmbeddingEncoder(64, emb_dims={768})
        x = torch.randn(4, 1200)
        with pytest.raises(KeyError, match="not divisible"):
            enc(x)

    def test_2d_unambiguous_match_with_multiple_projectors(self):
        """With {300, 768}, input [B, 900] only divisible by 300 -> unambiguous."""
        enc = SharedEmbeddingEncoder(64, emb_dims={300, 768})
        x = torch.randn(4, 900)
        out = enc(x)
        assert out.shape == (4, 3, 64)

    def test_2d_ambiguous_match_raises(self):
        """With {300, 600}, input [B, 1200] is divisible by both -> error."""
        enc = SharedEmbeddingEncoder(64, emb_dims={300, 600})
        x = torch.randn(4, 1200)
        with pytest.raises(KeyError, match="Ambiguous"):
            enc(x)

    def test_2d_no_projectors_raises(self):
        enc = SharedEmbeddingEncoder(64, emb_dims=set())
        x = torch.randn(4, 300)
        with pytest.raises(KeyError):
            enc(x)


class TestSharedEmbeddingEncoderGradients:

    def test_gradients_flow(self):
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        x = torch.randn(4, 3, 300)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert enc.projectors["300"].weight.grad is not None

    def test_2d_gradients_flow(self):
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        x = torch.randn(4, 1200)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert enc.projectors["300"].weight.grad is not None
