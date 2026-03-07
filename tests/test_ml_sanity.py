"""ML sanity checks for table-agnostic encoders.

These tests verify that the encoders behave correctly as *learning* components,
not just that they don't crash. They catch issues like:
- Representation collapse (everything maps to the same vector)
- Dead parameters (params that never receive gradients)
- Vanishing/exploding gradients
- Z-score normalization not actually normalizing
- Inability to overfit a trivial batch
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

from encoders import (
    NeighborNodeTypeEncoder,
    SharedNumericalEncoder,
    SharedCategoricalEncoder,
    SharedEmbeddingEncoder,
    SharedTimestampEncoder,
    NeighborTfsEncoder,
)
import torch_frame
from torch_frame.data.stats import StatType


# ── Helpers ──────────────────────────────────────────────────

class _FakeGlove:
    def __init__(self, device="cpu"):
        pass

    def __call__(self, names):
        out = torch.zeros(len(names), 300)
        for i, name in enumerate(names):
            torch.manual_seed(hash(name) % 2**32)
            out[i] = torch.randn(300)
        return out


def _make_type_encoder(node_type_map, embedding_dim=64):
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        return NeighborNodeTypeEncoder(node_type_map, embedding_dim)


def _make_tfs_encoder(node_type_map, col_names_dict, col_stats_dict, channels=32):
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        return NeighborTfsEncoder(
            channels=channels,
            node_type_map=node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
        )


# ═══════════════════════════════════════════════════════════
# 1. REPRESENTATION QUALITY
# ═══════════════════════════════════════════════════════════

class TestRepresentationQuality:
    """Verify embeddings are distinct and not collapsed."""

    def test_type_embeddings_well_separated(self):
        """GloVe vectors for semantically different names should have cos_sim < 0.95."""
        enc = _make_type_encoder({
            "drivers": 0, "races": 1, "circuits": 2, "customers": 3,
        })
        embs = enc.glove_embeddings[:4]  # exclude mask
        embs_norm = F.normalize(embs, dim=-1)
        cos_sim = embs_norm @ embs_norm.T

        # Off-diagonal similarities should be < 0.95 (not near-identical)
        mask = ~torch.eye(4, dtype=torch.bool)
        assert cos_sim[mask].max() < 0.95, (
            f"Max off-diagonal cosine similarity too high: {cos_sim[mask].max():.3f}"
        )

    def test_type_embeddings_not_collapsed(self):
        """All type embeddings should have non-trivial variance (not constant)."""
        enc = _make_type_encoder({"a": 0, "b": 1, "c": 2})
        embs = enc.glove_embeddings[:3]
        # Variance across types should be > 0 in most dimensions
        var_per_dim = embs.var(dim=0)
        nonzero_dims = (var_per_dim > 1e-6).sum()
        assert nonzero_dims > 100, (
            f"Only {nonzero_dims}/300 dimensions have variance — embeddings may be collapsed"
        )

    def test_numerical_encoder_different_inputs_different_outputs(self):
        """Encoder should not map all values to the same vector."""
        enc = SharedNumericalEncoder(64)
        x1 = torch.tensor([[0.0, 0.0]])
        x2 = torch.tensor([[100.0, 100.0]])
        out1 = enc(x1)
        out2 = enc(x2)
        assert not torch.allclose(out1, out2, atol=1e-4), (
            "Numerical encoder maps different inputs to same output — representation collapse"
        )

    def test_numerical_encoder_output_variance(self):
        """Outputs across a range of inputs should have meaningful spread."""
        enc = SharedNumericalEncoder(64)
        x = torch.linspace(-10, 10, 50).unsqueeze(1)  # [50, 1]
        out = enc(x)  # [50, 1, 64]
        # Variance across samples in each output dim
        var_per_dim = out.squeeze(1).var(dim=0)
        # At least half the output dims should have variance > 1e-4
        active_dims = (var_per_dim > 1e-4).sum()
        assert active_dims > 32, (
            f"Only {active_dims}/64 dims active — encoder may have dead neurons"
        )

    def test_embedding_encoder_preserves_input_differences(self):
        """Different embedding inputs should produce different outputs."""
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        x1 = torch.randn(4, 3, 300)
        x2 = torch.randn(4, 3, 300)
        out1 = enc(x1)
        out2 = enc(x2)
        # Outputs should differ
        diff = (out1 - out2).abs().mean()
        assert diff > 1e-4, "Embedding encoder produces identical output for different inputs"

    def test_categorical_encoder_different_categories_different_embeddings(self):
        """Different category indices should map to different vectors."""
        enc = SharedCategoricalEncoder(64)
        x1 = torch.tensor([[0, 1]])
        x2 = torch.tensor([[100, 200]])
        out1 = enc(x1)
        out2 = enc(x2)
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_type_projection_output_has_reasonable_magnitude(self):
        """Projected type embeddings should not be extremely large or tiny."""
        enc = _make_type_encoder({"drivers": 0, "races": 1}, embedding_dim=64)
        x = torch.tensor([[0, 1, 2]])
        out = enc(x)  # [1, 3, 64]
        rms = out.pow(2).mean().sqrt()
        assert 0.01 < rms < 100, f"Output RMS {rms:.4f} is outside reasonable range"


# ═══════════════════════════════════════════════════════════
# 2. GRADIENT HEALTH
# ═══════════════════════════════════════════════════════════

class TestGradientHealth:
    """Verify gradients are well-behaved across all encoder parameters."""

    def test_type_encoder_no_dead_params(self):
        """Every trainable parameter should receive a non-zero gradient."""
        enc = _make_type_encoder({"a": 0, "b": 1}, embedding_dim=64)
        x = torch.randint(0, 2, (4, 10))
        out = enc(x)
        loss = out.sum()
        loss.backward()

        for name, p in enc.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert p.grad.abs().sum() > 0, f"Parameter {name} has all-zero gradient"

    def test_numerical_encoder_no_dead_params(self):
        enc = SharedNumericalEncoder(64)
        x = torch.randn(8, 3)
        out = enc(x)
        loss = out.sum()
        loss.backward()

        for name, p in enc.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert p.grad.abs().sum() > 0, f"Parameter {name} has all-zero gradient"

    def test_embedding_encoder_no_dead_params(self):
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        x = torch.randn(4, 300)
        out = enc(x)
        loss = out.sum()
        loss.backward()

        for name, p in enc.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert p.grad.abs().sum() > 0, f"Parameter {name} has all-zero gradient"

    def test_no_exploding_gradients(self):
        """Gradient norms should stay bounded for reasonable inputs."""
        enc = SharedNumericalEncoder(64)
        x = torch.randn(16, 5)  # normal-scale input
        out = enc(x)
        loss = out.sum()
        loss.backward()

        for name, p in enc.named_parameters():
            grad_norm = p.grad.norm()
            assert grad_norm < 1e4, (
                f"Gradient norm for {name} is {grad_norm:.1f} — possible explosion"
            )

    def test_no_vanishing_gradients_through_projection(self):
        """Type encoder projection should pass meaningful gradients."""
        enc = _make_type_encoder({"a": 0, "b": 1, "c": 2}, embedding_dim=256)
        x = torch.randint(0, 3, (8, 50))
        out = enc(x)
        loss = out.mean()
        loss.backward()

        grad_norm = enc.proj.weight.grad.norm()
        assert grad_norm > 1e-8, (
            f"Projection gradient norm is {grad_norm:.2e} — vanishing gradient"
        )

    def test_glove_buffer_unchanged_after_backward(self):
        """GloVe embeddings should not change during training (they are buffers)."""
        enc = _make_type_encoder({"a": 0, "b": 1}, embedding_dim=64)
        original = enc.glove_embeddings.clone()

        optimizer = torch.optim.Adam(enc.parameters(), lr=0.01)
        for _ in range(5):
            x = torch.randint(0, 2, (4, 10))
            loss = enc(x).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert torch.allclose(enc.glove_embeddings, original), (
            "GloVe buffer was modified during training!"
        )


# ═══════════════════════════════════════════════════════════
# 3. OVERFIT TESTS
# ═══════════════════════════════════════════════════════════

class TestOverfit:
    """Verify encoders can memorize a small batch — proves learning signal flows."""

    def test_numerical_encoder_overfits_single_batch(self):
        """MLP should drive MSE loss toward 0 on a fixed tiny batch."""
        enc = SharedNumericalEncoder(64)
        target_proj = nn.Linear(64, 1)

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        targets = torch.tensor([[0.5], [1.5], [2.5]])

        params = list(enc.parameters()) + list(target_proj.parameters())
        optimizer = torch.optim.Adam(params, lr=0.01)

        initial_loss = None
        for step in range(100):
            out = enc(x)  # [3, 2, 64]
            pooled = out.mean(dim=1)  # [3, 64]
            pred = target_proj(pooled)  # [3, 1]
            loss = F.mse_loss(pred, targets)
            if step == 0:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.1, (
            f"Failed to overfit: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_type_encoder_projection_learns_to_separate(self):
        """After training, projection should push different types further apart."""
        enc = _make_type_encoder({"cat": 0, "dog": 1}, embedding_dim=32)
        optimizer = torch.optim.Adam(enc.parameters(), lr=0.01)

        # Contrastive-style: make type 0 output → [1,0,...] and type 1 → [0,1,...]
        target_0 = torch.zeros(32)
        target_0[0] = 1.0
        target_1 = torch.zeros(32)
        target_1[1] = 1.0

        for _ in range(200):
            out_0 = enc(torch.tensor([[0]]))  # [1, 1, 32]
            out_1 = enc(torch.tensor([[1]]))
            loss = F.mse_loss(out_0.squeeze(), target_0) + F.mse_loss(out_1.squeeze(), target_1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # After training, outputs for type 0 and type 1 should be distinct
        with torch.no_grad():
            final_0 = enc(torch.tensor([[0]])).squeeze()
            final_1 = enc(torch.tensor([[1]])).squeeze()
            cos_sim = F.cosine_similarity(final_0.unsqueeze(0), final_1.unsqueeze(0))
            assert cos_sim < 0.8, (
                f"Types not separated after training: cos_sim={cos_sim.item():.3f}"
            )

    def test_embedding_encoder_overfits(self):
        """Embedding projector should fit a tiny regression target."""
        enc = SharedEmbeddingEncoder(32, emb_dims={300})
        head = nn.Linear(32, 1)
        params = list(enc.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=0.01)

        x = torch.randn(4, 1, 300)
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

        initial_loss = None
        for step in range(100):
            out = enc(x).squeeze(1)  # [4, 32]
            pred = head(out)
            loss = F.mse_loss(pred, target)
            if step == 0:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss * 0.05, (
            f"Embedding encoder failed to overfit: {initial_loss:.4f} -> {loss.item():.4f}"
        )


# ═══════════════════════════════════════════════════════════
# 4. Z-SCORE NORMALIZATION EFFECTIVENESS
# ═══════════════════════════════════════════════════════════

class TestZScoreEffectiveness:
    """Verify Z-score normalization actually helps from an ML perspective."""

    def test_normalized_features_have_zero_mean_unit_var(self):
        """After Z-score, features should be approximately N(0,1)."""
        enc = _make_tfs_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x", "y"]}},
            col_stats_dict={"t": {
                "x": {StatType.MEAN: 1000.0, StatType.STD: 200.0},
                "y": {StatType.MEAN: 0.5, StatType.STD: 0.1},
            }},
        )

        # Simulate data drawn from the known distribution
        torch.manual_seed(42)
        feat = torch.cat([
            torch.normal(1000.0, 200.0, (100, 1)),
            torch.normal(0.5, 0.1, (100, 1)),
        ], dim=1)

        tf = MagicMock()
        tf.feat_dict = {torch_frame.numerical: feat}
        enc._normalize_numerical(tf, "t")
        normed = tf.feat_dict[torch_frame.numerical]

        # Mean should be near 0
        assert normed.mean(dim=0).abs().max() < 0.3, (
            f"Mean after Z-score: {normed.mean(dim=0).tolist()}"
        )
        # Std should be near 1
        assert (normed.std(dim=0) - 1.0).abs().max() < 0.3, (
            f"Std after Z-score: {normed.std(dim=0).tolist()}"
        )

    def test_zscore_makes_different_scales_comparable(self):
        """Features at wildly different scales should become similar after Z-score."""
        enc = _make_tfs_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["big", "small"]}},
            col_stats_dict={"t": {
                "big": {StatType.MEAN: 1e6, StatType.STD: 1e4},
                "small": {StatType.MEAN: 0.001, StatType.STD: 0.0001},
            }},
        )

        torch.manual_seed(0)
        feat = torch.cat([
            torch.normal(1e6, 1e4, (50, 1)),
            torch.normal(0.001, 0.0001, (50, 1)),
        ], dim=1)

        # Before normalization: scales differ by ~10 orders of magnitude
        range_before = feat.abs().max(dim=0).values
        scale_ratio_before = range_before.max() / range_before.min()

        tf = MagicMock()
        tf.feat_dict = {torch_frame.numerical: feat}
        enc._normalize_numerical(tf, "t")
        normed = tf.feat_dict[torch_frame.numerical]

        range_after = normed.abs().max(dim=0).values
        scale_ratio_after = range_after.max() / range_after.min()

        assert scale_ratio_after < scale_ratio_before * 0.01, (
            f"Z-score didn't reduce scale ratio: {scale_ratio_before:.0f} -> {scale_ratio_after:.2f}"
        )

    def test_zscore_preserves_relative_ordering(self):
        """Z-score is a monotonic transform — ordering within each column preserved."""
        enc = _make_tfs_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {
                "x": {StatType.MEAN: 50.0, StatType.STD: 10.0},
            }},
        )
        feat = torch.tensor([[10.0], [30.0], [50.0], [70.0], [90.0]])
        tf = MagicMock()
        tf.feat_dict = {torch_frame.numerical: feat.clone()}
        enc._normalize_numerical(tf, "t")
        normed = tf.feat_dict[torch_frame.numerical]

        # Check monotonicity: each row should be < the next
        for i in range(len(normed) - 1):
            assert normed[i, 0] < normed[i + 1, 0], "Z-score broke ordering"

    def test_numerical_encoder_more_stable_after_zscore(self):
        """SharedNumericalEncoder output should have less variance across
        differently-scaled inputs after Z-score normalization."""
        num_enc = SharedNumericalEncoder(32)

        # Two "tables" with wildly different scales
        feat_big = torch.tensor([[1e6, 2e6]])
        feat_small = torch.tensor([[0.001, 0.002]])

        # Without Z-score: output magnitudes may differ dramatically
        out_big_raw = num_enc(feat_big).detach()
        out_small_raw = num_enc(feat_small).detach()
        diff_raw = (out_big_raw - out_small_raw).abs().mean()

        # After Z-score (both become ~[0, 1])
        feat_big_z = (feat_big - 1.5e6) / 5e5
        feat_small_z = (feat_small - 0.0015) / 0.0005
        out_big_z = num_enc(feat_big_z).detach()
        out_small_z = num_enc(feat_small_z).detach()
        diff_z = (out_big_z - out_small_z).abs().mean()

        assert diff_z < diff_raw, (
            "Z-scored features did not produce more similar encoder outputs"
        )


# ═══════════════════════════════════════════════════════════
# 5. OUTPUT SANITY
# ═══════════════════════════════════════════════════════════

class TestOutputSanity:
    """Basic output properties that should always hold."""

    def test_type_encoder_no_nan_inf(self):
        enc = _make_type_encoder({"a": 0, "b": 1, "c": 2}, embedding_dim=64)
        x = torch.randint(0, 3, (8, 50))
        out = enc(x)
        assert torch.isfinite(out).all(), "Type encoder output contains NaN or Inf"

    def test_numerical_encoder_no_nan_from_nan_input(self):
        """NaN inputs should be handled (nan_to_num) and produce finite output."""
        enc = SharedNumericalEncoder(64)
        x = torch.tensor([[float("nan"), 1.0], [2.0, float("nan")]])
        out = enc(x)
        assert torch.isfinite(out).all(), "NaN input produced non-finite output"

    def test_numerical_encoder_no_nan_from_extreme_input(self):
        """Very large inputs should not cause overflow."""
        enc = SharedNumericalEncoder(64)
        x = torch.tensor([[1e10, -1e10]])
        out = enc(x)
        assert torch.isfinite(out).all(), "Extreme input caused non-finite output"

    def test_embedding_encoder_no_nan_inf(self):
        enc = SharedEmbeddingEncoder(64, emb_dims={300})
        x = torch.randn(4, 2, 300)
        out = enc(x)
        assert torch.isfinite(out).all()

    def test_timestamp_encoder_no_nan_inf(self):
        enc = SharedTimestampEncoder(64)
        x = torch.randn(8, 3)
        out = enc(x)
        assert torch.isfinite(out).all()

    def test_type_encoder_output_not_constant_across_batch(self):
        """Different samples with different types should produce different outputs."""
        enc = _make_type_encoder({"a": 0, "b": 1, "c": 2}, embedding_dim=64)
        x = torch.tensor([
            [0, 1, 2],
            [2, 1, 0],
            [1, 0, 1],
        ])
        out = enc(x)
        # Rows should not all be identical
        assert not torch.allclose(out[0], out[1], atol=1e-5)
        assert not torch.allclose(out[0], out[2], atol=1e-5)

    def test_numerical_encoder_equivariant_to_permutation(self):
        """Permuting columns should permute outputs (column-independent encoding)."""
        enc = SharedNumericalEncoder(64)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        x_perm = torch.tensor([[3.0, 1.0, 2.0]])
        out = enc(x)       # [1, 3, 64]
        out_perm = enc(x_perm)

        # out[:, 0, :] encoded value 1.0, out_perm[:, 1, :] also encoded 1.0
        assert torch.allclose(out[0, 0], out_perm[0, 1], atol=1e-5), (
            "Numerical encoder is not column-permutation equivariant"
        )
        assert torch.allclose(out[0, 1], out_perm[0, 2], atol=1e-5)
        assert torch.allclose(out[0, 2], out_perm[0, 0], atol=1e-5)
