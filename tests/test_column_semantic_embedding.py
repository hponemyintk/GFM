"""Tests for column-name semantic embeddings in NeighborTfsEncoder.

These tests verify that GloVe-based semantic embeddings of column names are:
1. Precomputed and stored as buffers at init time
2. Correctly looked up and added to value embeddings during forward
3. Well-behaved from an ML perspective (distinct, gradients flow, can overfit)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

from encoders import NeighborTfsEncoder, TableAgnosticStypeEncoder
import torch_frame
from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType


# ── Helpers ──────────────────────────────────────────────────

class _FakeGlove:
    """Deterministic GloVe mock: different names → different vectors."""
    def __init__(self, device="cpu"):
        pass

    def __call__(self, names):
        out = torch.zeros(len(names), 300)
        for i, name in enumerate(names):
            torch.manual_seed(hash(name) % 2**32)
            out[i] = torch.randn(300)
        return out


def _make_encoder(
    node_type_map,
    col_names_dict=None,
    col_stats_dict=None,
    channels=32,
):
    """Create NeighborTfsEncoder with mocked GloVe."""
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        enc = NeighborTfsEncoder(
            channels=channels,
            node_type_map=node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
        )
    return enc


def _make_tensorframe(feat_dict, col_names_dict):
    """Create a real TensorFrame with given features and column names."""
    return TensorFrame(
        feat_dict=feat_dict,
        col_names_dict=col_names_dict,
    )


# ═══════════════════════════════════════════════════════════
# 1. BUFFER REGISTRATION & INIT
# ═══════════════════════════════════════════════════════════

class TestColumnSemanticInit:
    """Verify GloVe embeddings for column names are precomputed at init."""

    def test_buffer_registered(self):
        """A GloVe embedding buffer for column names should exist."""
        enc = _make_encoder(
            node_type_map={"drivers": 0},
            col_names_dict={"drivers": {torch_frame.numerical: ["speed", "age"]}},
            col_stats_dict={"drivers": {
                "speed": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "age": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
        )
        assert hasattr(enc, "_col_glove_embeddings"), (
            "Missing _col_glove_embeddings buffer"
        )
        buffer_names = {n for n, _ in enc.named_buffers()}
        assert "_col_glove_embeddings" in buffer_names

    def test_buffer_is_not_parameter(self):
        """Column GloVe embeddings should be buffers, not trainable parameters."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
        )
        param_names = {n for n, _ in enc.named_parameters()}
        assert "_col_glove_embeddings" not in param_names

    def test_buffer_shape_matches_unique_columns(self):
        """Buffer rows = number of unique column names across all tables."""
        enc = _make_encoder(
            node_type_map={"drivers": 0, "races": 1},
            col_names_dict={
                "drivers": {torch_frame.numerical: ["speed", "age"]},
                "races": {torch_frame.numerical: ["laps", "distance"]},
            },
            col_stats_dict={
                "drivers": {
                    "speed": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                    "age": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                },
                "races": {
                    "laps": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                    "distance": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                },
            },
        )
        # 4 unique column names: speed, age, laps, distance
        assert enc._col_glove_embeddings.shape == (4, 300)

    def test_duplicate_column_names_deduplicated(self):
        """Same column name in two tables should share one buffer row."""
        enc = _make_encoder(
            node_type_map={"orders": 0, "returns": 1},
            col_names_dict={
                "orders": {torch_frame.numerical: ["price", "quantity"]},
                "returns": {torch_frame.numerical: ["price", "reason_code"]},
            },
            col_stats_dict={
                "orders": {
                    "price": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                    "quantity": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                },
                "returns": {
                    "price": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                    "reason_code": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                },
            },
        )
        # unique: price, quantity, reason_code → 3
        assert enc._col_glove_embeddings.shape == (3, 300)

    def test_columns_from_multiple_stypes_collected(self):
        """Column names from numerical, categorical, etc. should all be in buffer."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {
                torch_frame.numerical: ["num_a", "num_b"],
                torch_frame.categorical: ["cat_c"],
            }},
            col_stats_dict={"t": {
                "num_a": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "num_b": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "cat_c": {},
            }},
        )
        # 3 unique columns: num_a, num_b, cat_c
        assert enc._col_glove_embeddings.shape == (3, 300)

    def test_projection_layer_exists(self):
        """A Linear(300, channels) projection for column embeddings should exist."""
        channels = 64
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
            channels=channels,
        )
        assert hasattr(enc, "col_name_proj"), "Missing col_name_proj layer"
        assert enc.col_name_proj.in_features == 300
        assert enc.col_name_proj.out_features == channels

    def test_col_name_to_idx_mapping(self):
        """Internal mapping from column name to buffer index should be built."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["alpha", "beta"]}},
            col_stats_dict={"t": {
                "alpha": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "beta": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
        )
        assert hasattr(enc, "_col_name_to_idx")
        assert "alpha" in enc._col_name_to_idx
        assert "beta" in enc._col_name_to_idx
        # Indices should be distinct
        assert enc._col_name_to_idx["alpha"] != enc._col_name_to_idx["beta"]

    def test_none_col_names_dict_no_crash(self):
        """When col_names_dict is None, column semantic embeddings should be skipped."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict=None,
            col_stats_dict=None,
        )
        # Buffer should either not exist or be empty
        if hasattr(enc, "_col_glove_embeddings"):
            assert enc._col_glove_embeddings.shape[0] == 0

    def test_empty_col_names_dict(self):
        """Empty col_names_dict should not crash."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={},
            col_stats_dict={},
        )
        if hasattr(enc, "_col_glove_embeddings"):
            assert enc._col_glove_embeddings.shape[0] == 0


# ═══════════════════════════════════════════════════════════
# 2. REPRESENTATION QUALITY
# ═══════════════════════════════════════════════════════════

class TestColumnSemanticRepresentation:
    """Verify column name embeddings are distinct and meaningful."""

    def test_different_names_different_embeddings(self):
        """Columns with different names should have different GloVe vectors."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["price", "age", "height"]}},
            col_stats_dict={"t": {
                "price": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "age": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "height": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
        )
        embs = enc._col_glove_embeddings  # [3, 300]
        # All pairs should differ
        assert not torch.allclose(embs[0], embs[1], atol=1e-4)
        assert not torch.allclose(embs[0], embs[2], atol=1e-4)
        assert not torch.allclose(embs[1], embs[2], atol=1e-4)

    def test_same_name_across_tables_same_embedding(self):
        """'price' in table A and 'price' in table B should map to same buffer index."""
        enc = _make_encoder(
            node_type_map={"orders": 0, "returns": 1},
            col_names_dict={
                "orders": {torch_frame.numerical: ["price"]},
                "returns": {torch_frame.numerical: ["price"]},
            },
            col_stats_dict={
                "orders": {"price": {StatType.MEAN: 0.0, StatType.STD: 1.0}},
                "returns": {"price": {StatType.MEAN: 0.0, StatType.STD: 1.0}},
            },
        )
        idx_orders = enc._col_name_to_idx["price"]
        # Only one entry for "price" — deduplication
        assert enc._col_glove_embeddings.shape[0] == 1
        assert idx_orders == 0

    def test_embeddings_not_collapsed(self):
        """Column embeddings should have variance across dimensions (not constant)."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x", "y", "z"]}},
            col_stats_dict={"t": {
                "x": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "y": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "z": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
        )
        embs = enc._col_glove_embeddings  # [3, 300]
        var_per_dim = embs.var(dim=0)
        nonzero_dims = (var_per_dim > 1e-6).sum()
        assert nonzero_dims > 100, (
            f"Only {nonzero_dims}/300 dims have variance — column embeddings collapsed"
        )

    def test_projected_embeddings_reasonable_magnitude(self):
        """Projected column embeddings should not be extremely large or tiny."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["speed", "power"]}},
            col_stats_dict={"t": {
                "speed": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "power": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
            channels=64,
        )
        projected = enc.col_name_proj(enc._col_glove_embeddings)
        rms = projected.pow(2).mean().sqrt()
        assert 0.01 < rms < 100, f"Projected column embedding RMS {rms:.4f} out of range"


# ═══════════════════════════════════════════════════════════
# 3. GRADIENT HEALTH
# ═══════════════════════════════════════════════════════════

class TestColumnSemanticGradients:
    """Verify gradients flow correctly through column semantic embeddings."""

    def test_projection_receives_gradients(self):
        """col_name_proj should receive non-zero gradients."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["speed", "age"]}},
            col_stats_dict={"t": {
                "speed": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "age": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
            channels=32,
        )
        # Simulate: project column embeddings and compute loss
        projected = enc.col_name_proj(enc._col_glove_embeddings)
        loss = projected.sum()
        loss.backward()

        assert enc.col_name_proj.weight.grad is not None
        assert enc.col_name_proj.weight.grad.abs().sum() > 0
        assert enc.col_name_proj.bias.grad is not None

    def test_glove_buffer_no_gradient(self):
        """Column GloVe buffer should not receive gradients (it's a buffer)."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
        )
        projected = enc.col_name_proj(enc._col_glove_embeddings)
        loss = projected.sum()
        loss.backward()

        assert enc._col_glove_embeddings.grad is None

    def test_buffer_unchanged_after_training_steps(self):
        """GloVe column embeddings should not change during optimizer steps."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["speed", "age"]}},
            col_stats_dict={"t": {
                "speed": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "age": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
            channels=32,
        )
        original = enc._col_glove_embeddings.clone()
        optimizer = torch.optim.Adam(enc.parameters(), lr=0.01)

        for _ in range(5):
            projected = enc.col_name_proj(enc._col_glove_embeddings)
            loss = projected.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert torch.allclose(enc._col_glove_embeddings, original), (
            "Column GloVe buffer was modified during training!"
        )

    def test_no_exploding_gradients_through_projection(self):
        """Gradient norms should stay bounded for the column projection."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["a", "b", "c", "d"]}},
            col_stats_dict={"t": {
                "a": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "b": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "c": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "d": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
            channels=64,
        )
        projected = enc.col_name_proj(enc._col_glove_embeddings)
        loss = projected.sum()
        loss.backward()

        grad_norm = enc.col_name_proj.weight.grad.norm()
        assert grad_norm < 1e4, (
            f"Gradient norm {grad_norm:.1f} — possible explosion"
        )


# ═══════════════════════════════════════════════════════════
# 4. FORWARD INTEGRATION
# ═══════════════════════════════════════════════════════════

class TestColumnSemanticForward:
    """Verify semantic embeddings are added to value embeddings in forward."""

    def _run_forward_with_and_without_semantic(
        self, node_type_map, col_names_dict, col_stats_dict, big_tf, t_int, channels=32
    ):
        """Run the encoder on the same TF and return output with/without column semantics.

        Returns (output_with_semantics, output_without_semantics).
        'Without' is simulated by zeroing out the column GloVe buffer.
        """
        enc = _make_encoder(
            node_type_map=node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
            channels=channels,
        )
        enc.eval()

        # Forward with semantics
        B, K = 2, 3
        neighbor_types = torch.full((B, K), t_int, dtype=torch.long)

        # Build batch_dict: all neighbors are same type
        grouped_indices = {t_int: list(range(B * K))}
        flat_batch_idx = []
        flat_nbr_idx = []
        for b in range(B):
            for k in range(K):
                flat_batch_idx.append(b)
                flat_nbr_idx.append(k)

        batch_dict = {
            "grouped_tfs": {t_int: big_tf},
            "grouped_indices": grouped_indices,
            "flat_batch_idx": flat_batch_idx,
            "flat_nbr_idx": flat_nbr_idx,
        }

        with torch.no_grad():
            out_with = enc(batch_dict, neighbor_types).clone()

        # Zero out GloVe buffer to simulate "no column semantics"
        enc._col_glove_embeddings.zero_()
        batch_dict["grouped_tfs"] = {t_int: big_tf}
        with torch.no_grad():
            out_without = enc(batch_dict, neighbor_types).clone()

        return out_with, out_without

    def test_semantic_embedding_changes_output(self):
        """Output should differ when column semantic embeddings are present vs zeroed."""
        node_type_map = {"t": 0}
        col_names_dict = {"t": {torch_frame.numerical: ["speed", "age"]}}
        col_stats_dict = {"t": {
            "speed": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            "age": {StatType.MEAN: 0.0, StatType.STD: 1.0},
        }}

        feat = torch.randn(6, 2)  # B*K = 6 rows, 2 numerical cols
        big_tf = _make_tensorframe(
            feat_dict={torch_frame.numerical: feat},
            col_names_dict={torch_frame.numerical: ["speed", "age"]},
        )

        out_with, out_without = self._run_forward_with_and_without_semantic(
            node_type_map, col_names_dict, col_stats_dict, big_tf, t_int=0,
        )

        assert not torch.allclose(out_with, out_without, atol=1e-4), (
            "Column semantic embeddings had no effect on output"
        )

    def test_output_shape_unchanged(self):
        """Adding column semantics should not change output shape [B, K, channels]."""
        channels = 32
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x", "y"]}},
            col_stats_dict={"t": {
                "x": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "y": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
            channels=channels,
        )

        B, K = 4, 5
        neighbor_types = torch.zeros(B, K, dtype=torch.long)
        feat = torch.randn(B * K, 2)
        big_tf = _make_tensorframe(
            feat_dict={torch_frame.numerical: feat},
            col_names_dict={torch_frame.numerical: ["x", "y"]},
        )

        batch_dict = {
            "grouped_tfs": {0: big_tf},
            "grouped_indices": {0: list(range(B * K))},
            "flat_batch_idx": [b for b in range(B) for _ in range(K)],
            "flat_nbr_idx": [k for _ in range(B) for k in range(K)],
        }

        out = enc(batch_dict, neighbor_types)
        assert out.shape == (B, K, channels)

    def test_output_finite(self):
        """Column semantic embedding should not introduce NaN or Inf."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
            channels=32,
        )

        B, K = 2, 3
        neighbor_types = torch.zeros(B, K, dtype=torch.long)
        feat = torch.randn(B * K, 1)
        big_tf = _make_tensorframe(
            feat_dict={torch_frame.numerical: feat},
            col_names_dict={torch_frame.numerical: ["x"]},
        )

        batch_dict = {
            "grouped_tfs": {0: big_tf},
            "grouped_indices": {0: list(range(B * K))},
            "flat_batch_idx": [b for b in range(B) for _ in range(K)],
            "flat_nbr_idx": [k for _ in range(B) for k in range(K)],
        }

        out = enc(batch_dict, neighbor_types)
        assert torch.isfinite(out).all(), "Column semantic embedding produced NaN/Inf"

    def test_different_column_names_produce_different_cls_outputs(self):
        """Two single-column TFs with same value but different column names
        should produce different CLS readouts, proving the semantic signal
        propagates through the transformer.

        Uses separate table types so Z-score buffers (1 element each) match
        the 1-column TFs exactly, avoiding shape mismatches from broadcasting.
        """
        channels = 32
        enc = _make_encoder(
            node_type_map={"t_price": 0, "t_age": 1},
            col_names_dict={
                "t_price": {torch_frame.numerical: ["price"]},
                "t_age": {torch_frame.numerical: ["age"]},
            },
            col_stats_dict={
                "t_price": {"price": {StatType.MEAN: 0.0, StatType.STD: 1.0}},
                "t_age": {"age": {StatType.MEAN: 0.0, StatType.STD: 1.0}},
            },
            channels=channels,
        )
        enc.eval()

        feat = torch.ones(1, 1)  # Same value, single column

        tf_price = _make_tensorframe(
            feat_dict={torch_frame.numerical: feat.clone()},
            col_names_dict={torch_frame.numerical: ["price"]},
        )
        tf_age = _make_tensorframe(
            feat_dict={torch_frame.numerical: feat.clone()},
            col_names_dict={torch_frame.numerical: ["age"]},
        )

        B, K = 1, 1

        def _run(tf, t_int):
            neighbor_types = torch.full((B, K), t_int, dtype=torch.long)
            batch_dict = {
                "grouped_tfs": {t_int: tf},
                "grouped_indices": {t_int: [0]},
                "flat_batch_idx": [0],
                "flat_nbr_idx": [0],
            }
            with torch.no_grad():
                return enc(batch_dict, neighbor_types)

        out_price = _run(tf_price, 0)
        out_age = _run(tf_age, 1)

        assert not torch.allclose(out_price, out_age, atol=1e-4), (
            "Different column names produced same output — semantic embeddings not used"
        )


# ═══════════════════════════════════════════════════════════
# 5. UNSEEN COLUMN NAMES
# ═══════════════════════════════════════════════════════════

class TestUnseenColumnNames:
    """Verify behavior when forward receives column names not seen at init."""

    def test_unseen_column_produces_finite_output(self):
        """Column name not in the precomputed buffer should still produce finite output."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["speed"]}},
            col_stats_dict={"t": {"speed": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
            channels=32,
        )

        B, K = 1, 1
        neighbor_types = torch.zeros(B, K, dtype=torch.long)
        feat = torch.randn(1, 1)

        # TF has column "weight" which was NOT in col_names_dict at init
        big_tf = _make_tensorframe(
            feat_dict={torch_frame.numerical: feat},
            col_names_dict={torch_frame.numerical: ["weight"]},
        )

        batch_dict = {
            "grouped_tfs": {0: big_tf},
            "grouped_indices": {0: [0]},
            "flat_batch_idx": [0],
            "flat_nbr_idx": [0],
        }

        out = enc(batch_dict, neighbor_types)
        assert torch.isfinite(out).all(), (
            "Unseen column name produced NaN/Inf"
        )

    def test_unseen_column_output_differs_from_known(self):
        """An unseen column should not silently produce the same embedding as a known one."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["speed"]}},
            col_stats_dict={"t": {"speed": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
            channels=32,
        )
        enc.eval()

        B, K = 1, 1
        neighbor_types = torch.zeros(B, K, dtype=torch.long)
        feat = torch.ones(1, 1)

        def _run(col_name):
            tf = _make_tensorframe(
                feat_dict={torch_frame.numerical: feat.clone()},
                col_names_dict={torch_frame.numerical: [col_name]},
            )
            batch_dict = {
                "grouped_tfs": {0: tf},
                "grouped_indices": {0: [0]},
                "flat_batch_idx": [0],
                "flat_nbr_idx": [0],
            }
            with torch.no_grad():
                return enc(batch_dict, neighbor_types)

        out_known = _run("speed")
        out_unseen = _run("temperature")

        assert not torch.allclose(out_known, out_unseen, atol=1e-4), (
            "Unseen column 'temperature' produced same output as known 'speed'"
        )


# ═══════════════════════════════════════════════════════════
# 6. OVERFIT / LEARNING SIGNAL
# ═══════════════════════════════════════════════════════════

class TestColumnSemanticOverfit:
    """Verify the column semantic signal helps the model learn."""

    def test_column_projection_learns_to_separate(self):
        """After training, projected column embeddings for different names
        should be pushed apart by a contrastive-style loss."""
        channels = 32
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["price", "weight"]}},
            col_stats_dict={"t": {
                "price": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "weight": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
            channels=channels,
        )

        # Targets: "price" → first half hot, "weight" → second half hot
        target_price = torch.zeros(channels)
        target_price[:channels // 2] = 1.0
        target_weight = torch.zeros(channels)
        target_weight[channels // 2:] = 1.0

        optimizer = torch.optim.Adam([enc.col_name_proj.weight, enc.col_name_proj.bias], lr=0.01)

        for _ in range(200):
            projected = enc.col_name_proj(enc._col_glove_embeddings)  # [2, channels]
            loss = (
                F.mse_loss(projected[enc._col_name_to_idx["price"]], target_price) +
                F.mse_loss(projected[enc._col_name_to_idx["weight"]], target_weight)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            projected = enc.col_name_proj(enc._col_glove_embeddings)
            p_price = projected[enc._col_name_to_idx["price"]]
            p_weight = projected[enc._col_name_to_idx["weight"]]
            cos_sim = F.cosine_similarity(p_price.unsqueeze(0), p_weight.unsqueeze(0))
            assert cos_sim < 0.8, (
                f"Columns not separated after training: cos_sim={cos_sim.item():.3f}"
            )

    def test_semantic_embedding_helps_distinguish_columns(self):
        """A linear probe on top of column semantic embeddings should be able to
        distinguish which column a value came from."""
        channels = 32
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["col_a", "col_b"]}},
            col_stats_dict={"t": {
                "col_a": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "col_b": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
            channels=channels,
        )

        # The projected embeddings for col_a and col_b should be linearly separable
        probe = nn.Linear(channels, 2)
        optimizer = torch.optim.Adam(list(probe.parameters()) + [enc.col_name_proj.weight, enc.col_name_proj.bias], lr=0.01)

        for _ in range(100):
            projected = enc.col_name_proj(enc._col_glove_embeddings)  # [2, channels]
            logits = probe(projected)  # [2, 2]
            targets = torch.tensor([0, 1])  # col_a=class0, col_b=class1
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            projected = enc.col_name_proj(enc._col_glove_embeddings)
            logits = probe(projected)
            preds = logits.argmax(dim=-1)
            assert preds[0] == 0 and preds[1] == 1, (
                f"Probe failed to separate columns: preds={preds.tolist()}"
            )


# ═══════════════════════════════════════════════════════════
# 7. SERIALIZATION
# ═══════════════════════════════════════════════════════════

class TestColumnSemanticSerialization:
    """Verify column embeddings survive save/load."""

    def test_state_dict_contains_column_buffer(self):
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
        )
        state = enc.state_dict()
        assert "_col_glove_embeddings" in state
        assert "col_name_proj.weight" in state

    def test_column_buffer_survives_roundtrip(self):
        """Buffer values should be identical after state_dict save/load."""
        enc1 = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["speed", "age"]}},
            col_stats_dict={"t": {
                "speed": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "age": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
        )
        state = enc1.state_dict()

        enc2 = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["speed", "age"]}},
            col_stats_dict={"t": {
                "speed": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "age": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
        )
        enc2.load_state_dict(state)

        assert torch.allclose(
            enc1._col_glove_embeddings,
            enc2._col_glove_embeddings,
        )


# ═══════════════════════════════════════════════════════════
# 8. RESET PARAMETERS
# ═══════════════════════════════════════════════════════════

class TestColumnSemanticReset:

    def test_reset_parameters_resets_projection(self):
        """reset_parameters should reset col_name_proj without crashing."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
        )
        # Modify weights
        with torch.no_grad():
            enc.col_name_proj.weight.fill_(999.0)

        enc.reset_parameters()

        # After reset, weights should not be 999 anymore
        assert not torch.allclose(
            enc.col_name_proj.weight,
            torch.full_like(enc.col_name_proj.weight, 999.0),
        )

    def test_reset_does_not_change_glove_buffer(self):
        """reset_parameters should NOT modify the GloVe buffer."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
        )
        original = enc._col_glove_embeddings.clone()
        enc.reset_parameters()
        assert torch.allclose(enc._col_glove_embeddings, original)
