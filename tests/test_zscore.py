"""Tests for Z-score normalization in NeighborTfsEncoder."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import pytest
import torch
import torch_frame
from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType
from unittest.mock import patch, MagicMock

from encoders import NeighborTfsEncoder


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
    col_names_dict,
    col_stats_dict,
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


class TestZScoreBufferRegistration:

    def test_buffers_registered_for_numerical_table(self):
        enc = _make_encoder(
            node_type_map={"drivers": 0},
            col_names_dict={"drivers": {torch_frame.numerical: ["speed", "age"]}},
            col_stats_dict={"drivers": {
                "speed": {StatType.MEAN: 200.0, StatType.STD: 30.0},
                "age": {StatType.MEAN: 28.0, StatType.STD: 5.0},
            }},
        )
        assert hasattr(enc, "_num_mean_drivers")
        assert hasattr(enc, "_num_std_drivers")
        assert enc._num_mean_drivers.shape == (2,)
        assert enc._num_std_drivers.shape == (2,)

    def test_buffer_values_correct(self):
        enc = _make_encoder(
            node_type_map={"drivers": 0},
            col_names_dict={"drivers": {torch_frame.numerical: ["speed", "age"]}},
            col_stats_dict={"drivers": {
                "speed": {StatType.MEAN: 100.0, StatType.STD: 10.0},
                "age": {StatType.MEAN: 30.0, StatType.STD: 5.0},
            }},
        )
        assert torch.allclose(enc._num_mean_drivers, torch.tensor([100.0, 30.0]))
        assert torch.allclose(enc._num_std_drivers, torch.tensor([10.0, 5.0]))

    def test_safe_name_sanitization(self):
        """Special characters in table name are replaced with underscores."""
        enc = _make_encoder(
            node_type_map={"pit-stops": 0},
            col_names_dict={"pit-stops": {torch_frame.numerical: ["lap"]}},
            col_stats_dict={"pit-stops": {
                "lap": {StatType.MEAN: 50.0, StatType.STD: 10.0},
            }},
        )
        assert hasattr(enc, "_num_mean_pit_stops")
        assert hasattr(enc, "_num_std_pit_stops")

    def test_no_buffers_for_non_numerical_table(self):
        enc = _make_encoder(
            node_type_map={"users": 0},
            col_names_dict={"users": {torch_frame.categorical: ["country"]}},
            col_stats_dict={"users": {
                "country": {},
            }},
        )
        assert not hasattr(enc, "_num_mean_users")
        assert not hasattr(enc, "_num_std_users")

    def test_multiple_tables(self):
        enc = _make_encoder(
            node_type_map={"drivers": 0, "races": 1},
            col_names_dict={
                "drivers": {torch_frame.numerical: ["speed"]},
                "races": {torch_frame.numerical: ["laps"]},
            },
            col_stats_dict={
                "drivers": {"speed": {StatType.MEAN: 200.0, StatType.STD: 30.0}},
                "races": {"laps": {StatType.MEAN: 60.0, StatType.STD: 8.0}},
            },
        )
        assert hasattr(enc, "_num_mean_drivers")
        assert hasattr(enc, "_num_mean_races")
        assert not torch.allclose(enc._num_mean_drivers, enc._num_mean_races)

    def test_none_mean_defaults_to_zero(self):
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: None, StatType.STD: 1.0}}},
        )
        assert enc._num_mean_t.item() == 0.0

    def test_none_std_defaults_to_one(self):
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: None}}},
        )
        assert enc._num_std_t.item() == 1.0

    def test_zero_std_defaults_to_one(self):
        """Python `0.0 or 1.0` evaluates to 1.0, so zero std becomes 1.0."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 0.0}}},
        )
        # 0.0 is falsy in Python, so `float(0.0 or 1.0)` = 1.0
        assert enc._num_std_t.item() == 1.0

    def test_missing_stats_for_column(self):
        """Column exists in col_names but has no stats entry."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {}},  # no stats for "x"
        )
        # Should default to mean=0, std=1
        assert enc._num_mean_t.item() == 0.0
        assert enc._num_std_t.item() == 1.0

    def test_none_col_names_dict(self):
        """No crash when col_names_dict is None."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict=None,
            col_stats_dict=None,
        )
        assert not hasattr(enc, "_num_mean_t")

    def test_name_collision_raises(self):
        """'a-b' and 'a_b' both sanitize to 'a_b' — should raise ValueError."""
        with pytest.raises(ValueError, match="collision"):
            _make_encoder(
                node_type_map={"a-b": 0, "a_b": 1},
                col_names_dict={
                    "a-b": {torch_frame.numerical: ["x"]},
                    "a_b": {torch_frame.numerical: ["y"]},
                },
                col_stats_dict={
                    "a-b": {"x": {StatType.MEAN: 10.0, StatType.STD: 2.0}},
                    "a_b": {"y": {StatType.MEAN: 99.0, StatType.STD: 7.0}},
                },
            )


class TestNormalizeNumerical:

    def _build_encoder_and_tf(self, mean, std, feat_values):
        """Helper: build encoder with known stats and a mock TensorFrame."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x", "y"]}},
            col_stats_dict={"t": {
                "x": {StatType.MEAN: mean[0], StatType.STD: std[0]},
                "y": {StatType.MEAN: mean[1], StatType.STD: std[1]},
            }},
        )
        # Build a minimal mock TensorFrame
        tf = MagicMock()
        tf.feat_dict = {torch_frame.numerical: feat_values.clone()}
        return enc, tf

    def test_zscore_math(self):
        feat = torch.tensor([[100.0, 50.0], [110.0, 55.0]])
        enc, tf = self._build_encoder_and_tf(
            mean=[100.0, 50.0], std=[10.0, 5.0], feat_values=feat,
        )
        enc._normalize_numerical(tf, "t")
        result = tf.feat_dict[torch_frame.numerical]
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_negative_values(self):
        feat = torch.tensor([[80.0, 40.0]])
        enc, tf = self._build_encoder_and_tf(
            mean=[100.0, 50.0], std=[10.0, 5.0], feat_values=feat,
        )
        enc._normalize_numerical(tf, "t")
        result = tf.feat_dict[torch_frame.numerical]
        expected = torch.tensor([[-2.0, -2.0]])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_unseen_table_skipped(self):
        feat = torch.tensor([[1.0, 2.0]])
        enc, tf = self._build_encoder_and_tf(
            mean=[0.0, 0.0], std=[1.0, 1.0], feat_values=feat,
        )
        original = tf.feat_dict[torch_frame.numerical].clone()
        enc._normalize_numerical(tf, "unknown_table")
        # Should be unchanged
        assert torch.allclose(tf.feat_dict[torch_frame.numerical], original)

    def test_no_numerical_in_feat_dict(self):
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
        )
        tf = MagicMock()
        tf.feat_dict = {torch_frame.categorical: torch.tensor([[1, 2]])}
        # Should not crash — early return
        enc._normalize_numerical(tf, "t")

    def test_epsilon_prevents_division_by_zero(self):
        """Even if std buffer is manually set to 0, epsilon prevents inf."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}}},
        )
        # Override buffer properly (not shadowing with a plain tensor)
        enc._num_std_t.fill_(0.0)
        tf = MagicMock()
        tf.feat_dict = {torch_frame.numerical: torch.tensor([[5.0]])}
        enc._normalize_numerical(tf, "t")
        result = tf.feat_dict[torch_frame.numerical]
        assert torch.isfinite(result).all()

    def test_nan_propagates_through_zscore(self):
        """NaN values should remain NaN after Z-score so downstream
        SharedNumericalEncoder can detect them for missingness handling."""
        feat = torch.tensor([[float("nan"), 50.0], [100.0, float("nan")]])
        enc, tf = self._build_encoder_and_tf(
            mean=[100.0, 50.0], std=[10.0, 5.0], feat_values=feat,
        )
        enc._normalize_numerical(tf, "t")
        result = tf.feat_dict[torch_frame.numerical]
        # NaN positions should stay NaN (propagated for downstream detection)
        assert torch.isnan(result[0, 0]), "NaN should propagate through Z-score"
        assert torch.isnan(result[1, 1]), "NaN should propagate through Z-score"
        # Non-NaN values should be Z-scored normally
        assert result[0, 1].item() == pytest.approx(0.0, abs=1e-6)  # (50-50)/5 = 0
        assert result[1, 0].item() == pytest.approx(0.0, abs=1e-6)  # (100-100)/10 = 0

    def test_nan_not_silently_converted_to_outlier(self):
        """NaN must NOT become (0-mean)/std, which is an outlier for large means.
        Instead NaN should propagate through Z-score for downstream handling."""
        feat = torch.tensor([[float("nan")]])
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 1000.0, StatType.STD: 10.0}}},
        )
        tf = MagicMock()
        tf.feat_dict = {torch_frame.numerical: feat}
        enc._normalize_numerical(tf, "t")
        result = tf.feat_dict[torch_frame.numerical]
        # NaN should stay NaN, not become -100 ((0-1000)/10)
        assert torch.isnan(result).all(), (
            f"NaN became {result.item():.1f} — should stay NaN for missingness detection"
        )

    def test_in_place_mutation(self):
        """_normalize_numerical mutates feat_dict in place."""
        feat = torch.tensor([[100.0, 50.0]])
        enc, tf = self._build_encoder_and_tf(
            mean=[100.0, 50.0], std=[10.0, 5.0], feat_values=feat,
        )
        enc._normalize_numerical(tf, "t")
        # After normalization, the value should be different from original
        assert not torch.allclose(
            tf.feat_dict[torch_frame.numerical], feat
        )


class TestZScoreSerializationGuard:
    """Tests for the _num_zscore_tables guard that prevents silent normalization skip."""

    def test_zscore_table_count_buffer_registered(self):
        enc = _make_encoder(
            node_type_map={"drivers": 0, "races": 1},
            col_names_dict={
                "drivers": {torch_frame.numerical: ["speed"]},
                "races": {torch_frame.numerical: ["laps"]},
            },
            col_stats_dict={
                "drivers": {"speed": {StatType.MEAN: 200.0, StatType.STD: 30.0}},
                "races": {"laps": {StatType.MEAN: 60.0, StatType.STD: 8.0}},
            },
        )
        assert hasattr(enc, "_num_zscore_tables")
        assert enc._num_zscore_tables.item() == 2

    def test_zscore_table_count_zero_when_no_numerical(self):
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.categorical: ["c"]}},
            col_stats_dict={"t": {"c": {}}},
        )
        assert enc._num_zscore_tables.item() == 0

    def test_zscore_table_count_zero_when_none_dicts(self):
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict=None,
            col_stats_dict=None,
        )
        assert enc._num_zscore_tables.item() == 0

    def test_zscore_table_count_survives_state_dict_roundtrip(self):
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 5.0, StatType.STD: 2.0}}},
        )
        state = enc.state_dict()
        assert "_num_zscore_tables" in state
        assert state["_num_zscore_tables"].item() == 1

    def test_forward_crashes_when_loaded_without_col_dicts(self):
        """Simulate: train with stats, save, reload without col_names_dict."""
        # 1. Build encoder with stats
        enc_train = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 5.0, StatType.STD: 2.0}}},
        )
        state = enc_train.state_dict()

        # 2. Rebuild encoder WITHOUT col_names_dict (simulating inference setup)
        enc_infer = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict=None,
            col_stats_dict=None,
        )
        # Filter out column-semantic buffer (size mismatch: trained has columns,
        # inference has none). This is expected — column names must be provided
        # at construction time, just like Z-score stats.
        filtered_state = {k: v for k, v in state.items()
                          if not k.startswith('_col_glove')}
        enc_infer.load_state_dict(filtered_state, strict=False)

        # 3. _num_zscore_tables is loaded (=1), but _node_type_to_safe is empty
        assert enc_infer._num_zscore_tables.item() == 1
        assert len(enc_infer._node_type_to_safe) == 0

        # 4. Forward should crash with a clear error
        dummy_batch = {
            "grouped_tfs": {},
            "grouped_indices": {},
            "flat_batch_idx": [],
            "flat_nbr_idx": [],
        }
        with pytest.raises(RuntimeError, match="col_names_dict"):
            enc_infer.forward(dummy_batch, torch.zeros(1, 1, dtype=torch.long))

    def test_forward_ok_when_loaded_with_col_dicts(self):
        """Simulate: train with stats, save, reload WITH col_names_dict."""
        enc_train = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 5.0, StatType.STD: 2.0}}},
        )
        state = enc_train.state_dict()

        # Rebuild WITH col_names_dict
        enc_infer = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.numerical: ["x"]}},
            col_stats_dict={"t": {"x": {StatType.MEAN: 5.0, StatType.STD: 2.0}}},
        )
        enc_infer.load_state_dict(state, strict=False)

        # Should NOT crash — _node_type_to_safe is populated
        assert enc_infer._num_zscore_tables.item() == 1
        assert len(enc_infer._node_type_to_safe) > 0

    def test_no_crash_when_model_has_no_zscore_stats(self):
        """Model without Z-score stats should not crash on forward."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict=None,
            col_stats_dict=None,
        )
        assert enc._num_zscore_tables.item() == 0
        # Forward with empty batch should not trigger the guard
        dummy_batch = {
            "grouped_tfs": {},
            "grouped_indices": {},
            "flat_batch_idx": [],
            "flat_nbr_idx": [],
        }
        # Should not raise — guard only fires when _num_zscore_tables > 0
        enc.forward(dummy_batch, torch.zeros(1, 1, dtype=torch.long))
