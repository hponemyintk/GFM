"""Regression tests for PR 1.4 -- ``tools/compute_dataset_stats.py``.

The tool walks a TF-store directory and emits a ``col_stats_dict``
ready for ``NeighborTfsEncoder.register_dataset``. It's used at
adoption time (Phase 4) on a held-out dataset whose stats weren't
part of the training-time col_stats_dict.

Tests build synthetic TF-store directories and assert that the
emitted stats match expected values for each stype.

Cases:

  * ``test_numerical_mean_std_correct``
  * ``test_numerical_handles_all_nan_column``
  * ``test_numerical_handles_constant_column`` (std clamped to 1e-8)
  * ``test_categorical_level_count_correct``
  * ``test_embedding_emb_dim_from_offset``
  * ``test_name_prefix_applies``
  * ``test_output_is_register_dataset_compatible`` (pass directly into
    a fresh NeighborTfsEncoder.register_dataset and verify no
    KeyError or unexpected exception)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cross-file un-mock guard for torch_geometric.
from unittest.mock import MagicMock as _MagicMock
if isinstance(sys.modules.get("torch_geometric"), _MagicMock):
    for _k in [k for k in sys.modules if k.startswith("torch_geometric")]:
        del sys.modules[_k]
    import torch_geometric  # noqa: F401

from torch_frame.data.stats import StatType
from tools.compute_dataset_stats import (
    compute_dataset_stats,
    compute_table_stats,
)


def _write_table(
    root: Path,
    *,
    numerical: dict | None = None,    # {col_name: 1D np.array}
    categorical: dict | None = None,  # {col_name: 1D np.array of int64}
    embedding: dict | None = None,    # {col_name: 2D np.array of float32, [N, D_col]}
):
    """Materialize a synthetic TF-store table.

    Each kwarg is a dict from column name to its data (one column per
    entry). Layout matches what ``gfm_data.tf_store.build_tf_store``
    writes; ``compute_dataset_stats`` reads only meta.json + per-stype
    memmap files.
    """
    root.mkdir(parents=True, exist_ok=True)

    col_names_dict: dict = {}
    shapes: dict = {}
    n_rows = None
    embedding_offset = None

    if numerical:
        cols = list(numerical.keys())
        col_names_dict["numerical"] = cols
        arr = np.stack([numerical[c] for c in cols], axis=1).astype(np.float32)
        shapes["numerical"] = list(arr.shape)
        arr.tofile(root / "numerical.float32")
        n_rows = arr.shape[0]

    if categorical:
        cols = list(categorical.keys())
        col_names_dict["categorical"] = cols
        arr = np.stack([categorical[c] for c in cols], axis=1).astype(np.int64)
        shapes["categorical"] = list(arr.shape)
        arr.tofile(root / "categorical.int64")
        n_rows = n_rows if n_rows is not None else arr.shape[0]

    if embedding:
        cols = list(embedding.keys())
        col_names_dict["embedding"] = cols
        # Concatenate columns along the dim axis (matches the
        # MultiEmbeddingTensor layout: [N, total_dim]).
        cat = np.concatenate([embedding[c] for c in cols], axis=1).astype(np.float32)
        shapes["embedding_values"] = list(cat.shape)
        # offset: [C+1] cumulative dim boundaries.
        dims = [embedding[c].shape[1] for c in cols]
        offset = np.array([0] + list(np.cumsum(dims)), dtype=np.int64)
        shapes["embedding_offset"] = list(offset.shape)
        cat.tofile(root / "embedding.values.f32")
        offset.tofile(root / "embedding.offset.i64")
        embedding_offset = offset.tolist()
        n_rows = n_rows if n_rows is not None else cat.shape[0]

    meta = {
        "num_rows": int(n_rows or 0),
        "col_names_dict": col_names_dict,
        "stypes": list(col_names_dict.keys()),
        "shapes": shapes,
        "embedding_offset": embedding_offset,
        "multicategorical_num_cols": None,
        "layout_version": 1,
    }
    with open(root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def test_numerical_mean_std_correct(tmp_path):
    """Per-column mean/std must match np.mean/np.std on the raw arr."""
    root = tmp_path / "store" / "tableA"
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b = np.array([10.0, 12.0, 14.0, 16.0, 18.0], dtype=np.float32)
    _write_table(root, numerical={"a": a, "b": b})

    stats = compute_table_stats(root)
    assert pytest.approx(stats["a"][StatType.MEAN], rel=1e-6) == float(a.mean())
    assert pytest.approx(stats["a"][StatType.STD], rel=1e-6) == float(a.std())
    assert pytest.approx(stats["b"][StatType.MEAN], rel=1e-6) == float(b.mean())
    assert pytest.approx(stats["b"][StatType.STD], rel=1e-6) == float(b.std())


def test_numerical_handles_all_nan_column(tmp_path):
    """An all-NaN numerical column must produce mean=0, std=1 (safe
    identity transform downstream) without leaking NaN."""
    root = tmp_path / "store" / "tableA"
    a = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    _write_table(root, numerical={"a": a, "b": b})

    stats = compute_table_stats(root)
    assert stats["a"][StatType.MEAN] == 0.0
    assert stats["a"][StatType.STD] == 1.0
    assert pytest.approx(stats["b"][StatType.MEAN], rel=1e-6) == float(b.mean())
    # No NaN leakage anywhere in the output.
    for col, sd in stats.items():
        for k, v in sd.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"NaN in {col}.{k}"


def test_numerical_handles_constant_column(tmp_path):
    """A constant numerical column has std=0; the tool must clamp to
    a small floor so downstream (x - mean) / (std + 1e-8) stays finite."""
    root = tmp_path / "store" / "tableA"
    a = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    _write_table(root, numerical={"a": a})

    stats = compute_table_stats(root)
    assert stats["a"][StatType.MEAN] == 5.0
    # Floor: tool clamps to 1e-8 minimum.
    assert stats["a"][StatType.STD] >= 1e-8


def test_categorical_level_count_correct(tmp_path):
    """``StatType.COUNT`` must equal ``len(np.unique(col))``."""
    root = tmp_path / "store" / "tableA"
    a = np.array([0, 1, 2, 1, 0, 3, 4], dtype=np.int64)  # 5 distinct
    b = np.array([10, 10, 10, 10, 10, 10, 10], dtype=np.int64)  # 1 distinct
    _write_table(root, categorical={"a": a, "b": b})

    stats = compute_table_stats(root)
    assert stats["a"][StatType.COUNT] == 5
    assert stats["b"][StatType.COUNT] == 1


def test_embedding_emb_dim_from_offset(tmp_path):
    """``StatType.EMB_DIM`` per column must come from the offset
    diff (offset[i+1] - offset[i]) -- the per-column dim of the
    MultiEmbeddingTensor."""
    root = tmp_path / "store" / "tableA"
    e1 = np.random.randn(4, 16).astype(np.float32)  # dim 16
    e2 = np.random.randn(4, 32).astype(np.float32)  # dim 32
    _write_table(root, embedding={"e1": e1, "e2": e2})

    stats = compute_table_stats(root)
    assert stats["e1"][StatType.EMB_DIM] == 16
    assert stats["e2"][StatType.EMB_DIM] == 32


def test_dataset_walk_aggregates_tables(tmp_path):
    """``compute_dataset_stats`` must return one entry per table
    directory under the root."""
    root = tmp_path / "store"
    _write_table(
        root / "tableA",
        numerical={"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
    )
    _write_table(
        root / "tableB",
        categorical={"y": np.array([0, 1, 0, 1, 2], dtype=np.int64)},
    )

    out = compute_dataset_stats(str(root))
    assert set(out.keys()) == {"tableA", "tableB"}
    assert StatType.MEAN in out["tableA"]["x"]
    assert StatType.COUNT in out["tableB"]["y"]


def test_name_prefix_applies(tmp_path):
    """``--name_prefix`` should prepend to each table key."""
    root = tmp_path / "store"
    _write_table(
        root / "drivers",
        numerical={"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
    )
    out = compute_dataset_stats(str(root), name_prefix="rel-f1::")
    assert "rel-f1::drivers" in out
    assert "drivers" not in out


def test_output_is_register_dataset_compatible(tmp_path):
    """End-to-end: feed the tool's output straight into
    ``NeighborTfsEncoder.register_dataset(...)`` and verify it doesn't
    raise."""
    import torch_frame
    from encoders import NeighborTfsEncoder

    class _FakeGlove:
        def __init__(self, device="cpu"):
            pass
        def __call__(self, names):
            out = torch.zeros(len(names), 300)
            for i, name in enumerate(names):
                torch.manual_seed(hash(name) % 2**32)
                out[i] = torch.randn(300)
            return out

    root = tmp_path / "store"
    _write_table(
        root / "tableA",
        numerical={
            "p": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "q": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # constant
        },
    )

    # Compute via the tool.
    stats = compute_dataset_stats(str(root))
    # Build the col_names_dict in the format register_dataset expects
    # (stype-keyed; mirrors what make_pkey_fkey_graph emits).
    col_names_dict = {
        "tableA": {torch_frame.numerical: ["p", "q"]},
    }
    col_stats_dict = {"tableA": stats["tableA"]}

    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        enc = NeighborTfsEncoder(channels=32)
        enc.register_dataset(
            node_type_map={"tableA": 0},
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
        )

    # Buffers should be registered with the right values.
    assert hasattr(enc, "_num_mean_tableA")
    assert hasattr(enc, "_num_std_tableA")
    means = enc._num_mean_tableA
    stds = enc._num_std_tableA
    assert torch.allclose(means, torch.tensor([2.5, 0.0]))
    # First col std ~ 1.118; second is constant -> floor 1e-8.
    assert stds[0].item() == pytest.approx(float(np.std(np.array([1.0, 2.0, 3.0, 4.0]))), rel=1e-5)
    # Constant column gets clamped to ~1e-8 floor; FP32 quantization
    # may round to 9.99e-9, so just assert it's strictly positive
    # (avoid divide-by-zero downstream) rather than a magic threshold.
    assert stds[1].item() > 0.0
