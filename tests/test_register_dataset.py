"""Regression tests for PR 1.2 -- ``NeighborTfsEncoder.register_dataset``.

PR 1.2 pulled the per-prefixed-type buffer-registration code out of
``NeighborTfsEncoder.__init__`` into a public ``register_dataset``
method. The encoder can now be constructed with architectural args
only and have its schema state populated incrementally -- crucial for
adoption-time use where a second dataset's prefixed types need to be
registered after the backbone was built.

Tests in this module:

  * ``test_register_dataset_extends_buffers`` -- two disjoint registers
    extend the buffers (Z-score per table, GloVe per unique col, +
    counters) without collision.
  * ``test_register_dataset_safe_name_collision_raises`` -- two
    prefixed types whose underscored sanitization collide raise
    ValueError.
  * ``test_init_no_schema_args_skips_registration`` -- without
    col_names_dict/col_stats_dict, __init__ leaves the schema buffers
    empty (size-0).
  * ``test_init_with_full_schema_calls_register_dataset`` -- backward
    compat: passing schema args to __init__ produces the same final
    state as calling register_dataset() explicitly.
  * ``test_register_dataset_idempotent_on_repeat`` -- calling with the
    same prefixed type twice doesn't double-register or grow buffers.
  * ``test_register_dataset_extends_emb_dims`` -- a new emb_dim seen
    only by the second register_dataset gets a Linear projector added
    to SharedEmbeddingEncoder.
  * ``test_zscore_buffer_values_correct`` -- verifies the registered
    Z-score buffer contains the means/stds passed in.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Override conftest's torch_geometric mock so we can construct a real
# TableAgnosticStypeEncoder + NeighborTfsEncoder. (Linux torch_geometric
# imports cleanly; the conftest mock is M1-Mac defensive.) Only un-mock
# if torch_geometric is currently the MagicMock -- if a sibling test
# file already swapped in the real one, popping again would force a
# DataPipe re-registration which torch refuses ("batch_graphs already
# taken").
from unittest.mock import MagicMock as _MagicMock
if isinstance(sys.modules.get("torch_geometric"), _MagicMock):
    for _k in [k for k in sys.modules if k.startswith("torch_geometric")]:
        del sys.modules[_k]
    import torch_geometric  # noqa: F401  -- re-import the real one

import torch_frame
from torch_frame.data.stats import StatType


class _FakeGlove:
    """Deterministic GloVe mock used by the existing
    test_column_semantic_embedding tests; reused here so we don't pull
    sentence_transformers (mocked at conftest level) at test time."""
    def __init__(self, device="cpu"):
        pass

    def __call__(self, names):
        out = torch.zeros(len(names), 300)
        for i, name in enumerate(names):
            torch.manual_seed(hash(name) % 2**32)
            out[i] = torch.randn(300)
        return out


def _make_encoder(channels=32, **schema_kwargs):
    from encoders import NeighborTfsEncoder
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        enc = NeighborTfsEncoder(channels=channels, **schema_kwargs)
    return enc


def _register(enc, **kwargs):
    """Wrap register_dataset under the same GloVe mock so on-the-fly
    embedding resolution doesn't blow up in CI."""
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        enc.register_dataset(**kwargs)


def test_init_no_schema_args_skips_registration():
    """``__init__(channels=...)`` alone leaves the schema state empty.

    Phase-4 adoption code constructs encoders this way and calls
    register_dataset() per dataset. The schema-buffers must start at
    size 0 so the buffer-register-to-cat path in register_dataset
    initializes correctly.
    """
    enc = _make_encoder()
    assert enc.node_type_map == {}
    assert enc.inv_node_type_map == {}
    assert enc._node_type_to_safe == {}
    assert enc._col_name_to_idx == {}
    assert enc._col_glove_embeddings.shape == (0, 300)
    assert int(enc._num_zscore_tables.item()) == 0
    assert int(enc._num_col_semantic_cols.item()) == 0


def test_register_dataset_extends_buffers():
    """Register two disjoint datasets back-to-back; buffers extend
    correctly and the counters reflect both."""
    enc = _make_encoder()

    _register(enc,
        node_type_map={"ds1::A": 0},
        col_names_dict={"ds1::A": {torch_frame.numerical: ["x", "y"]}},
        col_stats_dict={"ds1::A": {
            "x": {StatType.MEAN: 1.0, StatType.STD: 2.0},
            "y": {StatType.MEAN: 3.0, StatType.STD: 4.0},
        }},
    )
    # After ds1: 1 table registered, 2 unique cols.
    assert enc.node_type_map == {"ds1::A": 0}
    assert enc.inv_node_type_map == {0: "ds1::A"}
    assert enc._node_type_to_safe == {"ds1::A": "ds1__A"}
    assert int(enc._num_zscore_tables.item()) == 1
    assert int(enc._num_col_semantic_cols.item()) == 2
    assert enc._col_glove_embeddings.shape == (2, 300)
    assert hasattr(enc, "_num_mean_ds1__A")
    assert hasattr(enc, "_num_std_ds1__A")

    _register(enc,
        node_type_map={"ds2::B": 1},
        col_names_dict={"ds2::B": {
            torch_frame.numerical: ["z"],         # new col
            torch_frame.categorical: ["x"],       # already-known col -> dedupe
        }},
        col_stats_dict={"ds2::B": {
            "z": {StatType.MEAN: 5.0, StatType.STD: 6.0},
            "x": {},
        }},
    )
    # After ds2: 2 tables, 3 unique cols (x dedupes against ds1::A's x).
    assert enc.node_type_map == {"ds1::A": 0, "ds2::B": 1}
    assert enc.inv_node_type_map == {0: "ds1::A", 1: "ds2::B"}
    assert enc._node_type_to_safe == {"ds1::A": "ds1__A", "ds2::B": "ds2__B"}
    assert int(enc._num_zscore_tables.item()) == 2
    assert int(enc._num_col_semantic_cols.item()) == 3
    assert enc._col_glove_embeddings.shape == (3, 300)
    assert hasattr(enc, "_num_mean_ds2__B")
    assert hasattr(enc, "_num_std_ds2__B")


def test_register_dataset_safe_name_collision_raises():
    """Two prefixed types whose underscored sanitization collide must
    raise on the second register_dataset call."""
    enc = _make_encoder()
    _register(enc,
        node_type_map={"rel-f1::drivers": 0},
        col_names_dict={"rel-f1::drivers": {torch_frame.numerical: ["x"]}},
        col_stats_dict={"rel-f1::drivers": {
            "x": {StatType.MEAN: 0.0, StatType.STD: 1.0},
        }},
    )
    # 'rel-f1::drivers' and 'rel-f1__drivers' both sanitize to
    # 'rel_f1__drivers' under the [^A-Za-z0-9] -> _ rule.
    with pytest.raises(ValueError, match="collision"):
        _register(enc,
            node_type_map={"rel-f1__drivers": 1},
            col_names_dict={"rel-f1__drivers": {
                torch_frame.numerical: ["y"],
            }},
            col_stats_dict={"rel-f1__drivers": {
                "y": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
        )


def test_init_with_full_schema_calls_register_dataset():
    """Backward-compat: passing the full schema to __init__ should
    leave the encoder in the same state as a fresh __init__() followed
    by register_dataset()."""
    schema = dict(
        node_type_map={"t::A": 0, "t::B": 1},
        col_names_dict={
            "t::A": {torch_frame.numerical: ["alpha", "beta"]},
            "t::B": {torch_frame.numerical: ["beta", "gamma"]},
        },
        col_stats_dict={
            "t::A": {
                "alpha": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "beta":  {StatType.MEAN: 0.0, StatType.STD: 1.0},
            },
            "t::B": {
                "beta":  {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "gamma": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            },
        },
    )

    enc1 = _make_encoder(**schema)

    enc2 = _make_encoder()
    _register(
        enc2,
        node_type_map=schema["node_type_map"],
        col_names_dict=schema["col_names_dict"],
        col_stats_dict=schema["col_stats_dict"],
    )

    assert enc1.node_type_map == enc2.node_type_map
    assert enc1.inv_node_type_map == enc2.inv_node_type_map
    assert enc1._node_type_to_safe == enc2._node_type_to_safe
    assert enc1._col_name_to_idx == enc2._col_name_to_idx
    assert enc1._col_glove_embeddings.shape == enc2._col_glove_embeddings.shape
    # GloVe is deterministic; the buffer values must match too.
    assert torch.allclose(
        enc1._col_glove_embeddings, enc2._col_glove_embeddings,
    )
    assert int(enc1._num_zscore_tables.item()) == int(enc2._num_zscore_tables.item())
    assert int(enc1._num_col_semantic_cols.item()) == int(enc2._num_col_semantic_cols.item())


def test_register_dataset_idempotent_on_repeat():
    """Re-registering an already-known prefixed type doesn't grow
    buffers or double-count. Useful for adoption code that may
    re-register on a second pass."""
    enc = _make_encoder()
    schema = dict(
        node_type_map={"t::A": 0},
        col_names_dict={"t::A": {torch_frame.numerical: ["x"]}},
        col_stats_dict={"t::A": {
            "x": {StatType.MEAN: 0.0, StatType.STD: 1.0},
        }},
    )
    _register(enc,**schema)
    z1 = int(enc._num_zscore_tables.item())
    c1 = int(enc._num_col_semantic_cols.item())
    glove_size_1 = enc._col_glove_embeddings.shape[0]

    _register(enc,**schema)  # same args
    z2 = int(enc._num_zscore_tables.item())
    c2 = int(enc._num_col_semantic_cols.item())
    glove_size_2 = enc._col_glove_embeddings.shape[0]

    assert z1 == z2, "z-score table count grew on re-register"
    assert c1 == c2, "column semantic count grew on re-register"
    assert glove_size_1 == glove_size_2, "GloVe buffer grew on re-register"


def test_register_dataset_extends_emb_dims():
    """A new emb_dim discovered only by the second register_dataset
    call should add a Linear projector to SharedEmbeddingEncoder."""
    enc = _make_encoder(channels=32)
    _register(enc,
        node_type_map={"t::A": 0},
        col_names_dict={"t::A": {torch_frame.embedding: ["text_col"]}},
        col_stats_dict={"t::A": {
            "text_col": {StatType.EMB_DIM: 64},
        }},
    )
    emb_enc = enc.table_agnostic_encoder.encoders[
        str(torch_frame.embedding)
    ]
    assert "64" in emb_enc.projectors
    assert "128" not in emb_enc.projectors

    _register(enc,
        node_type_map={"t::B": 1},
        col_names_dict={"t::B": {torch_frame.embedding: ["other_col"]}},
        col_stats_dict={"t::B": {
            "other_col": {StatType.EMB_DIM: 128},
        }},
    )
    assert "64" in emb_enc.projectors
    assert "128" in emb_enc.projectors
    # Both projectors map their dim to channels=32.
    assert emb_enc.projectors["64"].in_features == 64
    assert emb_enc.projectors["64"].out_features == 32
    assert emb_enc.projectors["128"].in_features == 128
    assert emb_enc.projectors["128"].out_features == 32


def test_zscore_buffer_values_correct():
    """The registered Z-score buffer should contain the exact means/
    stds passed in, in the column order of col_names_dict[t][numerical]."""
    enc = _make_encoder()
    _register(enc,
        node_type_map={"tab": 0},
        col_names_dict={"tab": {torch_frame.numerical: ["a", "b", "c"]}},
        col_stats_dict={"tab": {
            "a": {StatType.MEAN: 1.0, StatType.STD: 2.0},
            "b": {StatType.MEAN: 3.0, StatType.STD: 4.0},
            "c": {StatType.MEAN: 5.0, StatType.STD: 6.0},
        }},
    )
    means = enc._num_mean_tab
    stds = enc._num_std_tab
    assert torch.allclose(means, torch.tensor([1.0, 3.0, 5.0]))
    assert torch.allclose(stds, torch.tensor([2.0, 4.0, 6.0]))
