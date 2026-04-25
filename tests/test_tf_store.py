"""Tests T1-T7 for gfm_data/tf_store.py.

T2 (multi-categorical) is intentionally skipped in PR2 -- we raise
``NotImplementedError`` for that stype until rel-event lands in PR4.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from torch_frame import TensorFrame, stype
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfm_data.tf_store import TFStoreReader, build_tf_store  # noqa: E402


def _make_tf(num_rows=12):
    num = torch.arange(num_rows * 3, dtype=torch.float32).reshape(num_rows, 3)
    cat = torch.arange(num_rows * 2, dtype=torch.long).reshape(num_rows, 2)
    ts = torch.arange(num_rows * 1 * 7, dtype=torch.long).reshape(num_rows, 1, 7)
    # 2 embedding columns, dims 4 and 6.
    emb_values = torch.randn(num_rows, 4 + 6)
    emb_offset = torch.tensor([0, 4, 10], dtype=torch.long)
    emb = MultiEmbeddingTensor(num_rows=num_rows, num_cols=2,
                               values=emb_values, offset=emb_offset)
    feat = {
        stype.numerical: num,
        stype.categorical: cat,
        stype.timestamp: ts,
        stype.embedding: emb,
    }
    cnd = {
        stype.numerical: ["a", "b", "c"],
        stype.categorical: ["x", "y"],
        stype.timestamp: ["t"],
        stype.embedding: ["e1", "e2"],
    }
    return TensorFrame(feat_dict=feat, col_names_dict=cnd, num_rows=num_rows)


# ----------------------------------------------------------------- T1
def test_T1_view_matches_in_memory_tf():
    tf = _make_tf(num_rows=20)
    with tempfile.TemporaryDirectory() as root:
        build_tf_store(tf, root)
        r = TFStoreReader(root)
        assert len(r) == 20

        # Single-row, multi-row, and full-table views.
        for sel in ([0], [0, 1, 2], list(range(20))):
            v = r.view(sel)
            t = tf[sel]
            assert v.num_rows == t.num_rows
            assert torch.equal(v.feat_dict[stype.numerical], t.feat_dict[stype.numerical])
            assert torch.equal(v.feat_dict[stype.categorical], t.feat_dict[stype.categorical])
            assert torch.equal(v.feat_dict[stype.timestamp], t.feat_dict[stype.timestamp])
            ev, et = v.feat_dict[stype.embedding], t.feat_dict[stype.embedding]
            assert torch.equal(ev.values, et.values)
            assert torch.equal(ev.offset, et.offset)


# ----------------------------------------------------------------- T3
def test_T3_embedding_per_column_dim_preserved():
    tf = _make_tf(num_rows=8)
    with tempfile.TemporaryDirectory() as root:
        build_tf_store(tf, root)
        r = TFStoreReader(root)
        v = r.view([0, 1, 2])
        emb = v.feat_dict[stype.embedding]
        assert emb.num_cols == 2
        assert emb.values.shape == (3, 10)
        assert emb.offset.tolist() == [0, 4, 10]


# ----------------------------------------------------------------- T4
def test_T4_timestamp_int64_round_trip():
    tf = _make_tf(num_rows=5)
    with tempfile.TemporaryDirectory() as root:
        build_tf_store(tf, root)
        r = TFStoreReader(root)
        v = r.view(list(range(5)))
        # Order-preserving: comparison results match.
        a = tf.feat_dict[stype.timestamp]
        b = v.feat_dict[stype.timestamp]
        assert torch.equal((a < a.max()).to(torch.int8),
                           (b < b.max()).to(torch.int8))


# ----------------------------------------------------------------- T5
def test_T5_repeat_view_idempotent():
    """Successive view() calls return equal tensors (no state corruption)."""
    tf = _make_tf(num_rows=7)
    with tempfile.TemporaryDirectory() as root:
        build_tf_store(tf, root)
        r = TFStoreReader(root)
        v1 = r.view([2, 5])
        v2 = r.view([2, 5])
        for s in v1.feat_dict.keys():
            a, b = v1.feat_dict[s], v2.feat_dict[s]
            if isinstance(a, MultiEmbeddingTensor):
                assert torch.equal(a.values, b.values)
                assert torch.equal(a.offset, b.offset)
            else:
                assert torch.equal(a, b)


# ----------------------------------------------------------------- T6
def test_T6_two_readers_match():
    tf = _make_tf(num_rows=10)
    with tempfile.TemporaryDirectory() as root:
        build_tf_store(tf, root)
        r1 = TFStoreReader(root)
        r2 = TFStoreReader(root)
        v1 = r1.view([3, 7, 9])
        v2 = r2.view([3, 7, 9])
        assert torch.equal(v1.feat_dict[stype.numerical], v2.feat_dict[stype.numerical])
        assert torch.equal(v1.feat_dict[stype.embedding].values,
                           v2.feat_dict[stype.embedding].values)


# ----------------------------------------------------------------- T7 (lite)
def test_T7_random_row_view_is_independent_per_call():
    """Reading 100 random rows works without errors and matches in-memory tf."""
    tf = _make_tf(num_rows=100)
    rng = np.random.default_rng(0)
    sel = rng.integers(0, 100, size=64).tolist()
    with tempfile.TemporaryDirectory() as root:
        build_tf_store(tf, root)
        r = TFStoreReader(root)
        v = r.view(sel)
        t = tf[sel]
        assert torch.equal(v.feat_dict[stype.numerical], t.feat_dict[stype.numerical])


# Multicat sanity: builder must raise.
def test_multicat_builder_raises_NotImplementedError():
    """Until PR4, multi_categorical must fail loudly rather than silently drop.

    We bypass the real ``MultiNestedTensor`` here because torch_frame's
    validator is stricter than the builder's branch we want to exercise; the
    builder iterates ``tf.feat_dict`` and dispatches on the stype name, so a
    duck-typed object is sufficient.
    """
    class _FakeTF:
        feat_dict = {stype.multicategorical: object()}
        col_names_dict = {stype.multicategorical: ["m"]}
        num_rows = 4
    with tempfile.TemporaryDirectory() as root, pytest.raises(NotImplementedError):
        build_tf_store(_FakeTF(), root)
