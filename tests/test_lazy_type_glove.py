"""Regression tests for PR 1.3 -- ``NeighborNodeTypeEncoder`` name path.

The encoder now accepts either int-tensor indices (fast path, matches
pre-PR-1.3 behavior) or string lists (Phase-4 adoption path: known
names hit the precomputed buffer; unseen names get GloVe-embedded on
the fly into ``_unseen_cache``).

Tests in this module:

  * ``test_int_indices_path_unchanged`` -- existing tensor path
    produces the same output as before (golden test against the
    raw glove_embeddings buffer + proj).
  * ``test_unseen_type_no_keyerror`` -- forward with a name string
    not in node_type_map produces the right shape (no KeyError).
  * ``test_lazy_cache_hits`` -- second forward with the same unseen
    name reuses the cached vector.
  * ``test_lazy_output_matches_manual`` -- lazy-resolved output for
    name "c" equals proj(glove_embedder("c")).
  * ``test_known_type_does_not_populate_cache`` -- forward with a
    known name uses the buffer; cache stays empty.
  * ``test_mask_sentinel_resolves_via_buffer`` -- "mask" is in
    _name_to_idx; resolves to buffer[num_types], not via the cache.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cross-file unmock guard (matches test_drop_c_idx / test_register_dataset)
from unittest.mock import MagicMock as _MagicMock
if isinstance(sys.modules.get("torch_geometric"), _MagicMock):
    for _k in [k for k in sys.modules if k.startswith("torch_geometric")]:
        del sys.modules[_k]
    import torch_geometric  # noqa: F401


class _FakeGlove:
    """Deterministic GloVe mock: name -> seeded random vector."""
    def __init__(self, device="cpu"):
        pass

    def __call__(self, names):
        out = torch.zeros(len(names), 300)
        for i, name in enumerate(names):
            torch.manual_seed(hash(name) % 2**32)
            out[i] = torch.randn(300)
        return out


def _make_encoder(node_type_map=None, channels=32):
    from encoders import NeighborNodeTypeEncoder
    if node_type_map is None:
        node_type_map = {"a": 0, "b": 1}
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        return NeighborNodeTypeEncoder(
            node_type_map=node_type_map, embedding_dim=channels,
        )


def test_int_indices_path_unchanged():
    """Tensor input must produce buffer[idx] -> proj exactly. Catches
    accidental fallthrough into the name path on unexpected input
    types and any incidental change to the existing fast path."""
    enc = _make_encoder({"a": 0, "b": 1})
    indices = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 2 = "mask"
    out = enc(indices)
    assert out.shape == (2, 3, 32)
    # Verify against direct lookup + projection.
    expected = enc.proj(enc.glove_embeddings[indices])
    assert torch.allclose(out, expected)


def test_unseen_type_no_keyerror():
    """A name not in node_type_map must resolve via the lazy cache,
    not raise. Output shape must match [B, K, channels]."""
    enc = _make_encoder({"a": 0, "b": 1})
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        out = enc([["a", "c"], ["c", "b"]])
    assert out.shape == (2, 2, 32)
    assert torch.isfinite(out).all()


def test_lazy_cache_hits():
    """Second forward with the same unseen name reuses the cache, not
    a fresh GloVe call. Cache size must remain 1."""
    enc = _make_encoder({"a": 0, "b": 1})
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        enc([["c"]])
        assert len(enc._unseen_cache) == 1
        enc([["c", "c"]])
        assert len(enc._unseen_cache) == 1, (
            f"expected cache size 1, got {len(enc._unseen_cache)}"
        )


def test_lazy_output_matches_manual():
    """The vector cached for an unseen name must equal what the GloVe
    embedder would produce, and the projected output must equal
    proj(that vector)."""
    enc = _make_encoder({"a": 0, "b": 1})
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        out = enc([["c"]])
    # Manual compute via the same FakeGlove pattern.
    manual = _FakeGlove()(["c"])[0]
    cached = enc._unseen_cache["c"]
    assert torch.allclose(cached.cpu(), manual)
    expected_out = enc.proj(cached.unsqueeze(0).unsqueeze(0))  # [1, 1, channels]
    assert torch.allclose(out, expected_out)


def test_known_type_does_not_populate_cache():
    """Forward with names that are all in node_type_map (or 'mask')
    must leave _unseen_cache empty."""
    enc = _make_encoder({"a": 0, "b": 1})
    enc([["a", "b", "mask"]])
    assert enc._unseen_cache == {}


def test_mask_sentinel_resolves_via_buffer():
    """The 'mask' sentinel was inserted at index num_types in the
    buffer at __init__ time; the name-path must hit the buffer for it,
    not fall through to the lazy cache."""
    enc = _make_encoder({"a": 0, "b": 1})
    out = enc([["mask"]])
    assert out.shape == (1, 1, 32)
    assert "mask" not in enc._unseen_cache
    # Verify the value matches buffer[num_types] -> proj.
    num_types = max(enc._name_to_idx.values()) - 1 + 1  # = 2
    expected = enc.proj(enc.glove_embeddings[num_types].unsqueeze(0).unsqueeze(0))
    assert torch.allclose(out, expected)


def test_name_path_2d_list_shape_preserved():
    """A 3-row x 2-col input list must yield a [3, 2, channels] output
    in row-major order (matches how int-path indexing works)."""
    enc = _make_encoder({"a": 0, "b": 1})
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        out = enc([["a", "b"], ["b", "c"], ["c", "a"]])
    assert out.shape == (3, 2, 32)
    # Row 0 col 0 == buffer[0] proj == buffer["a"] proj.
    expected_a = enc.proj(enc.glove_embeddings[0].unsqueeze(0)).squeeze(0)
    assert torch.allclose(out[0, 0], expected_a)
    # Row 0 col 1 == buffer[1] proj == buffer["b"] proj.
    expected_b = enc.proj(enc.glove_embeddings[1].unsqueeze(0)).squeeze(0)
    assert torch.allclose(out[0, 1], expected_b)
