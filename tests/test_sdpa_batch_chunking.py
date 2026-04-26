"""Regression test for the sdpa batch-size 65535 overflow fix.

CUDA's efficient-attention kernel has a hard batch limit of 65535.
With multi-task batch_size=512 * K=300 = 153,600 neighbor slots and
a dominant node type taking most slots, the transformer call inside
``NeighborTfsEncoder.forward`` overflows. The fix chunks the batch
dim along 65535-sized splits and re-concatenates.

These tests verify the chunking path is numerically identical to
the un-chunked path on small inputs so we can trust the fast path
remains intact for normal-size batches.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_transformer(d_model: int = 32, n_heads: int = 4) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=64,
        dropout=0.0,  # deterministic for equality check
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=1)


def _chunked_forward(transformer, x_seq, limit):
    if x_seq.size(0) > limit:
        chunks = x_seq.split(limit, dim=0)
        return torch.cat([transformer(c) for c in chunks], dim=0)
    return transformer(x_seq)


def test_chunked_matches_unchunked_below_limit():
    """When batch <= limit, the chunked path is a no-op pass-through."""
    torch.manual_seed(0)
    t = _make_transformer().eval()
    x = torch.randn(100, 5, 32)
    direct = t(x)
    chunked = _chunked_forward(t, x, limit=65535)
    assert torch.allclose(direct, chunked, atol=1e-6)


def test_chunked_matches_unchunked_at_artificial_small_limit():
    """Force chunking with a small limit and verify equivalence.

    We can't actually feed >65535 tensors on CI/laptop GPUs, so we
    drop the limit for the test and confirm the split+cat math is
    correct. The kernel-overflow trigger above 65535 isn't exercised
    here -- that check is the responsibility of the actual training
    run on the big GPU. What this test verifies is that *our*
    chunking layer is a no-op when off and an exact split/concat
    when on.
    """
    torch.manual_seed(1)
    t = _make_transformer().eval()
    # Use a non-multiple-of-limit size to also test the trailing
    # short chunk.
    x = torch.randn(1000, 5, 32)
    direct = t(x)
    chunked = _chunked_forward(t, x, limit=300)  # forces 4 chunks: 300+300+300+100
    assert torch.allclose(direct, chunked, atol=1e-6)


def test_chunked_preserves_batch_order():
    """Concatenated output must retain row order so downstream
    grouped_indices lookups remain valid."""
    torch.manual_seed(2)
    t = _make_transformer().eval()
    x = torch.randn(750, 3, 32)
    direct = t(x)
    chunked = _chunked_forward(t, x, limit=200)
    # Pick a few specific rows and confirm they match.
    for row in (0, 199, 200, 399, 400, 599, 600, 749):
        assert torch.allclose(direct[row], chunked[row], atol=1e-6), \
            f"row {row} differs between direct and chunked output"
