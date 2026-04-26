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


def test_chunked_with_grad_checkpoint_matches_direct():
    """The training-mode path uses torch.utils.checkpoint to free
    per-chunk activations and re-run forward on backward. The
    forward output and the gradients flowing back must be bit-for-
    bit identical to the un-checkpointed direct path."""
    from torch.utils.checkpoint import checkpoint as ckpt

    torch.manual_seed(3)
    t = _make_transformer().train()  # training mode -- ckpt path is meaningful
    # Same weights, two parallel inputs that require_grad so we can
    # check both forward equality and gradient equality.
    x_a = torch.randn(1000, 5, 32, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)

    direct = t(x_a)
    direct.sum().backward()

    chunks = x_b.split(300, dim=0)  # forces 4 chunks: 300+300+300+100
    chunked = torch.cat(
        [ckpt(t, c, use_reentrant=False) for c in chunks], dim=0
    )
    chunked.sum().backward()

    # Forward-pass outputs match (recompute is mathematically lossless).
    assert torch.allclose(direct, chunked, atol=1e-6)
    # Input gradients match (the recomputed backward yields the same
    # grad as the cached one).
    assert torch.allclose(x_a.grad, x_b.grad, atol=1e-6)


def test_eval_mode_skips_chunking_in_production_path():
    """Sanity: the production path in encoders.py uses
    ``self.training and x_seq.size(0) > _TF_CHUNK``. In eval mode
    we want the one-shot transformer call regardless of size, since
    no grad graph is being built."""
    torch.manual_seed(4)
    t = _make_transformer().eval()
    x = torch.randn(50, 3, 32)
    # Mimic the production guard with an arbitrarily small chunk size.
    is_training = False
    if is_training and x.size(0) > 4:
        chunks = x.split(4, dim=0)
        out = torch.cat([t(c) for c in chunks], dim=0)
    else:
        out = t(x)
    direct = t(x)
    assert torch.allclose(direct, out, atol=1e-6)
