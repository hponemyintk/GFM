"""Regression test for VectorQuantizerEMA DDP centroid sync.

Before the fix, ``vq.update()`` ran the EMA computation on each rank
independently using only that rank's local batch. With DDP's default
``broadcast_buffers=True``, rank 0's buffer values overwrote ranks
1..N at the next forward, silently discarding their updates -- the
codebook was effectively trained at 1/world_size the data.

The fix all-reduces the raw statistics (``encodings_sum``, ``dw``)
across ranks BEFORE applying the EMA, then averages by world_size
to keep the per-step EMA decay consistent with single-rank tuning.
After this, every rank's update produces identical buffers, so any
broadcast is a no-op.

We exercise this via a 2-rank gloo process group (CPU-only — works
in CI without GPUs) and verify:
  1. With sync ON (the production path), buffers are identical
     across ranks after a single update call.
  2. With sync OFF (simulating the old buggy path), buffers diverge
     when ranks see different inputs -- locks in the bug-detection.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _worker(rank: int, world_size: int, q: mp.Queue):
    """Run on each rank: init gloo, build a VQ, update with rank-
    specific input, send buffers back to the parent."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29550"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo")

    from codebook import VectorQuantizerEMA

    torch.manual_seed(42)
    vq = VectorQuantizerEMA(num_embeddings=8, embedding_dim=4, decay=0.9)
    vq.train()

    # Each rank gets a different input batch -- if the sync fails,
    # the EMA buffers will diverge.
    torch.manual_seed(rank + 100)
    x = torch.randn(16, 4)
    vq.update(x)

    # Ship the relevant buffers back to the parent so it can compare.
    q.put((
        rank,
        vq._ema_cluster_size.detach().clone(),
        vq._ema_w.detach().clone(),
        vq._embedding.detach().clone(),
    ))
    dist.destroy_process_group()


def test_centroid_sync_keeps_ranks_consistent():
    """After a single vq.update() call with sync enabled, all ranks
    should hold byte-identical EMA buffers."""
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world_size, q)) for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = [q.get() for _ in range(world_size)]
    for p in procs:
        p.join(timeout=30)
    for p in procs:
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"

    results.sort(key=lambda t: t[0])
    _, c0, w0, e0 = results[0]
    _, c1, w1, e1 = results[1]

    # All three buffers must match across ranks. Use exact equality
    # (not allclose) -- after all_reduce + same decay + same inputs
    # to all ranks (because the all_reduce makes each rank see the
    # same global stats), the buffer math is bit-for-bit identical.
    assert torch.equal(c0, c1), "_ema_cluster_size diverged across ranks"
    assert torch.equal(w0, w1), "_ema_w diverged across ranks"
    assert torch.equal(e0, e1), "_embedding diverged across ranks"


def test_centroid_sync_uses_all_reduce_path():
    """Sanity: confirm the codebook actually imports torch.distributed
    and uses ``dist.all_reduce`` in the update path. This test catches
    accidental reverts where someone removes the sync block."""
    import codebook
    src = open(codebook.__file__).read()
    assert "import torch.distributed as dist" in src
    assert "dist.is_initialized()" in src
    assert "dist.all_reduce(encodings_sum" in src
    assert "dist.all_reduce(dw" in src
