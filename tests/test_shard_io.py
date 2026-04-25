"""Tests Sh1-Sh4 for gfm_data/shard_io.py."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfm_data.shard_io import ShardReader, ShardWriter  # noqa: E402


def _synth_samples(n: int, K: int, rng):
    """Return arrays for n samples + per-sample edge lists."""
    types = rng.integers(0, 8, size=(n, K), dtype=np.int16)
    indices = rng.integers(0, 1000, size=(n, K), dtype=np.int32)
    hops = rng.integers(0, 4, size=(n, K), dtype=np.int8)
    times = rng.standard_normal(size=(n, K)).astype(np.float32)
    edges_per_sample = []
    for _ in range(n):
        e = rng.integers(2, 6)
        edges_per_sample.append(rng.integers(0, K, size=(2, int(e)), dtype=np.int16))
    return types, indices, hops, times, edges_per_sample


# ----------------------------------------------------------------- Sh1
def test_Sh1_write_read_roundtrip_bit_identical():
    rng = np.random.default_rng(0)
    n, K = 128, 16
    types, indices, hops, times, edges = _synth_samples(n, K, rng)
    with tempfile.TemporaryDirectory() as root:
        w = ShardWriter(root, K=K, total_samples=n, shard_size=64)
        # Two shards: 0..64, 64..128.
        for s in range(w.num_shards):
            lo, hi = w.shard_range(s)
            w.write_shard(
                shard_idx=s,
                types=types[lo:hi],
                indices=indices[lo:hi],
                hops=hops[lo:hi],
                times=times[lo:hi],
                edges_per_sample=edges[lo:hi],
            )
        w.finalize()

        r = ShardReader(root)
        assert len(r) == n
        for i in range(n):
            s = r.read(i)
            assert np.array_equal(s["types"], types[i])
            assert np.array_equal(s["indices"], indices[i])
            assert np.array_equal(s["hops"], hops[i])
            assert np.array_equal(s["times"], times[i])
            assert np.array_equal(s["edge_index"], edges[i].astype(np.int16, copy=False))


# ----------------------------------------------------------------- Sh2
def test_Sh2_sample_at_shard_boundary():
    """Sample 99 (last of shard 0), 100 (first of shard 1) read correctly."""
    rng = np.random.default_rng(1)
    n, K = 200, 8
    types, indices, hops, times, edges = _synth_samples(n, K, rng)
    with tempfile.TemporaryDirectory() as root:
        w = ShardWriter(root, K=K, total_samples=n, shard_size=100)
        for s in range(w.num_shards):
            lo, hi = w.shard_range(s)
            w.write_shard(s, types[lo:hi], indices[lo:hi], hops[lo:hi],
                          times[lo:hi], edges[lo:hi])
        w.finalize()
        r = ShardReader(root)
        for i in (99, 100, 101):
            s = r.read(i)
            assert np.array_equal(s["types"], types[i])
            assert np.array_equal(s["edge_index"], edges[i].astype(np.int16, copy=False))


# ----------------------------------------------------------------- Sh3
def test_Sh3_empty_edges_handled():
    K, n = 8, 4
    rng = np.random.default_rng(2)
    types, indices, hops, times, _ = _synth_samples(n, K, rng)
    edges = [np.zeros((2, 0), dtype=np.int16) for _ in range(n)]
    with tempfile.TemporaryDirectory() as root:
        w = ShardWriter(root, K=K, total_samples=n, shard_size=n)
        w.write_shard(0, types, indices, hops, times, edges)
        w.finalize()
        r = ShardReader(root)
        for i in range(n):
            s = r.read(i)
            assert s["edge_index"].shape == (2, 0)


# ----------------------------------------------------------------- Sh4
def test_Sh4_concurrent_reads_match_single_reader():
    """Two ShardReaders on the same root return identical samples."""
    rng = np.random.default_rng(3)
    n, K = 64, 8
    types, indices, hops, times, edges = _synth_samples(n, K, rng)
    with tempfile.TemporaryDirectory() as root:
        w = ShardWriter(root, K=K, total_samples=n, shard_size=32)
        for s in range(w.num_shards):
            lo, hi = w.shard_range(s)
            w.write_shard(s, types[lo:hi], indices[lo:hi], hops[lo:hi],
                          times[lo:hi], edges[lo:hi])
        w.finalize()
        r1 = ShardReader(root)
        r2 = ShardReader(root)
        for i in range(n):
            s1 = r1.read(i)
            s2 = r2.read(i)
            assert np.array_equal(s1["types"], s2["types"])
            assert np.array_equal(s1["edge_index"], s2["edge_index"])
