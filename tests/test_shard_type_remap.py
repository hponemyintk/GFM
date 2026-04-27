"""Regression test for shard type-id remap (multi-dataset training).

Shards are written with per-cache LOCAL type ids (0..N-1 for that
dataset's types). When a unified_type_map is in play (multi-task
training across multiple datasets), the runtime uses the UNIFIED
type ids -- which differ because the union is sorted across all
datasets. Without the remap step in ``_sample_from_shards``, the
shards' "types" array holds local ids that the runtime decodes
against the unified map, returning the wrong type entirely.

These tests exercise:
  1. With a unified_type_map active, the remap table is built and
     correctly translates local -> unified.
  2. Without a unified_type_map (single-dataset / dev-kyaw), the
     remap is None and shard ids pass through unchanged.
  3. ``_sample_from_shards`` actually applies the remap on its
     output.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfm_data.graph_cache import DatasetGraphCache  # noqa: E402
from gfm_data.task_tokens import TaskTokens  # noqa: E402
from tests._fake_heterodata import make_toy_graph  # noqa: E402


def _build_shards_for_cache(cache: DatasetGraphCache, K: int, num_seeds: int,
                            split_dir: str) -> None:
    """Write a tiny memmap shard with per-cache local type ids.

    Mirrors what tools/precompute_shards.py would write for one split.
    Each seed gets random K-neighbor indices/types drawn from the
    cache's local type space [0, len(node_types)).
    """
    from gfm_data.shard_io import ShardWriter
    rng = np.random.default_rng(0)
    n_types = len(cache.node_types)
    os.makedirs(split_dir, exist_ok=True)
    writer = ShardWriter(split_dir, K=K, total_samples=num_seeds,
                         shard_size=num_seeds)
    types = rng.integers(0, n_types, size=(num_seeds, K), dtype=np.int16)
    indices = rng.integers(0, 100, size=(num_seeds, K), dtype=np.int32)
    hops = rng.integers(0, 4, size=(num_seeds, K), dtype=np.int8)
    times = rng.random(size=(num_seeds, K)).astype(np.float32)
    edges = [np.zeros((2, 0), dtype=np.int16) for _ in range(num_seeds)]
    writer.write_shard(0, types, indices, hops, times, edges)
    writer.finalize()


def _make_task_tokens(cache, K: int, shards_dir: str,
                      unified_type_map=None, num_seeds: int = 4) -> TaskTokens:
    """Build a TaskTokens via __new__ to bypass the relbench task plumbing."""
    tok = TaskTokens.__new__(TaskTokens)
    tok.cache = cache
    tok.data = cache.data
    tok.K = K
    tok.split = "val"
    tok.mode = "precomputed_shards"
    tok.shards_dir = shards_dir
    tok.precomputed_dir = None
    tok.precomputed_path = None
    tok.precompute = False
    tok.train_stage = "finetune"
    tok.task = None
    tok.task_id = 0
    tok.task_type_id = 1
    tok.target = None
    tok.time = None
    tok.transform = None
    tok.target_mean = None
    tok.target_std = None
    tok.node_idxs = torch.arange(num_seeds)
    if unified_type_map is not None:
        tok.node_type_to_index = dict(unified_type_map)
        tok.index_to_node_type = {v: k for k, v in unified_type_map.items()}
        tok.node_types = list(unified_type_map.keys())
    else:
        tok.node_type_to_index = dict(cache.node_type_to_index)
        tok.index_to_node_type = dict(cache.index_to_node_type)
        tok.node_types = list(cache.node_types)

    raw_seed_type = cache.prefixed_to_raw[cache.node_types[0]]
    tok.raw_node_type = raw_seed_type
    tok.node_type = cache.raw_to_prefixed[raw_seed_type]

    # Reproduce the parts of __init__ we care about.
    tok._create_global_mappings()

    tok._shard_type_remap = None
    from gfm_data.shard_io import ShardReader
    shard_split_dir = os.path.join(shards_dir, str(K), "val")
    tok._shard_reader = ShardReader(shard_split_dir)
    if unified_type_map is not None:
        max_local = max(cache.node_type_to_index.values()) + 1
        remap = np.zeros(max_local, dtype=np.int64)
        for prefixed, local_idx in cache.node_type_to_index.items():
            remap[local_idx] = unified_type_map[prefixed]
        tok._shard_type_remap = remap
    return tok


def test_unified_type_map_builds_remap():
    """Multi-dataset path: unified_type_map differs from cache local map,
    so a remap table is built."""
    K = 8
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True, name_prefix="rel-foo")
    tmp = tempfile.mkdtemp(prefix="remap_uni_")
    try:
        # Build shards using the cache's local ids.
        _build_shards_for_cache(cache, K, num_seeds=4,
                                split_dir=os.path.join(tmp, str(K), "val"))
        # Build a unified map where the SAME prefixed types are assigned
        # DIFFERENT ids (interleaved with another fake dataset's types).
        unified = {}
        cache_types = list(cache.node_types)
        # alternate cache types with fake "rel-bar::*" types.
        i = 0
        for t in cache_types:
            unified[f"rel-bar::other_{i}"] = i * 2
            unified[t] = i * 2 + 1
            i += 1
        tok = _make_task_tokens(cache, K, tmp, unified_type_map=unified)
        assert tok._shard_type_remap is not None
        # For every type in the cache, remap[local_id] must equal
        # unified[prefixed].
        for prefixed, local_id in cache.node_type_to_index.items():
            assert tok._shard_type_remap[local_id] == unified[prefixed]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_unified_map_means_no_remap():
    """Single-dataset path: shards are decoded directly via local ids,
    so no remap is built."""
    K = 8
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True, name_prefix=None)
    tmp = tempfile.mkdtemp(prefix="remap_none_")
    try:
        _build_shards_for_cache(cache, K, num_seeds=4,
                                split_dir=os.path.join(tmp, str(K), "val"))
        tok = _make_task_tokens(cache, K, tmp, unified_type_map=None)
        assert tok._shard_type_remap is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sample_from_shards_applies_remap():
    """Production path: ``_sample_from_shards`` returns remapped types
    when a unified map is in play."""
    K = 8
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True, name_prefix="rel-foo")
    tmp = tempfile.mkdtemp(prefix="remap_sample_")
    try:
        _build_shards_for_cache(cache, K, num_seeds=4,
                                split_dir=os.path.join(tmp, str(K), "val"))
        # Choose an offset that guarantees every cache-local id has a
        # different unified id (offset by +100).
        unified = {t: i + 100 for i, t in enumerate(cache.node_types)}
        tok = _make_task_tokens(cache, K, tmp, unified_type_map=unified)
        # Read seed 0 raw -- types should be in [0, n_types).
        raw = tok._shard_reader.read(0)["types"].astype(np.int64)
        assert raw.min() >= 0 and raw.max() < len(cache.node_types)
        # Now via the production path -- types should be shifted by +100.
        sample = tok._sample_from_shards(0)
        out_types = sample["types"].numpy()
        assert (out_types == raw + 100).all(), (
            f"remap not applied: raw={raw}, out={out_types}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
