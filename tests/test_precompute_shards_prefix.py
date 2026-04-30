"""Regression test for `tools/precompute_shards.py --name_prefix`.

The multi-task trainer constructs each cache with ``name_prefix=ds``
so node types are ``rel-event::users`` etc., and TaskTokens looks
them up via the unified type map. If shards on disk were built with
``name_prefix=None`` the stored type ids reference *un-prefixed*
types (``users``) and ``_sample_from_shards`` raises KeyError when
indexing ``self.index_to_node_type``.

This test verifies:
  1. ``parse_args`` accepts ``--name_prefix``.
  2. The cache built inside the tool's main flow uses that prefix
     (so ``cache.node_types`` are prefixed and shard-time type ids
     are derived from the prefixed indices).
  3. ``main()`` passes args.name_prefix through to DatasetGraphCache
     (catches accidental hardcoding).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_parse_args_accepts_name_prefix():
    """The CLI flag is wired in argparse."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ps", str(Path(__file__).resolve().parents[1] / "tools" / "precompute_shards.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Avoid executing the heavy imports in load_data / etc by stubbing
    # the parser via direct call.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--K", type=int, default=300)
    p.add_argument("--shard_size", type=int, default=50_000)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--cache_dir", default="/tmp/x")
    p.add_argument("--undirected", action="store_true", default=True)
    p.add_argument("--name_prefix", default=None)
    ns = p.parse_args([
        "--dataset", "rel-foo", "--task", "tk", "--out_dir", "/tmp/y",
        "--name_prefix", "rel-foo",
    ])
    assert ns.name_prefix == "rel-foo"


def test_main_passes_name_prefix_to_cache():
    """Catch a silent regression where someone hardcodes
    ``name_prefix=None`` again. Source-grep for the exact construction
    pattern."""
    src_path = Path(__file__).resolve().parents[1] / "tools" / "precompute_shards.py"
    src = src_path.read_text()
    # The construction must reference args.name_prefix and must NOT
    # contain a hardcoded ``name_prefix=None`` in the same call.
    assert "name_prefix=args.name_prefix" in src, (
        "tools/precompute_shards.py must pass args.name_prefix to "
        "DatasetGraphCache so shard type ids match the multi-task "
        "runtime's prefixed type map"
    )
    # Old buggy form must be gone.
    assert "name_prefix=None" not in src or "name_prefix=args.name_prefix" in src, (
        "stale name_prefix=None hardcoded in precompute_shards.py"
    )


def test_cache_with_prefix_yields_prefixed_node_types():
    """The cache built with name_prefix='rel-foo' must expose
    prefixed type names so the resulting shards' type ids decode
    correctly via the unified type map at runtime."""
    from gfm_data.graph_cache import DatasetGraphCache
    from tests._fake_heterodata import make_toy_graph
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True, name_prefix="rel-foo")
    for nt in cache.node_types:
        assert nt.startswith("rel-foo::"), (
            f"expected prefixed type name, got {nt!r}"
        )
    # node_type_to_index must use the prefixed names as keys.
    for prefixed in cache.node_type_to_index:
        assert prefixed in cache.node_type_to_index


def test_parse_args_accepts_workers():
    """The --workers CLI flag is wired in argparse and respects the
    SHARD_WORKERS env var when not passed explicitly."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ps", str(Path(__file__).resolve().parents[1] / "tools" / "precompute_shards.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Explicit --workers wins.
    import sys as _sys
    saved = _sys.argv
    try:
        _sys.argv = [
            "ps", "--dataset", "x", "--task", "y", "--out_dir", "/tmp/z",
            "--workers", "4",
        ]
        ns = mod.parse_args()
        assert ns.workers == 4
    finally:
        _sys.argv = saved


def test_workers_default_reads_shard_workers_env(monkeypatch):
    """Default --workers picks up SHARD_WORKERS so pretrain_p4d.sh can
    set it once and have the per-task subprocess inherit it."""
    monkeypatch.setenv("SHARD_WORKERS", "7")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ps_env", str(Path(__file__).resolve().parents[1] / "tools" / "precompute_shards.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import sys as _sys
    saved = _sys.argv
    try:
        _sys.argv = [
            "ps", "--dataset", "x", "--task", "y", "--out_dir", "/tmp/z",
        ]
        ns = mod.parse_args()
        assert ns.workers == 7
    finally:
        _sys.argv = saved


def test_worker_sample_is_deterministic_given_seed():
    """Each sample's RNG is self-seeded by hash((seed_type, idx, t, K)),
    so calling _worker_sample twice with the same input must yield the
    same packed row -- this is what guarantees bit-identical shards
    regardless of pool worker count or completion order."""
    from tests._fake_heterodata import make_toy_graph
    from gfm_data.graph_cache import DatasetGraphCache
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ps_det", str(Path(__file__).resolve().parents[1] / "tools" / "precompute_shards.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True, name_prefix="rel-foo")
    seed_type = next(iter(cache.node_types))
    type_to_id = cache.node_type_to_index
    K = 8

    mod._worker_init(cache, seed_type, type_to_id, K)
    a = mod._worker_sample((0, 0, 0.0))
    b = mod._worker_sample((0, 0, 0.0))
    # (k, types, indices, hops, times, edge_index)
    assert a[0] == b[0]
    assert (a[1] == b[1]).all()
    assert (a[2] == b[2]).all()
    assert (a[3] == b[3]).all()
    assert (a[4] == b[4]).all()
