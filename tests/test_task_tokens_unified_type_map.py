"""Regression test for the multi-dataset unified_type_map KeyError.

When TaskTokens is constructed with ``unified_type_map`` covering types
from multiple datasets, ``_create_global_mappings`` must NOT iterate
the whole unified map -- the cache is per-dataset and only knows its
own types. Iterating the whole map crashed with::

    KeyError: 'rel-event::users'   (when self.cache is the rel-f1 cache)

Fix: iterate only ``self.cache.node_types``, look up the type id
through the unified ``self.node_type_to_index``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfm_data.graph_cache import DatasetGraphCache  # noqa: E402
from gfm_data.task_tokens import TaskTokens  # noqa: E402
from tests._fake_heterodata import make_toy_graph  # noqa: E402


def test_unified_type_map_does_not_iterate_other_datasets():
    """Build two caches with disjoint prefixes, give a TaskTokens the
    rel-foo cache + a unified map that includes both, and verify the
    global mapping was built without crashing on rel-bar types."""
    g = make_toy_graph()
    cache_foo = DatasetGraphCache(data=g, undirected=True, name_prefix="rel-foo")
    cache_bar = DatasetGraphCache(data=g, undirected=True, name_prefix="rel-bar")

    # Build a unified vocabulary across both datasets.
    unified: dict = {}
    for c in (cache_foo, cache_bar):
        for t in c.node_types:
            if t not in unified:
                unified[t] = len(unified)
    # Sanity: it should contain both prefixes.
    assert any(t.startswith("rel-foo::") for t in unified)
    assert any(t.startswith("rel-bar::") for t in unified)

    # Construct a minimal TaskTokens directly (bypass relbench dependency).
    # We don't run __init__ -- we set just the attributes
    # _create_global_mappings reads.
    tok = TaskTokens.__new__(TaskTokens)
    tok.cache = cache_foo
    tok.node_type_to_index = dict(unified)  # full unified vocabulary
    tok.index_to_node_type = {i: t for t, i in unified.items()}
    tok.node_types = list(unified.keys())

    # Should not crash even though index_to_node_type has rel-bar:: keys
    # not in cache_foo.prefixed_to_raw.
    tok._create_global_mappings()

    # Verify the mapping covers exactly the foo cache's nodes.
    # NOTE: ``_total_global_nodes`` replaced the old
    # ``len(type_local_to_global)`` / ``len(global_to_type_local)``
    # check after the dict was switched to a (small) offset table to
    # avoid the rel-event 22 GiB-per-instance OOM.
    foo_total = cache_foo.num_nodes_total()
    assert tok._total_global_nodes == foo_total

    # Every type id in the offset table uses a UNIFIED type id (i.e.,
    # the same id this type would have in the rel-bar TaskTokens).
    for type_idx in tok._type_offset:
        prefixed = tok.index_to_node_type[type_idx]
        assert prefixed in cache_foo.node_types  # was iterated correctly
        assert tok.node_type_to_index[prefixed] == type_idx


def test_single_dataset_path_unaffected():
    """The single-dataset path (no unified_type_map) should still work
    with the new iteration. Maps over cache.node_types yields the same
    set as the per-cache index_to_node_type."""
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True, name_prefix=None)
    tok = TaskTokens.__new__(TaskTokens)
    tok.cache = cache
    tok.node_type_to_index = dict(cache.node_type_to_index)
    tok.index_to_node_type = dict(cache.index_to_node_type)
    tok.node_types = list(cache.node_types)

    tok._create_global_mappings()
    assert tok._total_global_nodes == cache.num_nodes_total()


# ---------------------------------------------------------------------------
# Regression tests for the proxy classes (introduced when the per-node
# global-id dicts were replaced with an offset table). The reverse proxy's
# bisect logic was previously broken at every type boundary including
# global_idx=0; the forward proxy silently aliased on out-of-range local_idx.

def test_offset_proxy_bounds_check():
    """``type_local_to_global[(t, l)]`` must KeyError for out-of-range l,
    matching the old dict's semantics."""
    from gfm_data.task_tokens import _OffsetProxy
    proxy = _OffsetProxy(offset={0: 0, 1: 5, 2: 8}, sizes={0: 5, 1: 3, 2: 2})

    assert proxy[(0, 0)] == 0
    assert proxy[(0, 4)] == 4   # last valid entry of type 0
    assert proxy[(1, 0)] == 5
    assert proxy[(2, 1)] == 9

    import pytest
    with pytest.raises(KeyError):
        _ = proxy[(0, 5)]   # past type 0's size
    with pytest.raises(KeyError):
        _ = proxy[(0, -1)]  # negative
    with pytest.raises(KeyError):
        _ = proxy[(99, 0)]  # unknown type

    assert (0, 0) in proxy
    assert (0, 5) not in proxy
    assert (99, 0) not in proxy


def test_reverse_offset_proxy_at_type_boundaries():
    """Regression: prior implementation used bisect_right(self._sorted,
    (global_idx,)) which lands too far left at every type boundary
    because Python tuple comparison treats (5,) < (5, 2) as True.
    With the (global_idx, math.inf) sentinel it lands correctly."""
    from gfm_data.task_tokens import _ReverseOffsetProxy
    # 3 types: type 0 spans [0, 5), type 2 spans [5, 10), type 3 spans [10, 12).
    proxy = _ReverseOffsetProxy(
        offset={0: 0, 2: 5, 3: 10},
        sizes={0: 5, 2: 5, 3: 2},
    )

    # Each boundary case the old buggy version got wrong.
    assert proxy[0] == (0, 0)    # was: KeyError
    assert proxy[4] == (0, 4)    # last entry of type 0
    assert proxy[5] == (2, 0)    # was: (0, 5) -- aliased into type 0
    assert proxy[9] == (2, 4)    # last entry of type 2
    assert proxy[10] == (3, 0)   # was: (2, 5) -- aliased into type 2
    assert proxy[11] == (3, 1)   # last entry of type 3

    # Out-of-range past the last type must KeyError.
    import pytest
    with pytest.raises(KeyError):
        _ = proxy[12]
    with pytest.raises(KeyError):
        _ = proxy[100]


def test_proxies_round_trip_consistent():
    """For every valid (type, local), forward then reverse returns
    the original pair."""
    from gfm_data.task_tokens import _OffsetProxy, _ReverseOffsetProxy
    offset = {0: 0, 1: 7, 5: 11}
    sizes = {0: 7, 1: 4, 5: 9}
    fwd = _OffsetProxy(offset, sizes)
    rev = _ReverseOffsetProxy(offset, sizes)
    for t, n in sizes.items():
        for l in range(n):
            g = fwd[(t, l)]
            assert rev[g] == (t, l), f"({t},{l}) round-trips through g={g} to {rev[g]}"
