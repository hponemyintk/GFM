"""Tests S1-S17 for the CSR sampler in gfm_data/sampler.py.

Bit-equivalence vs dev-kyaw is checked by importing ``utils`` directly; the
two implementations share the same Python process and PYTHONHASHSEED so set
iteration order is identical for identical-element sets.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfm_data.graph_cache import DatasetGraphCache  # noqa: E402
from gfm_data.sampler import (  # noqa: E402
    gather_1_and_2_hop,
    sample_local_subgraph,
)
from tests._fake_heterodata import (  # noqa: E402
    FakeEdgeStore,
    FakeHeteroData,
    FakeNodeStore,
    make_dense_graph,
    make_toy_graph,
)


# Helpers --------------------------------------------------------------------
def _multiset(tokens):
    """Comparable multiset of (type, idx, hop) for token equivalence checks."""
    return sorted((t, i, h) for (t, i, h, _t, _c) in tokens)


def _dev_kyaw_process_one_seed(adjacency, all_nodes, data, K, seed_type,
                                seed_idx, seed_t, seed_val):
    """Drive dev-kyaw's _process_one_seed without going through Pool.

    utils.py uses module-level globals that are normally set by the worker
    initializer; here we set them explicitly. Returns (final_tokens, edge_index).
    """
    import utils
    utils.GLOBAL_ADJ = adjacency
    utils.GLOBAL_ALL_NODES = all_nodes
    args = (data, K, seed_type, seed_idx, seed_t, seed_val)
    seed_node_type, seed_node_idx, final_tokens, edge_index = utils._process_one_seed(args)
    return final_tokens, edge_index


def _dev_kyaw_setup(g):
    from utils import build_adjacency_hetero
    adj = build_adjacency_hetero(g, undirected=True)
    all_nodes = []
    for nt in g.node_types:
        for i in range(g[nt].num_nodes):
            all_nodes.append((nt, i))
    return adj, all_nodes


# S1 -------------------------------------------------------------------------
def test_S1_time_filter_on_1hop():
    """All gathered 1-hop neighbors have time <= seed_time."""
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)
    # Seed A0 at time=15 -> only B0 (t=10) qualifies; B1 (t=20) excluded.
    random.seed(0)
    out = gather_1_and_2_hop(cache, "A", 0, seed_time=15.0)
    one_hop = [(t, i) for (t, i, h, _r, _c) in out if h == 1]
    assert ("B", 0) in one_hop
    assert ("B", 1) not in one_hop


# S2 -------------------------------------------------------------------------
def test_S2_1hop_cap_5000():
    """When 1-hop count exceeds threshold, exactly threshold are returned."""
    # Build a star: A0 connected to all 8000 B nodes, all time=0.
    n_b = 8000
    a = FakeNodeStore(num_nodes=1)
    b_time = torch.zeros(n_b)
    b = FakeNodeStore(num_nodes=n_b, time=b_time)
    src = torch.zeros(n_b, dtype=torch.long)
    dst = torch.arange(n_b, dtype=torch.long)
    g = FakeHeteroData(
        node_stores={"A": a, "B": b},
        edge_stores={("A", "to", "B"): FakeEdgeStore(torch.stack([src, dst]))},
    )
    cache = DatasetGraphCache(data=g, undirected=True)
    random.seed(123)
    out1 = gather_1_and_2_hop(cache, "A", 0, seed_time=10.0,
                              max_1hop_threshold=5000, max_2hop_threshold=0)
    one_hop = [(t, i) for (t, i, h, _r, _c) in out1 if h == 1]
    assert len(one_hop) == 5000
    # Reproducible under same seed.
    random.seed(123)
    out2 = gather_1_and_2_hop(cache, "A", 0, seed_time=10.0,
                              max_1hop_threshold=5000, max_2hop_threshold=0)
    one_hop_2 = [(t, i) for (t, i, h, _r, _c) in out2 if h == 1]
    assert sorted(one_hop) == sorted(one_hop_2)


# S3 -------------------------------------------------------------------------
def test_S3_2hop_cap_per_parent():
    """2-hop cap is applied per 1-hop parent (matches utils.py:95-96)."""
    # Build: A0 - B0; B0 has 1500 neighbors of type C with all-zero time.
    a = FakeNodeStore(num_nodes=1)
    b_time = torch.tensor([0.0])
    b = FakeNodeStore(num_nodes=1, time=b_time)
    n_c = 1500
    c_time = torch.zeros(n_c)
    c = FakeNodeStore(num_nodes=n_c, time=c_time)
    g = FakeHeteroData(
        node_stores={"A": a, "B": b, "C": c},
        edge_stores={
            ("A", "to", "B"): FakeEdgeStore(
                torch.tensor([[0], [0]], dtype=torch.long)),
            ("B", "to", "C"): FakeEdgeStore(
                torch.stack([
                    torch.zeros(n_c, dtype=torch.long),
                    torch.arange(n_c, dtype=torch.long),
                ])),
        },
    )
    cache = DatasetGraphCache(data=g, undirected=True)
    random.seed(42)
    out1 = gather_1_and_2_hop(cache, "A", 0, seed_time=10.0,
                              max_1hop_threshold=5000, max_2hop_threshold=1000)
    two_hop_1 = [(t, i) for (t, i, h, _r, _c) in out1 if h == 2]
    # Cap respected: at most 1000 (the 2-hop cap). May be slightly fewer because
    # the parent B0 also reverse-links to seed A0 under undirected=True; A0 can
    # be drawn into the 1000-sample and then filtered out as a self-loop
    # (matches utils.py:95-103).
    assert 990 <= len(two_hop_1) <= 1000
    # Reproducible under same seed.
    random.seed(42)
    out2 = gather_1_and_2_hop(cache, "A", 0, seed_time=10.0,
                              max_1hop_threshold=5000, max_2hop_threshold=1000)
    two_hop_2 = [(t, i) for (t, i, h, _r, _c) in out2 if h == 2]
    assert sorted(two_hop_1) == sorted(two_hop_2)


# S4, S5 ---------------------------------------------------------------------
def test_S4_S5_seed_token_format():
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)
    final, _ = sample_local_subgraph(cache, K=5, seed_node_type="A",
                                     seed_node_idx=0, seed_time=100.0,
                                     seed_val=7)
    assert len(final) == 5
    # S4: seed first, hop=0, rel_time=0.
    seed = final[0]
    assert seed[0] == "A"
    assert seed[1] == 0
    assert seed[2] == 0
    assert seed[3] == 0.0
    # S5: tuple shape and value ranges for all tokens.
    for (t, i, h, r, _c) in final:
        assert isinstance(t, str)
        assert isinstance(i, int)
        assert h in (0, 1, 2, 3)
        assert r >= 0


# S6 -------------------------------------------------------------------------
def test_S6_edge_index_is_K_local():
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)
    final, eidx = sample_local_subgraph(cache, K=5, seed_node_type="A",
                                        seed_node_idx=0, seed_time=100.0,
                                        seed_val=11)
    assert eidx.shape[0] == 2
    if eidx.shape[1] > 0:
        assert int(eidx.min()) >= 0
        assert int(eidx.max()) < len(final)


# S7 -------------------------------------------------------------------------
def test_S7_zero_neighbors_triggers_fallback_hop3():
    """Seed in an isolated graph -> all chosen tokens get hop=3."""
    a = FakeNodeStore(num_nodes=1)
    b = FakeNodeStore(num_nodes=3, time=torch.zeros(3))
    g = FakeHeteroData(
        node_stores={"A": a, "B": b},
        edge_stores={("A", "to", "B"): FakeEdgeStore(
            torch.zeros((2, 0), dtype=torch.long))},
    )
    cache = DatasetGraphCache(data=g, undirected=True)
    final, _ = sample_local_subgraph(cache, K=4, seed_node_type="A",
                                     seed_node_idx=0, seed_time=10.0,
                                     seed_val=99)
    # Seed at position 0 keeps hop=0; rest must all be hop=3.
    hops = [h for (_t, _i, h, _r, _c) in final]
    assert hops[0] == 0
    assert all(h == 3 for h in hops[1:])


# S8 -------------------------------------------------------------------------
def test_S8_determinism_same_seed_same_output():
    g = make_dense_graph()
    cache = DatasetGraphCache(data=g, undirected=True)
    out_a = sample_local_subgraph(cache, K=20, seed_node_type="A",
                                  seed_node_idx=3, seed_time=50.0, seed_val=2024)
    out_b = sample_local_subgraph(cache, K=20, seed_node_type="A",
                                  seed_node_idx=3, seed_time=50.0, seed_val=2024)
    assert _multiset(out_a[0]) == _multiset(out_b[0])
    assert np.array_equal(out_a[1], out_b[1])


# S9 -------------------------------------------------------------------------
def test_S9_gather_phase_dedup_multipath():
    """Multi-path 1-hop dedup: parallel edges A0-B0 produce one entry."""
    a = FakeNodeStore(num_nodes=1)
    b = FakeNodeStore(num_nodes=1, time=torch.tensor([0.0]))
    # Two parallel edges A0-B0.
    edges_ab = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    g = FakeHeteroData(
        node_stores={"A": a, "B": b},
        edge_stores={("A", "to", "B"): FakeEdgeStore(edges_ab)},
    )
    cache = DatasetGraphCache(data=g, undirected=True)
    random.seed(0)
    out = gather_1_and_2_hop(cache, "A", 0, seed_time=10.0)
    one_hop = [(t, i) for (t, i, h, _r, _c) in out if h == 1]
    assert one_hop == [("B", 0)]


# S10 ------------------------------------------------------------------------
def test_S10_gather_phase_2hop_intersect_1hop_dropped():
    """A node reachable as both 1-hop and 2-hop appears only as 1-hop."""
    # A0-B0, B0-A1, A0-A1 (triangle): A1 is 1-hop AND 2-hop of A0.
    a = FakeNodeStore(num_nodes=2)
    b = FakeNodeStore(num_nodes=1, time=torch.tensor([0.0]))
    edges_ab = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    edges_aa = torch.tensor([[0], [1]], dtype=torch.long)
    g = FakeHeteroData(
        node_stores={"A": a, "B": b},
        edge_stores={
            ("A", "to", "B"): FakeEdgeStore(edges_ab),
            ("A", "self", "A"): FakeEdgeStore(edges_aa),
        },
    )
    cache = DatasetGraphCache(data=g, undirected=True)
    random.seed(0)
    out = gather_1_and_2_hop(cache, "A", 0, seed_time=10.0)
    one_hop = [(t, i) for (t, i, h, _r, _c) in out if h == 1]
    two_hop = [(t, i) for (t, i, h, _r, _c) in out if h == 2]
    assert ("A", 1) in one_hop
    assert ("A", 1) not in two_hop


# S11 ------------------------------------------------------------------------
def test_S11_gather_phase_self_loop_excluded():
    """A 2-hop path back to seed is dropped (matches utils.py:102-103)."""
    # A0-B0, B0-A0 (undirected). 2-hop A0->B0->A0 must be skipped.
    a = FakeNodeStore(num_nodes=1)
    b = FakeNodeStore(num_nodes=1, time=torch.tensor([0.0]))
    edges_ab = torch.tensor([[0], [0]], dtype=torch.long)
    g = FakeHeteroData(
        node_stores={"A": a, "B": b},
        edge_stores={("A", "to", "B"): FakeEdgeStore(edges_ab)},
    )
    cache = DatasetGraphCache(data=g, undirected=True)
    random.seed(0)
    out = gather_1_and_2_hop(cache, "A", 0, seed_time=10.0)
    two_hop = [(t, i) for (t, i, h, _r, _c) in out if h == 2]
    assert ("A", 0) not in two_hop


# S12, S13, S14, S15 ---------------------------------------------------------
@pytest.mark.parametrize("regime", ["gt", "eq", "lt", "zero"])
def test_S12_S15_selection_regimes(regime):
    """Selection-phase cases match dev-kyaw bit-for-bit under fixed seed."""
    g = make_dense_graph(seed=11)
    cache = DatasetGraphCache(data=g, undirected=True)
    dev_adj, dev_all = _dev_kyaw_setup(g)

    # Pick K so we hit each regime by varying which seed we pick.
    seed_type = "A"
    if regime == "zero":
        # Build an isolated A node by clearing edges.
        a = FakeNodeStore(num_nodes=1)
        b = FakeNodeStore(num_nodes=1, time=torch.tensor([0.0]))
        g2 = FakeHeteroData(
            node_stores={"A": a, "B": b},
            edge_stores={("A", "to", "B"): FakeEdgeStore(
                torch.zeros((2, 0), dtype=torch.long))},
        )
        cache = DatasetGraphCache(data=g2, undirected=True)
        dev_adj, dev_all = _dev_kyaw_setup(g2)
        K = 5
        seed_idx = 0
    elif regime == "lt":
        K = 100  # large K relative to small per-node degree
        seed_idx = 7
    elif regime == "eq":
        # We rig K to exactly match (size_th + 1).
        random.seed(0)
        out = gather_1_and_2_hop(cache, seed_type, 7, seed_time=200.0)
        K = len(out) + 1
        seed_idx = 7
    else:  # "gt"
        K = 5
        seed_idx = 7

    # Run new and dev-kyaw under the same seed_val and compare multisets.
    seed_val = 31337
    final_new, eidx_new = sample_local_subgraph(
        cache, K=K, seed_node_type=seed_type, seed_node_idx=seed_idx,
        seed_time=200.0, seed_val=seed_val,
    )
    final_dev, eidx_dev = _dev_kyaw_process_one_seed(
        dev_adj, dev_all, cache.data, K, seed_type, seed_idx, 200.0, seed_val,
    )

    assert len(final_new) == K
    assert len(final_dev) == K
    assert _multiset(final_new) == _multiset(final_dev), regime

    # Selection invariants per regime:
    if regime == "zero":
        # All non-seed tokens must be hop=3.
        for tok in final_new[1:]:
            assert tok[2] == 3
    if regime == "lt":
        # random.choices semantics -> at least one duplicate exists in chosen.
        keys = [(t, i) for (t, i, h, _r, _c) in final_new[1:]]
        assert len(keys) != len(set(keys)), "expected duplicates from random.choices"


# S16 ------------------------------------------------------------------------
def test_S16_equivalence_vs_dev_kyaw_on_dense_graph():
    """End-to-end equivalence vs dev-kyaw across many seeds.

    This is the rel-f1-stand-in: rel-f1's full graph is too heavy for a unit
    test (loads relbench), so we use a dense synthetic graph with similar
    branching factor and assert dev-kyaw vs new produce the same multisets
    across 50 seeds spanning all regimes.
    """
    g = make_dense_graph(seed=7)
    cache = DatasetGraphCache(data=g, undirected=True)
    dev_adj, dev_all = _dev_kyaw_setup(g)

    K = 12
    n_a = g["A"].num_nodes
    rng = np.random.default_rng(0)
    seed_idxs = rng.integers(0, n_a, size=50)
    seed_times = rng.uniform(50.0, 100.0, size=50)

    for s_idx, s_t in zip(seed_idxs.tolist(), seed_times.tolist()):
        seed_val = hash(("A", int(s_idx), float(s_t), K)) & 0xFFFFFFFF
        f_new, _ = sample_local_subgraph(cache, K=K, seed_node_type="A",
                                         seed_node_idx=int(s_idx),
                                         seed_time=float(s_t),
                                         seed_val=seed_val)
        f_dev, _ = _dev_kyaw_process_one_seed(
            dev_adj, dev_all, cache.data, K, "A", int(s_idx),
            float(s_t), seed_val,
        )
        assert _multiset(f_new) == _multiset(f_dev), (s_idx, s_t)


# S17 ------------------------------------------------------------------------
def test_S17_per_regime_equivalence_vs_dev_kyaw():
    """Each of the four regimes hit on a hand-crafted toy graph."""
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)
    dev_adj, dev_all = _dev_kyaw_setup(g)

    cases = [
        # (K, seed_type, seed_idx, seed_time, regime_name)
        (3, "A", 0, 50.0, "gt_or_eq"),    # A0 has 2 1-hop neighbors at t<50
        (10, "A", 0, 50.0, "lt"),         # K-1=9 > num candidates -> random.choices
        (5, "A", 3, 50.0, "zero"),        # A3 is isolated -> fallback
    ]

    for K, st, si, t, name in cases:
        seed_val = 4242 + si
        f_new, _ = sample_local_subgraph(cache, K=K, seed_node_type=st,
                                         seed_node_idx=si, seed_time=t,
                                         seed_val=seed_val)
        f_dev, _ = _dev_kyaw_process_one_seed(
            dev_adj, dev_all, cache.data, K, st, si, t, seed_val,
        )
        assert _multiset(f_new) == _multiset(f_dev), name
