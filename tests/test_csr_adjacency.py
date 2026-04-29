"""Tests C1-C7 for the CSR adjacency in gfm_data/graph_cache.py."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfm_data.graph_cache import DatasetGraphCache  # noqa: E402
from tests._fake_heterodata import (  # noqa: E402
    FakeEdgeStore,
    FakeHeteroData,
    FakeNodeStore,
    make_toy_graph,
)


# ----------------------------------------------------------------- C1
def test_C1_csr_indptr_and_dst_match_expected():
    """Hand-built toy graph -> CSR matches expected indptr / dst arrays."""
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)

    # A has 4 nodes; with undirected edges A-B:
    #   A0 -> B0, B1
    #   A1 -> B0, B2
    #   A2 -> B1
    #   A3 -> (none)
    block_a = cache.csr["A"]
    assert block_a.indptr.tolist() == [0, 2, 4, 5, 5]
    # B has type id 1 (A=0).
    type_b = cache.node_type_to_index["B"]
    type_a = cache.node_type_to_index["A"]
    # Build (type_id, idx) sorted pairs grouped by source row.
    rows = []
    for s in range(4):
        start, end = block_a.indptr[s], block_a.indptr[s + 1]
        rows.append(sorted(zip(
            block_a.nbr_type_id[start:end].tolist(),
            block_a.nbr_idx[start:end].tolist(),
        )))
    assert rows[0] == sorted([(type_b, 0), (type_b, 1)])
    assert rows[1] == sorted([(type_b, 0), (type_b, 2)])
    assert rows[2] == sorted([(type_b, 1)])
    assert rows[3] == []

    # B has 3 nodes; reverse edges:
    #   B0 -> A0, A1
    #   B1 -> A0, A2
    #   B2 -> A1
    block_b = cache.csr["B"]
    rows_b = []
    for s in range(3):
        start, end = block_b.indptr[s], block_b.indptr[s + 1]
        rows_b.append(sorted(zip(
            block_b.nbr_type_id[start:end].tolist(),
            block_b.nbr_idx[start:end].tolist(),
        )))
    assert rows_b[0] == sorted([(type_a, 0), (type_a, 1)])
    assert rows_b[1] == sorted([(type_a, 0), (type_a, 2)])
    assert rows_b[2] == sorted([(type_a, 1)])


# ----------------------------------------------------------------- C2
def test_C2_roundtrip_dict_csr_dict():
    """dict-of-sets adjacency reconstructed from CSR equals dev-kyaw's adjacency."""
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)

    # dev-kyaw-equivalent dict-of-sets adjacency.
    expected = {
        "A": [set() for _ in range(4)],
        "B": [set() for _ in range(3)],
    }
    ei = g[("A", "to", "B")].edge_index.tolist()
    for s, d in zip(ei[0], ei[1]):
        expected["A"][s].add(("B", d))
        expected["B"][d].add(("A", s))

    for nt in ["A", "B"]:
        n = len(expected[nt])
        for i in range(n):
            assert cache.neighbors_set(nt, i) == expected[nt][i], (nt, i)


# ----------------------------------------------------------------- C3
def test_C3_zero_outdegree_returns_empty():
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)
    # A3 has no edges; B has all 3 nodes wired up; assert specifically.
    block_a = cache.csr["A"]
    assert block_a.indptr[3] == block_a.indptr[4]
    assert cache.neighbors_set("A", 3) == set()


# ----------------------------------------------------------------- C3b
def test_C3b_out_of_bounds_seed_returns_empty():
    """Layer-1 safety net: a seed past the truncated CSR returns
    an empty set instead of IndexError. This is what catches forgotten
    --full_graph for autocomplete tasks (docs/truncated_graph_caveat.md)."""
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)
    # Block A has 4 source rows (0..3); 4 and beyond are OOB.
    n_src_A = cache.csr["A"].num_src
    assert cache.neighbors_set("A", n_src_A) == set()
    assert cache.neighbors_set("A", n_src_A + 100) == set()
    assert cache.neighbors_set("A", -1) == set()


# ----------------------------------------------------------------- C4
def test_C4_dtype_overflow_guard_for_many_types():
    """If we ever exceed int16 type-id range we want an explicit failure.

    int16 covers ~32767 distinct types — far above RelBench v2's max — so
    rather than silently wrapping, we assert dtype assumptions via the
    CSRBlock constructor's checks.
    """
    # Construct a fake graph with a single type, but artificially produce a
    # large nbr_type_id and ensure CSRBlock's dtype assertion triggers.
    from gfm_data.graph_cache import CSRBlock
    indptr = np.array([0, 1], dtype=np.int64)
    nbr_type_id = np.array([5], dtype=np.int32)  # wrong dtype
    nbr_idx = np.array([0], dtype=np.int32)
    with pytest.raises(AssertionError):
        CSRBlock(indptr=indptr, nbr_type_id=nbr_type_id, nbr_idx=nbr_idx)


# ----------------------------------------------------------------- C5
def test_C5_neighbors_set_idempotent():
    """Repeated calls on the same source return identical sets.

    Stand-in for the disk-roundtrip C5: until PR2's npz cache lands, this
    pins the in-memory CSR's read consistency.
    """
    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)
    for nt in ["A", "B"]:
        for i in range(cache.csr[nt].num_src):
            s1 = cache.neighbors_set(nt, i)
            s2 = cache.neighbors_set(nt, i)
            assert s1 == s2 and s1 is not s2


# ----------------------------------------------------------------- C6
def test_C6_dataset_name_prefix_namespacing():
    """``name_prefix`` keeps two datasets in disjoint type namespaces."""
    g = make_toy_graph()
    c1 = DatasetGraphCache(data=g, undirected=True, name_prefix="rel-f1")
    c2 = DatasetGraphCache(data=g, undirected=True, name_prefix="rel-event")

    assert "rel-f1::A" in c1.csr and "rel-f1::B" in c1.csr
    assert "rel-event::A" in c2.csr and "rel-event::B" in c2.csr
    # Distinct namespaces.
    assert set(c1.csr.keys()).isdisjoint(set(c2.csr.keys()))
    # Same content modulo prefix.
    assert c1.neighbors_set("rel-f1::A", 0) == {("rel-f1::B", 0), ("rel-f1::B", 1)}
    assert c2.neighbors_set("rel-event::A", 0) == {("rel-event::B", 0), ("rel-event::B", 1)}


# ----------------------------------------------------------------- C7
def test_C7_equivalence_vs_dev_kyaw_build_adjacency():
    """``cache.neighbors_set`` matches dev-kyaw's ``build_adjacency_hetero`` for every node.

    This is the load-bearing equivalence test: the sampler relies on CSR
    producing the exact same dedup set as dev-kyaw at every src node.
    """
    # Import dev-kyaw's function directly. utils.py imports torch_geometric and
    # relbench at module load; the project conftest mocks both for fast tests,
    # which is fine here because we never call into them.
    from utils import build_adjacency_hetero

    g = make_toy_graph()
    cache = DatasetGraphCache(data=g, undirected=True)
    dev_adj = build_adjacency_hetero(g, undirected=True)

    for nt in g.node_types:
        n = g[nt].num_nodes
        for i in range(n):
            assert cache.neighbors_set(nt, i) == dev_adj[nt][i], (nt, i)
