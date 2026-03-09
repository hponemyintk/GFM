"""
Tests for neighbor sampling pipeline (utils.py).

Sections:
  1. build_adjacency_hetero   – baseline correctness
  2. gather_1_and_2_hop       – temporal filtering, hop structure
  3. _process_one_seed        – token output, edge index, determinism
  4. build_adjacency_csr      – new CSR format (skipped until implemented)
  5. gather vectorized         – new vectorized gather (skipped until implemented)
  6. ML sanity checks          – statistical / semantic invariants

NOTE: conftest.py mocks heavy deps (sentence_transformers, relbench) so
      imports are fast and tests don't need network/GPU.
"""

import sys, os
import pytest
import random
import numpy as np
import torch

# conftest.py mocks torch_geometric, sentence_transformers, relbench
# BEFORE utils is imported — no heavy imports needed here.
from utils import (
    build_adjacency_hetero,
    gather_1_and_2_hop_with_seed_time,
    _process_one_seed,
    init_worker_globals,
)

# New implementations — will be skipped until they exist
try:
    from utils import build_adjacency_csr
    HAS_CSR = True
except ImportError:
    HAS_CSR = False

try:
    from utils import gather_1_and_2_hop_vectorized
    HAS_VEC_GATHER = True
except ImportError:
    HAS_VEC_GATHER = False

needs_csr = pytest.mark.skipif(not HAS_CSR, reason="build_adjacency_csr not yet implemented")
needs_vec = pytest.mark.skipif(not HAS_VEC_GATHER, reason="gather_1_and_2_hop_vectorized not yet implemented")


# =====================================================================
# Section 1 — build_adjacency_hetero (baseline)
# =====================================================================
class TestBuildAdjacencyHetero:

    def test_all_node_types_present(self, small_hetero, small_adj):
        for nt in small_hetero.node_types:
            assert nt in small_adj

    def test_correct_list_length_per_type(self, small_hetero, small_adj):
        for nt in small_hetero.node_types:
            assert len(small_adj[nt]) == small_hetero[nt].num_nodes

    def test_entries_are_sets(self, small_adj):
        for nt in small_adj:
            for nbrs in small_adj[nt]:
                assert isinstance(nbrs, set)

    def test_undirected_symmetry(self, small_hetero, small_adj):
        for nt in small_hetero.node_types:
            for i, nbrs in enumerate(small_adj[nt]):
                for (nbr_t, nbr_i) in nbrs:
                    assert (nt, i) in small_adj[nbr_t][nbr_i], \
                        f"Missing reverse edge ({nbr_t},{nbr_i})->({nt},{i})"

    def test_directed_no_reverse(self, small_hetero, small_adj_directed):
        adj = small_adj_directed
        assert ("product", 0) in adj["user"][0]
        assert ("user", 0) not in adj["product"][0]

    def test_known_neighbors_user0(self, small_adj):
        nbrs = small_adj["user"][0]
        assert nbrs == {("product", 0), ("product", 1)}

    def test_known_neighbors_product1(self, small_adj):
        nbrs = small_adj["product"][1]
        assert nbrs == {("user", 0), ("user", 1), ("category", 0)}

    def test_known_neighbors_category0(self, small_adj):
        nbrs = small_adj["category"][0]
        assert nbrs == {("product", 0), ("product", 1), ("product", 6)}

    def test_isolated_type_empty(self, isolated_hetero):
        adj = build_adjacency_hetero(isolated_hetero, undirected=True)
        for i in range(isolated_hetero["Z"].num_nodes):
            assert len(adj["Z"][i]) == 0

    def test_total_edge_count_undirected(self, small_hetero, small_adj):
        total = sum(len(s) for nt in small_adj for s in small_adj[nt])
        num_directed_edges = sum(
            small_hetero[et].edge_index.shape[1]
            for et in small_hetero.edge_types
        )
        assert total == num_directed_edges * 2


# =====================================================================
# Section 2 — gather_1_and_2_hop_with_seed_time
# =====================================================================
class TestGather1And2Hop:

    # --- Temporal causality ---

    def test_temporal_causality_1hop(self, small_hetero, small_adj):
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 300.0
        )
        for (nbr_t, nbr_i, hop, _, _) in result:
            if hop == 1 and hasattr(small_hetero[nbr_t], "time"):
                assert small_hetero[nbr_t].time[nbr_i].item() <= 300.0

    def test_temporal_causality_2hop(self, small_hetero, small_adj):
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 300.0
        )
        for (nbr_t, nbr_i, hop, _, _) in result:
            if hop == 2 and hasattr(small_hetero[nbr_t], "time"):
                assert small_hetero[nbr_t].time[nbr_i].item() <= 300.0

    def test_strict_time_excludes_future(self, small_hetero, small_adj):
        """user[0] at t=100: only product[0] (t=50) passes; product[1] (t=150) fails."""
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 100.0
        )
        one_hop = {(t, i) for (t, i, h, _, _) in result if h == 1}
        assert ("product", 0) in one_hop
        assert ("product", 1) not in one_hop

    # --- Hop structure ---

    def test_no_overlap_1hop_2hop(self, small_hetero, small_adj):
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 300.0
        )
        one_hop = {(t, i) for (t, i, h, _, _) in result if h == 1}
        two_hop = {(t, i) for (t, i, h, _, _) in result if h == 2}
        assert one_hop.isdisjoint(two_hop)

    def test_seed_not_in_results(self, small_hetero, small_adj):
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 300.0
        )
        for (nbr_t, nbr_i, _, _, _) in result:
            assert not (nbr_t == "user" and nbr_i == 0)

    def test_hop_values_only_1_or_2(self, small_hetero, small_adj):
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 500.0
        )
        for (_, _, hop, _, _) in result:
            assert hop in {1, 2}

    # --- Known results ---

    def test_known_1hop_user0_t300(self, small_hetero, small_adj):
        """product[0] (t=50) and product[1] (t=150) both <= 300."""
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 300.0
        )
        one_hop = {(t, i) for (t, i, h, _, _) in result if h == 1}
        assert one_hop == {("product", 0), ("product", 1)}

    def test_known_2hop_user0_t300(self, small_hetero, small_adj):
        """
        Via product[0]: category[0] (no time), user[0] (self, excluded)
        Via product[1]: category[0] (already in 2-hop), user[1] (t=200<=300)
                        user[0] (self, excluded)
        2-hop set = {category[0], user[1]}
        """
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 300.0
        )
        two_hop = {(t, i) for (t, i, h, _, _) in result if h == 2}
        assert ("category", 0) in two_hop
        assert ("user", 1) in two_hop

    # --- Relative time ---

    def test_relative_time_values(self, small_hetero, small_adj):
        seed_time = 300.0
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, seed_time
        )
        for (nbr_t, nbr_i, _, rel_time, _) in result:
            if hasattr(small_hetero[nbr_t], "time"):
                nbr_time = small_hetero[nbr_t].time[nbr_i].item()
                expected = (seed_time - nbr_time) / (60 * 60 * 24)
                assert abs(rel_time - expected) < 1e-6
            else:
                assert rel_time == 0.0

    # --- No-time nodes ---

    def test_no_time_nodes_included(self, small_hetero, small_adj):
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 300.0
        )
        cat_nbrs = [(t, i) for (t, i, h, _, _) in result if t == "category"]
        assert len(cat_nbrs) > 0

    # --- Connecting 1-hops ---

    def test_2hop_has_connecting_set(self, small_hetero, small_adj):
        result = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 300.0
        )
        for (_, _, hop, _, c1hops) in result:
            if hop == 2:
                assert isinstance(c1hops, set) and len(c1hops) > 0
            elif hop == 1:
                assert c1hops is None

    # --- Threshold ---

    def test_1hop_threshold_cap(self, hub_hetero):
        adj = build_adjacency_hetero(hub_hetero, undirected=True)
        result = gather_1_and_2_hop_with_seed_time(
            adj, hub_hetero, "A", 0, 99999.0,
            max_1hop_threshold=50
        )
        one_hop = [r for r in result if r[2] == 1]
        assert len(one_hop) <= 50

    # --- Empty ---

    def test_empty_for_isolated_node(self, isolated_hetero):
        adj = build_adjacency_hetero(isolated_hetero, undirected=True)
        result = gather_1_and_2_hop_with_seed_time(
            adj, isolated_hetero, "Z", 0, 99999.0
        )
        assert len(result) == 0


# =====================================================================
# Section 3 — _process_one_seed
# =====================================================================
class TestProcessOneSeed:

    @pytest.fixture(autouse=True)
    def _globals(self, worker_globals_small):
        pass

    def _run(self, data, K, node_type, node_idx, seed_time, seed_val=None):
        if seed_val is None:
            seed_val = hash((node_type, node_idx, seed_time, K)) & 0xffffffff
        return _process_one_seed((K, node_type, node_idx, seed_time, seed_val))

    # --- Shape / format ---

    def test_output_has_K_tokens(self, small_hetero):
        K = 5
        _, _, tokens, _ = self._run(small_hetero, K, "user", 0, 300.0)
        assert len(tokens) == K

    def test_seed_is_first_with_hop0(self, small_hetero):
        _, _, tokens, _ = self._run(small_hetero, 5, "user", 2, 400.0)
        t_str, t_idx, hop, t_val, _ = tokens[0]
        assert (t_str, t_idx, hop, t_val) == ("user", 2, 0, 0.0)

    def test_edge_index_shape_2xE(self, small_hetero):
        _, _, _, ei = self._run(small_hetero, 5, "user", 0, 300.0)
        assert ei.ndim == 2
        assert ei.shape[0] == 2

    def test_edge_index_range_valid(self, small_hetero):
        K = 5
        _, _, tokens, ei = self._run(small_hetero, K, "user", 0, 300.0)
        if ei.size > 0:
            assert ei.min() >= 0
            assert ei.max() < len(tokens)

    def test_edge_index_no_self_loops(self, small_hetero):
        _, _, _, ei = self._run(small_hetero, 5, "user", 0, 300.0)
        if ei.shape[1] > 0:
            assert not np.any(ei[0] == ei[1])

    # --- Edge validity ---

    def test_edges_match_real_adjacency(self, small_hetero, small_adj):
        K = 5
        _, _, tokens, ei = self._run(small_hetero, K, "user", 0, 300.0)
        for e in range(ei.shape[1]):
            src_j, dst_j = ei[0, e], ei[1, e]
            src_t, src_i = tokens[src_j][0], tokens[src_j][1]
            dst_t, dst_i = tokens[dst_j][0], tokens[dst_j][1]
            assert (dst_t, dst_i) in small_adj[src_t][src_i]

    # --- Determinism ---

    def test_deterministic_same_seed(self, small_hetero):
        r1 = self._run(small_hetero, 5, "user", 0, 300.0, seed_val=42)
        r2 = self._run(small_hetero, 5, "user", 0, 300.0, seed_val=42)
        assert r1[2] == r2[2]
        np.testing.assert_array_equal(r1[3], r2[3])

    def test_different_seeds_same_start(self, small_hetero):
        r1 = self._run(small_hetero, 5, "user", 0, 300.0, seed_val=42)
        r2 = self._run(small_hetero, 5, "user", 0, 300.0, seed_val=99)
        # Both start with same seed node
        assert r1[2][0][:2] == r2[2][0][:2] == ("user", 0)

    # --- Fallback ---

    def test_fallback_hop3_for_isolated(self):
        import utils
        from torch_geometric.data import HeteroData as MockHeteroData
        iso_data = MockHeteroData()
        iso_data["X"].num_nodes = 3
        iso_data["X"].time = torch.tensor([100.0, 200.0, 300.0])
        iso_data["Y"].num_nodes = 2
        iso_data["Z"].num_nodes = 2
        iso_data["Z"].time = torch.tensor([50.0, 150.0])
        iso_data["X", "rel", "Y"].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )
        adj = build_adjacency_hetero(iso_data, undirected=True)
        all_nodes = []
        for nt in iso_data.node_types:
            for i in range(iso_data[nt].num_nodes):
                all_nodes.append((nt, i))
        init_worker_globals(adj, all_nodes, data=iso_data)
        utils.GLOBAL_TIME_ARRAYS = None  # use legacy path

        K = 4
        _, _, tokens, _ = _process_one_seed(
            (K, "Z", 0, 99999.0, 42)
        )
        assert len(tokens) == K
        assert tokens[0][2] == 0  # seed hop
        for tok in tokens[1:]:
            assert tok[2] == 3  # fallback hop

    # --- Return type consistency ---

    def test_return_node_type_matches(self, small_hetero):
        nt, idx, _, _ = self._run(small_hetero, 5, "user", 0, 300.0)
        assert nt == "user"
        assert idx == 0


# =====================================================================
# Section 4 — build_adjacency_csr (new, TDD)
# =====================================================================
@needs_csr
class TestBuildAdjacencyCSR:

    def test_all_node_types_present(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=True)
        for nt in small_hetero.node_types:
            assert nt in csr

    def test_required_keys(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=True)
        for nt in csr:
            for key in ("nbr_types", "nbr_indices", "offsets"):
                assert key in csr[nt], f"Missing key '{key}' for type '{nt}'"

    def test_offsets_length(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=True)
        for nt in small_hetero.node_types:
            assert len(csr[nt]["offsets"]) == small_hetero[nt].num_nodes + 1

    def test_offsets_start_zero(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=True)
        for nt in csr:
            assert csr[nt]["offsets"][0] == 0

    def test_offsets_monotonic(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=True)
        for nt in csr:
            assert np.all(np.diff(csr[nt]["offsets"]) >= 0)

    def test_array_lengths_consistent(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=True)
        for nt in csr:
            n = len(csr[nt]["nbr_types"])
            assert len(csr[nt]["nbr_indices"]) == n
            assert csr[nt]["offsets"][-1] == n

    def test_dtypes(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=True)
        for nt in csr:
            assert csr[nt]["nbr_types"].dtype == np.int8
            assert csr[nt]["nbr_indices"].dtype == np.int32
            assert csr[nt]["offsets"].dtype == np.int64

    def test_equivalence_with_old(self, small_hetero, small_adj):
        """CSR must represent the exact same graph as dict-of-sets."""
        csr = build_adjacency_csr(small_hetero, undirected=True)
        type_to_idx = {nt: i for i, nt in enumerate(small_hetero.node_types)}
        idx_to_type = {i: nt for nt, i in type_to_idx.items()}

        for nt in small_hetero.node_types:
            for node_i in range(small_hetero[nt].num_nodes):
                old_nbrs = small_adj[nt][node_i]

                start = csr[nt]["offsets"][node_i]
                end = csr[nt]["offsets"][node_i + 1]
                csr_nbrs = set()
                for k in range(start, end):
                    nbr_type_str = idx_to_type[int(csr[nt]["nbr_types"][k])]
                    nbr_idx = int(csr[nt]["nbr_indices"][k])
                    csr_nbrs.add((nbr_type_str, nbr_idx))

                assert old_nbrs == csr_nbrs, \
                    f"Mismatch at {nt}[{node_i}]: old={old_nbrs} csr={csr_nbrs}"

    def test_undirected_symmetry(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=True)
        type_to_idx = {nt: i for i, nt in enumerate(small_hetero.node_types)}
        idx_to_type = {i: nt for nt, i in type_to_idx.items()}

        for nt in small_hetero.node_types:
            for node_i in range(small_hetero[nt].num_nodes):
                start = csr[nt]["offsets"][node_i]
                end = csr[nt]["offsets"][node_i + 1]
                for k in range(start, end):
                    nbr_t = idx_to_type[int(csr[nt]["nbr_types"][k])]
                    nbr_i = int(csr[nt]["nbr_indices"][k])
                    # check reverse
                    r_start = csr[nbr_t]["offsets"][nbr_i]
                    r_end = csr[nbr_t]["offsets"][nbr_i + 1]
                    r_types = csr[nbr_t]["nbr_types"][r_start:r_end]
                    r_idxs = csr[nbr_t]["nbr_indices"][r_start:r_end]
                    found = any(
                        idx_to_type[int(r_types[j])] == nt and int(r_idxs[j]) == node_i
                        for j in range(len(r_types))
                    )
                    assert found

    def test_directed_no_reverse(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=False)
        type_to_idx = {nt: i for i, nt in enumerate(small_hetero.node_types)}
        idx_to_type = {i: nt for nt, i in type_to_idx.items()}

        # product[0] should NOT have user[0] in directed mode
        start = csr["product"]["offsets"][0]
        end = csr["product"]["offsets"][1]
        nbrs = set()
        for k in range(start, end):
            nbrs.add((idx_to_type[int(csr["product"]["nbr_types"][k])],
                       int(csr["product"]["nbr_indices"][k])))
        assert ("user", 0) not in nbrs

    def test_isolated_zero_degree(self, isolated_hetero):
        csr = build_adjacency_csr(isolated_hetero, undirected=True)
        for i in range(isolated_hetero["Z"].num_nodes):
            assert csr["Z"]["offsets"][i] == csr["Z"]["offsets"][i + 1]

    def test_total_edge_count_undirected(self, small_hetero):
        csr = build_adjacency_csr(small_hetero, undirected=True)
        total = sum(csr[nt]["offsets"][-1] for nt in csr)
        num_directed = sum(
            small_hetero[et].edge_index.shape[1]
            for et in small_hetero.edge_types
        )
        assert total == num_directed * 2

    def test_hub_node_degree(self, hub_hetero):
        """A[0] connects to all 200 B nodes (undirected adds B->A too)."""
        csr = build_adjacency_csr(hub_hetero, undirected=True)
        deg_a0 = csr["A"]["offsets"][1] - csr["A"]["offsets"][0]
        assert deg_a0 == 200


# =====================================================================
# Section 5 — gather_1_and_2_hop_vectorized (new, TDD)
# =====================================================================
@needs_vec
class TestGatherVectorized:

    def _make_helpers(self, data):
        csr = build_adjacency_csr(data, undirected=True)
        type_to_idx = {nt: i for i, nt in enumerate(data.node_types)}
        time_arrays = {}
        for nt in data.node_types:
            if hasattr(data[nt], "time"):
                time_arrays[nt] = data[nt].time.numpy()
        return csr, type_to_idx, time_arrays

    def _gather(self, data, node_type, node_idx, seed_time, **kwargs):
        csr, tti, ta = self._make_helpers(data)
        return gather_1_and_2_hop_vectorized(
            csr, ta, tti, data.node_types,
            node_type, node_idx, seed_time, **kwargs
        )

    # --- Temporal ---

    def test_temporal_causality(self, small_hetero):
        result = self._gather(small_hetero, "user", 0, 300.0)
        csr, tti, ta = self._make_helpers(small_hetero)
        for (nbr_t, nbr_i, hop, _, _) in result:
            if nbr_t in ta:
                assert ta[nbr_t][nbr_i] <= 300.0

    def test_strict_time_excludes_future(self, small_hetero):
        result = self._gather(small_hetero, "user", 0, 100.0)
        one_hop = {(t, i) for (t, i, h, _, _) in result if h == 1}
        assert ("product", 0) in one_hop
        assert ("product", 1) not in one_hop

    # --- Hop structure ---

    def test_no_1hop_2hop_overlap(self, small_hetero):
        result = self._gather(small_hetero, "user", 0, 300.0)
        one_hop = {(t, i) for (t, i, h, _, _) in result if h == 1}
        two_hop = {(t, i) for (t, i, h, _, _) in result if h == 2}
        assert one_hop.isdisjoint(two_hop)

    def test_seed_not_in_results(self, small_hetero):
        result = self._gather(small_hetero, "user", 0, 300.0)
        for (t, i, _, _, _) in result:
            assert not (t == "user" and i == 0)

    # --- Equivalence with old ---

    def test_equivalence_with_old_user0_t300(self, small_hetero, small_adj):
        old = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 300.0
        )
        new = self._gather(small_hetero, "user", 0, 300.0)
        old_set = {(t, i, h) for (t, i, h, _, _) in old}
        new_set = {(t, i, h) for (t, i, h, _, _) in new}
        assert old_set == new_set

    def test_equivalence_with_old_user0_t100(self, small_hetero, small_adj):
        old = gather_1_and_2_hop_with_seed_time(
            small_adj, small_hetero, "user", 0, 100.0
        )
        new = self._gather(small_hetero, "user", 0, 100.0)
        old_set = {(t, i, h) for (t, i, h, _, _) in old}
        new_set = {(t, i, h) for (t, i, h, _, _) in new}
        assert old_set == new_set

    def test_equivalence_all_users(self, small_hetero, small_adj):
        for uid in range(small_hetero["user"].num_nodes):
            st = small_hetero["user"].time[uid].item()
            old = gather_1_and_2_hop_with_seed_time(
                small_adj, small_hetero, "user", uid, st
            )
            new = self._gather(small_hetero, "user", uid, st)
            old_set = {(t, i, h) for (t, i, h, _, _) in old}
            new_set = {(t, i, h) for (t, i, h, _, _) in new}
            assert old_set == new_set, f"Mismatch for user[{uid}] t={st}"

    # --- Relative time ---

    def test_relative_time(self, small_hetero):
        csr, tti, ta = self._make_helpers(small_hetero)
        seed_time = 300.0
        result = gather_1_and_2_hop_vectorized(
            csr, ta, tti, small_hetero.node_types,
            "user", 0, seed_time
        )
        for (nbr_t, nbr_i, _, rel_time, _) in result:
            if nbr_t in ta:
                expected = (seed_time - ta[nbr_t][nbr_i]) / 86400.0
                assert abs(rel_time - expected) < 1e-6
            else:
                assert rel_time == 0.0

    # --- Edge cases ---

    def test_empty_for_isolated(self, isolated_hetero):
        result = self._gather(isolated_hetero, "Z", 0, 99999.0)
        assert len(result) == 0

    def test_threshold_cap(self, hub_hetero):
        result = self._gather(hub_hetero, "A", 0, 99999.0, max_1hop_threshold=50)
        one_hop = [r for r in result if r[2] == 1]
        assert len(one_hop) <= 50


# =====================================================================
# Section 6 — ML Sanity Checks
# =====================================================================
class TestMLSanity:
    """Statistical and semantic invariants for the sampling pipeline."""

    @pytest.fixture(autouse=True)
    def _globals(self, worker_globals_small):
        pass

    def _run(self, data, K, node_type, node_idx, seed_time):
        seed_val = hash((node_type, node_idx, seed_time, K)) & 0xffffffff
        return _process_one_seed((K, node_type, node_idx, seed_time, seed_val))

    # --- Temporal causality across all seeds ---

    def test_no_future_neighbors_any_user(self, small_hetero):
        K = 5
        for uid in range(small_hetero["user"].num_nodes):
            st = small_hetero["user"].time[uid].item()
            _, _, tokens, _ = self._run(small_hetero, K, "user", uid, st)
            for (t_str, t_idx, hop, _, _) in tokens[1:]:
                if hop in {1, 2} and hasattr(small_hetero[t_str], "time"):
                    nbr_time = small_hetero[t_str].time[t_idx].item()
                    assert nbr_time <= st, \
                        f"Temporal violation: user[{uid}] t={st}, " \
                        f"neighbor ({t_str},{t_idx}) t={nbr_time}"

    # --- Hop values ---

    def test_hop_values_in_valid_range(self, small_hetero):
        K = 5
        for uid in range(small_hetero["user"].num_nodes):
            st = small_hetero["user"].time[uid].item()
            _, _, tokens, _ = self._run(small_hetero, K, "user", uid, st)
            for (_, _, hop, _, _) in tokens:
                assert hop in {0, 1, 2, 3}

    def test_exactly_one_hop0_token(self, small_hetero):
        K = 5
        for uid in range(small_hetero["user"].num_nodes):
            st = small_hetero["user"].time[uid].item()
            _, _, tokens, _ = self._run(small_hetero, K, "user", uid, st)
            hop0 = [t for t in tokens if t[2] == 0]
            assert len(hop0) == 1

    # --- Relative time ---

    def test_relative_time_nonneg_for_hop12(self, small_hetero):
        K = 5
        for uid in range(small_hetero["user"].num_nodes):
            st = small_hetero["user"].time[uid].item()
            _, _, tokens, _ = self._run(small_hetero, K, "user", uid, st)
            for (t_str, _, hop, rel_time, _) in tokens:
                if hop in {1, 2} and hasattr(small_hetero[t_str], "time"):
                    assert rel_time >= -1e-9

    # --- Edge validity ---

    def test_all_subgraph_edges_within_range(self, small_hetero):
        K = 5
        for uid in range(small_hetero["user"].num_nodes):
            st = small_hetero["user"].time[uid].item()
            _, _, tokens, ei = self._run(small_hetero, K, "user", uid, st)
            if ei.shape[1] > 0:
                assert ei.min() >= 0
                assert ei.max() < len(tokens)

    def test_connected_seeds_have_edges(self, small_hetero):
        """user[0] at t=300 has neighbors — subgraph should have edges."""
        _, _, _, ei = self._run(small_hetero, 5, "user", 0, 300.0)
        assert ei.shape[1] > 0

    # --- Type diversity ---

    def test_neighbor_type_diversity(self, small_hetero):
        """user[0] at t=500 can reach product + category — expect >=2 types."""
        _, _, tokens, _ = self._run(small_hetero, 8, "user", 0, 500.0)
        types_seen = {t_str for (t_str, _, _, _, _) in tokens}
        assert len(types_seen) >= 2

    # --- No duplicates (when sampling without replacement) ---

    def test_no_duplicate_nodes_when_enough(self, small_hetero):
        """With K small enough relative to neighborhood, no duplicates."""
        K = 4
        _, _, tokens, _ = self._run(small_hetero, K, "user", 0, 500.0)
        node_ids = [(t, i) for (t, i, _, _, _) in tokens]
        assert len(node_ids) == len(set(node_ids))

    # --- Correctness of returned metadata ---

    def test_returned_node_type_and_idx(self, small_hetero):
        nt, idx, _, _ = self._run(small_hetero, 5, "user", 3, 400.0)
        assert nt == "user"
        assert idx == 3


# =====================================================================
# Section 7 — CSR-path integration tests for _process_one_seed
# =====================================================================
class TestProcessOneSeedCSR:
    """Tests _process_one_seed when globals use CSR adjacency + time_arrays."""

    @pytest.fixture(autouse=True)
    def _globals(self, worker_globals_csr):
        pass

    def _run(self, K, node_type, node_idx, seed_time, seed_val=None):
        if seed_val is None:
            seed_val = hash((node_type, node_idx, seed_time, K)) & 0xffffffff
        return _process_one_seed((K, node_type, node_idx, seed_time, seed_val))

    def test_output_has_K_tokens(self):
        _, _, tokens, _ = self._run(5, "user", 0, 300.0)
        assert len(tokens) == 5

    def test_seed_is_first_with_hop0(self):
        _, _, tokens, _ = self._run(5, "user", 2, 400.0)
        t_str, t_idx, hop, t_val, _ = tokens[0]
        assert (t_str, t_idx, hop, t_val) == ("user", 2, 0, 0.0)

    def test_edge_index_shape_and_range(self):
        K = 5
        _, _, tokens, ei = self._run(K, "user", 0, 300.0)
        assert ei.ndim == 2 and ei.shape[0] == 2
        if ei.size > 0:
            assert ei.min() >= 0
            assert ei.max() < len(tokens)

    def test_edge_index_no_self_loops(self):
        _, _, _, ei = self._run(5, "user", 0, 300.0)
        if ei.shape[1] > 0:
            assert not np.any(ei[0] == ei[1])

    def test_edges_match_real_adjacency(self, small_hetero, small_adj):
        """Edges from CSR path must correspond to real graph edges."""
        _, _, tokens, ei = self._run(5, "user", 0, 300.0)
        for e in range(ei.shape[1]):
            src_j, dst_j = ei[0, e], ei[1, e]
            src_t, src_i = tokens[src_j][0], tokens[src_j][1]
            dst_t, dst_i = tokens[dst_j][0], tokens[dst_j][1]
            assert (dst_t, dst_i) in small_adj[src_t][src_i]

    def test_deterministic_same_seed(self):
        r1 = self._run(5, "user", 0, 300.0, seed_val=42)
        r2 = self._run(5, "user", 0, 300.0, seed_val=42)
        assert r1[2] == r2[2]
        np.testing.assert_array_equal(r1[3], r2[3])

    def test_connected_seeds_have_edges(self):
        _, _, _, ei = self._run(5, "user", 0, 300.0)
        assert ei.shape[1] > 0

    def test_temporal_causality_all_users(self, small_hetero):
        K = 5
        for uid in range(small_hetero["user"].num_nodes):
            st = small_hetero["user"].time[uid].item()
            _, _, tokens, _ = self._run(K, "user", uid, st)
            for (t_str, t_idx, hop, _, _) in tokens[1:]:
                if hop in {1, 2} and hasattr(small_hetero[t_str], "time"):
                    assert small_hetero[t_str].time[t_idx].item() <= st

    def test_hop_values_valid(self, small_hetero):
        K = 5
        for uid in range(small_hetero["user"].num_nodes):
            st = small_hetero["user"].time[uid].item()
            _, _, tokens, _ = self._run(K, "user", uid, st)
            for (_, _, hop, _, _) in tokens:
                assert hop in {0, 1, 2, 3}

    def test_neighbor_type_diversity(self):
        _, _, tokens, _ = self._run(8, "user", 0, 500.0)
        types_seen = {t_str for (t_str, _, _, _, _) in tokens}
        assert len(types_seen) >= 2
