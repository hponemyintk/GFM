"""CSR-backed local-subgraph sampler.

This module is a **line-for-line port** of dev-kyaw's
``utils.gather_1_and_2_hop_with_seed_time`` and
``utils._process_one_seed`` (commit ``0359e08``), with the only change that
adjacency lookups go through ``DatasetGraphCache`` instead of an in-memory
``dict[node_type] -> list[set]``.

Dedup semantics (gather phase) and final K-1 selection semantics (sample
phase) are preserved exactly so that under fixed ``random.seed`` the new
sampler returns the same multiset of ``(type, idx, hop)`` tuples as
dev-kyaw — see tests/test_csr_sampler.py for the equivalence proofs.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from gfm_data.graph_cache import DatasetGraphCache

# A neighbor token: (node_type, node_idx, hop, relative_time_days, connecting_1hops)
# - hop ∈ {0, 1, 2, 3} where 3 == "global fallback"
# - connecting_1hops is ``None`` for 1-hop / fallback / seed; a ``set`` of 1-hop
#   ``(type, idx)`` pairs for 2-hop tokens (carried for the global module).
NbrToken = Tuple[str, int, int, float, Optional[Set[Tuple[str, int]]]]


# ------------------------------------------------------------- gather phase
def gather_1_and_2_hop(
    cache: DatasetGraphCache,
    node_type: str,
    node_idx: int,
    seed_time: float,
    max_1hop_threshold: int = 5000,
    max_2hop_threshold: int = 1000,
    out_neighbor_cache: Optional[Dict[Tuple[str, int], Set[Tuple[str, int]]]] = None,
) -> List[NbrToken]:
    """Mirror of ``utils.gather_1_and_2_hop_with_seed_time``.

    De-duplication semantics (set-based) are preserved:

    * a 1-hop neighbor reachable via multiple edges appears once in ``n1``
    * a node that is both 1-hop and 2-hop appears only as 1-hop
    * a 2-hop path that loops back to the seed is excluded

    Time filter: a neighbor is included iff ``data[nbr_t].time[nbr_i] <= seed_time``
    when ``data[nbr_t]`` has a ``time`` attribute; otherwise included unconditionally.

    ``out_neighbor_cache``: optional dict the caller can pass to capture the
    neighbor sets fetched during gather (the seed and every kept 1-hop).
    ``sample_local_subgraph`` reuses this in the edge-construction phase to
    skip its ~K redundant ``cache.neighbors_set`` calls.
    """
    # Resolve dict-of-arrays once so the inner loops just do scalar
    # numpy access -- no PyTorch dispatch, no .item() boxing.
    time_by_prefixed = cache.time_by_prefixed
    has_time_by_prefixed = cache.has_time_by_prefixed

    # ---- 1-hop candidates ----
    n1_full_set = cache.neighbors_set(node_type, node_idx)
    if out_neighbor_cache is not None:
        out_neighbor_cache[(node_type, node_idx)] = n1_full_set
    if len(n1_full_set) > max_1hop_threshold:
        # NOTE: random.sample(list(set), k) — list ordering is hash-determined
        # (CPython) and matches dev-kyaw for identical-element sets.
        n1_full = random.sample(list(n1_full_set), max_1hop_threshold)
    else:
        n1_full = list(n1_full_set)

    n1: Set[Tuple[str, int]] = set()
    for (nbr_t, nbr_i) in n1_full:
        if has_time_by_prefixed[nbr_t]:
            if time_by_prefixed[nbr_t][nbr_i] <= seed_time:
                n1.add((nbr_t, nbr_i))
        else:
            n1.add((nbr_t, nbr_i))

    # ---- 2-hop candidates ----
    n2: Dict[Tuple[str, int], Set[Tuple[str, int]]] = defaultdict(set)
    for (nbr_t, nbr_i) in n1:
        nbr2_full_set = cache.neighbors_set(nbr_t, nbr_i)
        if out_neighbor_cache is not None:
            out_neighbor_cache[(nbr_t, nbr_i)] = nbr2_full_set
        if len(nbr2_full_set) > max_2hop_threshold:
            nbr2_full = random.sample(list(nbr2_full_set), max_2hop_threshold)
        else:
            nbr2_full = list(nbr2_full_set)
        for (nbr2_t, nbr2_i) in nbr2_full:
            if (nbr2_t, nbr2_i) == (node_type, node_idx):
                continue  # self-loop
            if has_time_by_prefixed[nbr2_t]:
                if time_by_prefixed[nbr2_t][nbr2_i] <= seed_time:
                    n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))
            else:
                n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))

    # 2-hop ∩ 1-hop drop.
    n2 = {k: v for k, v in n2.items() if k not in n1}

    out: List[NbrToken] = []
    for (nbr_t, nbr_i) in n1:
        if has_time_by_prefixed[nbr_t]:
            nbr_time = float(time_by_prefixed[nbr_t][nbr_i])
            rel_days = (seed_time - nbr_time) / (60 * 60 * 24)
        else:
            rel_days = 0
        out.append((nbr_t, nbr_i, 1, rel_days, None))

    for (nbr2_t, nbr2_i), connecting_1hops in n2.items():
        if has_time_by_prefixed[nbr2_t]:
            nbr2_time = float(time_by_prefixed[nbr2_t][nbr2_i])
            rel_days = (seed_time - nbr2_time) / (60 * 60 * 24)
        else:
            rel_days = 0
        out.append((nbr2_t, nbr2_i, 2, rel_days, connecting_1hops))

    return out


# ------------------------------------------------------ final-selection phase
def sample_local_subgraph(
    cache: DatasetGraphCache,
    K: int,
    seed_node_type: str,
    seed_node_idx: int,
    seed_time: float,
    seed_val: int,
) -> Tuple[List[NbrToken], np.ndarray]:
    """Mirror of ``utils._process_one_seed``.

    Returns ``(final_tokens, edge_index_K_local)``.

    * ``final_tokens`` has length ``K`` with the seed at position 0
      (hop=0, rel_time=0).
    * ``edge_index_K_local`` is a ``[2, E]`` ``int32`` array of edges among
      the K tokens (using indices in ``[0, K)``).

    Bit-equal to dev-kyaw under the same ``random.seed(seed_val)``.
    """
    random.seed(seed_val)
    time_by_prefixed = cache.time_by_prefixed
    has_time_by_prefixed = cache.has_time_by_prefixed

    # Per-seed neighbor cache populated during gather. Skips ~K redundant
    # cache.neighbors_set calls during edge construction below (the seed
    # and every kept 1-hop already had their neighbor sets fetched in
    # gather; only 2-hop / fallback tokens fall through to the cache miss
    # path). Bit-exact: the cached set object is the same Python object
    # returned by the original call, so iteration order is identical.
    neighbor_cache: Dict[Tuple[str, int], Set[Tuple[str, int]]] = {}
    T_hat = gather_1_and_2_hop(
        cache, seed_node_type, seed_node_idx, seed_time,
        out_neighbor_cache=neighbor_cache,
    )
    T_hat_list = list(T_hat)
    size_th = len(T_hat_list)
    K_minus_1 = K - 1

    one_hop = [n for n in T_hat_list if n[2] == 1]
    two_hop = [n for n in T_hat_list if n[2] == 2]
    combined = one_hop + two_hop

    if size_th >= K_minus_1:
        chosen = random.sample(combined, K_minus_1)
    elif 0 < size_th < K_minus_1:
        chosen = random.choices(combined, k=K_minus_1)
    else:
        # Fallback from ALL nodes.
        if K_minus_1 <= len(cache.all_nodes):
            fallback = random.sample(cache.all_nodes, K_minus_1)
        else:
            fallback = random.choices(cache.all_nodes, k=K_minus_1)
        chosen = []
        for (ft, fi) in fallback:
            if has_time_by_prefixed[ft]:
                ft_time = float(time_by_prefixed[ft][fi])
                rel = (seed_time - ft_time) / (60 * 60 * 24)
            else:
                rel = 0
            chosen.append((ft, fi, 3, rel, None))

    # Assemble final tokens (seed first, then sampled rest).
    seed_token: NbrToken = (seed_node_type, seed_node_idx, 0, 0.0, 0)
    final_tokens: List[NbrToken] = [seed_token] + list(chosen)

    # Shuffle rest (preserve seed at position 0).
    if len(final_tokens) > 1:
        first = final_tokens[0]
        rest = final_tokens[1:]
        rest = random.sample(rest, len(rest))
        final_tokens = [first] + rest

    # Build local subgraph adjacency among the K tokens.
    local_map: Dict[Tuple[str, int], int] = {}
    for j, (t_str, i, _hop, _t_val, _c1) in enumerate(final_tokens):
        local_map[(t_str, i)] = j

    edges: List[Tuple[int, int]] = []
    for j_src, (t_str, i, _hop, _t_val, _c1) in enumerate(final_tokens):
        # The seed and every kept 1-hop had their neighbor set captured
        # during gather; reuse them here. 2-hop / fallback tokens were
        # never fetched, so fall through to cache.neighbors_set.
        nbrs = neighbor_cache.get((t_str, i))
        if nbrs is None:
            nbrs = cache.neighbors_set(t_str, i)
        for (nbr_t, nbr_i) in nbrs:
            if (nbr_t, nbr_i) in local_map:
                edges.append((j_src, local_map[(nbr_t, nbr_i)]))

    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int32)
    else:
        edge_index = np.array(edges, dtype=np.int32).T  # [2, E]

    return final_tokens, edge_index


# ---------------------------------------------------------- batched (driver)
def sample_seeds(
    cache: DatasetGraphCache,
    K: int,
    seed_node_type: str,
    seed_node_idxs: torch.Tensor,
    seed_times: torch.Tensor,
) -> Dict[int, Tuple[List[NbrToken], np.ndarray]]:
    """Sample local subgraphs for a batch of seeds. Sequential.

    For PR1 we keep this single-process: removing dev-kyaw's
    ``multiprocessing.Pool`` (which collided with DataLoader workers) is on
    the PR2 plan. This function is used by the offline precompute path.

    Returns
    -------
    dict mapping ``seed_idx -> (final_tokens, edge_index)``.
    """
    assert len(seed_node_idxs) == len(seed_times), \
        "Mismatch in seed_node_idxs vs seed_times"

    out: Dict[int, Tuple[List[NbrToken], np.ndarray]] = {}
    for i, node_idx_t in enumerate(seed_node_idxs):
        node_idx = int(node_idx_t.item() if isinstance(node_idx_t, torch.Tensor) else node_idx_t)
        seed_t = float(seed_times[i].item() if isinstance(seed_times[i], torch.Tensor) else seed_times[i])
        seed_val = hash((seed_node_type, node_idx, seed_t, K)) & 0xFFFFFFFF
        final_tokens, edge_index = sample_local_subgraph(
            cache, K, seed_node_type, node_idx, seed_t, seed_val
        )
        out[node_idx] = (final_tokens, edge_index)
    return out
