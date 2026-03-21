import random
import os
import time
from multiprocessing import Pool, cpu_count, get_context
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict

import numpy as np
import h5py

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

from relbench.base import Dataset, EntityTask
from relbench.modeling.graph import get_node_train_table_input

from collections import defaultdict

GLOBAL_ADJ = None
GLOBAL_ALL_NODES = None       # (node_type_counts, total_nodes) for compact fallback
GLOBAL_TIME_ARRAYS = None
GLOBAL_NODE_TYPES = None    # list of str, e.g. ["user", "product", ...]
GLOBAL_TYPE_TO_IDX = None   # {"user": 0, "product": 1, ...}
GLOBAL_IDX_TO_TYPE = None   # {0: "user", 1: "product", ...}

class GloveTextEmbedding:
        def __init__(self, device: torch.device):
            self.model = SentenceTransformer("sentence-transformers/average_word_embeddings_glove.6B.300d", device=device)
        
        def __call__(self, sentences: List[str]) -> Tensor:
            sentences = [str(s) if not isinstance(s, str) else s for s in sentences]
            return torch.from_numpy(self.model.encode(sentences))

def build_adjacency_hetero(hetero_data: HeteroData, undirected: bool = True):
    adjacency = {
        node_type: [set() for _ in range(hetero_data[node_type].num_nodes)]
        for node_type in hetero_data.node_types
    }
    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        if 'edge_index' not in hetero_data[edge_type]:
            continue
        edge_index = hetero_data[edge_type].edge_index
        src_list = edge_index[0].tolist()
        dst_list = edge_index[1].tolist()
        for s, d in zip(src_list, dst_list):
            adjacency[src_type][s].add((dst_type, d))
            if undirected:
                adjacency[dst_type][d].add((src_type, s))
    return adjacency

def build_adjacency_csr(hetero_data, undirected: bool = True):
    """
    Build adjacency in CSR (Compressed Sparse Row) format using numpy arrays.
    Fully vectorized — no per-element Python loops.

    Returns:
        {node_type: {"nbr_types": np.int8, "nbr_indices": np.int32, "offsets": np.int64}}
    """
    type_to_idx = {nt: i for i, nt in enumerate(hetero_data.node_types)}

    # Collect all edges per source node type as numpy arrays
    # Each entry: (source_node_id, nbr_type_id, nbr_node_id)
    edge_arrays = {nt: [] for nt in hetero_data.node_types}

    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        if 'edge_index' not in hetero_data[edge_type]:
            continue
        ei = hetero_data[edge_type].edge_index
        src_arr = ei[0].numpy().astype(np.int32)
        dst_arr = ei[1].numpy().astype(np.int32)
        n_edges = len(src_arr)
        dst_type_id = type_to_idx[dst_type]
        src_type_id = type_to_idx[src_type]

        # Forward edges: src_type[s] -> (dst_type_id, d)
        fwd = np.empty((n_edges, 3), dtype=np.int64)
        fwd[:, 0] = src_arr
        fwd[:, 1] = dst_type_id
        fwd[:, 2] = dst_arr
        edge_arrays[src_type].append(fwd)

        if undirected:
            rev = np.empty((n_edges, 3), dtype=np.int64)
            rev[:, 0] = dst_arr
            rev[:, 1] = src_type_id
            rev[:, 2] = src_arr
            edge_arrays[dst_type].append(rev)

    # Build CSR per node type using vectorized numpy ops
    csr = {}
    for nt in hetero_data.node_types:
        num_nodes = hetero_data[nt].num_nodes

        if not edge_arrays[nt]:
            # No edges for this type
            csr[nt] = {
                "nbr_types": np.empty(0, dtype=np.int8),
                "nbr_indices": np.empty(0, dtype=np.int32),
                "offsets": np.zeros(num_nodes + 1, dtype=np.int64),
            }
            continue

        all_edges = np.concatenate(edge_arrays[nt])  # (E, 3): [node_id, nbr_type, nbr_idx]

        # Deduplicate using composite keys
        # key = node_id * stride1 + nbr_type * stride2 + nbr_idx
        max_nbr_idx = int(all_edges[:, 2].max()) + 1 if len(all_edges) > 0 else 1
        max_nbr_type = int(all_edges[:, 1].max()) + 1 if len(all_edges) > 0 else 1
        stride2 = np.int64(max_nbr_idx)
        stride1 = np.int64(max_nbr_type) * stride2

        keys = all_edges[:, 0] * stride1 + all_edges[:, 1] * stride2 + all_edges[:, 2]
        _, unique_idx = np.unique(keys, return_index=True)
        all_edges = all_edges[unique_idx]

        # Sort by node_id for CSR grouping
        sort_idx = np.argsort(all_edges[:, 0], kind='stable')
        all_edges = all_edges[sort_idx]

        # Build offsets
        node_ids = all_edges[:, 0].astype(np.int64)
        offsets = np.zeros(num_nodes + 1, dtype=np.int64)
        if len(node_ids) > 0:
            unique_nodes, counts = np.unique(node_ids, return_counts=True)
            np.add.at(offsets, unique_nodes.astype(np.int64) + 1, counts)
            np.cumsum(offsets, out=offsets)

        csr[nt] = {
            "nbr_types": all_edges[:, 1].astype(np.int8),
            "nbr_indices": all_edges[:, 2].astype(np.int32),
            "offsets": offsets,
        }

    return csr


def _build_all_nodes_compact(data):
    """Build compact fallback representation: ([(type_str, count), ...], total_nodes)."""
    node_type_counts = [(nt, data[nt].num_nodes) for nt in data.node_types]
    total = sum(c for _, c in node_type_counts)
    return (node_type_counts, total)


def _sample_fallback_nodes(all_nodes_compact, k, rng=random):
    """
    Sample k nodes uniformly from the compact all_nodes representation.
    all_nodes_compact: (node_type_counts, total_nodes) where node_type_counts
                       is a list of (type_str, count) pairs.
    Uses the provided rng for random state compatibility.
    """
    node_type_counts, total_nodes = all_nodes_compact
    if k <= total_nodes:
        indices = rng.sample(range(total_nodes), k)
    else:
        indices = rng.choices(range(total_nodes), k=k)

    # Map flat indices to (type_str, local_idx) via cumulative counts
    result = []
    for idx in indices:
        remaining = idx
        for (nt, count) in node_type_counts:
            if remaining < count:
                result.append((nt, remaining))
                break
            remaining -= count
    return result


def init_worker_globals(adj, all_nodes, node_types=None, time_arrays=None):
    global GLOBAL_ADJ, GLOBAL_ALL_NODES, GLOBAL_TIME_ARRAYS
    global GLOBAL_NODE_TYPES, GLOBAL_TYPE_TO_IDX, GLOBAL_IDX_TO_TYPE
    GLOBAL_ADJ = adj
    GLOBAL_ALL_NODES = all_nodes
    if node_types is not None:
        GLOBAL_NODE_TYPES = list(node_types)
        GLOBAL_TYPE_TO_IDX = {nt: i for i, nt in enumerate(GLOBAL_NODE_TYPES)}
        GLOBAL_IDX_TO_TYPE = {i: nt for i, nt in enumerate(GLOBAL_NODE_TYPES)}
    if time_arrays is not None:
        GLOBAL_TIME_ARRAYS = time_arrays


def gather_1_and_2_hop_with_seed_time(
    adjacency: Dict[str, List[set]],
    data: HeteroData,
    node_type: str,
    node_idx: int,
    seed_time: float,
    max_1hop_threshold: int = 5000,
    max_2hop_threshold: int = 1000
) -> List[Tuple[str, int, int, float, Optional[set]]]:
    """
    Gather 1-hop and 2-hop neighbors with time condition.

    Returns:
        neighbors_with_time: List of tuples:
            (nbr_t, nbr_i, hop, relative_time_days, None) for 1-hop
            (nbr_t, nbr_i, hop, relative_time_days, connecting_1hop_tuple) for 2-hop
    """
    # Gather 1-hop neighbors satisfying the time condition
    n1_full = adjacency[node_type][node_idx]
    if len(n1_full) > max_1hop_threshold:
        n1_full = random.sample(list(n1_full), max_1hop_threshold)
    else:
        n1_full = list(n1_full)
        
    n1 = set()
    for (nbr_t, nbr_i) in n1_full:
        if hasattr(data[nbr_t], "time"):
            if data[nbr_t].time[nbr_i] <= seed_time:
                n1.add((nbr_t, nbr_i))
        else:
            n1.add((nbr_t, nbr_i))

    # Gather 2-hop neighbors satisfying the time condition
    n2 = defaultdict(set)  # Map 2-hop neighbor to set of connecting 1-hop neighbors
    for (nbr_t, nbr_i) in n1:
        nbr2_full = adjacency[nbr_t][nbr_i]
        if len(nbr2_full) > max_2hop_threshold:
            nbr2_full = random.sample(list(nbr2_full), max_2hop_threshold)
        else:
            nbr2_full = list(nbr2_full)
            
        for (nbr2_t, nbr2_i) in nbr2_full:
            # Skip if we loop back to the original node
            if (nbr2_t, nbr2_i) == (node_type, node_idx):
                continue
            if hasattr(data[nbr2_t], "time"):
                if data[nbr2_t].time[nbr2_i] <= seed_time:
                    n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))
            else:
                n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))

    # Remove overlaps: ensure 2-hop neighbors are not already in 1-hop
    n2 = {k: v for k, v in n2.items() if k not in n1}

    neighbors_with_time = []

    # Process 1-hop neighbors with hop distance 1
    for (nbr_t, nbr_i) in n1:
        if hasattr(data[nbr_t], "time"):
            nbr_time = data[nbr_t].time[nbr_i].item()
            relative_time_days = (seed_time - nbr_time) / (60 * 60 * 24)
        else:
            relative_time_days = 0  # no time entities
        # Append tuple with hop level 1 and no connecting 1-hop neighbor
        neighbors_with_time.append((nbr_t, nbr_i, 1, relative_time_days, None))

    # Process 2-hop neighbors with hop distance 2
    for (nbr2_t, nbr2_i), connecting_1hops in n2.items():
        if hasattr(data[nbr2_t], "time"):
            nbr2_time = data[nbr2_t].time[nbr2_i].item()
            relative_time_days = (seed_time - nbr2_time) / (60 * 60 * 24)
        else:
            relative_time_days = 0  # no time entities
        # If multiple connecting 1-hop neighbors, we will handle in sampling
        neighbors_with_time.append((nbr2_t, nbr2_i, 2, relative_time_days, connecting_1hops))

    return neighbors_with_time


def gather_1_and_2_hop_vectorized(
    csr_adj,
    time_arrays: Dict[str, np.ndarray],
    type_to_idx: Dict[str, int],
    node_types: list,
    node_type: str,
    node_idx: int,
    seed_time: float,
    max_1hop_threshold: int = 5000,
    max_2hop_threshold: int = 1000,
) -> List[Tuple[str, int, int, float, Optional[set]]]:
    """
    Vectorized version of gather_1_and_2_hop_with_seed_time.
    Uses CSR adjacency and pre-extracted numpy time arrays for speed.

    Args:
        csr_adj: output of build_adjacency_csr()
        time_arrays: {node_type_str: np.ndarray of timestamps}
        type_to_idx: {node_type_str: int}
        node_types: list of node type strings (index -> str)
        node_type: seed node type string
        node_idx: seed node local index
        seed_time: temporal cutoff
    """
    idx_to_type = {i: nt for nt, i in type_to_idx.items()}

    # --- 1-hop: slice from CSR ---
    csr_nt = csr_adj[node_type]
    start = int(csr_nt["offsets"][node_idx])
    end = int(csr_nt["offsets"][node_idx + 1])
    n1_types = csr_nt["nbr_types"][start:end].copy()
    n1_indices = csr_nt["nbr_indices"][start:end].copy()

    # Cap if too many
    if len(n1_types) > max_1hop_threshold:
        sel = np.random.choice(len(n1_types), max_1hop_threshold, replace=False)
        n1_types = n1_types[sel]
        n1_indices = n1_indices[sel]

    # Vectorized temporal filter per unique type
    mask = np.ones(len(n1_types), dtype=bool)
    for tid in np.unique(n1_types):
        t_str = idx_to_type[int(tid)]
        if t_str in time_arrays:
            t_mask = n1_types == tid
            times = time_arrays[t_str][n1_indices[t_mask]]
            mask[t_mask] = times <= seed_time

    n1_types = n1_types[mask]
    n1_indices = n1_indices[mask]

    # Build 1-hop set for fast lookup and overlap removal
    n1_set = set()
    for k in range(len(n1_types)):
        n1_set.add((int(n1_types[k]), int(n1_indices[k])))

    # --- 2-hop: batch CSR lookups by neighbor type ---
    n2_connecting = defaultdict(set)  # (type_str, idx) -> set of connecting 1-hop (type_str, idx)
    seed_key = (type_to_idx[node_type], node_idx)

    # Pre-compute all CSR offsets for 1-hop neighbors, grouped by type
    for tid in np.unique(n1_types):
        type_mask = n1_types == tid
        nbr_type_str = idx_to_type[int(tid)]
        nbr_indices_of_type = n1_indices[type_mask]
        csr_nbr = csr_adj[nbr_type_str]

        # Batch offset lookups for all neighbors of this type
        starts = csr_nbr["offsets"][nbr_indices_of_type].astype(np.int64)
        ends = csr_nbr["offsets"][nbr_indices_of_type + 1].astype(np.int64)
        degrees = ends - starts

        # Collect all 2-hop candidates with connecting labels
        all_n2_t = []
        all_n2_i = []
        all_connecting_k = []  # index into nbr_indices_of_type

        for local_k in range(len(nbr_indices_of_type)):
            s2, e2 = int(starts[local_k]), int(ends[local_k])
            deg = int(degrees[local_k])
            if deg == 0:
                continue
            n2_t = csr_nbr["nbr_types"][s2:e2]
            n2_i = csr_nbr["nbr_indices"][s2:e2]

            # Cap if too many
            if deg > max_2hop_threshold:
                sel2 = np.random.choice(deg, max_2hop_threshold, replace=False)
                n2_t = n2_t[sel2]
                n2_i = n2_i[sel2]

            all_n2_t.append(n2_t)
            all_n2_i.append(n2_i)
            all_connecting_k.append(np.full(len(n2_t), local_k, dtype=np.int32))

        if not all_n2_t:
            continue

        # Concatenate all 2-hop candidates for this neighbor type
        cat_n2_t = np.concatenate(all_n2_t)
        cat_n2_i = np.concatenate(all_n2_i)
        cat_conn_k = np.concatenate(all_connecting_k)

        # Vectorized temporal filter on the entire batch
        tmask = np.ones(len(cat_n2_t), dtype=bool)
        for tid2 in np.unique(cat_n2_t):
            t_str2 = idx_to_type[int(tid2)]
            if t_str2 in time_arrays:
                tm = cat_n2_t == tid2
                tmask[tm] = time_arrays[t_str2][cat_n2_i[tm]] <= seed_time
        cat_n2_t = cat_n2_t[tmask]
        cat_n2_i = cat_n2_i[tmask]
        cat_conn_k = cat_conn_k[tmask]

        # Vectorized seed/1-hop exclusion using composite keys
        candidate_keys = (cat_n2_t.astype(np.int64) << np.int64(32)) | cat_n2_i.astype(np.int64)
        seed_ck = (np.int64(seed_key[0]) << np.int64(32)) | np.int64(seed_key[1])

        # Build 1-hop composite key array for exclusion
        n1_ck = (n1_types.astype(np.int64) << np.int64(32)) | n1_indices.astype(np.int64)
        n1_ck_set = set(n1_ck.tolist())
        n1_ck_set.add(int(seed_ck))

        # Vectorized exclusion: remove seed and 1-hop nodes
        exclude_mask = np.array([int(ck) not in n1_ck_set for ck in candidate_keys], dtype=bool)
        cat_n2_t = cat_n2_t[exclude_mask]
        cat_n2_i = cat_n2_i[exclude_mask]
        cat_conn_k = cat_conn_k[exclude_mask]

        # Build n2_connecting dict from filtered results
        for j in range(len(cat_n2_t)):
            n2_key_str = idx_to_type[int(cat_n2_t[j])]
            connecting_nbr_idx = int(nbr_indices_of_type[cat_conn_k[j]])
            n2_connecting[(n2_key_str, int(cat_n2_i[j]))].add((nbr_type_str, connecting_nbr_idx))

    # --- Build output ---
    results = []

    # 1-hop
    for k in range(len(n1_types)):
        nbr_type_str = idx_to_type[int(n1_types[k])]
        nbr_idx = int(n1_indices[k])
        if nbr_type_str in time_arrays:
            rel_time = (seed_time - float(time_arrays[nbr_type_str][nbr_idx])) / 86400.0
        else:
            rel_time = 0.0
        results.append((nbr_type_str, nbr_idx, 1, rel_time, None))

    # 2-hop
    for (nbr2_type_str, nbr2_idx), connecting in n2_connecting.items():
        if nbr2_type_str in time_arrays:
            rel_time = (seed_time - float(time_arrays[nbr2_type_str][nbr2_idx])) / 86400.0
        else:
            rel_time = 0.0
        results.append((nbr2_type_str, nbr2_idx, 2, rel_time, connecting))

    return results


def _process_one_seed(args):
    """
    Worker function: gather neighbors for a single seed node + time,
    perform local nodes expansions up to K, apply fallback if necessary,
    then return a final list of neighbor tokens.

    Accepts 5-element tuple: (K, seed_node_type, seed_node_idx, seed_time, seed_val)
    or 6-element tuple: (K, seed_node_type, seed_node_idx, seed_time, seed_val, row_idx)

    Uses GLOBAL_ADJ (CSR), GLOBAL_NODE_TYPES, GLOBAL_TIME_ARRAYS, GLOBAL_ALL_NODES
    set via init_worker_globals — no pickling of heavy objects per task.
    """
    global GLOBAL_ADJ, GLOBAL_ALL_NODES, GLOBAL_TIME_ARRAYS
    global GLOBAL_NODE_TYPES, GLOBAL_TYPE_TO_IDX, GLOBAL_IDX_TO_TYPE

    if len(args) == 6:
        (K, seed_node_type, seed_node_idx, seed_time, seed_val, row_idx) = args
    else:
        (K, seed_node_type, seed_node_idx, seed_time, seed_val) = args
        row_idx = None
    random.seed(seed_val)
    np.random.seed(seed_val % (2**32))

    # 1. gather 1-hop and 2-hop — use vectorized path if CSR available
    if GLOBAL_TIME_ARRAYS is not None and isinstance(GLOBAL_ADJ, dict) and \
       len(GLOBAL_ADJ) > 0 and "offsets" in next(iter(GLOBAL_ADJ.values())):
        T_hat = gather_1_and_2_hop_vectorized(
            GLOBAL_ADJ, GLOBAL_TIME_ARRAYS, GLOBAL_TYPE_TO_IDX, GLOBAL_NODE_TYPES,
            seed_node_type, seed_node_idx, seed_time
        )
    else:
        raise RuntimeError(
            "Legacy dict-of-sets adjacency path is no longer supported in workers. "
            "Use build_adjacency_csr() to create CSR adjacency before spawning workers."
        )

    T_hat_list = list(T_hat)
    size_th = len(T_hat_list)
    K_minus_1 = K - 1

    # separate 1-hop, 2-hop
    one_hop_neighbors = [n for n in T_hat_list if n[2] == 1]
    two_hop_neighbors = [n for n in T_hat_list if n[2] == 2]
    combined_neighbors = one_hop_neighbors + two_hop_neighbors

    # 2. If we have enough neighbors => random.sample
    #    If not => random.choices
    if size_th >= K_minus_1:
        chosen_neighbors = random.sample(combined_neighbors, K_minus_1)
    elif 0 < size_th < K_minus_1:
        chosen_neighbors = random.choices(combined_neighbors, k=K_minus_1)
    else:
        # fallback from GLOBAL_ALL_NODES
        fallback = _sample_fallback_nodes(GLOBAL_ALL_NODES, K_minus_1, rng=random)
        chosen_neighbors = []
        time_arrays = GLOBAL_TIME_ARRAYS or {}
        for (ft, fi) in fallback:
            if ft in time_arrays:
                ft_time = float(time_arrays[ft][fi])
                rel_time = (seed_time - ft_time) / (60 * 60 * 24)
            else:
                rel_time = 0
            chosen_neighbors.append((ft, fi, 3, rel_time, None))

    # 3. Build final_tokens with subgraph adj for the seed node and its chosen neighbors
    final_tokens = []
    seed_token = (seed_node_type, seed_node_idx, 0, 0.0, 0)
    final_tokens.append(seed_token)
    final_tokens.extend(chosen_neighbors)

    # randomize order except keep seed first
    if len(final_tokens) > 1:
        first = final_tokens[0]
        rest = final_tokens[1:]
        rest = random.sample(rest, len(rest))
        final_tokens = [first] + rest

    # build adjacency among these K nodes using CSR + vectorized membership
    local_map = {}
    for j, (t_str, i, hop, t_val, c1hops) in enumerate(final_tokens):
        local_map[(t_str, i)] = j

    edges = []
    adj = GLOBAL_ADJ
    idx_to_type = GLOBAL_IDX_TO_TYPE
    type_to_idx = GLOBAL_TYPE_TO_IDX

    # Build composite key lookup for vectorized membership testing
    local_composite = {}  # composite_key -> local_j
    for (t_str, i), j in local_map.items():
        ck = (np.int64(type_to_idx[t_str]) << 32) | np.int64(i)
        local_composite[int(ck)] = j
    local_ck_array = np.array(list(local_composite.keys()), dtype=np.int64)

    for j_src, (t_str, i, hop, t_val, c1hops) in enumerate(final_tokens):
        csr_nt = adj[t_str]
        s = int(csr_nt["offsets"][i])
        e = int(csr_nt["offsets"][i + 1])
        if s == e:
            continue

        nbr_types_arr = csr_nt["nbr_types"][s:e]
        nbr_indices_arr = csr_nt["nbr_indices"][s:e]

        # Vectorized composite keys for all neighbors
        nbr_ck = (nbr_types_arr.astype(np.int64) << 32) | nbr_indices_arr.astype(np.int64)

        # Vectorized membership test
        mask = np.isin(nbr_ck, local_ck_array)
        for ck in nbr_ck[mask]:
            edges.append((j_src, local_composite[int(ck)]))

    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int32)
    else:
        arr = np.array(edges, dtype=np.int32)
        edge_index = arr.T  # shape [2, E]

    if row_idx is not None:
        return (row_idx, seed_node_type, seed_node_idx, final_tokens, edge_index)
    return (seed_node_type, seed_node_idx, final_tokens, edge_index)


def local_nodes_hetero(
    data,
    K: int,
    table_input_nodes: tuple,
    table_input_time: torch.Tensor,
    undirected: bool = True,
    num_workers: int = None,
    pool: Pool = None,
):
    """
    Produces a dictionary S[seed_node_type][idx] for each node in `table_input_nodes[1]`.
    local neighbor sampling up to 2-hop, with fallback if needed, in parallel.

    If `pool` is provided, reuses it (avoids re-pickling adj/data to workers).
    """
    seed_node_type, seed_node_idxs = table_input_nodes
    assert len(seed_node_idxs) == len(table_input_time), "Mismatch in seed_node_idxs vs table_input_time"

    # Prepare tasks — NO data object, just lightweight scalars
    tasks = []
    for i, node_idx_t in enumerate(seed_node_idxs):
        node_idx = node_idx_t.item()
        seed_t = table_input_time[i].item()
        seed_val = hash((seed_node_type, node_idx, seed_t, K)) & 0xffffffff
        tasks.append((K, seed_node_type, node_idx, seed_t, seed_val))

    _nw = num_workers or (getattr(pool, '_processes', None) if pool else None) or cpu_count()
    chunksize = max(1, len(tasks) // (_nw * 4))

    if pool is not None:
        # Reuse existing pool (globals already initialized)
        results = pool.map(_process_one_seed, tasks, chunksize=chunksize)
    else:
        # Legacy: create a pool on the fly (for backward compat)
        global GLOBAL_ADJ, GLOBAL_ALL_NODES, GLOBAL_TIME_ARRAYS
        global GLOBAL_NODE_TYPES, GLOBAL_TYPE_TO_IDX, GLOBAL_IDX_TO_TYPE

        if GLOBAL_ADJ is None:
            GLOBAL_ADJ = build_adjacency_csr(data, undirected=undirected)

        if GLOBAL_ALL_NODES is None:
            all_nodes_compact = _build_all_nodes_compact(data)
            GLOBAL_ALL_NODES = all_nodes_compact

        if GLOBAL_TIME_ARRAYS is None:
            GLOBAL_TIME_ARRAYS = {}
            for nt in data.node_types:
                if hasattr(data[nt], "time"):
                    GLOBAL_TIME_ARRAYS[nt] = data[nt].time.numpy().copy()

        node_types = list(data.node_types)
        GLOBAL_NODE_TYPES = node_types
        GLOBAL_TYPE_TO_IDX = {nt: i for i, nt in enumerate(node_types)}
        GLOBAL_IDX_TO_TYPE = {i: nt for i, nt in enumerate(node_types)}

        if num_workers is None:
            num_workers = max(1, min(cpu_count() - 1, len(tasks)))

        ctx = get_context("fork")
        chunksize = max(1, len(tasks) // (num_workers * 4))
        with ctx.Pool(
            processes=num_workers,
            initializer=init_worker_globals,
            initargs=(GLOBAL_ADJ, GLOBAL_ALL_NODES, node_types, GLOBAL_TIME_ARRAYS)
        ) as p:
            results = p.map(_process_one_seed, tasks, chunksize=chunksize)

    # Build the final dictionary S
    S = {seed_node_type: {}}
    for (nt, idx, final_list, edge_index) in results:
        S[nt][idx] = (final_list, edge_index)

    return S


########################################
#  The RelGTTokens dataset class
########################################
class RelGTTokens(Dataset):
    def __init__(
        self,
        data,
        task,
        K: int,
        split: str = "train",
        undirected: bool = True,
        num_workers: int = None,
        precompute: bool = True,
        precomputed_dir: str = None,
        train_stage: str = "finetune"
    ):
        super().__init__()
        self.data = data
        self.task = task
        self.split = split
        self.K = K
        self.undirected = undirected
        self.num_workers = num_workers
        self.precompute = precompute
        self.precomputed_dir = precomputed_dir

        # Retrieve the table and the seeds/targets
        self.table = self.task.get_table(split=self.split)
        self.table_input = get_node_train_table_input(self.table, self.task)
        self.node_type, self.node_idxs = self.table_input.nodes
        self.target = self.table_input.target if self.table_input.target is not None else None

        self.time = getattr(self.table_input, "time", None)
        self.transform = getattr(self.table_input, "transform", None)

        # type <-> index mappings
        self.node_types = self.data.node_types
        self.node_type_to_index = {nt: idx for idx, nt in enumerate(self.node_types)}
        self.index_to_node_type = {idx: nt for idx, nt in enumerate(self.node_types)}
        
        self.max_neighbor_hop = 2 + 1  # 2 for hop neighbors + 1 for random fallback

        # Create global index mappings for (type_id, local_id)
        self._create_global_mappings()

        # HDF5 path
        self.precomputed_path = self._construct_precomputed_path()
        
        self.train_stage = train_stage

        if self.precompute:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if os.path.exists(self.precomputed_path):
                print(f"[{self.split}] Found existing HDF5 at {self.precomputed_path}")
            elif world_size > 1 and torch.distributed.is_initialized():
                # All ranks participate in distributed precomputation
                print(f"[{self.split}] Rank {rank}: distributed precomputation (K={self.K})...")
                self._precompute_sampling_distributed()
            else:
                # Single-GPU: original path
                print(f"[{self.split}] Precomputing neighbor sampling (K={self.K})...")
                self._precompute_sampling()
            # Final sync so all ranks see the HDF5 before proceeding
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                if not os.path.exists(self.precomputed_path):
                    raise RuntimeError(
                        f"Rank {rank}: HDF5 file not found after precomputation: "
                        f"{self.precomputed_path}"
                    )

    def _create_global_mappings(self):
        """
        Create a dictionary that maps (type_idx, local_idx) -> global_idx, 
        and the reverse.
        """
        self.type_local_to_global = {}
        self.global_to_type_local = {}
        
        global_index = 0
        for type_idx, node_type in self.index_to_node_type.items():
            if 'x' in self.data[node_type]:
                num_nodes = self.data[node_type]['x'].size(0)
            else:
                num_nodes = self.data[node_type].num_nodes

            for local_idx in range(num_nodes):
                key = (type_idx, local_idx)
                self.type_local_to_global[key] = global_index
                self.global_to_type_local[global_index] = key
                global_index += 1

    def get_global_index(self, type_idxs: List[int], local_idxs: List[int]) -> List[int]:
        """
        Convert each (type_idx, local_idx) to a global ID for downstream usage.
        """
        out = []
        for t_i, l_i in zip(type_idxs, local_idxs):
            out.append(self.type_local_to_global[(t_i, l_i)])
        return out

    def _construct_precomputed_path(self) -> str:
        if not self.precomputed_dir:
            raise ValueError("must provide a 'precomputed_dir' to store expansions.")
        path = os.path.join(
            self.precomputed_dir,
            str(self.K),
            f"{self.split}.h5"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def __len__(self):
        return len(self.node_idxs)

    def _create_datasets(self, h5file: h5py.File, total_samples: int) -> dict:
        _chunk_size = min(total_samples, 10000)
        return {
            "types": h5file.create_dataset(
                "types", shape=(total_samples, self.K), dtype='int16',
                chunks=(_chunk_size, self.K)
            ),
            "indices": h5file.create_dataset(
                "indices", shape=(total_samples, self.K), dtype='int32',
                chunks=(_chunk_size, self.K)
            ),
            "hops": h5file.create_dataset(
                "hops", shape=(total_samples, self.K), dtype='int8',
                chunks=(_chunk_size, self.K)
            ),
            "times": h5file.create_dataset(
                "times", shape=(total_samples, self.K), dtype='float32',
                chunks=(_chunk_size, self.K)
            )
        }

    def _precompute_sampling(self):
        """
        Run neighbor sampling and store expansions in HDF5.

        Key optimizations:
        - CSR adjacency instead of dict-of-sets (less memory, faster pickle)
        - Pre-extracted numpy time arrays (vectorized temporal filtering)
        - All tasks submitted at once via imap_unordered (no chunk loop)
        - data/adj/time_arrays passed via init_worker_globals, NOT per-task
        - Atomic HDF5 write: writes to temp file, renames on success
        """
        total = len(self.node_idxs)
        data_cpu = self.data.to("cpu")
        t_total_start = time.time()

        # Build CSR adjacency + time arrays ONCE
        t_csr_start = time.time()
        csr_adj = build_adjacency_csr(data_cpu, undirected=self.undirected)
        time_arrays = {}
        for nt in data_cpu.node_types:
            if hasattr(data_cpu[nt], "time"):
                time_arrays[nt] = data_cpu[nt].time.numpy().copy()

        all_nodes_compact = _build_all_nodes_compact(data_cpu)

        print(f"[{self.split}] CSR adjacency built in {time.time() - t_csr_start:.1f}s")

        num_workers = self.num_workers
        if num_workers is None:
            num_workers = max(1, min(cpu_count() - 1, total))

        # Build ALL tasks at once with row_idx
        all_tasks = []
        for i, node_idx_t in enumerate(self.node_idxs):
            node_idx = node_idx_t.item()
            seed_t = self.time[i].item() if self.time is not None else 0.0
            seed_val = hash((self.node_type, node_idx, seed_t, self.K)) & 0xffffffff
            all_tasks.append((self.K, self.node_type, node_idx, seed_t, seed_val, i))

        chunksize = max(1, total // (num_workers * 4))

        print(f"[{self.split}] Starting sampling: {total} seeds, K={self.K}, {num_workers} workers")
        t_sample_start = time.time()

        # Allocate full numpy arrays for streaming results
        all_types = np.zeros((total, self.K), dtype=np.int16)
        all_indices = np.zeros((total, self.K), dtype=np.int32)
        all_hops = np.zeros((total, self.K), dtype=np.int8)
        all_times = np.zeros((total, self.K), dtype=np.float32)
        adjacency_all = [None] * total

        t_pool_start = time.time()
        node_types = list(data_cpu.node_types)
        # Use forkserver to avoid SIGBUS from fork-after-CUDA.
        # Workers are numpy-only so forkserver is safe; the startup cost
        # (re-importing modules) is acceptable for one-time precomputation.
        ctx = get_context("forkserver")
        with ctx.Pool(
            processes=num_workers,
            initializer=init_worker_globals,
            initargs=(csr_adj, all_nodes_compact, node_types, time_arrays)
        ) as pool:
            print(f"[{self.split}] Pool created in {time.time() - t_pool_start:.1f}s")

            for result in tqdm(
                pool.imap_unordered(_process_one_seed, all_tasks, chunksize=chunksize),
                total=total,
                desc=f"Precomputing '{self.split}'"
            ):
                row_idx, nt, idx, final_tokens, edge_index = result
                for j, (t_str, nbr_loc_idx, hop, t_val, c1hops) in enumerate(final_tokens):
                    all_types[row_idx, j] = self.node_type_to_index[t_str]
                    all_indices[row_idx, j] = nbr_loc_idx
                    all_hops[row_idx, j] = hop
                    all_times[row_idx, j] = t_val
                adjacency_all[row_idx] = edge_index

        # Atomic write: write to temp file first, then rename
        tmp_path = self.precomputed_path + f".tmp.{os.getpid()}"
        try:
            with h5py.File(tmp_path, 'w') as hf:
                hf.create_dataset("types", data=all_types)
                hf.create_dataset("indices", data=all_indices)
                hf.create_dataset("hops", data=all_hops)
                hf.create_dataset("times", data=all_times)

                # Compute edge offsets and write edges
                edge_counts = np.array(
                    [e.shape[1] if e is not None else 0 for e in adjacency_all],
                    dtype=np.uint64
                )
                offsets = np.zeros(total + 1, dtype=np.uint64)
                np.cumsum(edge_counts, out=offsets[1:])

                total_edges = int(offsets[-1])
                edges_dset = hf.create_dataset("edges", shape=(2, total_edges), dtype='int16')
                for i in range(total):
                    e_arr = adjacency_all[i]
                    start = int(offsets[i])
                    end_ = int(offsets[i + 1])
                    if e_arr is not None and e_arr.size > 0:
                        edges_dset[:, start:end_] = e_arr

                hf.create_dataset("edges_offsets", data=offsets)

            os.rename(tmp_path, self.precomputed_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        elapsed = time.time() - t_total_start
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        print(f"[{self.split}] Sampling complete: {total} seeds in {int(hrs)}h {int(mins)}m {secs:.1f}s "
              f"(sampling: {time.time() - t_sample_start:.1f}s)")

    def _precompute_sampling_distributed(self):
        """
        Distributed precomputation: each rank computes its shard of seeds,
        writes to a temp .npz file, then rank 0 merges all shards into
        the final HDF5.
        """
        import torch.distributed as dist

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        total = len(self.node_idxs)
        data_cpu = self.data.to("cpu")
        t_total_start = time.time()

        # All ranks build CSR adjacency (CPU work, fast)
        t_csr_start = time.time()
        csr_adj = build_adjacency_csr(data_cpu, undirected=self.undirected)
        time_arrays = {}
        for nt in data_cpu.node_types:
            if hasattr(data_cpu[nt], "time"):
                time_arrays[nt] = data_cpu[nt].time.numpy().copy()
        all_nodes_compact = _build_all_nodes_compact(data_cpu)
        print(f"[{self.split}] Rank {rank}: CSR built in {time.time() - t_csr_start:.1f}s")

        # Build all tasks, then take this rank's shard
        all_tasks = []
        for i, node_idx_t in enumerate(self.node_idxs):
            node_idx = node_idx_t.item()
            seed_t = self.time[i].item() if self.time is not None else 0.0
            seed_val = hash((self.node_type, node_idx, seed_t, self.K)) & 0xffffffff
            all_tasks.append((self.K, self.node_type, node_idx, seed_t, seed_val, i))

        my_tasks = all_tasks[rank::world_size]
        shard_size = len(my_tasks)

        num_workers = self.num_workers
        if num_workers is None:
            # Scale workers per rank to avoid oversubscribing CPUs
            num_workers = max(1, min(32, cpu_count() // max(world_size, 1)))

        chunksize = max(1, shard_size // (num_workers * 4))

        print(f"[{self.split}] Rank {rank}: sampling {shard_size}/{total} seeds, "
              f"K={self.K}, {num_workers} workers")
        t_sample_start = time.time()

        # Allocate shard arrays
        shard_types = np.zeros((shard_size, self.K), dtype=np.int16)
        shard_indices = np.zeros((shard_size, self.K), dtype=np.int32)
        shard_hops = np.zeros((shard_size, self.K), dtype=np.int8)
        shard_times = np.zeros((shard_size, self.K), dtype=np.float32)
        shard_row_idxs = np.zeros(shard_size, dtype=np.int64)
        shard_adjacency = [None] * shard_size

        node_types = list(data_cpu.node_types)
        # Use forkserver to avoid SIGBUS from fork-after-CUDA.
        ctx = get_context("forkserver")
        with ctx.Pool(
            processes=num_workers,
            initializer=init_worker_globals,
            initargs=(csr_adj, all_nodes_compact, node_types, time_arrays)
        ) as pool:
            for shard_idx, result in enumerate(tqdm(
                pool.imap_unordered(_process_one_seed, my_tasks, chunksize=chunksize),
                total=shard_size,
                desc=f"Rank {rank} '{self.split}'",
                disable=(rank != 0)
            )):
                row_idx, nt, idx, final_tokens, edge_index = result
                shard_row_idxs[shard_idx] = row_idx
                for j, (t_str, nbr_loc_idx, hop, t_val, c1hops) in enumerate(final_tokens):
                    shard_types[shard_idx, j] = self.node_type_to_index[t_str]
                    shard_indices[shard_idx, j] = nbr_loc_idx
                    shard_hops[shard_idx, j] = hop
                    shard_times[shard_idx, j] = t_val
                shard_adjacency[shard_idx] = edge_index

        print(f"[{self.split}] Rank {rank}: sampling done in "
              f"{time.time() - t_sample_start:.1f}s")

        # Pack adjacency edges into flat array with offsets
        edge_counts = np.array(
            [e.shape[1] if e is not None else 0 for e in shard_adjacency],
            dtype=np.uint64
        )
        shard_edge_offsets = np.zeros(shard_size + 1, dtype=np.uint64)
        np.cumsum(edge_counts, out=shard_edge_offsets[1:])
        total_shard_edges = int(shard_edge_offsets[-1])

        shard_edges = np.zeros((2, max(total_shard_edges, 1)), dtype=np.int16)
        for i in range(shard_size):
            e_arr = shard_adjacency[i]
            start = int(shard_edge_offsets[i])
            end_ = int(shard_edge_offsets[i + 1])
            if e_arr is not None and e_arr.size > 0:
                shard_edges[:, start:end_] = e_arr

        # Write shard to temp .npz
        shard_dir = os.path.dirname(self.precomputed_path)
        shard_path = os.path.join(shard_dir, f".shard_{self.split}_{rank}.npz")
        np.savez(
            shard_path,
            row_idxs=shard_row_idxs,
            types=shard_types,
            indices=shard_indices,
            hops=shard_hops,
            times=shard_times,
            edges=shard_edges[:, :total_shard_edges],
            edge_offsets=shard_edge_offsets
        )
        print(f"[{self.split}] Rank {rank}: shard written to {shard_path}")

        # Wait for all ranks to finish writing shards
        dist.barrier()

        # Rank 0 merges all shards into the final HDF5
        if rank == 0:
            print(f"[{self.split}] Rank 0: merging {world_size} shards...")
            t_merge_start = time.time()

            all_types = np.zeros((total, self.K), dtype=np.int16)
            all_indices = np.zeros((total, self.K), dtype=np.int32)
            all_hops = np.zeros((total, self.K), dtype=np.int8)
            all_times = np.zeros((total, self.K), dtype=np.float32)
            all_adjacency = [None] * total

            for r in range(world_size):
                r_path = os.path.join(shard_dir, f".shard_{self.split}_{r}.npz")
                shard = np.load(r_path)
                r_row_idxs = shard["row_idxs"]
                r_edges = shard["edges"]
                r_edge_offsets = shard["edge_offsets"]

                for si, row_idx in enumerate(r_row_idxs):
                    ri = int(row_idx)
                    all_types[ri] = shard["types"][si]
                    all_indices[ri] = shard["indices"][si]
                    all_hops[ri] = shard["hops"][si]
                    all_times[ri] = shard["times"][si]

                    start = int(r_edge_offsets[si])
                    end_ = int(r_edge_offsets[si + 1])
                    if start < end_:
                        all_adjacency[ri] = r_edges[:, start:end_]

                shard.close()
                os.remove(r_path)

            # Write final HDF5 atomically
            tmp_path = self.precomputed_path + f".tmp.{os.getpid()}"
            try:
                with h5py.File(tmp_path, 'w') as hf:
                    hf.create_dataset("types", data=all_types)
                    hf.create_dataset("indices", data=all_indices)
                    hf.create_dataset("hops", data=all_hops)
                    hf.create_dataset("times", data=all_times)

                    edge_counts = np.array(
                        [e.shape[1] if e is not None else 0 for e in all_adjacency],
                        dtype=np.uint64
                    )
                    offsets = np.zeros(total + 1, dtype=np.uint64)
                    np.cumsum(edge_counts, out=offsets[1:])

                    total_edges = int(offsets[-1])
                    edges_dset = hf.create_dataset(
                        "edges", shape=(2, total_edges), dtype='int16'
                    )
                    for i in range(total):
                        e_arr = all_adjacency[i]
                        start = int(offsets[i])
                        end_ = int(offsets[i + 1])
                        if e_arr is not None and e_arr.size > 0:
                            edges_dset[:, start:end_] = e_arr

                    hf.create_dataset("edges_offsets", data=offsets)

                os.rename(tmp_path, self.precomputed_path)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            print(f"[{self.split}] Merge complete in {time.time() - t_merge_start:.1f}s")

        elapsed = time.time() - t_total_start
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        if rank == 0:
            print(f"[{self.split}] Distributed sampling complete: {total} seeds across "
                  f"{world_size} ranks in {int(hrs)}h {int(mins)}m {secs:.1f}s")

    def __getitem__(self, idx: int):
        """
        Retrieve samples from HDF5 (row=idx) and the label from self.target[idx].
        """
        with h5py.File(self.precomputed_path, 'r') as hf:
            sample = {
                "types": torch.from_numpy(hf["types"][idx]).long(),         # [K]
                "indices": torch.from_numpy(hf["indices"][idx]).long(),     # [K]
                "hops": torch.from_numpy(hf["hops"][idx]).long(),           # [K]
                "times": torch.from_numpy(hf["times"][idx]),         # [K]
            }
            offsets = hf["edges_offsets"]
            edges_dset = hf["edges"]
            start = offsets[idx]
            end_ = offsets[idx+1]
            if start == end_:
                eidx = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_np = edges_dset[:, start:end_]
                eidx = torch.from_numpy(edge_np).long()
            sample["edge_index"] = eidx

        # retrieve label from self.target
        label = self.target[idx] if self.target is not None else None
        
        # needed for global module
        sample["first_type"] = sample["types"][0].item()
        sample["first_index"] = sample["indices"][0].item()

        sample["tfs"] = [
            self.data[self.index_to_node_type[t.item()]].tf[i.item()]
            for t, i in zip(sample["types"], sample["indices"])
        ]
        
        # store the "global index" for ordering
        # 'idx' is the local index within this split, but effectively
        # it's a stable ID for the sample's row 
        # used during evaluation/test to match predictions to the original table
        sample["global_idx"] = idx
        return sample, label

    def collate(self, batch: List[Tuple[dict, Optional[torch.Tensor]]]):
        samples, labels = zip(*batch)  
        
        neighbor_types = torch.stack([s["types"] for s in samples], dim=0)  # [B, K]
        neighbor_indices = torch.stack([s["indices"] for s in samples], dim=0)
        neighbor_hops = torch.stack([s["hops"] for s in samples], dim=0)
        neighbor_times = torch.stack([s["times"] for s in samples], dim=0)

        out = {
            "neighbor_types": neighbor_types,
            "neighbor_indices": neighbor_indices,
            "neighbor_hops": neighbor_hops,
            "neighbor_times": neighbor_times
        }

        # labels if available
        if self.target is not None:
            out["labels"] = torch.stack(labels, dim=0)
        else:
            out["labels"] = None

        # global node indices for the seed node (the first token)
        # needed for global module
        first_types = [s["first_type"] for s in samples]
        first_indices = [s["first_index"] for s in samples]
        out["node_indices"] = torch.tensor(
            self.get_global_index(first_types, first_indices),
            dtype=torch.long
        )

        B, K = neighbor_types.shape
        grouped_tfs = {}
        grouped_positions = {}
        for t_id in range(len(self.node_types)):
            # For each possible type, find which neighbors are that type
            mask = (neighbor_types == t_id)
            if not mask.any():
                continue
            
            local_idxs = neighbor_indices[mask]  # 1D, length = #neighbors-of-this-type
            type_str = self.index_to_node_type[t_id]

            # an offset for each [b, k]
            # 'torch.nonzero(mask, as_tuple=False)' gives shape [N, 2] with (b, k)
            offsets_list = []
            positions_2d = torch.nonzero(mask, as_tuple=False)
            for (b, k) in positions_2d.tolist():
                offset = b * K + k   # Flatten (b, k) into one integer
                offsets_list.append(offset)

            grouped_tfs[t_id] = self.data[type_str].tf[local_idxs]
            grouped_positions[t_id] = offsets_list

        flat_batch_idx = torch.arange(B).unsqueeze(1).expand(B, K).reshape(-1).tolist()
        flat_nbr_idx = torch.arange(K).repeat(B).tolist()
        
        global_idxs = [s["global_idx"] for s in samples]
        global_idxs = torch.tensor(global_idxs, dtype=torch.long)

        out.update({
            "grouped_tfs": grouped_tfs,
            "grouped_indices": grouped_positions,
            "flat_batch_idx": flat_batch_idx,
            "flat_nbr_idx": flat_nbr_idx,
            "global_idx": global_idxs,
        })
        
        # handling for subgraph adjs for each sample in a batch
        batched_edges = []
        batch_vec = []
        node_offset = 0

        for i, sample in enumerate(samples):
            eidx = sample["edge_index"]  # shape [2, E_i]
            K_i = sample["types"].size(0)  # # of nodes in this subgraph

            # shift
            shifted = eidx + node_offset
            batched_edges.append(shifted)

            # fill 'batch' vector
            sub_batch = torch.full((K_i,), i, dtype=torch.long)
            batch_vec.append(sub_batch)

            node_offset += K_i

        edge_index = torch.cat(batched_edges, dim=1) if batched_edges else torch.zeros((2,0), dtype=torch.long)
        batch_out = torch.cat(batch_vec, dim=0) if batch_vec else torch.zeros((0,), dtype=torch.long)

        out.update({
            "edge_index": edge_index,  # shape [2, total_E]
            "batch": batch_out         # shape [total_nodes]
        })

        return out