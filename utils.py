import random
import os
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict

import numpy as np
import h5py
import gc

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
GLOBAL_ALL_NODES = None
GLOBAL_DATA = None
GLOBAL_TIME_ARRAYS = None

class GloveTextEmbedding:
        def __init__(self, device: torch.device):
            self.model = SentenceTransformer("sentence-transformers/average_word_embeddings_glove.6B.300d", device=device)
        
        def __call__(self, sentences: List[str]) -> Tensor:
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

    Returns:
        {node_type: {"nbr_types": np.int8, "nbr_indices": np.int32, "offsets": np.int64}}
    """
    type_to_idx = {nt: i for i, nt in enumerate(hetero_data.node_types)}

    # Pass 1: collect edges into per-node lists (temporary)
    temp = {
        nt: [[] for _ in range(hetero_data[nt].num_nodes)]
        for nt in hetero_data.node_types
    }

    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        if 'edge_index' not in hetero_data[edge_type]:
            continue
        ei = hetero_data[edge_type].edge_index
        src_arr = ei[0].numpy()
        dst_arr = ei[1].numpy()
        dst_type_id = type_to_idx[dst_type]
        src_type_id = type_to_idx[src_type]

        for s, d in zip(src_arr, dst_arr):
            temp[src_type][int(s)].append((dst_type_id, int(d)))
            if undirected:
                temp[dst_type][int(d)].append((src_type_id, int(s)))

    # Pass 2: deduplicate and pack into contiguous numpy arrays
    csr = {}
    for nt in hetero_data.node_types:
        num_nodes = hetero_data[nt].num_nodes
        node_lists = temp[nt]

        offsets = np.zeros(num_nodes + 1, dtype=np.int64)
        # Deduplicate per node
        deduped = [list(set(node_lists[i])) for i in range(num_nodes)]
        for i in range(num_nodes):
            offsets[i + 1] = offsets[i] + len(deduped[i])

        total = int(offsets[-1])
        nbr_types = np.empty(total, dtype=np.int8)
        nbr_indices = np.empty(total, dtype=np.int32)

        for i in range(num_nodes):
            start = int(offsets[i])
            for j, (tid, nid) in enumerate(deduped[i]):
                nbr_types[start + j] = tid
                nbr_indices[start + j] = nid

        csr[nt] = {
            "nbr_types": nbr_types,
            "nbr_indices": nbr_indices,
            "offsets": offsets,
        }

    return csr


def init_worker_globals(adj, all_nodes, data=None, time_arrays=None):
    global GLOBAL_ADJ, GLOBAL_ALL_NODES, GLOBAL_DATA, GLOBAL_TIME_ARRAYS
    GLOBAL_ADJ = adj
    GLOBAL_ALL_NODES = all_nodes
    if data is not None:
        GLOBAL_DATA = data
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

    # --- 2-hop ---
    n2_connecting = defaultdict(set)  # (type_str, idx) -> set of connecting 1-hop (type_str, idx)
    seed_key = (type_to_idx[node_type], node_idx)

    for k in range(len(n1_types)):
        nbr_tid = int(n1_types[k])
        nbr_idx = int(n1_indices[k])
        nbr_type_str = idx_to_type[nbr_tid]

        csr_nbr = csr_adj[nbr_type_str]
        s2 = int(csr_nbr["offsets"][nbr_idx])
        e2 = int(csr_nbr["offsets"][nbr_idx + 1])
        n2_t = csr_nbr["nbr_types"][s2:e2]
        n2_i = csr_nbr["nbr_indices"][s2:e2]

        # Cap if too many
        if len(n2_t) > max_2hop_threshold:
            sel2 = np.random.choice(len(n2_t), max_2hop_threshold, replace=False)
            n2_t = n2_t[sel2]
            n2_i = n2_i[sel2]

        # Vectorized temporal filter
        mask2 = np.ones(len(n2_t), dtype=bool)
        for tid2 in np.unique(n2_t):
            t_str2 = idx_to_type[int(tid2)]
            if t_str2 in time_arrays:
                tm = n2_t == tid2
                mask2[tm] = time_arrays[t_str2][n2_i[tm]] <= seed_time
        n2_t = n2_t[mask2]
        n2_i = n2_i[mask2]

        connecting_1hop_str = (nbr_type_str, nbr_idx)
        for j in range(len(n2_t)):
            key = (int(n2_t[j]), int(n2_i[j]))
            # Skip seed node and 1-hop nodes
            if key == seed_key:
                continue
            if key in n1_set:
                continue
            n2_connecting[(idx_to_type[key[0]], key[1])].add(connecting_1hop_str)

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

    Uses GLOBAL_ADJ (CSR), GLOBAL_DATA, GLOBAL_TIME_ARRAYS, GLOBAL_ALL_NODES
    set via init_worker_globals — no pickling of heavy objects per task.
    """
    global GLOBAL_ADJ, GLOBAL_ALL_NODES, GLOBAL_DATA, GLOBAL_TIME_ARRAYS

    (K, seed_node_type, seed_node_idx, seed_time, seed_val) = args
    data = GLOBAL_DATA
    random.seed(seed_val)

    # 1. gather 1-hop and 2-hop — use vectorized path if CSR available
    if GLOBAL_TIME_ARRAYS is not None and isinstance(GLOBAL_ADJ, dict) and \
       len(GLOBAL_ADJ) > 0 and "offsets" in next(iter(GLOBAL_ADJ.values())):
        # CSR path
        type_to_idx = {nt: i for i, nt in enumerate(data.node_types)}
        T_hat = gather_1_and_2_hop_vectorized(
            GLOBAL_ADJ, GLOBAL_TIME_ARRAYS, type_to_idx, data.node_types,
            seed_node_type, seed_node_idx, seed_time
        )
    else:
        # Legacy path (old dict-of-sets adjacency)
        T_hat = gather_1_and_2_hop_with_seed_time(
            GLOBAL_ADJ, data, seed_node_type, seed_node_idx, seed_time
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
        if K_minus_1 <= len(GLOBAL_ALL_NODES):
            fallback = random.sample(GLOBAL_ALL_NODES, K_minus_1)
        else:
            fallback = random.choices(GLOBAL_ALL_NODES, k=K_minus_1)
        chosen_neighbors = []
        time_arrays = GLOBAL_TIME_ARRAYS or {}
        for (ft, fi) in fallback:
            if ft in time_arrays:
                ft_time = float(time_arrays[ft][fi])
                rel_time = (seed_time - ft_time) / (60 * 60 * 24)
            elif hasattr(data[ft], "time"):
                ft_time = data[ft].time[fi].item()
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

    # build adjacency among these K nodes using CSR
    local_map = {}
    for j, (t_str, i, hop, t_val, c1hops) in enumerate(final_tokens):
        local_map[(t_str, i)] = j

    edges = []
    adj = GLOBAL_ADJ
    is_csr = isinstance(adj, dict) and len(adj) > 0 and "offsets" in next(iter(adj.values()))
    if is_csr:
        idx_to_type = {ii: nt for ii, nt in enumerate(data.node_types)}

    for j_src, (t_str, i, hop, t_val, c1hops) in enumerate(final_tokens):
        if is_csr:
            csr_nt = adj[t_str]
            s = int(csr_nt["offsets"][i])
            e = int(csr_nt["offsets"][i + 1])
            for k in range(s, e):
                nbr_t = idx_to_type[int(csr_nt["nbr_types"][k])]
                nbr_i = int(csr_nt["nbr_indices"][k])
                if (nbr_t, nbr_i) in local_map:
                    edges.append((j_src, local_map[(nbr_t, nbr_i)]))
        else:
            for (nbr_t, nbr_i) in adj[t_str][i]:
                if (nbr_t, nbr_i) in local_map:
                    edges.append((j_src, local_map[(nbr_t, nbr_i)]))

    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int32)
    else:
        arr = np.array(edges, dtype=np.int32)
        edge_index = arr.T  # shape [2, E]

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

    if pool is not None:
        # Reuse existing pool (globals already initialized)
        results = pool.map(_process_one_seed, tasks)
    else:
        # Legacy: create a pool on the fly (for backward compat)
        global GLOBAL_ADJ, GLOBAL_ALL_NODES, GLOBAL_DATA, GLOBAL_TIME_ARRAYS

        if GLOBAL_ADJ is None:
            GLOBAL_ADJ = build_adjacency_csr(data, undirected=undirected)

        if GLOBAL_ALL_NODES is None:
            all_nodes = []
            for nt in data.node_types:
                for i in range(data[nt].num_nodes):
                    all_nodes.append((nt, i))
            GLOBAL_ALL_NODES = all_nodes

        if GLOBAL_TIME_ARRAYS is None:
            GLOBAL_TIME_ARRAYS = {}
            for nt in data.node_types:
                if hasattr(data[nt], "time"):
                    GLOBAL_TIME_ARRAYS[nt] = data[nt].time.numpy()

        GLOBAL_DATA = data

        if num_workers is None:
            num_workers = max(1, min(cpu_count() - 1, len(tasks)))

        with Pool(
            processes=num_workers,
            initializer=init_worker_globals,
            initargs=(GLOBAL_ADJ, GLOBAL_ALL_NODES, data, GLOBAL_TIME_ARRAYS)
        ) as p:
            results = p.map(_process_one_seed, tasks)

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
            if os.path.exists(self.precomputed_path):
                print(f"[{self.split}] Found existing HDF5 at {self.precomputed_path}")
            else:
                print(f"[{self.split}] Precomputing neighbor sampling (K={self.K})...")
                self._precompute_sampling()

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
        Run neighbor sampling in chunks and store expansions in HDF5.

        Key optimizations vs. original:
        - CSR adjacency instead of dict-of-sets (less memory, faster pickle)
        - Pre-extracted numpy time arrays (vectorized temporal filtering)
        - Single Pool created ONCE, reused across all chunks
        - data/adj/time_arrays passed via init_worker_globals, NOT per-task
        """
        total = len(self.node_idxs)
        chunk_size = 10000
        data_cpu = self.data.to("cpu")
        t_total_start = time.time()

        # Build CSR adjacency + time arrays ONCE
        t_csr_start = time.time()
        csr_adj = build_adjacency_csr(data_cpu, undirected=self.undirected)
        time_arrays = {}
        for nt in data_cpu.node_types:
            if hasattr(data_cpu[nt], "time"):
                time_arrays[nt] = data_cpu[nt].time.numpy()

        all_nodes = []
        for nt in data_cpu.node_types:
            for i in range(data_cpu[nt].num_nodes):
                all_nodes.append((nt, i))

        print(f"[{self.split}] CSR adjacency built in {time.time() - t_csr_start:.1f}s")

        num_workers = self.num_workers
        if num_workers is None:
            num_workers = max(1, min(cpu_count() - 1, total))

        print(f"[{self.split}] Starting sampling: {total} seeds, K={self.K}, {num_workers} workers")
        t_sample_start = time.time()

        with h5py.File(self.precomputed_path, 'w') as hf:
            datasets = self._create_datasets(hf, total)

            adjacency_all = [None] * total

            # Single Pool for ALL chunks — adjacency/data pickled only once
            with Pool(
                processes=num_workers,
                initializer=init_worker_globals,
                initargs=(csr_adj, all_nodes, data_cpu, time_arrays)
            ) as pool:

                with tqdm(total=total, desc=f"Precomputing '{self.split}'") as pbar:
                    for start_idx in range(0, total, chunk_size):
                        end_idx = min(start_idx + chunk_size, total)
                        size_chunk = end_idx - start_idx

                        chunk_node_idxs = self.node_idxs[start_idx:end_idx]
                        chunk_times = None
                        if self.time is not None:
                            chunk_times = self.time[start_idx:end_idx]

                        # Reuse the pool — no re-pickling of adj/data
                        S_chunk = local_nodes_hetero(
                            data=data_cpu,
                            K=self.K,
                            table_input_nodes=(self.node_type, chunk_node_idxs),
                            table_input_time=chunk_times,
                            undirected=self.undirected,
                            num_workers=num_workers,
                            pool=pool,
                        )

                        # arrays to fill
                        c_types = np.zeros((size_chunk, self.K), dtype=np.int16)
                        c_indices = np.zeros((size_chunk, self.K), dtype=np.int32)
                        c_hops = np.zeros((size_chunk, self.K), dtype=np.int8)
                        c_times = np.zeros((size_chunk, self.K), dtype=np.float32)

                        for i, node_id in enumerate(chunk_node_idxs):
                            final_nodes, edge_index = S_chunk[self.node_type][int(node_id)]
                            for j, (t_str, nbr_loc_idx, hop, t_val, c1hops) in enumerate(final_nodes):
                                c_types[i, j] = self.node_type_to_index[t_str]
                                c_indices[i, j] = nbr_loc_idx
                                c_hops[i, j] = hop
                                c_times[i, j] = t_val
                            adjacency_all[start_idx + i] = edge_index

                        datasets["types"][start_idx:end_idx] = c_types
                        datasets["indices"][start_idx:end_idx] = c_indices
                        datasets["hops"][start_idx:end_idx] = c_hops
                        datasets["times"][start_idx:end_idx] = c_times

                        pbar.update(size_chunk)

                        del S_chunk
                        gc.collect()

            # store adjacency in "edges" + "edges_offsets"
            offsets = np.zeros(total+1, dtype=np.uint64)
            for i in range(total):
                if adjacency_all[i] is not None:
                    E_i = adjacency_all[i].shape[1]  # [2, E]
                else:
                    E_i = 0
                offsets[i+1] = offsets[i] + E_i

            total_edges = offsets[-1]
            edges_dset = hf.create_dataset("edges", shape=(2, total_edges), dtype='int16')
            for i in range(total):
                e_arr = adjacency_all[i]
                start = offsets[i]
                end_ = offsets[i+1]
                if e_arr is not None and e_arr.size > 0:
                    edges_dset[:, start:end_] = e_arr

            hf.create_dataset("edges_offsets", data=offsets)

        elapsed = time.time() - t_total_start
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        print(f"[{self.split}] Sampling complete: {total} seeds in {int(hrs)}h {int(mins)}m {secs:.1f}s "
              f"(sampling: {time.time() - t_sample_start:.1f}s)")

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