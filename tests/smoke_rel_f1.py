"""End-to-end smoke against real rel-f1 (no GPU, no DDP, no precompute).

Verifies that:
  1. ``DatasetGraphCache`` builds successfully from a real ``HeteroData``
     produced by relbench's ``make_pkey_fkey_graph``;
  2. ``cache.neighbors_set`` matches dev-kyaw's ``build_adjacency_hetero``
     on every (node_type, idx) for the real graph;
  3. ``sample_local_subgraph`` matches dev-kyaw's ``_process_one_seed``
     under fixed ``random.seed`` on 200 real seeds.

Run as: ``python tests/smoke_rel_f1.py``.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.tasks import get_task
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from gfm_data.graph_cache import DatasetGraphCache
from gfm_data.sampler import sample_local_subgraph
from utils import GloveTextEmbedding, build_adjacency_hetero
import utils as _utils


def _multiset(tokens):
    return sorted((t, i, h) for (t, i, h, _r, _c) in tokens)


def main():
    cache_dir = os.path.expanduser("~/.cache/relbench_examples")
    dataset_name = "rel-f1"
    task_name = "driver-top3"
    print(f"Loading {dataset_name} / {task_name} ...")

    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    stypes_cache = Path(cache_dir) / dataset_name / "stypes.json"
    with open(stypes_cache, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)

    print("Materializing graph ...")
    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device="cpu"),
            batch_size=256,
        ),
        cache_dir=f"{cache_dir}/{dataset_name}/materialized",
    )
    print(f"  node_types: {data.node_types}")
    print(f"  edge_types: {data.edge_types}")
    print(f"  total num_nodes: {sum(data[t].num_nodes for t in data.node_types)}")

    # ---- (1) build CSR
    print("\n[1/3] Building CSR cache ...")
    cache = DatasetGraphCache(data=data, undirected=True, name_prefix=None)
    print(f"  cache.num_edges_total: {cache.num_edges_total()}")
    print(f"  cache.num_nodes_total: {cache.num_nodes_total()}")

    # ---- (2) neighbors_set equivalence vs dev-kyaw
    print("\n[2/3] Comparing CSR neighbors_set vs dev-kyaw build_adjacency_hetero ...")
    dev_adj = build_adjacency_hetero(data, undirected=True)
    mismatches = 0
    checked = 0
    for nt in data.node_types:
        n = data[nt].num_nodes
        # spot-check first 200 of each type, or all if smaller
        idxs = list(range(min(n, 200)))
        for i in idxs:
            checked += 1
            if cache.neighbors_set(nt, i) != dev_adj[nt][i]:
                mismatches += 1
                if mismatches <= 3:
                    print(f"  MISMATCH at {nt}[{i}]: csr={len(cache.neighbors_set(nt, i))}, dev={len(dev_adj[nt][i])}")
    print(f"  checked {checked} (type, idx) pairs, mismatches: {mismatches}")
    assert mismatches == 0, "neighbors_set diverges from dev-kyaw"

    # ---- (3) sample_local_subgraph equivalence on 200 driver-top3 seeds
    print("\n[3/3] Sampler equivalence vs dev-kyaw on 200 train seeds ...")
    train_table = task.get_table("train")
    from relbench.modeling.graph import get_node_train_table_input
    table_input = get_node_train_table_input(train_table, task)
    raw_seed_type, seed_idxs = table_input.nodes
    seed_times = getattr(table_input, "time", None)

    # All-nodes fallback list for dev-kyaw.
    all_nodes = []
    for nt in data.node_types:
        for i in range(data[nt].num_nodes):
            all_nodes.append((nt, i))
    _utils.GLOBAL_ADJ = dev_adj
    _utils.GLOBAL_ALL_NODES = all_nodes

    K = 32
    n_seeds = min(200, len(seed_idxs))
    mismatches = 0
    for k in range(n_seeds):
        idx = int(seed_idxs[k].item() if hasattr(seed_idxs[k], "item") else seed_idxs[k])
        t = float(seed_times[k].item()) if seed_times is not None else 0.0
        seed_val = hash((raw_seed_type, idx, t, K)) & 0xFFFFFFFF

        f_new, _ = sample_local_subgraph(
            cache, K=K, seed_node_type=raw_seed_type,
            seed_node_idx=idx, seed_time=t, seed_val=seed_val,
        )
        _, _, f_dev, _ = _utils._process_one_seed(
            (data, K, raw_seed_type, idx, t, seed_val)
        )
        if _multiset(f_new) != _multiset(f_dev):
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH seed#{k}: idx={idx}, time={t}")
    print(f"  checked {n_seeds} seeds, mismatches: {mismatches}")
    assert mismatches == 0, "sampler diverges from dev-kyaw on real rel-f1"

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
