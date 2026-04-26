"""Offline precompute: build memmap sample shards for one (dataset, task, split).

Replaces the in-process ``RelGTTokens._precompute_sampling`` HDF5 path with a
sharded layout consumed by ``gfm_data.shard_io.ShardReader``.

Usage::

    python tools/precompute_shards.py \\
        --dataset rel-f1 --task driver-top3 \\
        --K 300 \\
        --shard_size 50000 \\
        --out_dir ~/.cache/relbench_examples/shards/rel-f1/driver-top3 \\
        --splits train val test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.tasks import get_task

from gfm_data.graph_cache import DatasetGraphCache
from gfm_data.sampler import sample_local_subgraph
from gfm_data.shard_io import ShardWriter
from gfm_data.stypes import filter_to_db_columns, load_or_generate_stypes
from utils import GloveTextEmbedding


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--K", type=int, default=300)
    p.add_argument("--shard_size", type=int, default=50_000)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/relbench_examples"))
    p.add_argument("--undirected", action="store_true", default=True)
    p.add_argument(
        "--name_prefix",
        default=None,
        help="Prefix for node type names (e.g. dataset name). MUST match "
             "the name_prefix the multi-task trainer uses at runtime, "
             "otherwise the shard type ids won't match the unified type "
             "map and TaskTokens._sample_from_shards will KeyError. "
             "For the launcher's multi-task path, pass the dataset name "
             "(e.g. 'rel-f1', 'rel-event'). Single-task / dev-kyaw "
             "compatible runs leave it None.",
    )
    return p.parse_args()


def load_data(args):
    dataset = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)

    stypes_path = Path(args.cache_dir) / args.dataset / "stypes.json"
    cs = load_or_generate_stypes(stypes_path, dataset, upto_test_timestamp=False)

    # Use GPU for text embedding when available; major speedup on big
    # datasets (rel-event ~41M rows).
    import torch as _torch
    embed_device = "cuda" if _torch.cuda.is_available() else "cpu"

    # upto_test_timestamp=False so test seed indices don't overflow the
    # entity tables. Temporal leakage is prevented at sampling time via
    # per-row seed_time filtering.
    db = dataset.get_db(upto_test_timestamp=False)
    # RelBench tasks (e.g., results-position) strip leakage-risk
    # columns from the source table; the stypes JSON still references
    # the full schema. Filter so make_pkey_fkey_graph's torch_frame
    # Dataset.__init__ doesn't ValueError on missing columns.
    cs = filter_to_db_columns(cs, db)
    data, _ = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=cs,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=embed_device),
            batch_size=512,
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized_full",
    )

    # Shard building only needs the structural part (CSR adjacency +
    # per-node timestamps). The TF columns can be tens of GB on big
    # datasets (rel-event ~25 GB) and are never read during sampling,
    # so drop them right after load to keep per-process RAM bounded.
    # The training run loads its own TF later via TFStoreReader.
    #
    # IMPORTANT: torch_geometric's NodeStorage.num_nodes is a property
    # that infers the count from whatever tensors the store has --
    # often the TensorFrame itself. Once we delete .tf, tables without
    # an explicit num_nodes or fallback tensor (like x) start returning
    # None for num_nodes and crash gfm_data/graph_cache.py:_num_nodes_of
    # with TypeError: int() argument ... not 'NoneType'. Pin the count
    # explicitly before dropping the TF.
    import gc as _gc
    for _nt in list(data.node_types):
        store = data[_nt]
        if hasattr(store, "tf"):
            try:
                n = store.num_nodes
                if n is not None:
                    store.num_nodes = int(n)
            except Exception:
                pass
            try:
                del store["tf"]
            except Exception:
                try:
                    delattr(store, "tf")
                except Exception:
                    pass
    _gc.collect()
    return data, task


def precompute_split(cache, task, split, K, out_dir, shard_size):
    """Write one split's shards under ``<out_dir>/<K>/<split>/``."""
    table = task.get_table(split)
    table_input = get_node_train_table_input(table, task)
    raw_seed_type, seed_idxs = table_input.nodes
    seed_times = getattr(table_input, "time", None)

    n = len(seed_idxs)
    split_root = os.path.join(out_dir, str(K), split)
    os.makedirs(split_root, exist_ok=True)
    writer = ShardWriter(split_root, K=K, total_samples=n, shard_size=shard_size)

    print(f"[{split}] precomputing {n} samples into {writer.num_shards} shards "
          f"of up to {shard_size} ...")

    type_to_id = cache.node_type_to_index
    seed_node_type_prefixed = cache.raw_to_prefixed[raw_seed_type]

    for s_idx in range(writer.num_shards):
        lo, hi = writer.shard_range(s_idx)
        size = hi - lo
        types = np.zeros((size, K), dtype=np.int16)
        indices = np.zeros((size, K), dtype=np.int32)
        hops = np.zeros((size, K), dtype=np.int8)
        times = np.zeros((size, K), dtype=np.float32)
        edges = []

        for k in tqdm(range(size), desc=f"shard {s_idx:04d}", leave=False):
            global_k = lo + k
            node_idx_t = seed_idxs[global_k]
            node_idx = int(node_idx_t.item() if hasattr(node_idx_t, "item") else node_idx_t)
            seed_t = float(seed_times[global_k].item()) if seed_times is not None else 0.0
            seed_val = hash((seed_node_type_prefixed, node_idx, seed_t, K)) & 0xFFFFFFFF

            final_nodes, edge_index = sample_local_subgraph(
                cache, K=K, seed_node_type=seed_node_type_prefixed,
                seed_node_idx=node_idx, seed_time=seed_t, seed_val=seed_val,
            )
            for j, (t_str, nbr_loc, hop, t_val, _c) in enumerate(final_nodes):
                types[k, j] = type_to_id[t_str]
                indices[k, j] = nbr_loc
                hops[k, j] = hop
                times[k, j] = t_val
            edges.append(edge_index)

        writer.write_shard(s_idx, types, indices, hops, times, edges)
    writer.finalize()
    print(f"[{split}] done: {split_root}")


def main():
    args = parse_args()
    data, task = load_data(args)
    cache = DatasetGraphCache(
        data=data, undirected=args.undirected, name_prefix=args.name_prefix,
    )
    for split in args.splits:
        precompute_split(cache, task, split, args.K, args.out_dir, args.shard_size)


if __name__ == "__main__":
    main()
