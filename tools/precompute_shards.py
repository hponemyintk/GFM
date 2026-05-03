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
from gfm_data.task_tokens import coerce_string_target_to_numeric
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
        "--workers", type=int,
        default=int(os.environ.get("SHARD_WORKERS", 1)),
        help="Intra-shard parallelism via multiprocessing. Workers fork "
             "after the cache is built and inherit it via copy-on-write, "
             "so per-worker RSS is the inherited steady-state plus a few "
             "hundred MB of sampling scratch. Default 1 (sequential, "
             "matches the original behavior). Bit-exact regardless of "
             "worker count -- each sample is self-seeded by hash, results "
             "are gathered indexed by k.",
    )
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
    p.add_argument(
        "--full_graph", action="store_true", default=False,
        help="Use upto_test_timestamp=False so the entity tables include "
             "rows created after train_cutoff. Required for autocomplete "
             "tasks (users-birthyear, results-position, qualifying-position, "
             "transactions-price) whose val/test seeds reference entities "
             "added after the cutoff -- the truncated CSR adjacency would "
             "IndexError on those seeds. Temporal leakage is still "
             "prevented by the per-neighbor seed_time filter at "
             "gfm_data/sampler.py:69. Materialization is cached separately "
             "under <cache_dir>/<dataset>/materialized_full so the "
             "truncated build (default) is not overwritten.",
    )
    return p.parse_args()


def load_data(args):
    dataset = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)

    stypes_path = Path(args.cache_dir) / args.dataset / "stypes.json"
    upto = not args.full_graph
    cs = load_or_generate_stypes(stypes_path, dataset, upto_test_timestamp=upto)

    # Use GPU for text embedding when available; major speedup on big
    # datasets (rel-event ~41M rows).
    import torch as _torch
    embed_device = "cuda" if _torch.cuda.is_available() else "cpu"

    # upto_test_timestamp controls entity table truncation.
    # True (default): truncate at train_cutoff. Matches dev-kyaw / RelGT
    #   paper as a defense-in-depth guardrail. Works for forecasting
    #   tasks where seeds reference pre-existing entities.
    # False (--full_graph): keep all entities. Required for autocomplete
    #   tasks where val/test seeds reference entities created after
    #   train_cutoff (users-birthyear, results-position, ...).
    # In both cases the per-neighbor seed_time filter at
    # gfm_data/sampler.py:69 is the actual temporal leakage barrier.
    db = dataset.get_db(upto_test_timestamp=upto)
    # RelBench tasks (e.g., results-position) strip leakage-risk
    # columns from the source table; the stypes JSON still references
    # the full schema. Filter so make_pkey_fkey_graph's torch_frame
    # Dataset.__init__ doesn't ValueError on missing columns.
    cs = filter_to_db_columns(cs, db)
    mat_suffix = "materialized_full" if args.full_graph else "materialized"
    data, _ = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=cs,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=embed_device),
            batch_size=512,
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/{mat_suffix}",
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


# Module-level globals populated in worker processes via Pool's
# initializer callback. Forked workers inherit the cache via
# copy-on-write so we don't pickle DatasetGraphCache (which holds
# numpy arrays + a HeteroData reference). Avoids ~GB pickle overhead
# per task and matches the design in speedup-precompute-shards.md.
_W_CACHE = None
_W_SEED_TYPE = None
_W_TYPE_TO_ID = None
_W_K = None


def _worker_init(cache, seed_type_prefixed, type_to_id, K):
    global _W_CACHE, _W_SEED_TYPE, _W_TYPE_TO_ID, _W_K
    _W_CACHE = cache
    _W_SEED_TYPE = seed_type_prefixed
    _W_TYPE_TO_ID = type_to_id
    _W_K = K


def _worker_sample(item):
    """Sample one seed; pack the result into row arrays.

    Each sample's RNG seed is derived from (type, node_idx, seed_t, K)
    so the per-shard output is identical regardless of worker
    assignment / completion order -- callers index by ``k`` into the
    pre-allocated arrays.
    """
    k, node_idx, seed_t = item
    seed_val = hash((_W_SEED_TYPE, node_idx, seed_t, _W_K)) & 0xFFFFFFFF
    final_nodes, edge_index = sample_local_subgraph(
        _W_CACHE, K=_W_K, seed_node_type=_W_SEED_TYPE,
        seed_node_idx=node_idx, seed_time=seed_t, seed_val=seed_val,
    )
    types_row = np.zeros(_W_K, dtype=np.int16)
    indices_row = np.zeros(_W_K, dtype=np.int32)
    hops_row = np.zeros(_W_K, dtype=np.int8)
    times_row = np.zeros(_W_K, dtype=np.float32)
    for j, (t_str, nbr_loc, hop, t_val, _c) in enumerate(final_nodes):
        types_row[j] = _W_TYPE_TO_ID[t_str]
        indices_row[j] = nbr_loc
        hops_row[j] = hop
        times_row[j] = t_val
    return k, types_row, indices_row, hops_row, times_row, edge_index


def precompute_split(cache, task, split, K, out_dir, shard_size, workers=1):
    """Write one split's shards under ``<out_dir>/<K>/<split>/``.

    ``workers`` controls intra-shard parallelism. ``workers=1`` runs
    the original sequential loop (no fork, no pool overhead). With
    ``workers > 1`` the cache is shared via fork+COW; output is
    bit-identical because each sample is self-seeded by a hash of
    ``(seed_type, node_idx, seed_t, K)``.
    """
    table = task.get_table(split)
    coerce_string_target_to_numeric(table, task.target_col)
    table_input = get_node_train_table_input(table, task)
    raw_seed_type, seed_idxs = table_input.nodes
    seed_times = getattr(table_input, "time", None)

    n = len(seed_idxs)
    split_root = os.path.join(out_dir, str(K), split)
    os.makedirs(split_root, exist_ok=True)
    writer = ShardWriter(split_root, K=K, total_samples=n, shard_size=shard_size)

    print(f"[{split}] precomputing {n} samples into {writer.num_shards} shards "
          f"of up to {shard_size} (workers={workers}) ...")

    type_to_id = cache.node_type_to_index
    seed_node_type_prefixed = cache.raw_to_prefixed[raw_seed_type]

    pool = None
    if workers > 1:
        # 'fork' is required so workers share the cache via COW. The
        # parent must not have initialized CUDA before this point;
        # phase2_build_one in pretrain_p4d.sh already sets
        # CUDA_VISIBLE_DEVICES="" so this is safe in production.
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        pool = ctx.Pool(
            workers,
            initializer=_worker_init,
            initargs=(cache, seed_node_type_prefixed, type_to_id, K),
        )
    else:
        # Sequential path: set the module globals once, here, so
        # _worker_sample can be reused across every shard without
        # re-binding _W_CACHE / _W_SEED_TYPE / _W_TYPE_TO_ID / _W_K
        # on each iteration.
        _worker_init(cache, seed_node_type_prefixed, type_to_id, K)

    try:
        for s_idx in range(writer.num_shards):
            lo, hi = writer.shard_range(s_idx)
            size = hi - lo
            types = np.zeros((size, K), dtype=np.int16)
            indices = np.zeros((size, K), dtype=np.int32)
            hops = np.zeros((size, K), dtype=np.int8)
            times = np.zeros((size, K), dtype=np.float32)
            edges = [None] * size

            # Build the work list once per shard. Pull seed times out
            # eagerly so the worker payload is plain Python types
            # (no torch tensors crossing the fork boundary).
            work = []
            for k in range(size):
                global_k = lo + k
                node_idx_t = seed_idxs[global_k]
                node_idx = int(
                    node_idx_t.item() if hasattr(node_idx_t, "item")
                    else node_idx_t
                )
                seed_t = (
                    float(seed_times[global_k].item())
                    if seed_times is not None else 0.0
                )
                work.append((k, node_idx, seed_t))

            if pool is None:
                # Sequential: globals already set above.
                for item in tqdm(work, desc=f"shard {s_idx:04d}", leave=False):
                    k, t_row, i_row, h_row, ti_row, edge_index = (
                        _worker_sample(item)
                    )
                    types[k] = t_row
                    indices[k] = i_row
                    hops[k] = h_row
                    times[k] = ti_row
                    edges[k] = edge_index
            else:
                # Parallel: imap_unordered for throughput; chunksize
                # amortizes IPC overhead across small pure-Python
                # samples. Order doesn't matter -- we index by k.
                it = pool.imap_unordered(
                    _worker_sample, work, chunksize=256,
                )
                for k, t_row, i_row, h_row, ti_row, edge_index in tqdm(
                    it, total=size, desc=f"shard {s_idx:04d}", leave=False,
                ):
                    types[k] = t_row
                    indices[k] = i_row
                    hops[k] = h_row
                    times[k] = ti_row
                    edges[k] = edge_index

            writer.write_shard(s_idx, types, indices, hops, times, edges)
        writer.finalize()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    print(f"[{split}] done: {split_root}")


def main():
    args = parse_args()
    data, task = load_data(args)
    cache = DatasetGraphCache(
        data=data, undirected=args.undirected, name_prefix=args.name_prefix,
    )
    for split in args.splits:
        precompute_split(
            cache, task, split, args.K, args.out_dir, args.shard_size,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
