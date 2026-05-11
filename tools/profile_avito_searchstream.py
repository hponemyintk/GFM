"""Profile the sampler on rel-avito.searchstream-click without materializing TF.

Bypasses make_pkey_fkey_graph (which would run text embedding for ~10M rows
on CPU, taking hours). Builds HeteroData with just edges + num_nodes + time,
which is everything the structural sampler reads. Then runs cProfile on
N seeds and attributes cache.neighbors_set time by source node type and by
call site (gather vs edge-construction).
"""
from __future__ import annotations
import argparse
import contextlib
import cProfile
import io
import os
import pstats
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from gfm_data.graph_cache import DatasetGraphCache
from gfm_data import sampler as sampler_mod
from gfm_data.sampler import sample_local_subgraph
from gfm_data.task_tokens import coerce_string_target_to_numeric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="rel-avito")
    p.add_argument("--task", default="searchstream-click")
    p.add_argument("--K", type=int, default=300)
    p.add_argument("--n_seeds", type=int, default=200)
    p.add_argument("--split", default="train")
    p.add_argument("--top", type=int, default=25, help="cProfile top-N to print")
    p.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/relbench_examples"))
    return p.parse_args()


def build_fk_graph(db) -> HeteroData:
    """Build a HeteroData with FK edges + num_nodes + time, no TF columns."""
    data = HeteroData()
    pkey_idx_map: dict[str, pd.Series] = {}
    for tname, tbl in db.table_dict.items():
        df = tbl.df
        data[tname].num_nodes = int(len(df))
        if tbl.pkey_col is not None:
            pkey_idx_map[tname] = pd.Series(
                np.arange(len(df), dtype=np.int64),
                index=df[tbl.pkey_col].values,
            )
        if tbl.time_col is not None:
            t = pd.to_datetime(df[tbl.time_col]).astype("int64").to_numpy() / 1e9
            data[tname].time = torch.from_numpy(t.astype(np.float32))
    for tname, tbl in db.table_dict.items():
        df = tbl.df
        for fk_col, pk_table in tbl.fkey_col_to_pkey_table.items():
            src = np.arange(len(df), dtype=np.int64)
            mapped = pkey_idx_map[pk_table].reindex(df[fk_col].values).to_numpy()
            valid = ~np.isnan(mapped)
            src = src[valid]
            dst = mapped[valid].astype(np.int64)
            ei = torch.from_numpy(np.stack([src, dst]))
            data[(tname, f"f2p_{fk_col}", pk_table)].edge_index = ei
    return data


# --- per-call-site instrumentation for cache.neighbors_set ---
_SITE_STACK: list[str] = []  # "gather" / "edge"
_per_site_counts: dict[tuple[str, str], int] = defaultdict(int)
_per_site_fanout: dict[tuple[str, str], int] = defaultdict(int)
_per_site_time: dict[tuple[str, str], float] = defaultdict(float)


def make_instrumented_neighbors_set(orig):
    def inst(self, src_type, src_idx):
        block = self.csr[src_type]
        if 0 <= src_idx < block.indptr.shape[0] - 1:
            fanout = int(block.indptr[src_idx + 1] - block.indptr[src_idx])
        else:
            fanout = 0
        site = _SITE_STACK[-1] if _SITE_STACK else "?"
        t0 = time.perf_counter()
        res = orig(self, src_type, src_idx)
        dt = time.perf_counter() - t0
        key = (site, src_type)
        _per_site_counts[key] += 1
        _per_site_fanout[key] += fanout
        _per_site_time[key] += dt
        return res

    return inst


def main():
    args = parse_args()

    print(f"[1/4] Loading {args.dataset} parquet tables (no text embedding) ...")
    with contextlib.redirect_stdout(sys.stderr):
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task
        from relbench.modeling.graph import get_node_train_table_input

        ds = get_dataset(args.dataset, download=True)
        task = get_task(args.dataset, args.task, download=True)
        db = ds.get_db(upto_test_timestamp=True)

    t0 = time.perf_counter()
    data = build_fk_graph(db)
    print(f"  HeteroData built in {time.perf_counter() - t0:.1f}s "
          f"(node_types={len(data.node_types)} edge_types={len(data.edge_types)})")

    print(f"[2/4] Building DatasetGraphCache (CSR) ...")
    t0 = time.perf_counter()
    cache = DatasetGraphCache(data=data, undirected=True, name_prefix=args.dataset)
    print(f"  CSR built in {time.perf_counter() - t0:.1f}s "
          f"(num_edges_total={cache.num_edges_total():,})")

    table = task.get_table(args.split)
    coerce_string_target_to_numeric(table, task.target_col)
    table_input = get_node_train_table_input(table, task)
    raw_seed_type, seed_idxs = table_input.nodes
    seed_times = getattr(table_input, "time", None)
    seed_node_type_prefixed = cache.raw_to_prefixed[raw_seed_type]
    n_total = len(seed_idxs)

    random.seed(0)
    positions = random.sample(range(n_total), min(args.n_seeds, n_total))
    work = []
    for p in positions:
        nidx = int(seed_idxs[p].item() if hasattr(seed_idxs[p], "item") else seed_idxs[p])
        st = float(seed_times[p].item() if seed_times is not None else 0.0)
        sv = hash((seed_node_type_prefixed, nidx, st, args.K)) & 0xFFFFFFFF
        work.append((nidx, st, sv))

    print(f"[3/4] Sampling {len(work)} seeds (K={args.K}, seed_type={seed_node_type_prefixed}) ...")

    # Patch cache.neighbors_set with the instrumented version. Push site
    # tags from a wrapped sample_local_subgraph so we can split gather vs
    # edge-construction time.
    DatasetGraphCache.neighbors_set = make_instrumented_neighbors_set(
        DatasetGraphCache.neighbors_set
    )

    orig_gather = sampler_mod.gather_1_and_2_hop

    def gather_with_site(*a, **kw):
        _SITE_STACK.append("gather")
        try:
            return orig_gather(*a, **kw)
        finally:
            _SITE_STACK.pop()

    sampler_mod.gather_1_and_2_hop = gather_with_site

    def sample_with_site(c, K, seed_node_type, seed_node_idx, seed_time, seed_val):
        # everything outside gather is edge construction
        _SITE_STACK.append("edge")
        try:
            return sample_local_subgraph(c, K, seed_node_type, seed_node_idx,
                                         seed_time, seed_val)
        finally:
            _SITE_STACK.pop()

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    for nidx, st, sv in work:
        sample_with_site(cache, args.K, seed_node_type_prefixed, nidx, st, sv)
    wall = time.perf_counter() - t0
    pr.disable()

    print(f"\n=== {len(work)} seeds in {wall:.2f}s "
          f"({wall / max(len(work),1) * 1000:.1f} ms/seed) ===")

    # Per-call-site summary.
    print("\n=== cache.neighbors_set: by call-site x src_type ===")
    print(f"{'site':6s} {'src_type':40s} {'calls':>10s} {'avg_fanout':>12s} {'total_s':>10s} {'%seed':>7s}")
    rows = sorted(_per_site_time.items(), key=lambda x: -x[1])
    total_set_time = sum(_per_site_time.values())
    for (site, t), tt in rows:
        calls = _per_site_counts[(site, t)]
        avg_f = _per_site_fanout[(site, t)] / max(calls, 1)
        pct = 100 * tt / max(wall, 1e-9)
        print(f"{site:6s} {t:40s} {calls:>10,} {avg_f:>12,.0f} {tt:>10.3f} {pct:>6.1f}%")
    print(f"\nTotal neighbors_set time: {total_set_time:.2f}s "
          f"({100 * total_set_time / max(wall, 1e-9):.1f}% of seed wall)")

    # Site totals.
    print("\n=== site totals ===")
    site_totals: dict[str, float] = defaultdict(float)
    site_calls: dict[str, int] = defaultdict(int)
    for (site, _), tt in _per_site_time.items():
        site_totals[site] += tt
    for (site, _), c in _per_site_counts.items():
        site_calls[site] += c
    for site in ("gather", "edge"):
        print(f"  {site:6s}  calls={site_calls[site]:>10,}  "
              f"time={site_totals[site]:>8.3f}s  "
              f"{100*site_totals[site]/max(wall,1e-9):>5.1f}% of wall")

    print(f"\n=== cProfile cumulative top {args.top} ===")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(args.top)
    print(s.getvalue())

    print(f"=== cProfile self-time top {args.top} ===")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(args.top)
    print(s.getvalue())


if __name__ == "__main__":
    main()
