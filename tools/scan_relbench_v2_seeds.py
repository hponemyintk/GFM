"""Per-task seed counts + 1/2-hop neighbor distribution across RelBench v2.

For each BINARY_CLASSIFICATION / REGRESSION task, builds the FK-only adjacency
(skipping text embedding so it runs on CPU in seconds), samples N seeds from
the train split, and reports:

  - split sizes (train / val / test)
  - 1-hop raw fanout (set size before cap/filter)
  - 1-hop kept (after cap=5000 + time filter)
  - 2-hop kept (after cap=1000 per 1-hop + time filter + dedupe)
  - "high-fanout 2-hop" count per seed (CSR fanout >= HIGH_FANOUT)

Writes a Markdown table to stdout and a JSON dump to --out_json.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import json
import os
import random
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from gfm_data.graph_cache import DatasetGraphCache
from gfm_data.sampler import gather_1_and_2_hop
from gfm_data.task_tokens import coerce_string_target_to_numeric


DATASETS_DEFAULT = [
    "rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm",
    "rel-stack", "rel-trial", "rel-arxiv", "rel-ratebeer",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--n_seeds", type=int, default=200)
    p.add_argument("--high_fanout", type=int, default=50_000,
                   help="Threshold for counting 'high-fanout 2-hop' tokens per seed")
    p.add_argument("--out_json", default="results/relbench_v2_seed_scan.json")
    p.add_argument("--out_md",   default="results/relbench_v2_seed_scan.md")
    return p.parse_args()


def build_fk_graph(db) -> HeteroData:
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


def stats(arr) -> dict:
    if not arr:
        return {"n": 0}
    a = np.asarray(arr)
    return {
        "n": int(a.size),
        "median": int(np.median(a)),
        "mean": float(a.mean()),
        "p95": int(np.percentile(a, 95)),
        "max": int(a.max()),
    }


def csr_fanout(cache: DatasetGraphCache, t_str: str, idx: int) -> int:
    block = cache.csr[t_str]
    if 0 <= idx < block.indptr.shape[0] - 1:
        return int(block.indptr[idx + 1] - block.indptr[idx])
    return 0


def scan_task(cache: DatasetGraphCache, ds_name: str, task, task_name: str,
              n_seeds: int, high_fanout: int) -> dict:
    """Return per-task summary dict; raises if anything goes wrong."""
    from relbench.modeling.graph import get_node_train_table_input

    # Split sizes (cheap).
    sizes: dict[str, int | None] = {}
    for split in ("train", "val", "test"):
        try:
            tab = task.get_table(split)
            sizes[split] = int(len(tab.df))
        except Exception:
            sizes[split] = None

    # Sample N seeds from train.
    train_tab = task.get_table("train")
    coerce_string_target_to_numeric(train_tab, task.target_col)
    ti = get_node_train_table_input(train_tab, task)
    raw_seed_type, seed_idxs = ti.nodes
    seed_times = getattr(ti, "time", None)
    seed_type_p = cache.raw_to_prefixed[raw_seed_type]

    n_total = len(seed_idxs)
    random.seed(0)
    positions = random.sample(range(n_total), min(n_seeds, n_total))

    raw_1hop, kept_1hop, kept_2hop, hf_count = [], [], [], []

    t0 = time.perf_counter()
    for p in positions:
        nidx_v = seed_idxs[p]
        nidx = int(nidx_v.item() if hasattr(nidx_v, "item") else nidx_v)
        st = float(seed_times[p].item() if seed_times is not None else 0.0)
        # raw 1-hop fanout (before cap/filter)
        raw_1hop.append(csr_fanout(cache, seed_type_p, nidx))
        # full gather (post-cap, post-time-filter, post-dedupe)
        tokens = gather_1_and_2_hop(cache, seed_type_p, nidx, st)
        ones = [t for t in tokens if t[2] == 1]
        twos = [t for t in tokens if t[2] == 2]
        kept_1hop.append(len(ones))
        kept_2hop.append(len(twos))
        # high-fanout 2-hop tokens (e.g. Category-like)
        hf = 0
        for (t_str, idx2, _hop, _t, _c) in twos:
            if csr_fanout(cache, t_str, idx2) >= high_fanout:
                hf += 1
        hf_count.append(hf)
    wall = time.perf_counter() - t0

    return {
        "dataset": ds_name,
        "task": task_name,
        "task_type": str(task.task_type).split(".")[-1],
        "entity_table": getattr(task, "entity_table", "?"),
        "split_sizes": sizes,
        "1hop_raw": stats(raw_1hop),
        "1hop_kept": stats(kept_1hop),
        "2hop_kept": stats(kept_2hop),
        "high_fanout_2hop_per_seed": stats(hf_count),
        "scan_wall_s": round(wall, 2),
        "scan_n_sampled": len(positions),
    }


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Quiet relbench loader prints (they go to stdout).
    with contextlib.redirect_stdout(sys.stderr):
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task_names, get_task
        from relbench.base import TaskType

    ALLOWED = {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}

    out: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for ds_name in datasets:
        print(f"\n=== {ds_name} ===", flush=True)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                ds = get_dataset(ds_name, download=True)
                db = ds.get_db(upto_test_timestamp=True)
            t0 = time.perf_counter()
            data = build_fk_graph(db)
            t_build = time.perf_counter() - t0
            t0 = time.perf_counter()
            cache = DatasetGraphCache(data, undirected=True, name_prefix=ds_name)
            t_csr = time.perf_counter() - t0
            print(f"  graph: {len(data.node_types)} types, "
                  f"{cache.num_edges_total():,} edges  "
                  f"(build={t_build:.1f}s csr={t_csr:.1f}s)", flush=True)

            try:
                task_names = list(get_task_names(ds_name))
            except Exception as e:
                print(f"  ! cannot list tasks: {e}", flush=True)
                failures.append({"dataset": ds_name, "task": "*", "error": str(e)})
                continue

            for tn in task_names:
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        task = get_task(ds_name, tn, download=True)
                    if task.task_type not in ALLOWED:
                        print(f"  - {tn} skipped (task_type={task.task_type})", flush=True)
                        continue
                    summary = scan_task(cache, ds_name, task, tn,
                                        args.n_seeds, args.high_fanout)
                    out.append(summary)
                    s = summary
                    print(f"  ✓ {tn:36s}  "
                          f"train={s['split_sizes']['train']:>10}  "
                          f"val={s['split_sizes']['val']:>9}  "
                          f"test={s['split_sizes']['test']:>9}  "
                          f"1hop_med={s['1hop_kept']['median']:>5}  "
                          f"2hop_med={s['2hop_kept']['median']:>5}  "
                          f"hf2hop_med={s['high_fanout_2hop_per_seed']['median']}", flush=True)
                except Exception as e:
                    print(f"  ✗ {tn}: {e}", flush=True)
                    traceback.print_exc(file=sys.stderr)
                    failures.append({"dataset": ds_name, "task": tn, "error": str(e)})
        except Exception as e:
            print(f"  !! dataset failed: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)
            failures.append({"dataset": ds_name, "task": "<dataset-load>", "error": str(e)})
        finally:
            for v in ("data", "cache", "db", "ds"):
                if v in locals():
                    del locals()[v]
            gc.collect()

    # Write JSON.
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"results": out, "failures": failures,
                   "args": {"n_seeds": args.n_seeds,
                            "high_fanout": args.high_fanout}},
                  f, indent=2)
    print(f"\nJSON: {args.out_json}", flush=True)

    # Write Markdown table.
    lines = [
        "# RelBench v2 — per-task seed counts + 1/2-hop neighbor distribution\n",
        f"_n_seeds_sampled_per_task = {args.n_seeds}; "
        f"high_fanout_threshold = {args.high_fanout:,} (CSR row size); "
        f"caps: 5000 (1-hop), 1000 (per 1-hop, 2-hop)._\n",
        "| dataset | task | type | seed table | train | val | test | 1h raw med | 1h raw p95 | 1h raw max | 1h kept med | 2h kept med | 2h kept p95 | hi-fanout 2h med |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in out:
        sz = s["split_sizes"]
        lines.append(
            f"| {s['dataset']} | {s['task']} | {s['task_type']} | "
            f"{s['entity_table']} | "
            f"{sz['train'] if sz['train'] is not None else '—'} | "
            f"{sz['val']   if sz['val']   is not None else '—'} | "
            f"{sz['test']  if sz['test']  is not None else '—'} | "
            f"{s['1hop_raw']['median']} | {s['1hop_raw']['p95']} | {s['1hop_raw']['max']} | "
            f"{s['1hop_kept']['median']} | "
            f"{s['2hop_kept']['median']} | {s['2hop_kept']['p95']} | "
            f"{s['high_fanout_2hop_per_seed']['median']} |"
        )
    if failures:
        lines.append("\n## Failures\n")
        lines.append("| dataset | task | error |\n|---|---|---|")
        for fr in failures:
            lines.append(f"| {fr['dataset']} | {fr['task']} | {fr['error']} |")
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown: {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
