"""
Compute average number of 2-hop neighbors for seed nodes in RelBench tasks.

The graph is built from all primary key - foreign key relationships in the
database, treated as an undirected graph.  Temporal filtering is applied per
seed node: a neighbor is only counted if its row timestamp < the seed node's
cutoff time (rows in tables with no time_col are always included).

For each seed node the count is the number of unique nodes reachable within
1 or 2 hops (seed node itself excluded).
"""

import argparse
import os
import statistics
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from relbench.datasets import get_dataset, get_dataset_names
from relbench.tasks import get_task, get_task_names


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_undirected_adj(db):
    """
    Build an undirected adjacency list and a per-node timestamp lookup.

    Each node is a tuple (table_name, row_index).  For every FK column in
    every table we add a bidirectional edge between the FK row and the
    referenced PK row.

    Returns
    -------
    adj : dict[tuple, set[tuple]]
        Maps each node to its set of directly connected neighbours.
    node_time : dict[tuple, pd.Timestamp | None]
        Per-node timestamp.  None means the node's table has no time column
        and the node should never be filtered out on temporal grounds.
    """
    adj = defaultdict(set)
    node_time = {}

    for table_name, table in db.table_dict.items():
        df = table.df
        times = df[table.time_col].values if table.time_col is not None else None

        # Record timestamp for every node in this table.
        # Store None (not pd.NaT) for missing/absent timestamps so that
        # the temporal filter (t < cutoff) never silently excludes them —
        # pd.NaT < any_timestamp evaluates to False, which would wrongly
        # drop nodes whose time is unknown.
        for row_idx in range(len(df)):
            if times is None or pd.isnull(times[row_idx]):
                node_time[(table_name, row_idx)] = None
            else:
                node_time[(table_name, row_idx)] = pd.Timestamp(times[row_idx])

        # Add edges for every FK relationship
        for fkey_col, pkey_table in table.fkey_col_to_pkey_table.items():
            col = df[fkey_col]
            valid_mask = col.notna()
            row_indices = np.where(valid_mask)[0]
            pkey_vals = col[valid_mask].astype(int).values

            for row_idx, pkey_val in zip(row_indices, pkey_vals):
                src = (table_name, int(row_idx))
                dst = (pkey_table, int(pkey_val))
                adj[src].add(dst)
                adj[dst].add(src)

    return adj, node_time


# ---------------------------------------------------------------------------
# 2-hop neighbour counting with temporal filtering
# ---------------------------------------------------------------------------

def count_2hop_neighbors_for_nodes(seed_nodes_with_cutoff, adj, node_time):
    """
    For each (seed_node, cutoff_time) pair, count unique nodes reachable
    within 1 or 2 hops where every traversed neighbor satisfies:

        node_time[neighbor] < cutoff_time   (if the neighbor has a timestamp)
        always included                      (if the neighbor has no timestamp)

    The seed node itself is not counted.

    Parameters
    ----------
    seed_nodes_with_cutoff : list[tuple[tuple, pd.Timestamp]]
        Each entry is ((table_name, row_idx), cutoff_time).
    adj : dict[tuple, set[tuple]]
    node_time : dict[tuple, pd.Timestamp | None]

    Returns
    -------
    counts : list[int]
        Total (1-hop + 2-hop) neighbour count per seed.
    hop1_counts : list[int]
        1-hop neighbour count per seed.
    hop2_counts : list[int]
        2-hop-only neighbour count per seed.
    hop1_type_per_seed : list[Counter]
        Per-seed node-type counts for 1-hop neighbours.
    hop2_type_per_seed : list[Counter]
        Per-seed node-type counts for 2-hop neighbours.
    """
    counts = []
    hop1_counts = []
    hop2_counts = []
    hop1_type_per_seed = []
    hop2_type_per_seed = []

    for node, cutoff in seed_nodes_with_cutoff:

        def temporally_valid(nbr):
            t = node_time.get(nbr)
            return t is None or t < cutoff

        # --- 1-hop ---
        one_hop = {nbr for nbr in adj[node] if temporally_valid(nbr)}

        visited = {node} | one_hop

        # --- 2-hop ---
        two_hop_new = set()
        for nbr in one_hop:
            for nbr2 in adj[nbr]:
                if nbr2 not in visited and temporally_valid(nbr2):
                    two_hop_new.add(nbr2)

        total = len(one_hop) + len(two_hop_new)
        counts.append(total)
        hop1_counts.append(len(one_hop))
        hop2_counts.append(len(two_hop_new))

        h1_ct = Counter()
        for nbr in one_hop:
            h1_ct[nbr[0]] += 1
        hop1_type_per_seed.append(h1_ct)

        h2_ct = Counter()
        for nbr in two_hop_new:
            h2_ct[nbr[0]] += 1
        hop2_type_per_seed.append(h2_ct)

    return counts, hop1_counts, hop2_counts, hop1_type_per_seed, hop2_type_per_seed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_avg_2hop_neighbors(dataset_task_dict, verbose=True, max_seeds=None):
    """
    Compute the average number of 2-hop neighbours for seed nodes in each
    split (train / val / test) for every (dataset, task) pair.

    Parameters
    ----------
    dataset_task_dict : dict[str, list[str]]
        Maps dataset names to lists of task names, e.g.
        {"rel-avito": [" ad-ctr"]}.
    verbose : bool
    max_seeds : int or None
        If set, randomly downsample each split to at most this many seeds
        before running the traversal.  Useful for approximating neighbourhood
        statistics on large datasets without paying the full O(N) cost.

    Returns
    -------
    results : dict
        results[dataset][task][split] = {
            "avg_2hop_neighbors":    float,
            "std_2hop_neighbors":    float,
            "median_2hop_neighbors": float,
            "mode_2hop_neighbors":   int,
            "num_seeds":             int,
            "sampled":               bool,
        }
    """
    results = {}

    for dataset_name, task_names in dataset_task_dict.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")

        dataset = get_dataset(dataset_name, download=True)
        db = dataset.get_db()

        if verbose:
            print("Building undirected adjacency list …")

        adj, node_time = build_undirected_adj(db)

        if verbose:
            print(f"  Graph has {len(adj):,} distinct nodes")

        results[dataset_name] = {}

        for task_name in task_names:
            if verbose:
                print(f"\n  Task: {task_name}")

            try:
                task = get_task(dataset_name, task_name, download=True)
            except Exception as exc:
                print(f"    [SKIP] could not load task: {exc}")
                continue

            if not hasattr(task, "entity_col"):
                print(
                    f"    [SKIP] task type {type(task).__name__} not supported "
                    "(only EntityTask); skipping."
                )
                continue

            entity_col   = task.entity_col
            entity_table = task.entity_table
            time_col     = task.time_col

            results[dataset_name][task_name] = {}

            for split in ("train", "val", "test"):
                mask_input = split == "test"
                try:
                    table = task.get_table(split, mask_input_cols=mask_input)
                except Exception as exc:
                    print(f"    [SKIP] {split}: {exc}")
                    continue
                df = table.df.dropna(subset=[entity_col])

                # One entry per task-table row; duplicates are intentionally kept.
                # Use vectorized column access instead of iterrows() — iterrows()
                # boxes every value into Python objects and is ~100x slower.
                entity_ids = df[entity_col].values.astype(int)
                cutoff_times = pd.to_datetime(df[time_col]).values

                seed_nodes_with_cutoff = [
                    ((entity_table, int(eid)), pd.Timestamp(ts))
                    for eid, ts in zip(entity_ids, cutoff_times)
                ]

                sampled = False
                if max_seeds is not None and len(seed_nodes_with_cutoff) > max_seeds:
                    rng = np.random.default_rng(seed=42)
                    idx = rng.choice(len(seed_nodes_with_cutoff), size=max_seeds, replace=False)
                    seed_nodes_with_cutoff = [seed_nodes_with_cutoff[i] for i in idx]
                    sampled = True

                counts, hop1_counts, hop2_counts, hop1_type_per_seed, hop2_type_per_seed = \
                    count_2hop_neighbors_for_nodes(
                        seed_nodes_with_cutoff, adj, node_time
                    )

                avg    = float(np.mean(counts))
                std    = float(np.std(counts))
                median = float(np.median(counts))
                mode   = int(statistics.mode(counts))

                all_node_types = set()
                for ct in hop1_type_per_seed:
                    all_node_types.update(ct.keys())
                for ct in hop2_type_per_seed:
                    all_node_types.update(ct.keys())

                hop1_type_stats = {}
                hop2_type_stats = {}
                for nt in all_node_types:
                    h1_arr = np.array([ct.get(nt, 0) for ct in hop1_type_per_seed])
                    h2_arr = np.array([ct.get(nt, 0) for ct in hop2_type_per_seed])
                    hop1_type_stats[nt] = {"mean": float(h1_arr.mean()), "std": float(h1_arr.std())}
                    hop2_type_stats[nt] = {"mean": float(h2_arr.mean()), "std": float(h2_arr.std())}

                results[dataset_name][task_name][split] = {
                    "avg_2hop_neighbors":    avg,
                    "std_2hop_neighbors":    std,
                    "median_2hop_neighbors": median,
                    "mode_2hop_neighbors":   mode,
                    "num_seeds":             len(seed_nodes_with_cutoff),
                    "sampled":               sampled,
                    "avg_1hop":              float(np.mean(hop1_counts)),
                    "std_1hop":              float(np.std(hop1_counts)),
                    "avg_2hop_only":         float(np.mean(hop2_counts)),
                    "std_2hop_only":         float(np.std(hop2_counts)),
                    "hop1_type_stats":       hop1_type_stats,
                    "hop2_type_stats":       hop2_type_stats,
                }

                if verbose:
                    sampled_tag = f" (sampled from {len(df):,})" if sampled else ""
                    print(
                        f"    {split:5s} | seeds: {len(seed_nodes_with_cutoff):,}{sampled_tag} | "
                        f"avg: {avg:10.2f} ± {std:.2f} | "
                        f"median: {median:.2f} | mode: {mode}"
                    )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mean_std_per_task(results, out_dir="plots"):
    """
    For each dataset, create 3 side-by-side subplots (1-hop, 2-hop-only, total)
    with grouped bars for train / val / test per task, showing mean ± std.
    """
    os.makedirs(out_dir, exist_ok=True)
    splits = ["train", "val", "test"]
    colors = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
    metrics = [
        ("1-hop neighbours",      "avg_1hop",            "std_1hop"),
        ("2-hop-only neighbours",  "avg_2hop_only",       "std_2hop_only"),
        ("Total (1+2 hop)",        "avg_2hop_neighbors",  "std_2hop_neighbors"),
    ]

    for dataset_name, tasks in results.items():
        task_names = list(tasks.keys())
        n_tasks = len(task_names)
        x = np.arange(n_tasks)
        width = 0.25

        fig, axes = plt.subplots(1, 3, figsize=(max(14, n_tasks * 6), 5))

        for ax, (title, mean_key, std_key) in zip(axes, metrics):
            for i, split in enumerate(splits):
                means = [tasks[t][split][mean_key] for t in task_names]
                stds  = [tasks[t][split][std_key] for t in task_names]
                ax.bar(
                    x + i * width, means, width,
                    yerr=stds, label=split,
                    color=colors[split], capsize=4, edgecolor="black", linewidth=0.5,
                )
            ax.set_xticks(x + width)
            ax.set_xticklabels(task_names, rotation=30, ha="right")
            ax.set_ylabel("Count (mean ± std)")
            ax.set_title(title)
            ax.legend()

        fig.suptitle(f"{dataset_name} — neighbour statistics per split", fontsize=14)
        fig.tight_layout()

        path = os.path.join(out_dir, f"{dataset_name}_neighbor_mean_std_by_split.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


def plot_node_type_tallies(results, out_dir="plots"):
    """
    For each (dataset, task), create a single plot with all splits and hops
    shown side by side per node type: 6 bars per node type
    (train-1hop, train-2hop, val-1hop, val-2hop, test-1hop, test-2hop).
    """
    os.makedirs(out_dir, exist_ok=True)
    split_order = ["train", "val", "test"]
    split_colors_1hop = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
    split_colors_2hop = {"train": "#7BA7D7", "val": "#8FD4A4", "test": "#E08888"}

    for dataset_name, tasks in results.items():
        for task_name, splits_data in tasks.items():
            all_types = set()
            for split_data in splits_data.values():
                all_types.update(split_data["hop1_type_stats"].keys())
                all_types.update(split_data["hop2_type_stats"].keys())
            all_types = sorted(all_types)
            if not all_types:
                continue

            n_bars = 6
            width = 0.12
            x = np.arange(len(all_types))

            fig, ax = plt.subplots(figsize=(max(10, len(all_types) * 2.5), 6))

            for si, split in enumerate(split_order):
                sd = splits_data[split]
                h1_means = [sd["hop1_type_stats"].get(nt, {"mean": 0})["mean"] for nt in all_types]
                h1_stds  = [sd["hop1_type_stats"].get(nt, {"std": 0})["std"] for nt in all_types]
                h2_means = [sd["hop2_type_stats"].get(nt, {"mean": 0})["mean"] for nt in all_types]
                h2_stds  = [sd["hop2_type_stats"].get(nt, {"std": 0})["std"] for nt in all_types]

                offset_1hop = (si * 2) * width - (n_bars - 1) * width / 2
                offset_2hop = (si * 2 + 1) * width - (n_bars - 1) * width / 2

                ax.bar(x + offset_1hop, h1_means, width, yerr=h1_stds,
                       label=f"{split} 1-hop", color=split_colors_1hop[split],
                       capsize=3, edgecolor="black", linewidth=0.5)
                ax.bar(x + offset_2hop, h2_means, width, yerr=h2_stds,
                       label=f"{split} 2-hop", color=split_colors_2hop[split],
                       capsize=3, edgecolor="black", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(all_types, rotation=45, ha="right")
            ax.set_ylabel("Neighbours per seed (mean ± std)")
            ax.set_title(f"{dataset_name} / {task_name} — node type by hop & split")
            ax.legend(ncol=3, fontsize=8)
            fig.tight_layout()

            path = os.path.join(out_dir, f"{dataset_name}_{task_name}_node_type_by_hop_and_split.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute average 2-hop neighbour count for RelBench tasks."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run over every dataset and task registered in relbench.",
    )
    mode.add_argument(
        "--datasets",
        nargs="+",
        metavar="DATASET",
        help=(
            "Run over all tasks for the given dataset(s), e.g. "
            "rel-f1 rel-amazon"
        ),
    )
    mode.add_argument(
        "--dataset-task",
        nargs="+",
        metavar="DATASET:TASK",
        help=(
            "One or more DATASET:TASK pairs, e.g. "
            "rel-f1:driver-top3  rel-amazon:user-churn"
        ),
    )
    parser.add_argument(
        "--sample-seeds",
        type=int,
        metavar="N",
        default=None,
        help=(
            "If set, randomly downsample each split to at most N seeds before "
            "running the 2-hop traversal.  Gives a fast approximation of "
            "neighbourhood statistics on large datasets (e.g. --sample-seeds 10000)."
        ),
    )
    args = parser.parse_args()

    if args.all:
        dataset_task_dict = {
            ds: get_task_names(ds) for ds in get_dataset_names()
        }
    elif args.datasets:
        dataset_task_dict = {
            ds: get_task_names(ds) for ds in args.datasets
        }
    elif args.dataset_task:
        dataset_task_dict = defaultdict(list)
        for pair in args.dataset_task:
            dataset_name, task_name = [s.strip() for s in pair.split(":", 1)]
            dataset_task_dict[dataset_name].append(task_name)
        dataset_task_dict = dict(dataset_task_dict)
    else:
        # default predefined selection
        dataset_task_dict = {
            "rel-f1":     ["driver-position", "driver-dnf", "driver-top3"],
            "rel-avito":  ["ad-ctr", "user-clicks", "user-visits"],
            "rel-event":  ["user-attendance", "user-repeat", "user-ignore"],
            "rel-trial":  ["study-adverse", "study-outcome", "site-success"],
            "rel-amazon": ["user-ltv", "item-ltv", "user-churn", "item-churn"],
            "rel-stack":  ["post-votes", "user-engagement", "user-badge"],
            "rel-hm":     ["item-sales", "user-churn"],
        }

    results = compute_avg_2hop_neighbors(dataset_task_dict, max_seeds=args.sample_seeds)

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset_name, tasks in results.items():
        for task_name, splits in tasks.items():
            print(f"\n{dataset_name} / {task_name}")
            header = (
                f"  {'split':<6}  {'seeds':>8}  {'avg 2-hop nbrs':>14}  "
                f"{'std':>10}  {'median':>10}  {'mode':>8}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for split, m in splits.items():
                sampled_tag = "*" if m.get("sampled") else " "
                print(
                    f"  {split:<6}  {m['num_seeds']:>8}{sampled_tag} "
                    f"{m['avg_2hop_neighbors']:>14.2f}  "
                    f"{m['std_2hop_neighbors']:>10.2f}  "
                    f"{m['median_2hop_neighbors']:>10.2f}  "
                    f"{m['mode_2hop_neighbors']:>8}"
                )

    print("\nGenerating plots …")
    plot_mean_std_per_task(results)
    plot_node_type_tallies(results)


if __name__ == "__main__":
    main()
