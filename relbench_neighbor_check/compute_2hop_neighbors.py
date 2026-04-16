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
from collections import defaultdict

import numpy as np
import pandas as pd

from relbench.datasets import get_dataset
from relbench.tasks import get_task


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
    """
    counts = []

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

    return counts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_avg_2hop_neighbors(dataset_task_dict, verbose=True):
    """
    Compute the average number of 2-hop neighbours for seed nodes in each
    split (train / val / test) for every (dataset, task) pair.

    Parameters
    ----------
    dataset_task_dict : dict[str, list[str]]
        Maps dataset names to lists of task names, e.g.
        {"rel-f1": ["driver-top3"]}.
    verbose : bool

    Returns
    -------
    results : dict
        results[dataset][task][split] = {
            "avg_2hop_neighbors": float,
            "std_2hop_neighbors": float,
            "num_seeds":          int,
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

            task = get_task(dataset_name, task_name, download=True)

            if not hasattr(task, "entity_col"):
                raise NotImplementedError(
                    f"Task type {type(task)} not yet supported; "
                    "only EntityTask is handled."
                )

            entity_col   = task.entity_col
            entity_table = task.entity_table
            time_col     = task.time_col

            results[dataset_name][task_name] = {}

            for split in ("train", "val", "test"):
                mask_input = split == "test"
                table = task.get_table(split, mask_input_cols=mask_input)
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

                counts = count_2hop_neighbors_for_nodes(
                    seed_nodes_with_cutoff, adj, node_time
                )

                avg = float(np.mean(counts))
                std = float(np.std(counts))

                results[dataset_name][task_name][split] = {
                    "avg_2hop_neighbors": avg,
                    "std_2hop_neighbors": std,
                    "num_seeds":          len(seed_nodes_with_cutoff),
                }

                if verbose:
                    print(
                        f"    {split:5s} | seeds: {len(seed_nodes_with_cutoff):4d} | "
                        f"avg 2-hop neighbors: {avg:10.2f} ± {std:.2f}"
                    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute average 2-hop neighbour count for RelBench tasks."
    )
    parser.add_argument(
        "--dataset-task",
        nargs="+",
        metavar="DATASET:TASK",
        default=["rel-f1:driver-top3"],
        help=(
            "One or more DATASET:TASK pairs, e.g. "
            "rel-f1:driver-top3  rel-amazon:user-churn"
        ),
    )
    args = parser.parse_args()

    dataset_task_dict = defaultdict(list)
    for pair in args.dataset_task:
        dataset_name, task_name = pair.split(":", 1)
        dataset_task_dict[dataset_name].append(task_name)

    results = compute_avg_2hop_neighbors(dict(dataset_task_dict))

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset_name, tasks in results.items():
        for task_name, splits in tasks.items():
            print(f"\n{dataset_name} / {task_name}")
            header = f"  {'split':<6}  {'seeds':>8}  {'avg 2-hop nbrs':>14}  {'std':>10}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for split, m in splits.items():
                print(
                    f"  {split:<6}  {m['num_seeds']:>8}  "
                    f"{m['avg_2hop_neighbors']:>14.2f}  "
                    f"{m['std_2hop_neighbors']:>10.2f}"
                )


if __name__ == "__main__":
    main()
