"""Aggregate parity-sweep JSON results into docs/parity_results.md.

Each main_node_ddp.py run writes ``<out_dir>/rel-f1/<task>/<seed>.json``
containing ``test_metrics`` and ``val_metrics``. We pick the test metric
of interest per task type and compute mean / std across seeds.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict


_METRIC = {
    "driver-position": "mae",        # regression: lower is better
    "driver-top3": "roc_auc",        # binary: higher is better
    "driver-dnf": "roc_auc",
}
_DIRECTION = {
    "driver-position": "lower",
    "driver-top3": "higher",
    "driver-dnf": "higher",
}


def gather(root: str):
    """root/<pipeline_tag>/rel-f1/<task>/<seed>.json"""
    out = defaultdict(lambda: defaultdict(list))  # pipeline -> task -> [(seed, metric)]
    for pip in os.listdir(root):
        pip_dir = os.path.join(root, pip)
        if not os.path.isdir(pip_dir):
            continue
        for path in glob.glob(os.path.join(pip_dir, "rel-f1", "*", "*.json")):
            task = os.path.basename(os.path.dirname(path))
            seed = int(os.path.splitext(os.path.basename(path))[0])
            with open(path) as f:
                d = json.load(f)
            metric_name = _METRIC.get(task)
            if metric_name is None or metric_name not in d.get("test_metrics", {}):
                continue
            out[pip][task].append((seed, float(d["test_metrics"][metric_name])))
    return out


def render_table(stats):
    """Produce the markdown table for docs/parity_results.md §Results."""
    lines = ["| Task | Pipeline | Seeds | Mean | Std | Notes |",
             "|---|---|---|---|---|---|"]
    for task in sorted({t for d in stats.values() for t in d.keys()}):
        m_name = _METRIC[task]
        for pip in ("devkyaw", "new"):
            samples = [v for (_s, v) in stats.get(pip, {}).get(task, [])]
            if not samples:
                lines.append(f"| rel-f1 / {task} ({m_name}) | {pip} | 0 | _missing_ | _missing_ | |")
                continue
            mu = statistics.fmean(samples)
            sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
            lines.append(
                f"| rel-f1 / {task} ({m_name}) | {pip} | {len(samples)} | "
                f"{mu:.4f} | {sd:.4f} | {sorted(_s for _s, _v in stats.get(pip, {}).get(task, []))} |"
            )
    return "\n".join(lines)


def acceptance(stats) -> str:
    """Apply the §6.3.5 68%-CI overlap criterion."""
    out = []
    for task in sorted({t for d in stats.values() for t in d.keys()}):
        a = [v for (_s, v) in stats.get("devkyaw", {}).get(task, [])]
        b = [v for (_s, v) in stats.get("new", {}).get(task, [])]
        if len(a) < 2 or len(b) < 2:
            out.append(f"- **{task}**: not enough seeds to evaluate ({len(a)} dev-kyaw, {len(b)} new)")
            continue
        mu_a, sd_a = statistics.fmean(a), statistics.stdev(a)
        mu_b, sd_b = statistics.fmean(b), statistics.stdev(b)
        # Means within 1 std of each other (CI overlap proxy).
        gap = abs(mu_a - mu_b)
        ok = gap <= max(sd_a, sd_b)
        out.append(
            f"- **{task}** ({_METRIC[task]}, {_DIRECTION[task]} is better): "
            f"dev-kyaw {mu_a:.4f}±{sd_a:.4f} vs new {mu_b:.4f}±{sd_b:.4f} -- "
            f"gap {gap:.4f}, threshold {max(sd_a, sd_b):.4f} -> "
            f"{'PASS' if ok else 'FAIL'}"
        )
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="Root of parity results (contains devkyaw/ and new/)")
    args = p.parse_args()

    stats = gather(args.root)
    print("Per-pipeline counts:")
    for pip, by_task in stats.items():
        for task, items in by_task.items():
            print(f"  {pip} / {task}: {len(items)} seeds, values={[round(v,4) for _,v in items]}")
    print()
    print("=== TABLE ===")
    print(render_table(stats))
    print()
    print("=== ACCEPTANCE ===")
    print(acceptance(stats))


if __name__ == "__main__":
    main()
