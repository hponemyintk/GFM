#!/usr/bin/env python
"""Aggregate driver-top3 baseline-fix sweep results.

Reads per-seed result JSONs at:
  debug_runs/results/<run_name>/rel-f1/driver-top3/<seed>.json
where <run_name> matches {devkyaw,ogpass}_dt3_bl_n{K}_ep{E}_s{SEED}.

Usage:
  summarize_dt3.py --k 1 --epochs 20
  summarize_dt3.py --k 10 --epochs 20
  summarize_dt3.py --k 50 --epochs 20
"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "debug_runs" / "results"
SUMMARY_MD = REPO / "experiment_results_baseline_fix.md"
DATASET = "rel-f1"
TASK = "driver-top3"
METRICS = ["average_precision", "roc_auc", "f1", "accuracy"]
METRIC_LABELS = {"average_precision": "AP", "roc_auc": "AUC", "f1": "F1", "accuracy": "Acc"}
# higher = better for all classification metrics
HIGHER_BETTER = {m: True for m in METRICS}
MAX_SEEDS = 20
BRANCHES = [("devkyaw", "dev-kyaw"), ("ogpass", "ogPASS")]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _read_seed(run_name: str, seed: int) -> dict | None:
    path = RESULTS / run_name / DATASET / TASK / f"{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("test_metrics", {})


def _collect(k: int, epochs: int) -> dict[str, list[dict]]:
    out = {}
    for slug, _label in BRANCHES:
        rows = []
        for seed in range(MAX_SEEDS):
            run = f"{slug}_dt3_bl_n{k}_ep{epochs}_s{seed}"
            m = _read_seed(run, seed)
            if m is None:
                continue
            rows.append({"seed": seed, "run": run, "metrics": m})
        out[slug] = rows
    return out


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def _per_seed_table(rows: list[dict]) -> str:
    hdr = "| seed | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"
    sep = "|---:" + "|---:" * len(METRICS) + "|"
    lines = [hdr, sep]
    for r in rows:
        m = r["metrics"] or {}
        cells = " | ".join(_fmt(m.get(metric)) for metric in METRICS)
        lines.append(f"| {r['seed']} | {cells} |")
    return "\n".join(lines)


def _aggregate(rows: list[dict]) -> dict[str, tuple[float, float]]:
    agg = {}
    for metric in METRICS:
        vals = [
            r["metrics"][metric]
            for r in rows
            if r["metrics"] and metric in r["metrics"]
        ]
        agg[metric] = _mean_std(vals)
    return agg


def _sweep_section(k: int, epochs: int) -> str:
    collected = _collect(k, epochs)
    sweep = f"n{k}_ep{epochs}"
    lines = [f"## Sweep `{sweep}` — K={k}, epochs={epochs}", ""]

    if k == 1:
        phase = "K=1 seed-only. PASS `sampler_only` is a no-op; schedule shape kept."
    else:
        w = max(1, epochs // 5)
        s = max(1, epochs // 5)
        j = epochs - w - s
        phase = f"ogPASS 3-phase: warmup={w}, sampler_only={s}, joint={j}."
    lines += [f"_{phase}_", ""]

    n_seeds = max((len(collected[slug]) for slug, _ in BRANCHES), default=0)
    labels = [METRIC_LABELS[m] + " ↑" for m in METRICS]
    lines += [
        f"### Aggregated (mean ± std over {n_seeds} seeds)",
        "",
        "| branch | " + " | ".join(labels) + " |",
        "|---|" + "|".join(["---"] * len(METRICS)) + "|",
    ]
    aggs: dict[str, dict[str, tuple[float, float]]] = {}
    n_found: dict[str, int] = {}
    for slug, label in BRANCHES:
        rows = collected[slug]
        agg = _aggregate(rows)
        aggs[slug] = agg
        n_found[slug] = sum(1 for r in rows if r["metrics"])
        cells = []
        for metric in METRICS:
            m, s = agg[metric]
            if math.isnan(m):
                cells.append("—")
            else:
                cells.append(f"{m:.4f} ± {s:.4f}")
        lines.append(f"| {label} (n={n_found[slug]}) | " + " | ".join(cells) + " |")
    lines.append("")

    # delta row
    dev = aggs.get("devkyaw", {})
    og = aggs.get("ogpass", {})
    if dev and og and not any(math.isnan(v[0]) for v in dev.values()) and not any(math.isnan(v[0]) for v in og.values()):
        dlabels = ["Δ" + METRIC_LABELS[m] for m in METRICS]
        lines += [
            "### Δ (ogPASS − dev-kyaw) — positive is better for all metrics",
            "",
            "| " + " | ".join(dlabels) + " |",
            "|" + "|".join(["---"] * len(METRICS)) + "|",
            "| " + " | ".join(
                f"{og[m][0] - dev[m][0]:+.4f}" for m in METRICS
            ) + " |",
            "",
        ]

        # variance ratio
        def _ratio(a, b):
            if b == 0 or math.isnan(a) or math.isnan(b):
                return float("nan")
            return a / b

        ratios = [_ratio(og[m][1], dev[m][1]) for m in METRICS]
        lines += [
            "### Variance ratio (ogPASS std ÷ dev-kyaw std — lower = ogPASS more stable)",
            "",
            "| " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |",
            "|" + "|".join(["---"] * len(METRICS)) + "|",
            "| " + " | ".join(
                f"{r:.2f}×" if not math.isnan(r) else "—" for r in ratios
            ) + " |",
            "",
        ]

    # per-seed
    lines += ["### Per-seed", ""]
    for slug, label in BRANCHES:
        lines += [f"#### {label}", "", _per_seed_table(collected[slug]), ""]

    return "\n".join(lines)


HEADER = """# ogPASS vs dev-kyaw — driver-top3 baseline-fix results

Head-to-head on `rel-f1 / driver-top3` (binary classification). Tune metric is
**AP** (higher = better); we also report AUC, F1, and Accuracy. 5 seeds (0–4)
per branch per sweep.

**Fixes applied:** REINFORCE EMA baseline (--use_reinforce_baseline),
tune_metric changed from AUC to AP.

Common config: batch_size 32, num_layers 4, channels 512, max_steps_per_epoch
1000, lr 1e-4, warmup_steps 100, ff_dropout 0.3, attn_dropout 0.3, single-GPU
DDP, precompute cache cleared between seeds. ogPASS uses the 3-phase schedule
with 20/20/60 % of epochs spent on warmup / sampler_only / joint.
"""


def _replace_or_append(full_md: str, sweep_md: str, sweep: str) -> str:
    marker_open = f"<!-- BEGIN SWEEP {sweep} -->"
    marker_close = f"<!-- END SWEEP {sweep} -->"
    block = f"{marker_open}\n{sweep_md}\n{marker_close}\n"
    if marker_open in full_md and marker_close in full_md:
        pattern = re.compile(
            re.escape(marker_open) + r".*?" + re.escape(marker_close) + r"\n?",
            re.DOTALL,
        )
        return pattern.sub(block, full_md)
    return full_md.rstrip() + "\n\n" + block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    args = ap.parse_args()

    sweep = f"n{args.k}_ep{args.epochs}"
    sweep_md = _sweep_section(args.k, args.epochs)

    if SUMMARY_MD.exists():
        full_md = SUMMARY_MD.read_text()
        if not full_md.strip():
            full_md = HEADER
    else:
        full_md = HEADER

    full_md = _replace_or_append(full_md, sweep_md, sweep)
    SUMMARY_MD.write_text(full_md)
    print(f"[summarize_dt3] wrote sweep `{sweep}` to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
