#!/usr/bin/env python3
"""Plot mean ± std train loss and val AP curves across seeds for a given K sweep.

Usage:
  python plot_training_curves.py --k 50 --epochs 30
  python plot_training_curves.py --k 10 --epochs 30
"""
from __future__ import annotations
import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "debug_runs" / "logs"
OUT_DIR = REPO / "debug_runs"
DATASET, TASK = "rel-f1", "driver-top3"
BRANCHES = [("devkyaw", "dev-kyaw"), ("ogpass", "ogPASS")]
COLORS = {"devkyaw": "#2196F3", "ogpass": "#FF5722"}
MAX_SEEDS = 20

RE_EPOCH = re.compile(
    r"^Epoch:\s*(\d+),\s*Train loss:\s*([\d.]+),\s*Val metrics:.*?"
    r"'average_precision':\s*(?:np\.float64\()?([\d.]+)\)?"
)
RE_PHASE = re.compile(r"^\[phase\] epoch (\d+): (\w+)")


def parse_log(path: Path) -> dict:
    epochs, train_loss, val_ap, phases = [], [], [], {}
    with open(path) as f:
        for line in f:
            m = RE_EPOCH.match(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_ap.append(float(m.group(3)))
                continue
            m = RE_PHASE.match(line)
            if m:
                phases[int(m.group(1))] = m.group(2)
    return {"epochs": epochs, "train_loss": train_loss, "val_ap": val_ap, "phases": phases}


def collect(k: int, epochs: int) -> dict[str, list[dict]]:
    data = {}
    for slug, _ in BRANCHES:
        runs = []
        for seed in range(MAX_SEEDS):
            log = LOGS / f"{slug}_dt3_bl_n{k}_ep{epochs}_s{seed}.log"
            if not log.exists():
                continue
            parsed = parse_log(log)
            if len(parsed["epochs"]) == 0:
                continue
            runs.append(parsed)
        data[slug] = runs
    return data


def mean_std_curves(runs: list[dict], key: str, n_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.full((len(runs), n_epochs), np.nan)
    for i, r in enumerate(runs):
        for ep, val in zip(r["epochs"], r[key]):
            if 1 <= ep <= n_epochs:
                mat[i, ep - 1] = val
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=1)
    count = np.sum(~np.isnan(mat), axis=0)
    return mean, std, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    data = collect(args.k, args.epochs)
    ep_axis = np.arange(1, args.epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training curves — K={args.k}, epochs={args.epochs} (mean ± std across seeds)", fontsize=13)

    keys = [("train_loss", "Train Loss", axes[0]), ("val_ap", "Val AP (average precision)", axes[1])]

    for key, ylabel, ax in keys:
        for slug, label in BRANCHES:
            runs = data[slug]
            if not runs:
                continue
            mean, std, count = mean_std_curves(runs, key, args.epochs)
            n = len(runs)
            color = COLORS[slug]
            ax.plot(ep_axis, mean, color=color, linewidth=2, label=f"{label} (n={n})")
            ax.fill_between(ep_axis, mean - std, mean + std, color=color, alpha=0.15)

        # Draw ogPASS phase boundaries if available
        og_runs = data.get("ogpass", [])
        if og_runs:
            phases = og_runs[0].get("phases", {})
            phase_colors = {"sampler_only": "#9C27B0", "joint": "#4CAF50"}
            phase_labels = {"sampler_only": "sampler_only phase", "joint": "joint phase"}
            for ep_start, phase_name in sorted(phases.items()):
                if phase_name in phase_colors:
                    ax.axvline(ep_start, color=phase_colors[phase_name], linestyle="--",
                               linewidth=1.2, alpha=0.7, label=phase_labels.pop(phase_name, None))

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, args.epochs)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / f"training_curves_n{args.k}_ep{args.epochs}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
