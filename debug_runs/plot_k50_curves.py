#!/usr/bin/env python
"""Plot train loss and val AP curves for K=50 runs (both branches, all seeds)."""
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

LOGS = Path(__file__).resolve().parent / "logs"
SEEDS = range(5)
BRANCHES = [
    ("devkyaw_dt3_bl_n50_ep20", "dev-kyaw"),
    ("ogpass_dt3_bl_n50_ep20", "ogPASS"),
]


def parse_log(path):
    epochs, train_losses, val_aps = [], [], []
    pattern = re.compile(
        r"Epoch:\s+(\d+),\s+Train loss:\s+([\d.]+),.*"
        r"'average_precision':\s+np\.float64\(([\d.]+)\)"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_aps.append(float(m.group(3)))
    return epochs, train_losses, val_aps


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors_dk = plt.cm.Blues(np.linspace(0.4, 0.9, len(SEEDS)))
colors_og = plt.cm.Oranges(np.linspace(0.4, 0.9, len(SEEDS)))

for b_idx, (slug, label) in enumerate(BRANCHES):
    colors = colors_dk if b_idx == 0 else colors_og
    ax_loss = axes[0][b_idx]
    ax_ap = axes[1][b_idx]

    for s_idx, seed in enumerate(SEEDS):
        log_path = LOGS / f"{slug}_s{seed}.log"
        if not log_path.exists():
            continue
        epochs, train_losses, val_aps = parse_log(log_path)
        ax_loss.plot(epochs, train_losses, color=colors[s_idx], label=f"s{seed}", alpha=0.8)
        ax_ap.plot(epochs, val_aps, color=colors[s_idx], label=f"s{seed}", alpha=0.8)

    ax_loss.set_title(f"{label} — Train Loss (K=50)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_ap.set_title(f"{label} — Val AP (K=50)")
    ax_ap.set_xlabel("Epoch")
    ax_ap.set_ylabel("Val AP")
    ax_ap.legend(fontsize=8)
    ax_ap.grid(True, alpha=0.3)

# Shared y-axis ranges for comparison
all_losses, all_aps = [], []
for slug, _ in BRANCHES:
    for seed in SEEDS:
        log_path = LOGS / f"{slug}_s{seed}.log"
        if log_path.exists():
            _, tl, va = parse_log(log_path)
            all_losses.extend(tl)
            all_aps.extend(va)

if all_losses:
    for ax in axes[0]:
        ax.set_ylim(min(all_losses) * 0.95, max(all_losses) * 1.05)
if all_aps:
    for ax in axes[1]:
        ax.set_ylim(min(all_aps) * 0.9, max(all_aps) * 1.1)

fig.suptitle("K=50 Training Curves: dev-kyaw vs ogPASS (AP-tuned + REINFORCE baseline)", fontsize=13)
fig.tight_layout()

out = Path(__file__).resolve().parent / "k50_training_curves.png"
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
