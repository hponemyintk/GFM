"""Plot diagnostics for the distilled-sampler three-phase pipeline.

Parses per-phase stdout logs (teacher.log, distill.log, joint.log) for
per-epoch train/val losses, and reads distill_diagnostic.npz for the teacher
vs sampler scatter. Does NOT depend on wandb or any CSV produced by the
training code.

Usage:
    python plot_distilled_sampler.py --out_dir results/<run>
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---- Regex patterns matching main_node_ddp.py's existing print() lines ----
# Teacher / joint: "Epoch: 02, Train loss: 0.6735, Val metrics: {'roc_auc': 0.71, ...}"
_SUPERVISED_RE = re.compile(
    r"Epoch:\s*(\d+),\s*Train loss:\s*([-\d.eE+]+),\s*Val metrics:\s*(\{[^}]*\})")
# Distill: "Epoch 02 distill_train=0.12345 distill_val=0.11111"
_DISTILL_RE = re.compile(
    r"Epoch\s*(\d+)\s+distill_train=([-\d.eE+]+)\s+distill_val=([-\d.eE+]+)")


def _parse_val_metrics(s: str):
    """Turn "{'roc_auc': 0.71, 'ap': 0.5}" into a dict of floats (first key wins)."""
    out = {}
    for m in re.finditer(r"'([^']+)'\s*:\s*([-\d.eE+]+)", s):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def parse_supervised_log(path):
    """Return (epochs, train, val_metric_name, val_metric_values)."""
    if not os.path.exists(path):
        return None
    epochs, trains, vals, metric_name = [], [], [], None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = _SUPERVISED_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            trains.append(float(m.group(2)))
            metrics = _parse_val_metrics(m.group(3))
            if not metrics:
                continue
            if metric_name is None:
                metric_name = next(iter(metrics.keys()))
            vals.append(metrics.get(metric_name, float("nan")))
    if not epochs:
        return None
    return epochs, trains, metric_name, vals


def parse_distill_log(path):
    """Return (epochs, train, val)."""
    if not os.path.exists(path):
        return None
    epochs, trains, vals = [], [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = _DISTILL_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            trains.append(float(m.group(2)))
            vals.append(float(m.group(3)))
    if not epochs:
        return None
    return epochs, trains, vals


def plot_training_curves(out_dir: str):
    teacher = parse_supervised_log(os.path.join(out_dir, "teacher.log"))
    distill = parse_distill_log(os.path.join(out_dir, "distill.log"))
    joint = parse_supervised_log(os.path.join(out_dir, "joint.log"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Phase 1: teacher (task loss + val metric). ---
    ax = axes[0]
    if teacher is not None:
        ep, tr, name, vl = teacher
        ax.plot(ep, tr, "o-", color="tab:blue", label="train loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Train loss", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(ep, vl, "s--", color="tab:red", label=f"val {name}")
        ax2.set_ylabel(f"val {name}", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(f"Phase 1: teacher ({len(ep)} epochs)")
    else:
        ax.text(0.5, 0.5, "teacher.log not found", ha="center", va="center")
        ax.set_title("Phase 1: teacher")
    ax.grid(True, alpha=0.3)

    # --- Phase 2: distill (MSE train + val). ---
    ax = axes[1]
    if distill is not None:
        ep, tr, vl = distill
        ax.plot(ep, tr, "o-", color="tab:orange", label="distill train (MSE)")
        ax.plot(ep, vl, "s--", color="tab:orange", alpha=0.6, label="distill val (MSE)")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (log)")
        ax.set_title(f"Phase 2: distill ({len(ep)} epochs)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "distill.log not found", ha="center", va="center")
        ax.set_title("Phase 2: distill")
    ax.grid(True, alpha=0.3)

    # --- Phase 3: joint (task loss + val metric). ---
    ax = axes[2]
    if joint is not None:
        ep, tr, name, vl = joint
        ax.plot(ep, tr, "o-", color="tab:green", label="train loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Train loss", color="tab:green")
        ax.tick_params(axis="y", labelcolor="tab:green")
        ax2 = ax.twinx()
        ax2.plot(ep, vl, "s--", color="tab:red", label=f"val {name}")
        ax2.set_ylabel(f"val {name}", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(f"Phase 3: joint ({len(ep)} epochs)")
    else:
        ax.text(0.5, 0.5, "joint.log not found", ha="center", va="center")
        ax.set_title("Phase 3: joint")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


def _pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 0 else 0.0


def _spearman(x, y):
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    return _pearson(rx, ry)


def plot_sampler_scatter(out_dir: str):
    # distill_diagnostic.npz is written inside the {dataset}/{task} nesting.
    candidates = [os.path.join(out_dir, "distill_diagnostic.npz")]
    candidates += sorted(
        glob.glob(os.path.join(out_dir, "*/*/distill_diagnostic.npz")))
    npz_path = next((p for p in candidates if os.path.exists(p)), None)
    if npz_path is None:
        print(f"[skip] {npz_path} not found (distill phase may have failed)")
        return
    z = np.load(npz_path)
    teacher, q_imp = z["teacher_logits"], z["q_imp"]
    B, K = teacher.shape
    assert q_imp.shape == (B, K)

    # Exclude seed self-entry at index 0.
    t = teacher[:, 1:].reshape(-1)
    p = q_imp[:, 1:].reshape(-1)

    pr = _pearson(t, p)
    sr = _spearman(t, p)

    def topk_recall(k):
        hits = 0
        for b in range(B):
            gold = np.argsort(-teacher[b, 1:])[:k]
            pred = np.argsort(-q_imp[b, 1:])[:k]
            hits += len(np.intersect1d(gold, pred))
        return hits / (B * k)

    recalls = {k: topk_recall(k) for k in [5, 10, 25, 50] if k <= K - 1}

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(t, p, s=3, alpha=0.25)
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    m, c = np.polyfit(t, p, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, m * xs + c, "r-", lw=1.5, label=f"fit: y = {m:.3f}x + {c:.3f}")
    ax.plot(xs, xs, "k--", lw=1.0, alpha=0.6, label="y = x")
    title = (f"Teacher vs sampler logits (B={B}, K={K}, n={t.size})\n"
             f"Pearson ρ = {pr:.3f}   Spearman ρ = {sr:.3f}")
    if recalls:
        title += "\nTop-K recall: " + "  ".join(f"@{k}={v:.2f}" for k, v in recalls.items())
    ax.set_title(title)
    ax.set_xlabel("Teacher attention logit (last-layer Q·K/√d, mean over heads)")
    ax.set_ylabel("DistillSampler q_imp")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = os.path.join(out_dir, "sampler_scatter.png")
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")
    print(f"  Pearson={pr:.4f}  Spearman={sr:.4f}  n={t.size}  recalls={recalls}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    plot_training_curves(args.out_dir)
    plot_sampler_scatter(args.out_dir)


if __name__ == "__main__":
    main()
