"""Aggregate the distilled-sampler sweep across seeds and temperatures.

Expected layout (temp-sweep variant):
    <sweep_root>/seed_<s>/teacher.log
    <sweep_root>/seed_<s>/distill.log
    <sweep_root>/seed_<s>/distill_diagnostic.npz
    <sweep_root>/seed_<s>/<temp_label>/joint.log
    ...
Falls back to flat layout (no temp subdirs) if <sweep_root>/seed_<s>/joint.log exists.

Produces in <sweep_root>/aggregate/:
    - training_curves.png : mean±std across seeds for phases 1/2; mean±std
                            across seeds per temp for phase 3, overlaid.
    - sampler_scatter.png : pooled scatter across seeds (temp-independent).
    - summary.txt         : per-seed Pearson/Spearman + per-temp phase-3 summary.

Usage:
    python plot_distilled_sampler_sweep.py --sweep_root results/<sweep>
"""
import argparse
import glob
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_distilled_sampler import (
    parse_supervised_log, parse_distill_log, _pearson, _spearman,
)


TEMP_COLORS = {
    "det": "tab:purple",
    "t1":  "tab:green",
    "t2":  "tab:olive",
    "t5":  "tab:red",
}
TEMP_LABELS = {
    "det": "deterministic",
    "t1":  "Gumbel  T=1",
    "t2":  "Gumbel  T=2",
    "t5":  "Gumbel  T=5",
}


def _pad_nan(arrs: List[np.ndarray]) -> np.ndarray:
    if not arrs:
        return np.zeros((0, 0))
    mx = max(a.size for a in arrs)
    out = np.full((len(arrs), mx), np.nan)
    for i, a in enumerate(arrs):
        out[i, :a.size] = a
    return out


def _mean_std(arr: np.ndarray):
    if arr.size == 0:
        return np.array([]), np.array([])
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _shade(ax, x, mu, sd, *, label, color, ls="-"):
    ax.plot(x, mu, marker="o", ms=4, ls=ls, color=color, lw=1.8, label=label)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18)


def _collect_sup(paths):
    trains, vals, metric = [], [], None
    for p in paths:
        parsed = parse_supervised_log(p)
        if parsed is None:
            continue
        _ep, tr, name, vl = parsed
        trains.append(np.asarray(tr, dtype=float))
        vals.append(np.asarray(vl, dtype=float))
        if metric is None:
            metric = name
    return trains, vals, metric


def _collect_distill(paths):
    trains, vals = [], []
    for p in paths:
        parsed = parse_distill_log(p)
        if parsed is None:
            continue
        _ep, tr, vl = parsed
        trains.append(np.asarray(tr, dtype=float))
        vals.append(np.asarray(vl, dtype=float))
    return trains, vals


def _detect_temp_modes(seed_dirs):
    modes = []
    for d in seed_dirs:
        for label in TEMP_COLORS.keys():
            if os.path.isfile(os.path.join(d, label, "joint.log")):
                if label not in modes:
                    modes.append(label)
    return modes


def plot_training_curves(sweep_root: str, seed_dirs: List[str], out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- Phase 1: teacher ---
    ax = axes[0]
    teacher_logs = [os.path.join(d, "teacher.log") for d in seed_dirs]
    tr, vl, name = _collect_sup(teacher_logs)
    if tr:
        x = np.arange(1, max(a.size for a in tr) + 1)
        mu, sd = _mean_std(_pad_nan(tr))
        _shade(ax, x, mu, sd, label=f"train (n={len(tr)})", color="tab:blue")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Train loss", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        if vl and name is not None:
            ax2 = ax.twinx()
            mu2, sd2 = _mean_std(_pad_nan(vl))
            _shade(ax2, x, mu2, sd2, label=f"val {name}", color="tab:red")
            ax2.set_ylabel(f"val {name}", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(f"Phase 1: teacher  (mean ± std, n={len(tr)})")
    else:
        ax.text(0.5, 0.5, "teacher.log not found", ha="center", va="center")
    ax.grid(True, alpha=0.3)

    # --- Phase 2: distill ---
    ax = axes[1]
    distill_logs = [os.path.join(d, "distill.log") for d in seed_dirs]
    d_tr, d_vl = _collect_distill(distill_logs)
    if d_tr:
        x = np.arange(1, max(a.size for a in d_tr) + 1)
        mu_tr, sd_tr = _mean_std(_pad_nan(d_tr))
        _shade(ax, x, mu_tr, sd_tr, label="distill train (MSE)", color="tab:orange")
        if d_vl:
            mu_vl, sd_vl = _mean_std(_pad_nan(d_vl))
            _shade(ax, x, mu_vl, sd_vl, label="distill val (MSE)", color="tab:brown")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (log)")
        ax.set_title(f"Phase 2: distill  (n={len(d_tr)})")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "distill.log not found", ha="center", va="center")
    ax.grid(True, alpha=0.3)

    # --- Phase 3: joint, one curve per temperature mode ---
    ax = axes[2]
    temps = _detect_temp_modes(seed_dirs)
    if temps:
        ax2 = ax.twinx()
        train_metric_name = None
        for tm in temps:
            logs = [os.path.join(d, tm, "joint.log") for d in seed_dirs]
            tr, vl, metric = _collect_sup(logs)
            if not tr:
                continue
            x = np.arange(1, max(a.size for a in tr) + 1)
            mu_tr, sd_tr = _mean_std(_pad_nan(tr))
            _shade(ax, x, mu_tr, sd_tr,
                   label=f"{TEMP_LABELS[tm]} train", color=TEMP_COLORS[tm])
            if vl and metric is not None:
                mu_vl, sd_vl = _mean_std(_pad_nan(vl))
                _shade(ax2, x, mu_vl, sd_vl,
                       label=f"{TEMP_LABELS[tm]} val", color=TEMP_COLORS[tm], ls="--")
                train_metric_name = metric
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        if train_metric_name is not None:
            ax2.set_ylabel(f"val {train_metric_name}")
        ax.set_title(f"Phase 3: joint  (mean ± std, n={len(seed_dirs)} per temp)")
        # Combine legends.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    else:
        # fallback flat layout
        logs = [os.path.join(d, "joint.log") for d in seed_dirs]
        tr, vl, name = _collect_sup(logs)
        if tr:
            x = np.arange(1, max(a.size for a in tr) + 1)
            mu, sd = _mean_std(_pad_nan(tr))
            _shade(ax, x, mu, sd, label="train", color="tab:green")
            ax.set_title(f"Phase 3: joint  (mean ± std, n={len(tr)})")
        else:
            ax.text(0.5, 0.5, "no joint logs found", ha="center", va="center")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Distilled-sampler sweep — {os.path.basename(sweep_root)}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_sampler_scatter(seed_dirs: List[str], out_dir: str, summary_path: str):
    per_seed = []
    all_t, all_p, all_id = [], [], []
    for s_idx, d in enumerate(seed_dirs):
        # Search both flat and nested {dataset}/{task}/ layout.
        candidates = [os.path.join(d, "distill_diagnostic.npz")]
        candidates += sorted(glob.glob(os.path.join(d, "*/*/distill_diagnostic.npz")))
        npz = next((c for c in candidates if os.path.exists(c)), None)
        if npz is None:
            continue
        z = np.load(npz)
        teacher, q_imp = z["teacher_logits"], z["q_imp"]
        B, K = teacher.shape
        t = teacher[:, 1:].reshape(-1)
        p = q_imp[:, 1:].reshape(-1)

        pr = _pearson(t, p); sr = _spearman(t, p)

        def topk_recall(k):
            hits = 0
            for b in range(B):
                gold = np.argsort(-teacher[b, 1:])[:k]
                pred = np.argsort(-q_imp[b, 1:])[:k]
                hits += len(np.intersect1d(gold, pred))
            return hits / (B * k)

        recalls = {k: topk_recall(k) for k in [5, 10, 25, 50] if k <= K - 1}
        per_seed.append({
            "seed_dir": os.path.basename(d), "B": B, "K": K,
            "pearson": pr, "spearman": sr, "recalls": recalls,
            "q_imp_std": float(np.std(q_imp[:, 1:])),
            "teacher_std": float(np.std(teacher[:, 1:])),
        })
        all_t.append(t); all_p.append(p)
        all_id.append(np.full_like(t, s_idx, dtype=int))

    if not per_seed:
        print("[skip] no distill_diagnostic.npz files found")
        return

    all_t = np.concatenate(all_t); all_p = np.concatenate(all_p)
    all_id = np.concatenate(all_id)
    pooled_pr = _pearson(all_t, all_p); pooled_sr = _spearman(all_t, all_p)
    per_pr = np.array([s["pearson"] for s in per_seed])
    per_sr = np.array([s["spearman"] for s in per_seed])

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, s in enumerate(per_seed):
        mask = all_id == i
        ax.scatter(all_t[mask], all_p[mask], s=3, alpha=0.25, color=cmap(i),
                   label=f"{s['seed_dir']}  ρ={s['pearson']:.2f}")
    lo, hi = float(all_t.min()), float(all_t.max())
    m, c = np.polyfit(all_t, all_p, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, m * xs + c, "r-", lw=1.5, label=f"pooled fit: y={m:.3f}x+{c:.3f}")
    ax.plot(xs, xs, "k--", lw=1.0, alpha=0.6, label="y = x")

    title = (
        f"Teacher vs sampler logits — {len(per_seed)} seeds pooled  (n={all_t.size})\n"
        f"Pooled Pearson ρ = {pooled_pr:.3f}   "
        f"per-seed mean ± std = {per_pr.mean():.3f} ± {per_pr.std():.3f}\n"
        f"Pooled Spearman ρ = {pooled_sr:.3f}   "
        f"per-seed mean ± std = {per_sr.mean():.3f} ± {per_sr.std():.3f}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Teacher attention logit (last-layer Q·K/√d, mean over heads)")
    ax.set_ylabel("DistillSampler q_imp")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, "sampler_scatter.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")

    lines = [f"Distilled-sampler sweep summary  (seeds={len(per_seed)})", ""]
    for s in per_seed:
        rec = "  ".join(f"@{k}={v:.3f}" for k, v in s["recalls"].items())
        lines.append(
            f"[{s['seed_dir']}] B={s['B']}, K={s['K']}  "
            f"Pearson={s['pearson']:.4f}  Spearman={s['spearman']:.4f}  "
            f"std(q_imp)={s['q_imp_std']:.3f}  std(teacher)={s['teacher_std']:.3f}  {rec}"
        )
    lines += ["",
              f"Pooled Pearson   = {pooled_pr:.4f}",
              f"Pooled Spearman  = {pooled_sr:.4f}",
              f"Per-seed Pearson   = {per_pr.mean():.4f} ± {per_pr.std():.4f}",
              f"Per-seed Spearman  = {per_sr.mean():.4f} ± {per_sr.std():.4f}"]
    ks_common = set.intersection(*(set(s["recalls"].keys()) for s in per_seed))
    for k in sorted(ks_common):
        arr = np.array([s["recalls"][k] for s in per_seed])
        lines.append(f"Top-{k} recall     = {arr.mean():.4f} ± {arr.std():.4f}")

    # Per-temp phase-3 val metric at final epoch.
    temps = _detect_temp_modes(seed_dirs)
    if temps:
        lines.append("")
        lines.append("Phase-3 val metric at final epoch (mean ± std across seeds):")
        for tm in temps:
            logs = [os.path.join(d, tm, "joint.log") for d in seed_dirs]
            _, vl, name = _collect_sup(logs)
            if not vl or name is None:
                continue
            finals = np.array([v[-1] for v in vl if v.size > 0])
            if finals.size:
                lines.append(
                    f"  {TEMP_LABELS[tm]:<18}  {name}  = {finals.mean():.4f} ± {finals.std():.4f}"
                )

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {summary_path}")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", required=True)
    args = ap.parse_args()

    seed_dirs = sorted(glob.glob(os.path.join(args.sweep_root, "seed_*")))
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories under {args.sweep_root}")
    print(f"Aggregating {len(seed_dirs)} seeds: {[os.path.basename(d) for d in seed_dirs]}")

    out_dir = os.path.join(args.sweep_root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    plot_training_curves(args.sweep_root, seed_dirs, out_dir)
    plot_sampler_scatter(seed_dirs, out_dir,
                         summary_path=os.path.join(out_dir, "summary.txt"))


if __name__ == "__main__":
    main()
