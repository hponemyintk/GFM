"""Scatter plots of teacher per-head logits vs distilled sampler predictions.

Loads distill_diagnostic.npz (shape [B, H, K] for both teacher_logits and q_imp)
and writes a PNG with:
  - Top row (4 panels):   per-head scatter (teacher vs sampler), Pearson + MSE.
  - Bottom left:          mean-over-heads scatter (the signal used for top-K).
  - Bottom right:         sorted-by-teacher comparison for one example seed.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="distill_diagnostic.npz path")
    p.add_argument("--out", default=None, help="output PNG path")
    p.add_argument("--sample_seed_idx", type=int, default=0,
                   help="which seed (batch index) to use for the sorted line plot")
    args = p.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(args.npz), "distill_scatter.png")

    z = np.load(args.npz)
    t, q = z["teacher_logits"], z["q_imp"]          # [B, H, K]
    assert t.shape == q.shape and t.ndim == 3, f"unexpected shapes: {t.shape} / {q.shape}"
    B, H, K = t.shape

    # Drop self (column 0) for all analysis; it's degenerate (Q_seed·K_seed).
    tc = t[:, :, 1:].reshape(B, H, -1)
    qc = q[:, :, 1:].reshape(B, H, -1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # --- Per-head scatter ---
    lims = (min(tc.min(), qc.min()) - 0.2, max(tc.max(), qc.max()) + 0.2)
    for h in range(H):
        ax = axes[0, h]
        tt = tc[:, h].reshape(-1)
        qq = qc[:, h].reshape(-1)
        ax.scatter(tt, qq, s=4, alpha=0.25, color=f"C{h}")
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.6, label="y=x")
        pear = np.mean([pearsonr(tc[b, h], qc[b, h]).statistic for b in range(B)])
        mse = ((tt - qq) ** 2).mean()
        ax.set_title(f"head {h}   Pearson={pear:+.3f}   MSE={mse:.3f}")
        ax.set_xlabel("teacher logit")
        ax.set_ylabel("sampler q_imp")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(loc="upper left", fontsize=8)

    # --- Bottom-left: mean-over-heads scatter (the top-K selection signal) ---
    tm = tc.mean(axis=1)                            # [B, K-1]
    qm = qc.mean(axis=1)
    ax = axes[1, 0]
    ax.scatter(tm.reshape(-1), qm.reshape(-1), s=5, alpha=0.3, color="C4")
    lims_m = (min(tm.min(), qm.min()) - 0.05, max(tm.max(), qm.max()) + 0.05)
    ax.plot(lims_m, lims_m, "k--", lw=0.8, alpha=0.6)
    pear_m = np.mean([pearsonr(tm[b], qm[b]).statistic for b in range(B)])
    ax.set_title(f"mean over heads   Pearson={pear_m:+.3f}\n(this feeds Gumbel-Top-K)")
    ax.set_xlabel("teacher (mean over heads)")
    ax.set_ylabel("sampler (mean over heads)")
    ax.set_xlim(lims_m); ax.set_ylim(lims_m)

    # --- Bottom-middle: sorted-by-teacher line plot for one seed ---
    s = args.sample_seed_idx
    order = np.argsort(tm[s])                       # ascending by teacher
    ax = axes[1, 1]
    ax.plot(tm[s][order], "o-", ms=3, lw=1, label="teacher (sorted)")
    ax.plot(qm[s][order], "s-", ms=3, lw=1, alpha=0.7, label="sampler at same positions")
    ax.set_title(f"seed {s}: sampler vs teacher, ordered by teacher rank")
    ax.set_xlabel("candidate rank (by teacher mean)")
    ax.set_ylabel("mean-over-heads logit")
    ax.legend(fontsize=8)

    # --- Bottom-right: per-head distribution (histogram) ---
    ax = axes[1, 2]
    for h in range(H):
        ax.hist(tc[:, h].reshape(-1), bins=40, histtype="step",
                label=f"head {h}", color=f"C{h}", alpha=0.8)
    ax.set_title("per-head teacher logit distributions")
    ax.set_xlabel("logit")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    # --- Far-right bottom: top-K identification ---
    # Highlight how teacher's top-16 candidates fall in sampler ranking.
    ax = axes[1, 3]
    k = 16
    in_teacher_top = []
    sampler_rank_of_teacher_top = []
    for b in range(B):
        teacher_top = set(np.argsort(-tm[b])[:k])
        sampler_order = np.argsort(-qm[b])
        for rank, idx in enumerate(sampler_order):
            if idx in teacher_top:
                sampler_rank_of_teacher_top.append(rank)
    h_counts, h_edges = np.histogram(sampler_rank_of_teacher_top,
                                     bins=np.linspace(0, tm.shape[1], 33))
    ax.bar(h_edges[:-1], h_counts, width=np.diff(h_edges), align="edge",
           color="C3", alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.axvline(k, color="k", ls="--", lw=0.8, label=f"ideal (rank<{k})")
    ax.set_title(f"where teacher's top-{k} land in sampler's ranking\n"
                 f"(pooled over {B} seeds)")
    ax.set_xlabel("sampler's rank")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Phase-2 distillation diagnostic   B={B}  H={H}  K-1={K-1}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
