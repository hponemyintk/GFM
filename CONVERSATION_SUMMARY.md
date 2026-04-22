# Distilled Sampler — Investigation Summary

Session date: 2026-04-21 → 2026-04-22
Task: `rel-f1/driver-top3` (binary classification; tune metric `roc_auc`)
Branch: `distilled_sampler` (worktree at `.claude/worktrees/distilled_sampler`)

## Problem

The three-phase pipeline (`teacher → distill → joint`) was underperforming:
- Phase-1 teacher achieved test AUC ≈ 0.80–0.82.
- Phase-3 joint student (trained on neighbors curated by the distilled sampler) was ~4 AUC points below on some seeds (seed 1 teacher 0.824 → joint 0.784) and its `det` mode collapsed to F1=0 (majority class). Question: **why is the distilled sampler yielding worse neighborhoods than the baseline, and how do we close the gap?**

## Architecture recap

- Phase 1: teacher RelGT trained on `local_nodes_hetero(K=128)` neighborhoods.
- Phase 2: distill a small sampler `q_imp = f(seed, cand)` to predict the teacher's last-layer seed-to-all attention logits.
- Phase 3: for each seed, score a scope pool of 1024 candidates with the sampler, Gumbel-Top-K pick 128, retrain the teacher-initialized student on curated HDF5s. Temp modes: `det / t1 / t2 / t5` (sample_temp ∈ {0, 1, 2, 5}).

## Investigation timeline + key learnings

### 1. Initial diagnosis (rejected)

Original sampler: single shared `Ws`, unnormalized dot product, MSE on mean-over-heads pre-softmax logits.

My first hypothesis was "scope mismatch" (sampler trained at K=128 can't generalize to S=1024). The user correctly pushed back — the *whole point* of the distilled sampler is to cheaply score a larger pool than it was trained on. Dropped this framing.

### 2. Real architectural issues identified

- **Shared `Ws` gives rank-deficient symmetric scorer** (`Ws^T Ws` is PSD, bounded rank). The teacher's Q·K^T is asymmetric across heads.
- **Unnormalized dot has a popularity/scale bias** — candidates with large `‖Ws·e_c‖` win for every seed. At scope=1024, this bias has more candidates to dominate → `det` mode collapses.
- **MSE on raw logits is not ideal for top-K retrieval** — but see (6) below; MSE turned out to be fine once the target was right.

### 3. Per-type projections + L2-norm (backed off on L2-norm)

Introduced per-node-type `W_t` (one `nn.Linear` per table in the database). Initially added L2-norm + learnable scale. The user's question — "doesn't pytorch_frame already normalize?" — led to realizing that upstream `LayerNorm` on each of the 4 base components (type/hop/time/tfs) gives consistent per-token norms; L2-norm on the sampler output was redundant.

**Kept**: per-type `W_t`. **Dropped**: L2-norm, scale, bias. MSE target and temperatures stayed.

### 4. Per-head distillation target

Key diagnostic finding: on seed 0, the original distillation target — **mean over heads of pre-softmax logits** — has Pearson **0.964** with gold, so it's a near-perfect ranker. But teacher's mean-over-heads `std = 0.135`, and sampler MSE floored around 0.13 — sampler's noise (~√0.13 ≈ 0.36) was 2.7× larger than the signal it was trying to match. MSE convergence was dominated by predicting the mean; the per-seed Pearson(sampler ↔ teacher) was only **0.17**.

Changed the distillation target from `dots.mean(dim=1)` to `dots` — return full `[B, H, K]` per-head logits (in `local_module.py`). Each head individually has std ~0.3–0.6 (stronger signal). The sampler architecture was extended to output `[B, H, K]` — each `Ws[t]` projection is split into H heads of `hidden_dim/H` and produces per-head dot products with `1/√d_head` scaling (mirrors teacher). Curate-time top-K uses the mean over heads of the sampler's per-head outputs.

Seed-0 diagnostic after this change: per-head Pearson(sampler ↔ teacher) = [0.78, 0.74, 0.74, 0.87]. Mean-reduced sampler vs gold Pearson = **0.93** (was 0.17). Huge improvement.

### 5. 3-seed sweep revealed a catastrophic failure on seed 1

Seed 0: joint `det / t1 / t2 / t5` = 0.803 / 0.791 / 0.789 / 0.816 AUC vs teacher 0.800 — student matches/exceeds teacher.

Seed 1 (in the same sweep): joint results = 0.619 / 0.594 / 0.553 / 0.508 AUC vs teacher 0.793 — monotonic collapse with temperature toward random (AUC 0.5).

Phase-2 diagnostic on seed 1 showed **two dead heads**:
- head 0: `Pearson(teacher vs gold) = +0.019` (noise)
- head 2: `Pearson(teacher vs gold) = -0.140` (anti-correlated)

Per-head distillation forced the sampler to spend capacity fitting these noise/anti-signal heads. Mean-reducing the sampler's per-head output at inference dragged the signal from the two useful heads down.

### 6. Option B: variance-weighted per-head MSE (implemented, then found insufficient)

Implemented `head_weights ∝ Var(teacher_head)` normalized to mean 1, applied in both the phase-2 loss (mean of `w_h * MSE_h`) and the phase-3 curate reduction. Calibrated from the first training batch via `set_head_weights(per_head_std)`.

5-seed partial sweep (stopped after seed 1) with variance weighting:
- Seed 0: beat teacher — det 0.832 vs teacher 0.762 (weights: [0.91, 1.10, 0.44, 1.55]).
- Seed 1: still collapsed — det 0.684 vs teacher 0.805 (weights: [1.69, 1.17, 0.30, 0.84]).

Diagnostic analysis: for seed 1 this run, **all four heads had strong teacher-vs-gold Pearson (0.58–0.66)** — no head death. Variance-based weighting produced arbitrary skewed weights that *hurt* performance (mean-reduced sampler vs gold Pearson: 0.148 with variance weighting vs 0.191 with uniform mean). **Variance ≠ useful signal.** A high-variance anti-correlated head gets a large weight from variance weighting but should be discounted.

### 7. Ridge-regression head weights (current approach, next sweep)

Key insight from the user: fit head weights by solving
$$w^* = \arg\min_w \lVert Q w - g \rVert^2 + \lambda \lVert w \rVert^2$$
where `Q ∈ ℝ^{(B·K)×H}` stacks the trained sampler's per-head outputs on a val calibration pass, and `g ∈ ℝ^{B·K}` is the teacher's gold (sum over heads of post-softmax attention). This is the *optimal linear reduction* of the sampler's heads for approximating gold — handles head redundancy, anti-correlation (sign flip), and head-quality differences in one unified way.

Flow implemented (`main_node_ddp.py` phase 2):
1. Train with uniform per-head MSE (`head_weights = [1,1,1,1]`).
2. After training, reload the best-val checkpoint, forward on the val set to collect `(q, gold)` pairs, solve `(QᵀQ + λI)⁻¹ Qᵀg`, overwrite `head_weights`, re-save the checkpoint.
3. Phase 3 curate uses `sampler.reduce_heads` which applies the ridge weights.

`λ = 1e-3 · mean(diag(QᵀQ))` for numerical stability; the ridge fit is a 4-dim solve on ~18 val batches × 16 × 127 ≈ 38k rows.

## State of the code

Files modified during this session (all on `distilled_sampler` branch, origin in sync):

| File | Purpose |
|---|---|
| `distill_sampler.py` | `DistillSampler` with per-type `Ws`, multi-head `[B,H,K]` output, `head_weights` buffer, `set_head_weights`, `reduce_heads`, instance `distillation_loss` |
| `local_module.py` | Teacher returns per-head seed_logits `[B,H,L]` (no longer mean-over-heads) |
| `main_node_ddp.py` | Pass `num_heads`, thread `neighbor_types` to sampler forward, mean-reduce with `sampler.reduce_heads` in curate, ridge fit at end of distill |
| `dump_sampler_diagnostic.py` | Updated to new constructor + types input |
| `test_distillation.py` | Updated shape assertions and synthetic teacher geometry |
| `expts/full-sweep-distilled-sampler.sh` | 20 seeds, temps `det / t1 / t2 / t5` |
| `diagnose_teacher_heads.py` | Per-head teacher analysis via forward hooks (no model edits) |
| `plot_distill_scatter.py` | Multi-head diagnostic scatter/CDF plot from NPZ |

## Relevant commits

- `d18390a` — Per-head distillation + per-type projections for sampler
- `a2be9b1` — Signal-weighted per-head distillation MSE (variance-based)
- `3f701db` — Ridge-regression head weights for sampler→gold reduction ← current HEAD

## What's running now

- Background bash `bmgzoklky` — full sweep with 20 seeds × 4 temp modes, ridge head-weights.
- Log: `/tmp/sweep_run.log`.
- Results dir: `results/full-sweep-distilled-sampler-rel-f1-driver-top3/seed_{0..19}/`.
- Per-seed diagnostic NPZ: `seed_{i}/rel-f1/driver-top3/distill_diagnostic.npz`.
- Estimated wall time: ~8–10 h (≈25 min/seed × 20 seeds).

No live monitoring attached. Completion notification will fire when the background bash exits.

## Suggested analysis flow when sweep completes

1. Verify all 20 seed directories have the four joint subdirs populated with `joint.log`.
2. Parse `Best Test metrics` from each `joint.log`, aggregate mean ± std per temp mode (across seeds).
3. Compare to per-seed teacher AUC (`teacher.log`) for the gap distribution.
4. Inspect `distill_diagnostic.npz` + `distill_scatter.png` per seed to identify any seeds where:
   - sampler-vs-gold Pearson is notably below cohort median (potential sampler failure);
   - ridge weights are extreme (a sign of a quirky teacher).
5. The `diagnose_teacher_heads.py` script can be re-run per seed to see per-head teacher→gold correlations if a seed looks anomalous.

## Open questions / follow-ups not addressed this session

- **Phase-1 ↔ phase-3 distribution shift at the student level.** For some teachers (observed on old seed 1), the student warm-started from `phase1.pt` doesn't transfer well to curated neighborhoods despite the teacher being strong. Options to try if the 20-seed sweep still shows seed-dependent collapse:
  - Train student from scratch (no warm-start) on curated data.
  - Train teacher on scope-1024 neighborhoods directly (curating its own top-128) so there's no distribution shift between phases.
  - Mix `local_nodes_hetero` and curated data in the first few phase-3 epochs.
- **Sampler capacity**: currently one `Linear` per node-type; a small MLP (`Linear → GELU → Linear`) could lift per-head Pearson. Not expected to be the bottleneck based on diagnostics so far.
- **Head weights at training time**: the current flow uses uniform MSE during training and ridge only at inference. If ridge consistently helps inference, a two-stage approach (retrain sampler after fitting ridge, with a loss weighted by `|w_h|`) might give a second-order bump. Low priority unless inference-only ridge leaves a clear gap.
- **Alternate distillation targets** explored but not selected:
  - Softmax-KL (user rejected: pool-size coupled).
  - MarginMSE (user rejected: wanted absolute attention-magnitude interpretability).
  - Per-seed standardized MSE (also pool-coupled by the user's objection to pool normalization).

## Diagnostic commands to remember

```bash
# Per-head teacher analysis (requires GPU)
python diagnose_teacher_heads.py --dataset rel-f1 --task driver-top3 \
    --out_dir results/full-sweep-distilled-sampler-rel-f1-driver-top3/seed_<i> \
    --num_neighbors 128 --num_layers 3 --channels 256 --num_heads 4 \
    --ff_dropout 0.3 --attn_dropout 0.3 --gt_conv_type full --ablate none \
    --batch_size 16 --seed <i> --split val

# Scatter plot from NPZ (CPU only)
python plot_distill_scatter.py --npz <path>/distill_diagnostic.npz
```

## Key diagnostic numbers (reference)

Old sweep (mean-over-heads target, pre-rewrite):
- seed 1: teacher 0.824 · det 0.784 · t1 0.770
- sampler-vs-teacher Pearson (per-seed mean): ~0.17

Post-rewrite per-head target, uniform weights:
- seed 0: teacher 0.800 · det 0.803 · t1 0.791 · t2 0.789 · t5 0.816
- seed 1: teacher 0.793 · det 0.619 · t1 0.594 · t2 0.553 · t5 0.508 (collapse)
- seed-1 teacher had 2 dead heads out of 4

Variance-weighted head distillation, partial sweep:
- seed 0: teacher 0.762 · det 0.832 · t1 0.791 · t2 0.835 · t5 0.828 (exceeds teacher)
- seed 1: teacher 0.805 · det 0.684 (still 0.12 below teacher; no head death this seed)
- variance weights skewed on heads whose sampler fit was weak → regression

Ridge-weighted (current, 20-seed sweep pending): results TBD.
