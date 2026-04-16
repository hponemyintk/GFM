# PASS Sampler Diagnostic + REINFORCE Baseline Fix — Experiment Plan

## Context

Across all prior sweeps on `rel-f1/driver-top3` (15 seeds × 4 configs)
and `rel-f1/driver-position` (10 seeds × 2 configs), the PASS sampler
consistently acts as a **variance regularizer** rather than a
**discriminative policy**: lower seed-to-seed variance, slightly better
AUC, but worse AP (the metric that matters for imbalanced classification)
and worse MAE (after baseline-offset correction).

**Hypothesis:** The sampler's REINFORCE gradient is too noisy to learn
meaningful importance weights. Evidence from a saved checkpoint
(`ogpass_s14`) shows `as_` = [0.483, 0.483] → softmax ≈ [0.50, 0.50],
meaning the sampler converged to a 50/50 mix of importance and uniform
sampling — it never learned to lean toward importance-based selection.

This plan covers three phases:
1. **Diagnose** — inspect all existing checkpoints to confirm the sampler
   isn't learning (are Ws/as_ near-init? is selection ≈ random?)
2. **Fix** — add REINFORCE moving-average baseline + tune on AP
3. **Rerun** — driver-top3 sweep with fixes, compare against prior results

## Phase 1: Diagnostic script (`debug_runs/diagnose_sampler.py`)

Write a script that loads all existing ogPASS `pass_sampler.pt`
checkpoints and reports:

### 1a. Weight inspection (all checkpoints)

For each `pass_sampler.pt` found under `debug_runs/results/ogpass_*/`:

- **`as_` mixing weights:** compute `softmax(as_)`. If importance weight
  ≈ 0.50 (same as init), the sampler never learned to prefer importance
  over uniform. Report: raw `as_` values, softmax values, delta from init.
- **`Ws` projection matrix:** compute L2 norm, min, max, std. If near-zero
  (zero-init), the importance distribution is near-uniform regardless of
  `as_`. Compare across seeds and sweeps — does `Ws` grow with more epochs?
- **`type_embeddings`:** norm per type. Do some node types get stronger
  embeddings (= more important in the importance computation)?

Output: a summary table printed to stdout and saved to
`debug_runs/sampler_diagnostic.md`.

### 1b. Selection comparison (3 representative checkpoints)

For 3 checkpoints (one each from baseline_n50, warmup_n10, dp_n10), load
the sampler and the model, run a forward pass on a small batch of val
data, and compare:
- What neighbors PASS selects vs. what random would select
- The entropy of the importance distribution (high entropy ≈ uniform)
- Overlap percentage between PASS top-K and random top-K

**Skip 1b if 1a clearly shows near-init weights** — if `as_` is 50/50
and `Ws` norm is small, the sampler IS random and selection comparison
adds no information.

### Critical files for Phase 1

- `pass_sampler.py` — `PASSHeteroSampler.__init__` (Ws zero-init line 45,
  as_ init [0.5, 0.5] line 48), `forward()` (importance computation
  lines 128–217), `reinforce_loss()` (lines 219–236)
- Checkpoints at `debug_runs/results/ogpass_*/rel-f1/*/pass_sampler.pt`
  (~20 files across driver-top3 and driver-position runs)
- Checkpoint load logic: `main_node_ddp.py` lines 757–770 (filters
  tfs_encoder keys out with `own_keys`)

## Phase 2: Code fixes

### 2a. REINFORCE moving-average baseline (`pass_sampler.py`)

**Problem:** The current `reinforce_loss()` (lines 219–236) computes:
```
sample_loss = mean(loss_up @ (logp * selected_embeds))
```
with no baseline subtraction. This is raw REINFORCE — high variance,
the absolute magnitude of `loss_up` dominates the gradient direction
rather than the relative quality of selected neighbors.

**Fix:** Add exponential moving-average (EMA) baseline to reduce
gradient variance. Changes to `pass_sampler.py`:

1. In `__init__`, add:
   ```python
   self.register_buffer('baseline_ema', torch.tensor(0.0))
   self.register_buffer('baseline_initialized', torch.tensor(False))
   self.baseline_momentum = 0.99
   ```

2. In `reinforce_loss()`, before the final loss computation (line 235):
   ```python
   # Compute per-sample rewards: inner product of task gradient with
   # policy-weighted embeddings
   rewards = (loss_up * X).sum(dim=-1)  # [B]

   # Update EMA baseline
   with torch.no_grad():
       batch_mean = rewards.mean()
       if not self.baseline_initialized:
           self.baseline_ema.copy_(batch_mean)
           self.baseline_initialized.fill_(True)
       else:
           self.baseline_ema.mul_(self.baseline_momentum).add_(
               batch_mean, alpha=1 - self.baseline_momentum
           )

   # Advantage = reward - baseline
   advantages = rewards - self.baseline_ema  # [B]

   # Policy gradient: advantage * log_prob (summed over selections)
   logp_sum = logp.sum(dim=1)  # [B] — sum log-probs of K-1 selections
   sample_loss = (advantages.detach() * logp_sum).mean()
   ```

   This replaces the current `torch.bmm(loss_up, X)` formulation with
   an advantage-weighted log-prob, which is the standard REINFORCE with
   baseline form.

### 2b. Tune on AP instead of AUC (`main_node_ddp.py`)

**Problem:** Line 247 sets `tune_metric = "roc_auc"` for binary
classification. Checkpoints are selected by best val AUC, but AP is
the metric we care about. AUC-optimal checkpoints may be AP-suboptimal.

**Fix:** Change line 247:
```python
tune_metric = "average_precision"
```

This is a one-line change. Both metrics are already computed by
RelBench's `task.evaluate()`.

### 2c. Add `--use_reinforce_baseline` flag (optional, for A/B testing)

Add a CLI flag so we can compare with/without baseline in the same sweep:
```python
parser.add_argument("--use_reinforce_baseline", action="store_true",
                    help="Enable EMA baseline for REINFORCE sampler loss")
```
Pass to `PASSHeteroSampler.__init__`, gate the baseline logic on this
flag. This lets us run the sweep with and without the fix in the same
experiment.

## Phase 3: Rerun driver-top3 sweep

### 3a. Config

Same setup as prior sweeps but on driver-top3 with AP as tune metric.
5 seeds (0–4) × 2 branches × 3 K values. ogPASS uses 3-phase schedule
(20/20/60 %).

| exp | K | epochs | PASS warmup | sampler_only | joint | baseline flag |
|---:|---:|---:|---:|---:|---:|---|
| A1 | 1  | 20 | 4 | 4 | 12 | --use_reinforce_baseline |
| A2 | 10 | 20 | 4 | 4 | 12 | --use_reinforce_baseline |
| A3 | 50 | 20 | 4 | 4 | 12 | --use_reinforce_baseline |

dev-kyaw runs remain unchanged (no sampler). The only differences from
prior driver-top3 sweeps are:
1. Tune metric = AP (was AUC)
2. REINFORCE baseline enabled (was no baseline)

### 3b. Comparison table (to produce)

After the sweep, produce this table comparing old vs new ogPASS:

| K | dev-kyaw AP | ogPASS (old, AUC-tuned) | ogPASS (new, AP-tuned + baseline) | Δold | Δnew |
|---|---|---|---|---|---|

The prior sweep data is in `experiment_results.md` (warmup_n10_ep30 for
K=10, fix12_n30_ep30 for K=30, baseline_n50_ep10 for K=50).

### 3c. Infrastructure

- **run_one.sh:** Revert `TASK` back to `driver-top3` (it's currently
  hardcoded to `driver-position`).
- **New sweep scripts:** `sweep_dt3_baseline_n{1,10,50}.sh` — same
  pattern as `sweep_dp_*.sh` but for driver-top3, passing
  `--use_reinforce_baseline`.
- **Summarizer:** Adapt `summarize_dp.py` or write a new one for
  driver-top3 metrics (AP, AUC, F1, Acc). Output to
  `experiment_results_baseline_fix.md`.
- **Driver script:** `run_dt3_baseline_sweeps.sh` — runs all 3
  sequentially, summarizes after each.
- **Detached launch:** `setsid --fork nohup` pattern (proven to survive
  session resets).

### 3d. Wall-clock estimate

From prior driver-top3 runs (~40-60s/epoch at K=50, 233 steps/epoch):
- 5 seeds × 2 branches × 20 epochs × 3 configs = 600 total epochs
- At ~50s/epoch avg + precompute overhead: ~10-12 hours

## Run order

1. **Diagnostic** (~30 min): run `diagnose_sampler.py`, inspect output.
   If `as_` ≈ [0.50, 0.50] and `Ws` norm is small across most
   checkpoints → confirms the sampler never learned, proceed to fix.
2. **Code changes** (~30 min): implement 2a (baseline), 2b (tune metric),
   2c (flag). Test with a 2-epoch probe.
3. **Sweep** (~15-18h): launch detached, summarize after each K.
4. **Analysis**: produce comparison table, write up in
   `experiment_results_baseline_fix.md`.

## Expected outputs

- `debug_runs/sampler_diagnostic.md` — weight inspection report
- `pass_sampler.py` — modified with EMA baseline (gated by flag)
- `main_node_ddp.py` — tune_metric changed to AP for binary classification
- `debug_runs/logs/{devkyaw,ogpass}_dt3_bl_n{1,10,50}_ep30_s{0..4}.log`
- `experiment_results_baseline_fix.md` — comparison of old vs new ogPASS

## Verification

- After diagnostic: confirm `as_` softmax values and `Ws` norms are
  reported for all checkpoints. Check that near-init values are flagged.
- After code changes: run 2-epoch probe, verify that `baseline_ema` is
  being updated (print it), and that `sample_loss` magnitude is different
  from before (should be smaller with baseline subtraction).
- After sweep: confirm all result JSONs contain `average_precision` and
  `roc_auc`. Verify tune_metric = AP by checking that val_metrics logs
  show checkpoint selection based on AP, not AUC.
