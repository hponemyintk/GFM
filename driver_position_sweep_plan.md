# ogPASS vs dev-kyaw — driver-position Sweep Plan

## Context

`experiment_results.md` compares the `ogPASS` learned sampler against
`dev-kyaw` random sampling on `rel-f1 / driver-top3` (binary, imbalanced).
Across every sweep at K>1, ogPASS loses on AP — the right metric for that
imbalanced task. The user now wants to rerun the comparison on
`rel-f1 / driver-position`, which is a **regression** task (metrics:
`mae`, `rmse`, `r2`; tune metric = `mae`, lower is better). This sidesteps
class imbalance and gives a cleaner read on whether the learned sampler
helps when the signal is a continuous target.

The goal is a fresh head-to-head on `driver-position` across a few
neighbor-budget / epoch settings, with the 3-phase PASS schedule
(warmup → sampler_only → joint) allocated as 20/20/60 % of epochs so
`x_set.grad` stabilizes before the sampler starts updating, and the
sampler converges on a frozen reward before joint training.

## Experiments

3 configs total, both branches (`dev-kyaw`, `ogPASS`). Originally 5
configs were discussed; exp 4 and 5 were dropped to keep wall-clock
tractable. Exp 2 and 3 were further shortened from 10 seeds × 30 epochs
→ 5 seeds × 20 epochs (~12–13 h → ~8–9 h) so they finish inside an
overnight run.

| exp | K (num_neighbors) | epochs | seeds | PASS warmup | PASS sampler_only | PASS joint |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 1  | 10 | 10 (0–9) | 2  | 2  | 6  |
| 2  | 10 | 20 |  5 (0–4) | 4  | 4  | 12 |
| 3  | 50 | 20 |  5 (0–4) | 4  | 4  | 12 |

PASS split is 20 / 20 / 60 % of total epochs. `dev-kyaw` ignores the
phase flags (no such args on that branch). Exp 1 with K=1 is a seed-only
floor; the `sampler_only` phase is a no-op at K=1 but kept to match the
schedule shape.

**Status:** Exp 1 completed 2026-04-15 (`experiment_results_driver_position.md`,
sweep `n1_ep10`). Exp 2 and 3 pending.

## Fixed config (matches existing `debug_runs/run_one.sh`)

- `--dataset rel-f1 --task driver-position`
- `--batch_size 32 --num_layers 4 --channels 512`
- `--max_steps_per_epoch 1000 --lr 0.0001 --warmup_steps 100`
- `--ff_dropout 0.3 --attn_dropout 0.3`
- `--precompute`, single-GPU DDP, cache cleared between seeds
- Tune on val `mae` (lower is better); report best-val test `mae/rmse/r2`

## Critical files (read-only reference; no code edits needed)

- `main_node_ddp.py` — entry point.
  - `--num_neighbors`, `--epochs`, `--sampler_warmup_epochs`,
    `--sampler_only_epochs` args (lines 60–102).
  - `_set_pass_phase` toggles `requires_grad` for 3-phase schedule
    (line 339).
  - Phase scheduler at line 710 (`warmup_end`, `sampler_only_end`).
  - Regression branch sets `tune_metric = "mae"`, `higher_is_better = False`
    (lines 249–253).
  - "Best Test metrics:" printed at line 781 — grep target for parsing.
- `debug_runs/run_one.sh` — per-run launcher; takes worktree dir, run_name,
  seed, epochs, plus extra args. Clears precompute cache, logs to
  `debug_runs/logs/${run_name}.log`, writes outputs under
  `debug_runs/results/${run_name}/`. **Edited** to hardcode
  `TASK="driver-position"` on line 16 and echo it per-run for visibility.
  `--num_neighbors` is overridden via CLI extras (argparse last-wins).
- `debug_runs/sweep_warmup_n10.sh` — template for new sweep scripts.
- `/tmp/gfm-dev-kyaw` — existing worktree on `dev-kyaw` branch
  (verified via `git worktree list`).

## Work to do

### 1. `debug_runs/run_one.sh` — done
Hardcoded `TASK="driver-position"` (line 16) and added an echo for
visibility. `--num_neighbors` overriding the hardcoded default via CLI
extras works thanks to argparse last-wins semantics.

### 2. Three sweep scripts under `debug_runs/` — done

- `sweep_dp_n1_ep10.sh`  — K=1,  epochs=10, seeds 0..9, PASS 2/2/6
- `sweep_dp_n10_ep20.sh` — K=10, epochs=20, seeds 0..4, PASS 4/4/12
- `sweep_dp_n50_ep20.sh` — K=50, epochs=20, seeds 0..4, PASS 4/4/12

Each runs dev-kyaw first then ogPASS, clears precompute cache (via
`run_one.sh`), writes logs to `debug_runs/logs/${run_name}.log`.
Run-name scheme: `{devkyaw|ogpass}_dp_n{K}_ep{E}_s{SEED}`.

### 3. A summary writer: `debug_runs/summarize_dp.py`

Short python script that:
1. Walks `debug_runs/results/<run_name>/rel-f1/driver-position/<seed>.json`
   (auto-detects which seeds are present, up to `MAX_SEEDS=20`).
2. Emits per-seed tables + mean±std rows for both branches, plus
   Δ(ogpass−devkyaw) and variance-ratio rows.
3. Appends to `experiment_results_driver_position.md` (created fresh)
   using replaceable `<!-- BEGIN/END SWEEP <name> -->` markers so a
   re-summarize overwrites just the matching sweep block.

### 4. Driver scripts: `debug_runs/run_dp_sweeps*.sh`

- `run_dp_sweeps.sh` — runs all 3 sweeps sequentially (used for the initial
  launch that was killed after Exp 1).
- `run_dp_sweeps_23.sh` — runs only Exp 2 and Exp 3 sequentially. Used
  after Exp 1 completed so we can resume without redoing Exp 1.

Each invokes `summarize_dp.py` after every sweep so partial results are
written incrementally. Per-sweep driver logs land at
`debug_runs/logs/sweep_<name>.out`; master driver log at
`debug_runs/logs/run_dp_sweeps*.master.log`.

### 5. Timing probe (done)

K=1 dev-kyaw 2 epochs on `driver-position` → ~42 s/epoch at K=1 with
233 train steps (full dataset; `max_steps_per_epoch=1000` not reached).
Log at `debug_runs/logs/probe_dp_n1.log`.

## Run order

1. ~~Timing probe~~ (done).
2. ~~Exp 1 (K=1, ep=10, seeds 0..9)~~ (done — sweep `n1_ep10` in results md).
3. ~~Exp 2 (K=10, ep=20, seeds 0..4)~~ (done 8/10 — sweep `n10_ep20`).
4. Exp 3 (K=50, ep=20, seeds 0..4) — still running via detached nohup/setsid.

## Outcome (2026-04-16)

**driver-position has near-zero graph signal.** The paper reports only
2.61% MAE improvement from RDL → RelGT at K=300. Our K=1 seed-only
baseline reaches MAE ~4.09 (ogPASS) / ~4.25 (dev-kyaw), already close
to the paper's K=300 RDL result of 4.02.

At K=10, dev-kyaw barely improves over K=1 (Δ = −0.02 MAE), and ogPASS
actually gets worse (Δ = +0.11 MAE). After subtracting the K=1 baseline
offset, the sampler effect at K=10 is **+0.13** (harmful). This task
does not have enough graph signal for any sampler to demonstrate value.

See `experiment_results_driver_position.md` for full data tables and
task selection analysis recommending `driver-top3` with AP as the best
available RelBench task for sampler evaluation.
- Final check: `experiment_results_driver_position.md` contains three
  sweep sections, mean±std rows for both branches, and Δ/variance-ratio
  rows computed from the per-seed data.
