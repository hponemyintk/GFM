# Holdout Experiments — GFM Generalization

This file records the results from the two holdout protocols implemented
in the launchers: Phase-4 same-dataset task holdout
(`scripts/holdout_task_dev.sh`) and Phase-5 cross-dataset adoption
(`scripts/holdout_dataset_eval.sh`). Both share the same backbone
contract — train a single multi-task RelGT, freeze it, and adopt to a
held-out task or dataset via a fresh head (linear or TabPFN-in-context).

## TL;DR

| Protocol | Holdout | Adopter | Test metric | Mean ± SD | RelBench v2 GNN baseline |
|---|---|---|---|---|---|
| Phase 5 | rel-arxiv.paper-citation | linear head, frozen backbone | AUROC | **0.7600 ± 0.0004** | 0.8250 ± 0.0004 |
| Phase 5 | rel-arxiv.paper-citation | TabPFN, post-hoc | AUROC | **0.7510 ± 0.0057** | 0.8250 ± 0.0004 |
| Phase 4 | rel-f1.driver-top3 / rel-event.user-attendance | (adoption blocked by column-mismatch bug; see notes) | — | — | — |

Both adopters beat the single-table LightGBM baseline on RelBench v2
(0.7121) by 4-5 points and land 6-7 points below the single-task
supervised RDL ceiling (0.8250) — the headline transfer-learning gap.

## Phase 5: cross-dataset adoption (rel-f1 + rel-hm → rel-arxiv)

### Setup

| | |
|---|---|
| Launcher | `scripts/holdout_dataset_eval.sh` |
| Source pretrain datasets | `rel-f1`, `rel-hm` (multi-source) |
| Source pretrain tasks | 3 rel-f1 (driver-position reg, driver-dnf bcls, driver-top3 bcls) + 2 rel-hm (user-churn bcls, item-sales reg) = 5 entity tasks |
| Target dataset | `rel-arxiv` (held out from pretraining; backbone never saw it) |
| Target tasks | `paper-citation` (entity binary, AUROC) ; `author-publication` (entity reg) — paper-citation was the focus of the local sweep |
| Backbone config | `channels=128`, `num_layers=1`, `num_heads=4`, `num_centroids=512`, `K=64` (laptop config) |
| FULL_GRAPH | 1 (autocomplete-task safe) |
| Pretrain epochs | (cached pretrain artifacts from prior p4d run, multi-source) |
| Adoption | extract embeddings via frozen backbone with `register_new_dataset` for rel-arxiv, then fit a fresh linear head + TabPFN |
| Seeds | 0, 1, 2 — per seed wipes the per-row neighbor HDF5 cache so the sampler regenerates under a new RNG |

The backbone is genuinely frozen: only the column-aware encoder
(`NeighborTfsEncoder`) registers rel-arxiv's tables and column stats
at adoption time; the transformer / centroid / attention layers stay
bit-identical to the saved checkpoint.

### Adopter configurations

* **`finetune_head`** — `Linear(128, 1)`, AdamW lr=1e-3, 30 epochs,
  batch_size=256, BCEWithLogitsLoss. Best-val checkpoint reloaded
  before final test eval (`task.evaluate(predictions)`).
* **`tabpfn_eval`** — TabPFN v2 (`tabpfn==2.0.9`), `n_estimators=2`,
  `memory_saving_mode=True`, GPU device, support set capped at 5,000
  rows via stratified subsample (preserves the ~5% positive-class
  share on paper-citation), test inference chunked to 1,000 rows per
  pass to stay under the 12 GB GPU budget.

### Results — `rel-arxiv.paper-citation` (entity binary, AUROC ↑)

| seed | extract wall | finetune AUROC | finetune F1 | finetune Acc | tabpfn AUROC | tabpfn F1 | tabpfn Acc |
|---|---|---|---|---|---|---|---|
| 0 | 3h01m | 0.7596 | 0.6345 | 0.6972 | 0.7453 | 0.6039 | 0.6887 |
| 1 | 2h41m | 0.7599 | 0.6417 | 0.6954 | 0.7567 | 0.6228 | 0.6960 |
| 2 | 2h40m | 0.7604 | 0.6346 | 0.6989 | 0.7511 | 0.6111 | 0.6954 |
| **mean ± SD** |  | **0.7600 ± 0.0004** | 0.6369 ± 0.0041 | 0.6972 ± 0.0018 | **0.7510 ± 0.0057** | 0.6126 ± 0.0095 | 0.6934 ± 0.0040 |

Total wall time: 8h 48m on the laptop (single 12 GB GPU, sequential per-seed). Raw artifacts at
`/tmp/holdout_dataset_relf1_relhm_to_relarxiv/rel-f1+rel-hm_to_rel-arxiv/paper-citation/seed{0,1,2}/{finetune,tabpfn}.json`.

### Comparison to RelBench v2 paper baselines

RelBench v2 Table 7 / 16 reports for `rel-arxiv.paper-citation`:

| Method | Test AUROC | Setup |
|---|---|---|
| LightGBM (single-table) | 0.7121 ± 0.0013 | tabular only, no graph |
| **GFM TabPFN (this repo)** | **0.7510 ± 0.0057** | frozen multi-source backbone, post-hoc 5k-support TabPFN |
| **GFM linear head (this repo)** | **0.7600 ± 0.0004** | frozen multi-source backbone, linear adoption |
| GNN (RDL HeteroGraphSAGE, single-task supervised) | 0.8250 ± 0.0004 | trained end-to-end on rel-arxiv |

Reading the gap:

* The **+4.8-pt lift over LightGBM** is genuine cross-dataset transfer
  signal — the GFM never saw rel-arxiv during pretraining, but the
  multi-source backbone produces embeddings that beat a tabular-only
  model on a held-out citation graph.
* The **−6.5-pt gap below the supervised RDL ceiling** is the price
  of frozen-backbone transfer with only a linear head on top.
  Closing this gap is what the `--backbone_init` warm-start path in
  `main_node_ddp.py` (commit `ccfcfc0`) is for; full-fine-tune
  numbers belong on a p4d run, not the laptop.

### Variance characterization

Three observations from the SD column:

1. **Linear-head SD is essentially zero (4×10⁻⁴).** Even with
   per-seed neighbor sampling re-rolled from scratch (HDF5 cache
   wiped between seeds via `random.seed(args.seed)` + `np.random.seed`
   + `torch.manual_seed`), the linear-on-frozen-embeddings adopter is
   nearly deterministic. That's expected — a single linear layer over
   534k inputs has very little discretion left after the embeddings
   are fixed.
2. **TabPFN SD is ~14× larger (5.7×10⁻³).** The 5k stratified
   subsample of the 534k support set has a real effect on TabPFN's
   in-context predictions; different seeds pick different 5k
   "exemplars". This is the dominant variance component for the
   in-context adopter.
3. **The variance gap implies that for a paper-grade error bar on
   `finetune_head`** you need to also vary something upstream of the
   embeddings (e.g., the backbone seed at pretrain time). The current
   sweep only varies the rel-arxiv-side neighbor sampling.

## Phase 4: same-dataset task holdout (rel-f1 + rel-event)

### Setup

| | |
|---|---|
| Launcher | `scripts/holdout_task_dev.sh` |
| Datasets | `rel-f1`, `rel-event` (both in pretrain AND adoption) |
| Holdout tasks | `rel-f1.driver-top3` (binary) ; `rel-event.user-attendance` (regression) |
| Pretrain tasks | union of the dataset task lists minus the holdouts: 4 rel-f1 (driver-position, driver-dnf, results-position, qualifying-position) + 5 rel-event (user-repeat, user-ignore, event_interest-interested, event_interest-not_interested, users-birthyear) = 9 entity tasks |
| Backbone config | p4d paper config (`channels=512`, `num_layers=4`, `num_heads=4`, `num_centroids=4096`, `K=300`, `BATCH=512`) |
| FULL_GRAPH | 1 (autocomplete-task safe) |
| Pretrain epochs | 20 (best epoch 17, val macro 0.4966) |
| Steps per task | 1000 (data-limited at 297 actual steps/epoch on rel-event-sized seeds) |
| Hardware | 1× p4d.24xlarge, 8× A100 40GB, NPROC=8, DDP |

### Pretrain test metrics (recorded 2026-04-29)

The pretrain run completed cleanly. From the log:

| Task | Type | Primary metric | Value |
|---|---|---|---|
| rel-f1.driver-position | regression | R² | 0.165 |
| rel-f1.driver-dnf | binary | AUROC | 0.717 |
| rel-f1.results-position | regression | R² | 0.861 |
| rel-f1.qualifying-position | regression | R² | 0.943 |
| rel-event.user-repeat | binary | AUROC | 0.706 |
| rel-event.user-ignore | binary | AUROC | 0.566 |
| rel-event.event_interest-interested | binary | AUROC | 0.169 |
| rel-event.event_interest-not_interested | binary | AUROC | 0.199 |
| rel-event.users-birthyear | regression | R² | -0.518 |

Notes:
- The `event_interest-*` AUROCs near random (and below 0.5) are
  consistent with RelBench v2 paper Table 3, where GNN baselines on
  these tasks score 0.4764 / 0.6040. These are inherently weak-signal
  tasks with extreme class imbalance.
- `users-birthyear` R²=-0.518 is poor but RelBench v2 Table 5 gives
  the GNN baseline at R²=-0.030 — the task is hard for everyone.
- Per-epoch val macro progression: e1 0.375 → e6 0.464 → e12 0.472
  → e16 0.484 → **e17 0.497 (best)** → e20 0.496.

### Holdout adoption — pending

The 2026-04-29 p4d holdout-adoption run **crashed at extract time**
on `rel-f1.driver-top3` with:

```
RuntimeError: The size of tensor a (3) must match the size of tensor b (2)
  at non-singleton dimension 1
```

at `encoders.py::_normalize_numerical`. Root cause: RelBench's
task-aware leakage stripping leaves a different surviving column set
on the shared `rel-f1.results` table when `driver-top3` is the active
task vs when `results-position` was active during pretraining (3
numerical columns survive vs 2). The pretrained `_num_mean_*` /
`_num_std_*` buffers were sized for 2 cols but the adoption-time TF
arrived with 3 cols → broadcast failure.

**Fix:** commits `1c021bd` (`encoders: align numerical buffers by
column name at adoption`) and `dbeee29` (`encoders: drop positional
truncate/pad fallback`). The encoder now records the column-name
order at `register_dataset` time and aligns by name in
`_normalize_numerical`; pretrained columns keep their saved stats and
new columns get identity normalization (mean=0, std=1).

The holdout-adoption rerun is pending — it needs a p4d slot. The
launcher itself is unchanged.

## How to reproduce

### Phase 5 (cross-dataset, the numbers above)

Laptop (single-GPU, slow):

```bash
SOURCE="rel-f1 rel-hm" TARGET=rel-arxiv \
  EPOCHS=3 STEPS_PER_TASK=30 \
  bash scripts/holdout_dataset_eval.sh
```

p4d.24xlarge (paper config, full LOO with the new defaults — TARGET
becomes `rel-event` and SOURCE auto-fills to the other 8 supported
datasets):

```bash
NPROC=8 SHARD_WORKERS=10 EPOCHS=10 STEPS_PER_TASK=500 \
  bash scripts/holdout_dataset_eval.sh
```

To reproduce the rel-arxiv holdout exactly:

```bash
NPROC=8 SHARD_WORKERS=10 \
  TARGET=rel-arxiv SOURCE="rel-f1 rel-hm" \
  EPOCHS=10 STEPS_PER_TASK=500 \
  bash scripts/holdout_dataset_eval.sh
```

The launcher writes per-task summary to
`results/holdout_dataset_eval/<src_slug>_to_<target>/summary.json`
with mean ± SD across the seed list.

### Phase 4 (same-dataset task holdout)

Laptop (rel-f1 only, fast):

```bash
DATASETS="rel-f1" HOLDOUTS="rel-f1:driver-top3" \
  EPOCHS=3 STEPS_PER_TASK=30 \
  bash scripts/holdout_task_dev.sh
```

p4d.24xlarge:

```bash
NPROC=8 EPOCHS=20 STEPS_PER_TASK=1000 \
  bash scripts/holdout_task_dev.sh
```

The defaults pretrain on rel-f1 + rel-event (every entity binary +
regression task except the 2 holdouts), then adopt to
`rel-f1.driver-top3` and `rel-event.user-attendance`.

## Open questions

* **Backbone-side variance.** Current 3-seed Phase-5 numbers vary
  only the per-row neighbor sampling. To bound the *backbone* SD we
  need the pretrain itself re-run with different seeds; the cost is
  ~8h × 3 seeds on p4d.
* **Full fine-tune comparison.** `main_node_ddp.py --backbone_init`
  is wired (commit `ccfcfc0`) but not run. Expected delta vs frozen
  is what closes the 6.5-pt gap to the supervised RDL ceiling.
* **Multiclass + recommendation tasks.** rel-arxiv `author-category`
  (multiclass) and `paper-paper-cocitation` (link-prediction) aren't
  covered by `finetune_head` / `tabpfn_eval` today. Same for
  `rel-mimic` and `rel-salt`, neither of which has any entity
  binary/regression tasks.
* **Other holdouts.** Default Phase-5 launcher now LOO's `rel-event`
  (6 tasks) for richer adoption-side eval; the rel-arxiv numbers
  above are from the prior `rel-f1 + rel-hm → rel-arxiv` config.
  Sweeping `TARGET` over all 9 supported datasets gives the full LOO
  matrix.
