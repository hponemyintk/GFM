# Multi-Task, Multi-Dataset Refactor — Final Plan

**Branch:** `gfm_test_using-table_agnostic_model-branch`
**Reference branch for sampler/baseline parity:** `dev-kyaw`
**Reference papers:** Ranjan et al., "Relational Transformer", ICLR 2026 (`rfm_2510.06377v3.pdf`); Fey et al., "KumoRFM" (`kumo_relational_foundation_model.pdf`)
**Author:** Kyaw — 2026-04-25

---

## 1. Goals

1. Train **one model jointly across multiple (dataset, task) pairs** in RelBench, restricted in v0 to **regression and binary classification**.
2. Make preprocessing **memory-efficient enough to scale from this laptop → 8×A100 / 1TB box → eventually all of RelBench v2** without code rewrites.
3. **Do not modify** `encoders.py`, `model.py`, `codebook.py`. All changes are in new modules under `gfm_data/`, `heads/`, `losses/`, plus a thinned `main_node_ddp.py`.
4. Keep existing tests in `tests/` green; add the test suite in §6.

In-scope task list for v0:

| Dataset | Task | Type |
|---|---|---|
| rel-f1 | `driver-position` | regression |
| rel-f1 | `driver-dnf` | binary |
| rel-f1 | `driver-top3` | binary |
| rel-event | `user-attendance` | regression |
| rel-event | `user-repeat` | binary |
| rel-event | `user-ignore` | binary |

(Exact task names will be re-confirmed against `relbench.tasks` registry at PR1 time.)

**Out of scope (this round):** in-context learning / forward-looking samplers / PQL / multi-class / multi-label / link prediction / model-side changes.

---

## 2. Architecture overview

```
main_node_ddp.py  ── argparse list of (dataset, task) ──┐
                                                         ▼
              gfm_data/registry.py        gfm_data/multi_task_dataset.py
              (loads & caches each ds)   (ConcatDataset-like, weighted sampler)
                                                         ▼
              gfm_data/graph_cache.py  ──── one DatasetGraphCache per dataset (CSR adj + memmap TFs)
              gfm_data/task_tokens.py        one TaskTokens per (dataset, task, split)
                                                         ▼
              gfm_data/sampler.py            CSR-based 1/2-hop sampler (replaces utils.py:59-213)
              gfm_data/collate.py            multi-task aware collate (replaces utils.py:522-615)
              gfm_data/shard_io.py           memmap shard reader/writer
              gfm_data/tf_store.py           disk-backed TensorFrame columns
              tools/precompute_shards.py offline sharded sampler
              tools/build_tf_store.py    offline TF memmap builder
                                                         ▼
              RelGT backbone  (UNCHANGED — encoders.py / model.py / codebook.py)
                                                         ▼
              heads/multi_task_head.py   per-datatype shared heads (numeric + boolean)
              losses/multi_task_loss.py  Huber + BCE, batch-mean, RT formula
```

Node types are **prefixed with the dataset name** (`"rel-f1::drivers"`) at graph-cache build time so encoder/collate grouping stays correct across datasets — the only cross-cutting change to non-touched files is the type strings, which encoders treat opaquely.

---

## 3. Preprocessing redesign

### 3.1 Diagnosed bottlenecks (today)

- **Adjacency dict** in `build_adjacency_hetero` (`utils.py:35-51`): `dict[node_type, list[set[(neighbor_type, idx)]]]`. ~80 B/edge. For rel-event (41M rows) this dict alone is tens of GB.
- **TensorFrames in CPU RAM**: `data[node_type].tf` holds whole-table column tensors. Independent of seed count.
- **HDF5 precomputation** loads 10k-sample chunks into RAM, then writes one big file per task; tasks of the same dataset re-do the work.

### 3.2 Replacements

**A. CSR adjacency** (replaces dict).

For each `(src_type, edge_type, dst_type)`:
- `indptr: int64[N_src + 1]`
- `dst_idx: int32[E]`
- `dst_time: float32[E]` (seconds, or int32 unix-day if it packs better)

~12 B/edge instead of ~80 B/edge. Built once per dataset, cached to `<cache>/<dataset>/adj.npz`, mmap-loaded.

**B. Memmap-backed TensorFrame columns** (`gfm_data/tf_store.py`).

- On first build, write each numerical/categorical/embedding/timestamp column to a memmap file:
  `<cache>/<dataset>/tf/<table>/<column>.{i32,f16,...}`
- For variable-length columns (multi-categorical), also write `<column>.offsets.i64`.
- For embedding columns, write a per-column `dim` to `<table>/meta.json`.
- `DatasetGraphCache.tf_view(table, row_idx)` assembles a `torch_frame.TensorFrame` slice on demand. Same shape/dtypes the encoder expects → **zero changes to encoders.py**.

This is what makes `--max_rows_per_task` actually bound resident memory: the OS pages in only the K-hop closure of the active seeds.

**C. Sharded sample I/O** (`gfm_data/shard_io.py`).

When precompute is enabled, write shards instead of one HDF5:
- `shard_NNNN.types.i16`, `.indices.i32`, `.hops.i8`, `.times.f32`, `.edges.i16`, `.edges_offsets.u64`
- One shard per N samples (e.g. 50k). Workers `mmap` only their shards. No HDF5 GIL contention.

**D. Streaming mode.**

Add `--mode streaming` that runs sampling in DataLoader workers without writing shards. Memory cost = O(K) per sample. This is the laptop default.

**E. Offline tools** (`tools/`).

Move precompute logic out of `RelGTTokens.__init__` into:
- `tools/precompute_shards.py` (was `_precompute_sampling`, `utils.py:399-479`)
- `tools/build_tf_store.py` (new)

Training only reads. Cleaner DDP. Simpler memory accounting.

### 3.3 New file layout

```
gfm_data/
  __init__.py
  graph_cache.py        # DatasetGraphCache: CSR adj + tf_view + col_stats + stypes
  task_tokens.py        # TaskTokens: (entity_id, t, label) table per (dataset, task, split)
  multi_task_dataset.py # MultiTaskConcat + DistributedMultiTaskSampler
  sampler.py            # CSR-based gather_1_and_2_hop (replaces utils.py:59-213)
  collate.py            # multi-task aware collate (replaces utils.py:522-615)
  shard_io.py           # memmap shard reader/writer
  tf_store.py           # disk-backed TensorFrame
heads/
  multi_task_head.py    # shared numeric + boolean heads, dispatched by task_type
losses/
  multi_task_loss.py    # Huber + BCE, batch-mean (RT formula)
tools/
  precompute_shards.py
  build_tf_store.py
```

`utils.py` shrinks to genuinely shared helpers (or re-exports during the migration so existing tests don't break).

---

## 4. Multi-task batching

### 4.1 Composition: mixed batches via weighted task sampling

`MultiTaskWeightedSampler` (DDP-aware) draws sample indices proportional to `w_t · |D_t|` per task. **A single batch contains rows from many tasks.** Collate emits two extra per-row fields:

```python
"task_id":   int64[B]   # which (dataset, task) pair
"task_type": int8[B]    # 0 = regression, 1 = binary
```

Existing collate keys (`neighbor_types`, `neighbor_indices`, `neighbor_hops`, `neighbor_times`, `grouped_tfs`, `grouped_indices`, `flat_batch_idx`, `flat_nbr_idx`, `edge_index`, `batch`, `labels`, `node_indices`, `global_idx`) are preserved. `grouped_tfs` is keyed by **prefixed** node type so two datasets don't collide.

Default task weights: `w_t = 1 / num_tasks` (equal). Override via `--task_weights "rel-f1.driver-top3:1.0,rel-event.user-repeat:2.0,..."`.

### 4.2 DDP correctness

`DistributedMultiTaskSampler`:
- Picks tasks using a global RNG seeded by `epoch` so **all ranks pick the same task at the same step** (avoids DDP gradient-shape mismatch).
- For the chosen task, partitions the per-task index list across ranks like `DistributedSampler`.
- Reshuffles within tasks via `set_epoch`.

---

## 5. Heads & loss (RT-style)

### 5.1 Heads

`heads/multi_task_head.py`:

```python
class MultiTaskHead(nn.Module):
    """
    Two shared heads keyed by task datatype, mirroring RT (Ranjan et al., §3.3):
      - numeric head:  Linear(channels, 1)  → real-valued output for regression
      - boolean head:  Linear(channels, 1)  → logit for binary classification
    Dispatched per row via task_type vector.
    """
```

Wrapper class `MultiTaskRelGT(backbone, head)` lives in `main_node_ddp.py` (or a thin new file). `model.py` is **not** edited; backbone `out_channels=channels` and emits the pre-head embedding the heads consume.

### 5.2 Loss (RT formula)

`losses/multi_task_loss.py`:

```
L = (1/B) * (Σ_{i: regression}  HuberLoss(r_i, r'_i)  +  Σ_{i: binary}  BCE(1{r_i > 0}, r'_i))
```

This is exactly RT's "the overall loss is the mean over all masked cells in the batch" (§3.3 of `rfm_2510.06377v3.pdf`). Plain batch mean works because:

1. Regression targets are **z-score normalized per (dataset, task)** before loss. Stats fit on the train split only, stored in `TaskTokens`, applied in collate, denormalized for metrics.
2. **HuberLoss** (not L1) bounds outlier gradients — output is O(1) after normalization.
3. BCE is O(0.7).

So Huber and BCE rows live on the same scale. No need for weighting in v0.

`--loss_balance` flag retained for ablations: `none` (default, RT-style) | `per_task_mean` | `uncertainty` (Kendall et al., 2 params per task) | `fixed:w1,w2,...`.

### 5.3 Metrics

Per-task metric (AUROC for binary, MAE/RMSE for regression — matches `task.metrics`). Predictions for regression are **denormalized** before metric. Per-task and macro-mean logged to wandb each eval epoch.

---

## 6. Test plan (exhaustive)

Tagged P1/P2/P3/P4 per the phased delivery in §8.

### 6.1 Unit tests

#### 6.1.1 CSR adjacency `gfm_data/graph_cache.py` — P1

| # | Test |
|---|---|
| C1 | Toy hand-built hetero graph → CSR matches expected `indptr`/`dst_idx`/`dst_time` |
| C2 | Round-trip: dict-adjacency → CSR → dict-adjacency identical to original |
| C3 | Nodes with zero out-edges: `indptr[i] == indptr[i+1]`; sampler returns empty |
| C4 | Edge dtype overflow guard: build with > 2^15 of one type, assert dtype promotes |
| C5 | Memmap save/load round-trip is bit-identical |
| C6 | Cross-dataset namespace: prefixed `"rel-f1::drivers"` and `"rel-event::users"` are distinct keys |
| C7 | **Equivalence vs `dev-kyaw`**: on a deterministic seed-list of 200 rel-f1 nodes, the CSR `(dst_idx, dst_time)` arrays for every node match `build_adjacency_hetero(...)` from `dev-kyaw` (modulo set ordering, which we sort by `(dst_type, dst_idx)` for both before comparing) |

#### 6.1.2 CSR sampler `gfm_data/sampler.py` — P1

These are the most safety-critical because we have to match `dev-kyaw`'s `gather_1_and_2_hop_with_seed_time` + `_process_one_seed` behavior exactly.

**Scope note on de-duplication.** Dev-kyaw's sampler has two distinct phases, with different dedup semantics:

1. **Gather/exploration phase** (`gather_1_and_2_hop_with_seed_time`, `utils.py:59–135`) — builds the candidate sets `n1` and `n2` using Python `set`s. Here, **no double-counting**: each unique `(type, idx)` appears at most once across the gathered candidates, regardless of how many edges or paths reach it.
2. **Final K-1 selection phase** (`_process_one_seed`, `utils.py:163–172`) — given the deduped candidate list of size `size_th`:
   - `size_th >= K-1` → `random.sample` (no replacement, distinct K-1 tokens)
   - `0 < size_th < K-1` → **`random.choices` (with replacement) — duplicates in the final K-1 are expected and required**
   - `size_th == 0` → fallback from `GLOBAL_ALL_NODES`, hop=3

The new CSR sampler must reproduce **both** phases exactly. Tests S9–S11 below check phase-1 dedup; S12–S15 check phase-2 selection (including expected duplicates in the `< K-1` case).

| # | Test |
|---|---|
| S1 | Toy graph, seed at time t: returned 1-hop neighbors all have `time ≤ t` |
| S2 | Seed with > 5000 1-hop neighbors → exactly 5000 returned, sample reproducible w/ seed |
| S3 | Same as S2 for 2-hop cap (1000), applied **per 1-hop parent** |
| S4 | Final K=300 token list has the seed itself at position 0 with hop=0 and rel_time=0 |
| S5 | Tuples are `(neighbor_type, neighbor_idx, hop, rel_time_days, c1hops)` with `hop ∈ {0,1,2,3}`, `rel_time_days ≥ 0` |
| S6 | Edge_index returned is **K-local** (indices in `[0, K)`), not global |
| S7 | Seed with zero neighbors → fallback path triggered, all returned tokens have `hop == 3` |
| S8 | Determinism: same `(seed, time, rng_seed)` → same output across two runs / two processes |
| S9 | **Gather-phase dedup 1 (multi-path)**: if a 1-hop neighbor `(t, i)` is reachable via two distinct edges from the seed, it appears **exactly once** in the gathered 1-hop set `n1` (matches `utils.py:83–89` set semantics). *Scope: gather phase only — does NOT apply to the final K-1 token list.* |
| S10 | **Gather-phase dedup 2 (2-hop ∩ 1-hop)**: a node that is both 1-hop and 2-hop appears only as 1-hop in the gathered candidate list (`utils.py:111`). *Scope: gather phase only.* |
| S11 | **Gather-phase dedup 3 (self-loop)**: a 2-hop path that loops back to the seed is excluded from `n2` (`utils.py:102–103`). *Scope: gather phase only.* |
| S12 | **Final-selection case `size_th > K-1`**: behavior matches `random.sample(combined, K-1)` from dev-kyaw (`utils.py:163–164`) — chosen K-1 tokens are distinct *because the gathered candidate list itself is deduped*; under fixed `random.seed`, the chosen multiset matches dev-kyaw exactly |
| S13 | **Final-selection case `size_th == K-1`**: chosen list equals the gathered list as a multiset (matches dev-kyaw under fixed seed) |
| S14 | **Final-selection case `0 < size_th < K-1`**: behavior matches `random.choices(combined, k=K-1)` from dev-kyaw (`utils.py:165–166`) — **duplicates ARE present in the final K-1 tokens, by design**. Test asserts: (a) `len(final_tokens) == K-1`; (b) the multiset of `(type, idx, hop)` from the new sampler equals dev-kyaw's under the same `random.seed`; (c) at least one duplicate is observed when `size_th < K-1` (sanity that `random.choices` semantics are reproduced and not silently swapped for `random.sample`) |
| S15 | **Final-selection case `size_th == 0`**: fallback from `GLOBAL_ALL_NODES` (`utils.py:167–180`), hop=3 on every fallback token; under fixed seed the chosen fallback tokens match dev-kyaw exactly |
| S16 | **End-to-end equivalence vs `dev-kyaw`** on 1000 deterministic seeds from rel-f1 train split, fixed `random.seed`: for each seed the new sampler returns the **same multiset of `(type, idx, hop)` tuples** as dev-kyaw's sampler — including duplicates where the `0 < size_th < K-1` case fires. We compare as sorted tuple lists (token order may differ due to the post-selection `random.sample` shuffle of `rest` in `utils.py:189–193`); times match within fp32 epsilon |
| S17 | **Per-regime equivalence vs `dev-kyaw`** — manually construct toy graphs that hit each of the four regimes (`size_th > K-1`, `= K-1`, `0 < size_th < K-1`, `= 0`), run both samplers under fixed seed, assert identical multiset of `(type, idx, hop)` outputs. Critically for the `< K-1` regime, both pipelines must produce the **same set of duplicated tokens** (i.e. the random.choices draw is reproduced bit-for-bit) |

#### 6.1.3 Memmap TF store `gfm_data/tf_store.py` — P2

| # | Test |
|---|---|
| T1 | Build memmap from in-memory TF, then `tf_view(table, [i,j,k])` returns TensorFrame element-equal to `tf[[i,j,k]]` for every stype |
| T2 | Multi-categorical column (variable-length): offsets array correct; reading row i returns same sequence as original |
| T3 | Embedding column (per-column dim): dim recorded, view returns correct shape |
| T4 | Timestamp column: `datetime64` → `int64` epoch round-trip preserves comparison ordering |
| T5 | `col_stats` recomputed on memmap matches `col_stats` from the original TF within 1e-6 |
| T6 | Concurrent reads from 8 worker processes — no corruption, all reads match single-worker baseline |
| T7 | Page-fault budget: read 10k random rows; resident set ≤ `1.5 × rows × bytes-per-row` (catches accidental whole-table pre-read) |

#### 6.1.4 Sample shard I/O `gfm_data/shard_io.py` — P2

| # | Test |
|---|---|
| Sh1 | Write 1k samples → read back → bit-identical types/indices/hops/times/edges |
| Sh2 | Sample straddling shard boundary (shard size 100; read sample 99, 100, 101) → correct |
| Sh3 | Empty edge list for a sample → `edges_offsets` correctly encodes zero-width slice |
| Sh4 | Concurrent reads identical to single-reader |

#### 6.1.5 Multi-task collate `gfm_data/collate.py` — P3

| # | Test |
|---|---|
| Co1 | Mixed batch (rows from 2 datasets, 2 tasks): per-row `task_id`, `task_type` correct; `grouped_tfs` keys are **prefixed** node types |
| Co2 | Edge offsets across mixed batch: concatenated `edge_index` has correct per-sample offsets, `batch` vector lengths sum correctly |
| Co3 | All-regression batch: `task_type == 0` everywhere, labels `float32` (post-normalization) |
| Co4 | All-binary batch: `task_type == 1` everywhere, labels `float32` in `{0,1}` |
| Co5 | Single-task batch: output identical (modulo extra `task_id`/`task_type` keys) to old `collate` from `utils.py:522-615` |
| Co6 | Targets normalized for regression rows, untouched for binary rows |

#### 6.1.6 Multi-task DDP sampler `gfm_data/multi_task_dataset.py` — P3

| # | Test |
|---|---|
| D1 | Mocked `world_size=2`: both ranks draw the **same task id** at the same step |
| D2 | For each step's task, per-rank index lists are disjoint and union = full task index list |
| D3 | Statistical: with weights `{A:1, B:3}`, over 100k draws task counts within 3σ of expected |
| D4 | Reproducible with `(seed, epoch)` — same sequence |
| D5 | `set_epoch(e)` reshuffles within tasks but doesn't change task draw order across ranks |

#### 6.1.7 Heads & loss — P3

| # | Test |
|---|---|
| H1 | Numeric head + Huber matches `torch.nn.HuberLoss(delta=1.0)` on synthetic data |
| H2 | Boolean head + BCE matches `BCEWithLogitsLoss(reduction='none').mean()` on synthetic data |
| H3 | Mixed batch: total loss == `(Σ Huber(reg rows) + Σ BCE(bin rows)) / B` (RT formula) |
| H4 | Regression rows produce zero grad on boolean head's params; binary rows produce zero grad on numeric head's params |
| H5 | Backbone params receive gradient from BOTH task types in a mixed batch |
| H6 | Z-score normalize → predict → denormalize: round-trip identity within 1e-5 |
| H7 | Stats fitted on train labels only (mocked val set with different distribution → val metrics computed on raw scale) |

#### 6.1.8 Per-task metrics — P3

| # | Test |
|---|---|
| M1 | Synthetic: 100 binary rows with known logits → per-task AUROC matches `sklearn.metrics.roc_auc_score` |
| M2 | Synthetic regression with known preds → MAE matches `torch.mean(torch.abs(...))` after denorm |
| M3 | Multi-rank gather: predictions/labels gathered per-task across 2 mocked ranks → metric equals single-rank computation on combined data |
| M4 | Macro average across tasks = unweighted mean of per-task metrics |

### 6.2 Equivalence / integration tests

| # | Test |
|---|---|
| E1 | `streaming` mode batch == `precomputed` mode batch for the same `(seed, time, rng_seed)` set on rel-f1 |
| E2 | Multi-task pipeline with **one task** == old single-task pipeline on rel-f1 driver-top3 (loss, metric, gradient norm) for first 50 steps within 1e-5 |
| E3 | Memmap-TF training step == in-RAM-TF training step (same batch, same seeds) — gradients match within 1e-5 |
| E4 | DDP single-rank loss == multi-rank emulated loss (`world_size=2` torchrun fake) over 10 steps |
| E5 | Save shards → kill process → restart → resume from same step yields identical loss |

### 6.3 ML sanity checks (`tests/ml/`, gated `pytest -m ml_sanity`)

#### 6.3.1 Single-task baselines

| # | Check |
|---|---|
| ML1 | Overfit a 64-row subset of rel-f1 driver-top3 (binary): loss → ~0, AUROC → 1.0 within 200 steps |
| ML2 | Overfit a 64-row subset of rel-f1 driver-position (regression): Huber → ~0 after normalization, MAE on raw → small |
| ML3 | Curve match: train rel-f1 driver-top3 for 1k steps with old vs new pipeline — per-step loss curves within 5% relative |
| **ML3.5** | **Performance parity vs `dev-kyaw` baseline** (NEW — see §6.3.5) |

#### 6.3.2 Multi-task balance

| # | Check |
|---|---|
| ML4 | Train on `{driver-top3, driver-position}` jointly on tiny set: both per-task losses decrease monotonically over 200 steps |
| ML5 | Per-row loss magnitudes after target normalization: Huber rows in `[0, 5]`, BCE rows in `[0, 5]` |
| ML6 | Per-datatype head gradient norms within an order of magnitude after warmup (10k steps) |
| ML7 | Multi-task model on driver-top3 ≥ 95% of single-task AUROC after equal training budget on a tiny train subset |

#### 6.3.3 Gradient health (extends existing `test_ml_sanity.py`)

| # | Check |
|---|---|
| ML8 | No dead parameters (zero grad over 10 steps) in the multi-task setting |
| ML9 | No exploding (>1e3) or vanishing (<1e-7) per-layer grad norms over warmup |
| ML10 | Loss is finite for first 100 steps on every task (catch NaN from log of zero, division by zero σ, all-zero column) |

#### 6.3.4 Cross-dataset sanity (P4)

| # | Check |
|---|---|
| ML11 | Model trained on rel-f1 + rel-event evaluated on rel-f1 driver-top3 ≥ 90% of single-task baseline AUROC |
| ML12 | Type embedding for `"rel-f1::drivers"` and `"rel-event::users"` cosine similarity < 0.8 after 1k steps |
| ML13 | Per-task validation curves all show monotone improvement for first N epochs on the 8×A100 box |

#### 6.3.5 Performance parity vs `dev-kyaw` baseline (NEW)

This is the headline check: the refactor must not silently regress quality on rel-f1.

**Setup.** Pick one regression task and one classification task from rel-f1:
- Regression: `driver-position`
- Classification: `driver-top3`

**Protocol.** For each task:
1. Run `dev-kyaw` `main_node_ddp.py` on this task with seeds `{0, 1, 2, 3, 4}`, fixed hyperparams, full train/val/test, until convergence (or fixed epoch budget). Record best test metric per seed.
2. Run **new** pipeline (single-task mode = degenerate multi-task with one task) on same task with the same seeds and hyperparams. Record best test metric per seed.
3. Compute mean ± std across seeds for each pipeline.

**Acceptance.** For each task, the new pipeline's `(mean - std)` must be ≥ `dev-kyaw`'s `(mean - std)` and the means must be within `1 × std` of each other (i.e., 68%-CI overlap). MAE/AUROC sign convention handled appropriately.

**Reporting.** A small markdown table checked into `docs/parity_results.md`, one row per task per pipeline:

| Task | Pipeline | Seeds | Mean | Std | Notes |
|---|---|---|---|---|---|
| rel-f1 / driver-position | dev-kyaw | 5 | … | … | |
| rel-f1 / driver-position | new | 5 | … | … | |
| rel-f1 / driver-top3 | dev-kyaw | 5 | … | … | |
| rel-f1 / driver-top3 | new | 5 | … | … | |

This is the gate to merge **PR1** (preprocessing refactor) and again to merge **PR3** (multi-task plumbing in single-task mode).

#### 6.3.6 Memory / scaling smoke

| # | Check |
|---|---|
| MS1 | Resident set on laptop while training rel-f1 + 5k-seed rel-event (memmap mode) stays below `4 × num_workers × batch_size × K × 8 KB` ceiling — measured via `psutil` |
| MS2 | Throughput on 8×A100: ≥ 1 batch/sec at batch 256, K=300, 6 tasks active |
| MS3 | OOM-free 100-step run on rel-event-only with `--max_rows_per_task 5000` on the laptop |

### 6.4 What we are NOT testing (and why)

- **Encoder internals** (`encoders.py`) — untouched. Existing `test_ml_sanity.py`, `test_zscore.py`, etc. cover.
- **Model internals** (`model.py`, `codebook.py`) — untouched. Existing tests cover.
- **RelBench library correctness** — out of scope; we trust their splits/labels.
- **Determinism under DataLoader shuffling with `num_workers > 0`** — well-known torch quirk; we set generator seeds and accept residual nondeterminism.

---

## 7. Scaling

| Box | Mode | Datasets | Notes |
|---|---|---|---|
| Laptop, pre-PR2 | streaming | rel-f1 only | rel-f1 fits in RAM trivially; rel-event TFs would not |
| Laptop, post-PR2 | streaming + memmap-TF | rel-f1 full + rel-event with `--max_rows_per_task 5000` | OS pages in only the K-hop closure of the active seeds |
| 8×A100, 1TB | precomputed shards + memmap-TF | rel-f1 full + rel-event full | DDP via `torchrun --nproc_per_node 8`, AMP=bf16 |
| Eventual: all of RelBench v2 | precomputed shards + memmap-TF | all | RAM cost no longer scales with #datasets thanks to memmap-TFs |

`--max_rows_per_task N` semantics: caps the seed-row count in the **task split**. Combined with memmap-TFs (PR2), this is sufficient to bound resident memory — no cross-table row scrubbing required, because the kernel pages in only what the BFS-reached rows actually need.

---

## 8. Phased delivery

| PR | Scope | Test gates |
|---|---|---|
| **PR1** | Refactor preprocessing into `gfm_data/` package, no behavior change. CSR adj replaces dict adj. Single dataset, single task. | §6.1.1, §6.1.2 (S1–S17 incl. equivalence vs `dev-kyaw`), §6.1.5(Co5), §6.2(E2), §6.3.1(ML1, ML3, **ML3.5**), §6.3.3(ML8–10) |
| **PR2** | Sharded streaming/precompute split + offline tools. Memmap-backed TFs. | §6.1.3, §6.1.4, §6.2(E1, E3), §6.3.6(MS1, MS3) |
| **PR3** | Multi-task plumbing: `MultiTaskConcat`, `DistributedMultiTaskSampler`, multi-task collate, `MultiTaskHead`, `MultiTaskLoss` (RT formula). Train rel-f1 (3 tasks) jointly. | §6.1.5–§6.1.8, §6.2(E2, E4), §6.3.2(ML4–7), §6.3.3 multi-task extension. Re-run **ML3.5** in single-task mode |
| **PR4** | Add rel-event, full 6-task pretraining run on 8×A100. Tune task weights / loss balance. | §6.3.4(ML11–13), §6.3.6(MS2). Plus a wandb run we eyeball for sanity |

---

## 9. Risks / open questions

- **Node-type namespace collisions** across datasets — solved by prefixing, but watch for hardcoded type strings anywhere downstream (none found in the survey, but PR1 should grep for any).
- **Column-name semantic GloVe** (added in `78d572e`) keys off column names; same-name columns across datasets with different meanings ("id", "name") will be lossy. Acceptable for v0; revisit if metrics suffer.
- **Determinism vs `dev-kyaw`** in S16/S17: our sampler uses CSR-derived sets, dev-kyaw uses Python sets. Iteration order may differ even under the same seed. Sort both outputs canonically before comparing, but document any unavoidable divergence as a deliberate decision.
- **`int16` dtype** for type IDs is fine for ≤32k types (RelBench has ≤500 total) but assumed throughout the pipeline; new shard format keeps this.
- **HuberLoss `delta`** — RT paper does not specify. Default to `delta=1.0` (PyTorch default). Expose as flag.

---

## 10. Future work (post-PR4)

- **In-context learning** (Kumo-RFM-style or RT-style "task table prompting"): sample `(entity', t', label')` historical triples per query, attach context labels as additional tokens. Touches `model.py` (context-vs-query mask). Out of scope this round.
- **Multi-class / multi-label / link prediction** — additional heads + loss dispatch.
- **Forward-looking samplers** for online label generation.
- **PQL** parser front-end.
