# GFM zero-shot execution plan

Status: ready to execute (2026-04-28). Companion to
`docs/gfm_zero_shot_plan.md`, which holds the architectural analysis
(*why* each refactor matters; what the alternatives were). This doc is
the *what to do* — a PR-by-PR breakdown with gates between phases.

## Goal

Pretrain a single RelGT backbone on multiple RelBench-v2 datasets with
per-task supervised heads (the current pipeline), then freeze the
backbone for two downstream consumption paths on unseen schemas:

1. **Embedding → TabPFN** as a post-hoc evaluator. No backbone gradient,
   no fine-tuning. TabPFN consumes `[N, channels]` as features.
2. **Embedding → fresh head**. `Linear(channels, 1)` or 2-layer MLP
   trained on the new dataset's labels with the backbone frozen.
   Optional `--unfreeze_after_epoch K` for warmed unfreezing.

Adoption is validated in two stages:

- **Holdout-task (Phase 4)** — same dataset, hold out one task.
  Validates that the embedding is task-agnostic. Fast laptop loop.
- **Holdout-dataset (Phase 5)** — train on rel-f1, adopt on rel-event
  (and reverse). Validates cross-schema transfer. Real benchmark needs
  AWS p4d; laptop dev validates the launcher with rel-f1 ↔ rel-hm.

## Phase order

```
Phase 1 (arch refactors)
  → Phase 2 (checkpoint)
    → Phase 3 (tooling)
      → Phase 4 (holdout-task)
        → Phase 5 (holdout-dataset)
          → Phase 6 (deferred SSL etc.)
```

Architectural refactors come first so that benchmarks already exercise
the refactored code paths — any in-distribution regression surfaces
before adoption tooling layers on top.

Each phase gates the next via a 1-epoch val-metric comparison against
the pre-refactor baseline (same seed). Tolerance: identical-within-
numerical-noise where the refactor is functionally equivalent (PRs 1.1,
1.2, 1.3); deeper validation where new behavior is introduced.

### Cross-branch parity gate (per PR, hard requirement)

In addition to unit + ML-sanity + intuition checks, **every PR in
Phases 1-2 must pass a parity sweep against the `dev-kyaw` baseline**
on rel-f1 driver-position (regression) and driver-top3 (classification)
before it merges. This is the same protocol used for PR1+PR2 (see
`docs/parity_results.md`):

- 5 seeds × 2 tasks × 5 epochs per pipeline.
- `dev-kyaw` runs are cached at `results/parity/devkyaw/` and reused
  across PRs (the baseline doesn't move).
- The PR branch's runs go to `results/parity/new/`; this directory is
  **wiped before each PR's sweep** so we measure the current code, not
  a prior PR's cached result.
- Aggregate via `scripts/aggregate_parity.py`.
- **Acceptance criterion**: `|μ_new − μ_dev| ≤ max(σ_dev, σ_new)` per
  task. This is the "1× std overlap" rule; matches PR1+PR2 protocol.
- **On failure**: halt the automation, surface the diff, ask for
  human direction. Do not commit a regressing PR.

Tools-only PRs (e.g. PR 1.4 which adds a script and changes no
training code) are exempt from the parity sweep but still gated on
unit tests + a smoke run that exercises the new tooling end-to-end.

## Phase 1 — adoption-ready encoder refactor (~3-4 days, four PRs)

### PR 1.1 — drop `c_idx` (~½ day)

The `c_idx` `[num_nodes]` int64 buffer is parallel to and superseded by
`vq._ema_cluster_size`. It's used only for the popularity-bias term in
`RelGTLayer.global_forward`. The VQ already tracks the same information
correctly (drift-consistent, DDP-synced, Laplace-smoothed).

Changes:

- `model.py`: delete `register_buffer("c_idx", ...)` (line 65-66) and
  the write at line 149.
- Replace lines 133-138 with:
  ```python
  centroid_count = self.vq._ema_cluster_size.clamp(min=1)
  dots = dots + torch.log(centroid_count).view(1, 1, -1)
  ```
- Remove `num_nodes` arg from `RelGTLayer.__init__` (line 26),
  `RelGT.__init__` (line 167); drop `num_nodes_total` computation in
  `train_multi_task.py:591`.
- Keep `node_indices` arg threaded through `global_forward` and
  `RelGT.forward` for now — it's now unused but ripples into
  `collate.py` and `task_tokens.get_global_index`. Remove in a separate
  cleanup sweep later.

Gate:

- 1-epoch laptop pretrain on rel-f1 (5 tasks, same seed). Per-task val
  MAE/AUROC must match the pre-refactor baseline within numerical
  noise. Any drift signals a logic mismatch.

### PR 1.2 — `NeighborTfsEncoder.register_dataset()` method (~1.5 days)

Per-prefixed-type buffer registration currently lives inside `__init__`
(`encoders.py:516-598`), keyed by the training-time `node_type_map`.
For adoption on a new dataset, we need to register additional buffers
(Z-score mean/std per new table; column-name GloVe embeddings for new
columns) after construction.

Changes:

- Pull buffer-registration code from `NeighborTfsEncoder.__init__` into
  a `register_dataset(col_names_dict, col_stats_dict)` method.
  `__init__` now takes only architectural constants.
- The method:
  - Appends to `self._node_type_to_safe`, raising on collision.
  - Registers new `_num_mean_<safe>` / `_num_std_<safe>` buffers per
    new table.
  - Extends `self._col_name_to_idx` and re-registers
    `_col_glove_embeddings` with the concatenated tensor.
  - Updates `_num_zscore_tables` / `_num_col_semantic_cols` counters so
    the runtime guards (`encoders.py:693-709`) reflect post-registration
    state.
- Training entry point: `_build_model` (`train_multi_task.py:516-535`)
  calls `tfs_encoder.register_dataset(col_names_dict, col_stats_dict)`
  once with the train-time union, after construction.

Gate:

- 1-epoch laptop pretrain. Metrics identical to PR 1.1 baseline (this
  refactor is functionally equivalent at training time).

### PR 1.3 — `NeighborNodeTypeEncoder` lazy GloVe for unseen types (~½ day)

Mirror of the unseen-column-name lazy cache in `NeighborTfsEncoder`
(`encoders.py:586-589`). At forward time, if `type_indices` references a
type not in the precomputed `glove_embeddings` buffer, fall back to
on-the-fly GloVe + projection through the existing `self.proj` layer,
cached per-rank.

Changes:

- Add `_type_unseen_cache: Dict[str, Tensor]` to
  `NeighborNodeTypeEncoder`.
- Modify `forward` to accept either integer `type_indices` (existing
  path) or string `type_names`. When type names are passed, resolve via
  buffer + lazy cache.
- Update collate to pass type-name strings alongside or in lieu of
  integer ids (mirroring the `col_names_dict` pattern).

Gate:

- 1-epoch pretrain — lazy path never triggers, metrics identical.
- Unit test: instantiate with `node_type_map={"a": 0, "b": 1}`, call
  with type name `"c"`, assert output shape `[B, K, channels]` and no
  KeyError.

### PR 1.4 — `tools/compute_dataset_stats.py` (~½ day)

Used at adoption time on the held-out dataset to feed
`register_dataset`. No training-time effect.

Changes:

- New script: walks a TF store directory, computes per-column mean/std
  (numerical) and level count (categorical / multicategorical) per
  prefixed-type. Outputs `col_stats_dict` saved as `.pt`.
- Output schema matches `make_pkey_fkey_graph`'s `col_stats` (StatType
  keys, float values).

Gate:

- Regression test: recompute stats on rel-f1, assert match against the
  cached `col_stats` from `make_pkey_fkey_graph` within tolerance.

## Phase 2 — checkpoint plumbing + adoption helpers (~1 day, two PRs)

### PR 2.1 — best-val checkpoint to disk (~½ day)

Replace the in-memory `copy.deepcopy(model.module.state_dict())`
(`train_multi_task.py:826`) with on-disk saves:

- `<run>/best_full.pt` — full `MultiTaskRelGT` state (heads +
  backbone). Used for resume / final test eval.
- `<run>/best_backbone.pt` — `model.module.backbone.state_dict()` only.
  This is the adoption-portable artifact.
- `<run>/backbone_meta.json` — JSON-serializable architectural config:
  `{channels, num_centroids, num_layers, num_heads, K, ablate,
  gnn_pe_dim, attn_dropout, ff_dropout, max_neighbor_hop,
  node_type_map, registered_datasets: [{name, col_names_keys}],
  best_epoch, best_val_macro}`.
- `<run>/backbone_schema.pt` — sidecar with `col_names_dict` +
  `col_stats_dict`. Stored as `.pt` because `StatType` enum keys
  aren't JSON-friendly and the col_stats can contain NaN/inf floats.
  Roundtrip via `torch.save` / `torch.load`.

Free the in-memory copy via `gc.collect()` after each save. Final test
eval reloads from `best_full.pt`; existing `dist.broadcast` pattern
(`train_multi_task.py:837-841`) stays — broadcast happens once after
rank-0 loads.

Gate:

- Full pretrain run completes. Final test metrics with the reload-from-
  disk path match what the in-memory deepcopy produced (golden test
  before/after).

### PR 2.2 — `load_backbone` helpers (~½ day)

- `RelGT.load_backbone(meta_json_path, weights_path, schema_pt_path)` —
  classmethod. Reads meta + schema, constructs
  `RelGT(out_channels=channels)` with architectural constants from
  meta, calls `tfs_encoder.register_dataset(col_names_dict,
  col_stats_dict)` from schema, loads `state_dict`. Returns a ready-to-
  forward backbone.
- `MultiTaskRelGT.load_backbone(...)` — same plumbing without
  reconstructing per-task heads (heads are throwaway at adoption).

Gate:

- Roundtrip test: train 1 epoch, save, load via `load_backbone`, run
  forward on a fresh batch, assert outputs match the live model within
  FP tolerance.

## Phase 3 — adoption tooling (~2 days, three PRs)

### PR 3.1 — `tools/extract_embeddings.py`

Args: `--backbone_meta`, `--backbone_weights`, `--backbone_schema`,
`--dataset`, `--task`, `--split {train,val,test,all}`, `--out_dir`,
`--register_new_dataset` (flag).

If `--register_new_dataset` is set and the requested dataset isn't in
the meta's `registered_datasets`, runs `tools/compute_dataset_stats.py`
on the new dataset and calls `register_dataset` before forward. This is
the cross-schema adoption path; for holdout-task on a seen dataset it
stays off.

Builds TF store + shards via existing tools (idempotent / cached). Runs
`forward(task_id=None)` over the requested splits. Dumps per-split
`.pt`: `{embeddings: [N, channels], labels: [N], global_idx: [N],
split: str, task: str, channels: int}`.

### PR 3.2 — `tools/finetune_head.py`

Args: `--embeddings_dir`, `--head {linear, mlp2}`, `--epochs`, `--lr`,
`--unfreeze_after_epoch K` (optional; needs backbone weights/meta/schema
when unfreezing).

Trains a fresh head on train embeddings, picks best-val epoch on val
embeddings, evaluates via `task.evaluate(...)` from RelBench. Reports
metrics relative to a from-scratch single-task baseline if available.

### PR 3.3 — `tools/tabpfn_eval.py`

Pure post-hoc evaluator. Loads embeddings, fits TabPFN, evaluates via
`task.evaluate(...)`. No backbone gradient anywhere.

Args: `--embeddings_dir`, `--projector {none, pca64}`. Default `none`
(raw `[B, channels]`). If TabPFN errors on >100-dim input, fall back to
`pca64` — PCA-64 fitted on the train embeddings, applied to val/test.

## Phase 4 — holdout-TASK dev loop (~½ day)

`scripts/holdout_task_dev.sh`:

- **rel-f1**: pretrain on `{driver-position, driver-dnf,
  driver-circuit-compete, results-position, qualifying-position}` (5
  tasks). Hold out `driver-top3` (binary → AUROC signal). Then extract
  → finetune-head + TabPFN.
- **rel-hm**: pretrain on `{item-sales, transactions-price}` (2 tasks).
  Hold out `user-churn` (binary). Then extract → finetune-head +
  TabPFN.

Compares against from-scratch single-task baseline (existing
`pretrain_laptop.sh` with single-task input) for each held-out task.

Goal: end-to-end signal that the embedding-extraction path produces
useful representations for held-out tasks on a seen schema. Validates
Phases 1-3 before investing in cross-dataset.

## Phase 5 — holdout-DATASET benchmark (~½ day code + pretrain wallclock)

`scripts/holdout_dataset_eval.sh`:

- Train on rel-f1 (all 6 tasks) → adopt on each rel-event task: extract
  → finetune-head + TabPFN. Mirror direction (train rel-event, adopt
  rel-f1).
- Reports cross-dataset adoption metrics vs from-scratch single-task
  baseline.

Run targets:

- **Laptop dev**: rel-f1 ↔ rel-hm to validate the launcher (rel-event
  won't fit — see `pretrain_laptop.sh` comments about 25 GB peak RSS).
- **Real benchmark**: rel-f1 ↔ rel-event on AWS p4d via existing
  `scripts/pretrain_p4d.sh` infra.

## Phase 6 — deferred (research-grade quality)

- Self-supervised auxiliary losses: masked column prediction, masked
  node prediction, contrastive seed (B1-B4 in `gfm_zero_shot_plan.md`).
- Text-of-level categorical encoding for human-meaningful categoricals
  (current path is hash-bucket only; collisions on novel levels).
- Full 6-vs-1 holdout protocol across all RelBench v2 datasets.
- Learned projector tuning for TabPFN if PCA-64 falls short.

## Decisions locked in

These were debated and resolved during plan finalization; recording
them here so future context can pick up cold:

- `col_stats_dict` saved as `.pt` sidecar, not JSON. Reason: `StatType`
  enum keys + occasional NaN/inf floats aren't cleanly JSON-roundtrip-
  able. Inspection cost is acceptable since the meta JSON has the
  human-readable architectural constants.
- TabPFN is **pure post-hoc only**. No ICL-as-loss during pretraining.
- Per-phase validation gate is **1 full epoch on rel-f1** (laptop
  scale, ~5-10 min per attempt), not a 50-step smoke. Trades iteration
  speed for confidence.
- Phase 1 ships as **four separate PRs** (1.1 through 1.4), not one
  bundled PR. Lets a regression bisect cleanly.
- Held-out task picks: `driver-top3` (rel-f1, binary), `user-churn`
  (rel-hm, binary). Both binary → AUROC is the cleanest signal.
- Architectural refactors **before** holdout-task benchmarks, so the
  benchmarks already exercise the refactored encoder paths.
