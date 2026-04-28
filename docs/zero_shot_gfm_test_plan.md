# GFM zero-shot test plan

Companion to `docs/zero_shot_gfm_execution_plan.md`. Each phase's
verification falls into four categories:

- **Unit tests** — narrow correctness checks. Sub-second, deterministic,
  no GPU. Live in `tests/` as `test_*.py`. Run with `pytest tests/`.
- **ML sanity** — broader integration tests against real data: shapes,
  distributions, gradient flow, loss curves, before/after metric
  parity. Seconds-to-minutes, GPU required. Gated behind a
  `@pytest.mark.sanity` marker; run manually before merging each PR.
- **dev-kyaw parity sweep** (training-touching PRs only) — 5 seeds × 2
  tasks × 5 epochs against the `dev-kyaw` reference branch. **Hard
  gate per PR**: `|μ_new − μ_dev| ≤ max(σ_dev, σ_new)` on both
  rel-f1 driver-position (MAE↓) and driver-top3 (AUROC↑). On failure,
  automation halts and asks for human direction. Tools-only PRs are
  exempt (no training code touched).
- **Intuition checks** — exploratory experiments validating the
  *meaning* of the change: does the embedding capture structure? does
  cluster popularity look heavy-tailed? does cross-task transfer
  actually work? Researcher-owned, not CI. Recorded as scripts or
  notebooks; results pasted into the PR description.

The phase gate (1-epoch val-metric parity vs pre-refactor baseline) is
the **ML sanity layer** for each PR. Unit tests catch logic errors
fast; ML sanity catches silent regressions; the parity sweep catches
glaring metric drift against a known-good reference; intuition checks
catch design flaws that pass all three.

---

## Phase 1 — adoption-ready encoder refactor

### PR 1.1 — drop `c_idx`

#### Unit tests (`tests/test_drop_c_idx.py`)

- `test_global_forward_uses_ema_cluster_size`: instantiate `RelGTLayer`
  with `conv_type="global"`. Manually set
  `layer.vq._ema_cluster_size` to a known unbalanced distribution
  (e.g., `[10.0, 1.0, 1.0, ...]`). Run a forward pass with random
  inputs. Assert the bias term added to dot-products equals
  `log(_ema_cluster_size.clamp(min=1))` (probe via a hook on the
  attention pre-softmax tensor).
- `test_no_c_idx_attribute_after_refactor`: instantiate `RelGT`,
  recursively walk modules, assert no `c_idx` buffer exists on any
  submodule.
- `test_relgt_init_signature_drops_num_nodes`: introspect
  `inspect.signature(RelGT.__init__)`, assert `num_nodes` is not in
  parameters. Same for `RelGTLayer.__init__`.
- `test_train_multi_task_drops_num_nodes_total`: parse
  `_build_model` (`train_multi_task.py`), assert no reference to
  `num_nodes_total` in its body.

#### ML sanity (`tests/test_pr11_sanity.py`, marked `sanity`)

- `test_train_loss_decreases_50_steps`: 50-step laptop-scale pretrain
  on rel-f1 (5 tasks, batch=32). Assert epoch-mean train_loss at step
  50 is strictly less than at step 5.
- `test_val_metric_parity_with_baseline`: 1-epoch laptop pretrain on
  rel-f1, same seed as a pre-refactor baseline run captured before the
  PR lands. Assert per-task val MAE/AUROC matches baseline within
  numerical noise (±1% relative for AUROC, ±2% for MAE/MSE).
- `test_ddp_centroid_count_synced`: spawn 2 ranks with `torchrun`, run
  10 training steps, assert `vq._ema_cluster_size` is bit-for-bit
  identical across ranks at every step (extends the existing
  `test_centroid_sync.py`).

#### Intuition

- **Cluster popularity distribution**: at the end of a 1-epoch
  pretrain, log the histogram of `vq._ema_cluster_size`. Expectation:
  heavy-tailed (a handful of high-mass centroids, long tail of small
  ones). If uniform, the popularity-bias term is computationally
  expensive but informationally inert — flag for follow-up.
- **Centroid drift inspection**: log centroid embeddings at step 0,
  step T/2, step T. Compute pairwise cosine sim of corresponding
  centroid slots across snapshots. If two slots' embeddings cross over
  each other (high cross-slot similarity, low same-slot similarity),
  drift is real and `c_idx`'s stale-label bug was actually mattering.

### PR 1.2 — `NeighborTfsEncoder.register_dataset()` method

#### Unit tests (`tests/test_register_dataset.py`)

- `test_register_dataset_extends_buffers`: construct encoder with
  empty schema. Call `register_dataset(col_names_A, col_stats_A)` then
  `register_dataset(col_names_B, col_stats_B)` (disjoint table names).
  Assert: new `_num_mean_<safe_a>`, `_num_mean_<safe_b>` buffers exist
  with correct shapes; `_col_glove_embeddings.shape[0]` equals
  total unique columns; `_col_name_to_idx` length matches.
- `test_register_dataset_safe_name_collision_raises`: register table
  `"rel-f1::drivers"`, then attempt `"rel-f1__drivers"` (sanitizes to
  same `rel_f1__drivers`). Assert `ValueError` with the colliding
  names mentioned.
- `test_register_dataset_handles_torch_frame_stype_keys`: pass a
  `col_names_dict` with `torch_frame.numerical` and
  `torch_frame.categorical` keys, run register, then forward over a
  matching `TensorFrame`. Assert no KeyError on stype lookup.
- `test_init_alone_does_not_register`: construct encoder, do NOT call
  `register_dataset`, attempt `forward`. Assert `RuntimeError` from
  the runtime guards (`encoders.py:693-709`).
- `test_runtime_guards_pass_after_register`: construct, register,
  forward — guards must not fire.
- `test_zscore_buffer_values_correct`: register a synthetic table
  with known means `[1.0, 2.0]` and stds `[3.0, 4.0]`. Assert the
  registered `_num_mean_<safe>` and `_num_std_<safe>` buffers contain
  exactly these values.

#### ML sanity (`tests/test_pr12_sanity.py`)

- `test_pretrain_metrics_match_pr11_baseline`: 1-epoch pretrain, same
  seed as PR 1.1 baseline. Assert val metrics IDENTICAL within
  numerical noise. This refactor moves code without changing
  computation — anything beyond noise signals a bug.
- `test_z_score_applied_correctly_in_forward`: register with known
  stats, run forward on a batch with known numerical values, capture
  the post-normalize tensor via a hook. Assert
  `(value - mean) / (std + 1e-8)` matches.

#### Intuition

- **Lazy buffer extension non-interference**: register dataset A, run
  forward on an A batch, capture output `O_A`. Then register dataset
  B (disjoint tables). Run forward on the SAME A batch again, capture
  `O_A2`. Assert `O_A == O_A2` within FP tolerance — registering B
  must not change A's processing path. (This is implicit in the
  buffer-naming scheme but worth a runtime assertion since adoption
  depends on it.)

### PR 1.3 — `NeighborNodeTypeEncoder` lazy GloVe for unseen types

#### Unit tests (`tests/test_lazy_type_glove.py`)

- `test_unseen_type_no_keyerror`: instantiate with
  `node_type_map={"a": 0, "b": 1}`. Call `forward` with a batch
  including the type-name string `"c"`. Assert output shape
  `[B, K, channels]` and no KeyError.
- `test_lazy_cache_hits`: call forward with `"c"` twice. Assert
  `_type_unseen_cache` size is 1 after both calls (cache hit on the
  second).
- `test_lazy_output_matches_manual`: compute `proj(glove("c"))`
  manually using the encoder's `glove_embedder` + `proj` layer.
  Assert lazy-path output for `"c"` matches.
- `test_known_type_does_not_populate_cache`: call forward with a
  known type `"a"`. Assert `_type_unseen_cache` remains empty.
- `test_int_indices_path_unchanged`: call forward with integer
  `type_indices` (existing path). Assert output unchanged from before
  the refactor (golden test).

#### ML sanity (`tests/test_pr13_sanity.py`)

- `test_pretrain_metrics_match_pr12_baseline`: 1-epoch pretrain,
  metrics identical to PR 1.2 baseline (lazy path never triggers).
- `test_lazy_path_works_in_ddp`: 2-rank training with a synthetic
  unseen type forced into a few batches. Assert no rank divergence,
  no NCCL timeout.

#### Intuition

- **DDP cache consistency**: with 2 ranks both encountering unseen
  type `"c"`, assert `_type_unseen_cache["c"]` matches across ranks
  (GloVe is deterministic, so this should hold trivially — but worth
  a check before adoption depends on it).

### PR 1.4 — `tools/compute_dataset_stats.py`

#### Unit tests (`tests/test_compute_dataset_stats.py`)

- `test_stats_match_make_pkey_fkey_graph`: run on cached rel-f1 TF
  store. Compare per-column mean/std/level-count against the cached
  `col_stats` from `make_pkey_fkey_graph`. Tolerance: 1e-4 relative
  for floats, exact match for level counts.
- `test_handles_all_nan_column`: synthetic TF store with one
  numerical column entirely NaN. Assert mean=0.0, std=1.0 (or
  whatever the convention is), no NaN in output.
- `test_categorical_level_count_correct`: synthetic categorical
  column with 5 unique values. Assert
  `output["categorical"][col][StatType.COUNT] == 5`.
- `test_output_format_register_dataset_compatible`: compute stats,
  immediately call `tfs_encoder.register_dataset(...)` with the
  output. Assert no KeyError or type error.
- `test_multicategorical_handled`: synthetic multicategorical column
  (variable-length lists). Assert level count and stats reflect
  flattened element distribution.

#### ML sanity (`tests/test_pr14_sanity.py`)

- `test_compute_then_register_then_forward_endtoend`: build TF store
  on rel-hm (small subset), compute stats, register on a fresh
  backbone, run forward over a batch. Assert no errors and output
  shape `[B, channels]`.

#### Intuition

- **Sanity-check distributions**: plot per-column histograms of
  computed mean/std for rel-f1 vs rel-hm vs rel-event. Sanity:
  numerical columns should have comparable scales after z-score; if
  rel-event has columns with std ~1e6 while rel-f1 is ~1, z-score
  dynamic range is fine but the categorical level count distribution
  may suggest rel-event needs hash-bucket scaling.

---

## Phase 2 — checkpoint plumbing + adoption helpers

### PR 2.1 — best-val checkpoint to disk

#### Unit tests (`tests/test_checkpoint_to_disk.py`)

- `test_save_load_full_state_roundtrip`: train 1 step, save
  `best_full.pt`, load into a fresh model, assert state_dict
  bit-equal.
- `test_meta_json_schema_complete`: save, parse JSON, assert all
  required keys present (channels, num_centroids, num_layers,
  num_heads, K, ablate, gnn_pe_dim, attn_dropout, ff_dropout,
  max_neighbor_hop, node_type_map, registered_datasets, best_epoch,
  best_val_macro). Assert all values JSON-roundtrip.
- `test_schema_pt_roundtrip_with_stype_keys`: save a `col_stats_dict`
  with `StatType.MEAN`, `StatType.STD`, `StatType.COUNT` keys and
  occasional NaN values. Load. Assert keys match (enum equality), NaN
  preserved.
- `test_best_val_writes_all_four_files`: invoke the best-val save
  block once. Assert all four files (`best_full.pt`,
  `best_backbone.pt`, `backbone_meta.json`, `backbone_schema.pt`)
  exist at expected paths with non-zero size.
- `test_in_memory_state_freed_after_save`: monitor process RSS
  before/after save. Assert RSS does not grow by ~`sizeof(state_dict)`
  (i.e., the save is followed by a `gc.collect()` that frees the
  in-memory copy).
- `test_overwrite_on_new_best`: trigger best-val save twice with
  different state. Assert the second save overwrites the first
  (single canonical `best_*` file, not numbered).

#### ML sanity (`tests/test_pr21_sanity.py`)

- `test_full_pretrain_disk_path_matches_in_memory`: golden test —
  run full pretrain twice on rel-f1 (5 tasks, 3 epochs), once with
  the legacy in-memory deepcopy path, once with the new disk-save
  path. Same seed, same DDP world size. Assert final test metrics
  match exactly per task.
- `test_resume_from_disk_checkpoint`: train 2 epochs, save, kill
  process, resume from `best_full.pt`, train 1 more epoch. Assert no
  errors and final test metrics within FP tolerance of an
  uninterrupted 3-epoch run.

#### Intuition

- **Memory profile delta**: log peak RSS over a full pretrain run
  with the new disk-save path. Assert peak ≤ pre-refactor peak by at
  least 1× model state-dict size (since we're freeing the in-memory
  deepcopy).

### PR 2.2 — `load_backbone` helpers

#### Unit tests (`tests/test_load_backbone.py`)

- `test_load_backbone_constructs_correct_arch`: save backbone, load
  via `RelGT.load_backbone(...)`. Assert architectural constants
  (channels, num_centroids, etc.) on the loaded model match the
  saved meta JSON.
- `test_load_backbone_calls_register_dataset`: load. Assert the
  loaded `tfs_encoder` has `_node_type_to_safe`, `_col_name_to_idx`
  populated to match the saved schema.
- `test_load_state_dict_strict_match`: save weights, load via
  `load_backbone`. Assert all parameters and buffers equal within FP
  tolerance (1e-7 absolute for fp32).
- `test_load_multitask_skips_heads`: save `MultiTaskRelGT`. Load via
  `MultiTaskRelGT.load_backbone(...)`. Assert no `head.task_heads`
  weights loaded (or assert heads are present but freshly
  initialized — confirm intended behavior in PR description).
- `test_load_backbone_raises_on_arch_mismatch`: save with
  channels=64. Attempt to load into a `RelGT` constructed with
  channels=128. Assert clear error message about mismatch.

#### ML sanity (`tests/test_pr22_sanity.py`)

- `test_loaded_backbone_forward_matches_live`: train 1 epoch, save,
  load via `load_backbone`. Run forward on the same fixed batch
  through both the live model and the loaded one. Assert outputs
  match within 1e-5 absolute.
- `test_loaded_backbone_eval_mode`: assert loaded model has
  `training=False` by default (adoption protocol expects frozen).

#### Intuition

- **Schema-portable check**: save backbone trained on rel-f1, load.
  Inspect `tfs_encoder._node_type_to_safe` to confirm it lists
  `"rel-f1::drivers"` etc. — this is the receipt that adoption code
  will call `register_dataset` for new prefixed types on top of.

---

## Phase 3 — adoption tooling

### PR 3.1 — `tools/extract_embeddings.py`

#### Unit tests (`tests/test_extract_embeddings.py`)

- `test_output_shape_and_keys`: small synthetic dataset+task. Run
  extract. Assert output `.pt` has keys
  `{embeddings, labels, global_idx, split, task, channels}`. Shape
  `[N, channels]`. `len(labels) == len(global_idx) == N`.
- `test_register_new_dataset_path`: run with
  `--register_new_dataset` on a dataset not in saved meta. Assert
  `compute_dataset_stats` was called and the backbone forward
  succeeds without runtime guards firing.
- `test_extract_idempotent`: run twice with same seed. Assert
  outputs bit-equal.
- `test_no_backbone_grad`: run extract. Assert that `requires_grad=False`
  on all backbone parameters during the forward calls (probe via a
  hook).
- `test_split_filtering`: run with `--split val`. Assert the dumped
  global_idx is a subset of the val seeds for that task.
- `test_label_alignment`: assert that `labels[i]` corresponds to
  `global_idx[i]` (cross-check against task's raw labels).

#### ML sanity (`tests/test_pr31_sanity.py`)

- `test_extracted_embeddings_distinct`: extract on 100 seeds.
  Compute pairwise cosine similarity. Assert NOT all identical
  (median sim < 0.95 — embeddings are non-degenerate).
- `test_embeddings_deterministic_across_runs`: same seed, two runs.
  Assert bit-equal.

#### Intuition

- **Embedding cluster structure**: extract on rel-f1
  driver-position. Color points by driver-id. Run UMAP/t-SNE.
  Visually inspect: drivers with similar career trajectories should
  cluster (champions vs back-markers). Save the plot to
  `results/intuition/extract_clustering_relf1.png`.
- **Linear probe sanity**: fit a `LogisticRegression` on extracted
  embeddings against the held-out task's labels. Assert AUROC > 0.5
  (better than random) before bothering to run TabPFN or finetune-
  head. If this fails, the embedding has no useful signal at all and
  Phase 4 will not produce results.

### PR 3.2 — `tools/finetune_head.py`

#### Unit tests (`tests/test_finetune_head.py`)

- `test_synthetic_linear_separable_converges`: synthetic embeddings
  with a clear linear relationship to binary labels. 10 epochs.
  Assert val AUROC > 0.9.
- `test_unfreeze_after_epoch_unfreezes_backbone`: pass
  `--unfreeze_after_epoch=2`. After epoch 2, assert at least one
  backbone parameter has non-None `.grad`.
- `test_best_val_picks_early_stop_point`: train with intentional
  overfitting (tiny train set, large head). Assert
  `best_val_epoch < final_epoch`.
- `test_head_choices_construct_correctly`: pass `--head=linear` vs
  `--head=mlp2`. Assert head module class differs accordingly.

#### ML sanity (`tests/test_pr32_sanity.py`)

- `test_finetune_on_holdout_task_beats_random`: pre-extracted
  embeddings on rel-f1 driver-top3 (held out from a small
  pretrain). Finetune linear head. Assert AUROC > 0.55.
- `test_finetune_with_unfreeze_no_nan`: enable unfreeze after epoch 2.
  Assert no NaN in gradients or losses across 5 epochs.

#### Intuition

- **Frozen vs unfrozen-after-warm**: side-by-side AUROC. Hypothesis:
  unfreeze catches up to or exceeds frozen by epoch ~10. Confirms
  backbone has useful structure that can be specialized further.
- **Head architecture comparison**: linear vs 2-layer MLP head.
  Hypothesis: MLP wins by a small margin (1-3 AUROC pts) on tasks
  with non-linear label structure. If MLP is dramatically better,
  the backbone embedding is under-utilized — flag for backbone
  improvement.

### PR 3.3 — `tools/tabpfn_eval.py`

#### Unit tests (`tests/test_tabpfn_eval.py`)

- `test_tabpfn_synthetic`: synthetic 64-d embeddings, 1k binary
  labels. Run TabPFN. Assert non-NaN predictions of expected shape.
- `test_pca64_fit_on_train_only`: synthetic train + val + test
  embeddings. Assert the PCA transformer's `fit` was called only on
  train (probe via a mock or check fit-state).
- `test_no_backbone_loaded_in_tabpfn_path`: run tabpfn_eval. Assert
  no `RelGT` instance is constructed in the script's namespace
  (TabPFN consumes `.pt` files only).
- `test_pca64_fallback_triggered_on_dim_overflow`: pass embeddings
  with channels > TabPFN's input cap. Assert PCA-64 fallback path
  was taken (logged or returned in output meta).

#### ML sanity (`tests/test_pr33_sanity.py`)

- `test_tabpfn_beats_random_on_holdout`: pre-extracted rel-f1
  driver-top3 embeddings. Run TabPFN. Assert AUROC > 0.55.
- `test_tabpfn_consistent_across_seeds`: TabPFN is deterministic;
  assert two runs match.

#### Intuition

- **Raw vs PCA-64**: side-by-side AUROC. Hypothesis: raw 512-d
  works if TabPFN accepts it; PCA-64 loses 1-3 AUROC pts but is
  required if TabPFN errors on raw. Document the actual TabPFN
  behavior in the PR (does it error, warn, or silently underperform
  on >100-d?).
- **TabPFN vs finetune-head**: same embeddings, both downstream
  paths. Hypothesis: TabPFN competitive at small train sizes
  (~1k-10k), finetune-head wins on larger. Concrete data points:
  rel-f1 driver-top3 (small, ~5k train) and rel-hm item-sales (large,
  ~M train). Use as a guideline for when to recommend each at
  adoption time.

---

## Phase 4 — holdout-TASK dev loop

### Unit tests (`tests/test_holdout_task_launcher.py`)

- `test_holdout_split_excludes_correctly`: parse the
  `holdout_task_dev.sh` TASKS list. Assert `driver-top3` is not in
  the pretrain set; assert it's the held-out task fed to extract.

### ML sanity

- `test_holdout_task_pipeline_endtoend`: 1-epoch pretrain on rel-f1
  (5 tasks). Extract on driver-top3. Finetune linear head. Assert
  AUROC > 0.55.
- `test_holdout_task_pipeline_relhm`: same on rel-hm
  (pretrain 2 tasks, holdout user-churn). Assert AUROC > 0.55.

### Intuition (the central claim — record results in PR description)

- **Pretraining helps the held-out task**: compare AUROC on
  driver-top3 from:
  - **(a) pretrain-on-5-tasks → embedding → finetune linear head**.
  - **(b) pretrain-on-driver-top3-only → embedding → finetune linear head**
    (sanity oracle — backbone trained directly on the held-out task).
  - **(c) from-scratch single-task** (existing baseline).

  Hypothesis: (a) > (c) (multi-task pretraining helps via shared
  structure). (a) ≈ (b) (multi-task isn't significantly worse than
  single-task on the same task). If (a) << (c), the per-task heads
  absorbed too much task-specific information; flag for backbone
  regularization or smaller heads.
- **Embedding 2D projection**: UMAP of held-out task embeddings,
  colored by label. Compare pretrained-backbone vs random-init-
  backbone. Pretrained should show partial class separability;
  random shouldn't.
- **TabPFN-vs-finetune comparison**: log AUROC for both downstream
  paths. Use to inform Phase 5 protocol (which path to default to
  at adoption).

---

## Phase 5 — holdout-DATASET benchmark

### Unit tests (`tests/test_holdout_dataset_launcher.py`)

- `test_register_new_dataset_at_extraction`: run extract with
  `--register_new_dataset` on rel-hm using a backbone trained on
  rel-f1. Assert `register_dataset` was invoked for rel-hm prefixed
  types and forward succeeds.
- `test_launcher_args_specify_different_datasets`: parse
  `holdout_dataset_eval.sh`. Assert pretrain_dataset !=
  adopt_dataset.

### ML sanity

- `test_cross_dataset_extract_no_crash`: train rel-f1 (small
  config), extract on rel-hm (subset). Assert no errors and output
  shape correct.
- `test_cross_dataset_finetune_beats_random`: train rel-f1, extract
  on rel-hm.user-churn, finetune head. Assert AUROC > 0.5.

### Intuition (the central cross-schema claim)

- **Cross-schema transfer signal**: compare AUROC on rel-hm.user-churn
  from:
  - **(a) train rel-f1 → extract → finetune head** (cross-dataset).
  - **(b) train rel-hm-all-tasks-but-user-churn → extract → finetune head**
    (Phase 4 holdout-task — same-schema upper bound).
  - **(c) from-scratch single-task** (baseline).

  Hypothesis: (a) > (c) means cross-dataset transfer works. (a) close
  to (b) means cross-schema is as good as cross-task. (a) << (c)
  means schema-specific structure is load-bearing — motivates Phase 6
  (text-of-level cats, SSL).
- **Embedding distribution shift**: compute centroid of rel-f1 train
  extraction-embeddings vs rel-hm extraction-embeddings. Distance >>
  expected → backbone produces OOD embeddings on the new schema.
  Action: inspect whether z-score normalization with the new
  dataset's stats brings them closer. (Confirms register_dataset is
  doing useful work.)
- **Codebook utilization at adoption**: log centroid attention
  weights at forward time on rel-hm. Compare to rel-f1's. If only a
  handful of centroids fire on rel-hm (mode collapse), the
  popularity bias may be dominating. If utilization is similar,
  transfer is working at the global-attention level.
- **Bidirectional sanity**: rel-f1 → rel-hm AND rel-hm → rel-f1.
  Strong asymmetry suggests one direction is harder than the other
  due to schema complexity / data scale (informative but not a
  blocker).

---

## Phase 6 — deferred (test categories only)

When SSL losses (B1-B4) and text-of-level categoricals land, expect a
similar three-tier test plan per change:

- **Unit**: synthetic data, assert forward shapes / loss signs / no
  NaN.
- **ML sanity**: 100-step synthetic train, assert SSL loss decreases.
  Mixed-loss schedule sanity: per-objective contributions are
  tracked and balanced.
- **Intuition**: does SSL pretraining (without supervised) on
  rel-f1 produce embeddings that already linearly-probe for
  driver-top3? If yes, SSL is doing real work.

---

## dev-kyaw parity sweep (per training-touching PR)

The reference baseline is `dev-kyaw` at the worktree
`/home/jedi/research_repos/GFM/.claude/worktrees/dev-kyaw`. The sweep
is launched via `scripts/parity_sweep.sh` and aggregated by
`scripts/aggregate_parity.py`. Both sides write to
`results/parity/{devkyaw,new}/rel-f1/{task}/{seed}.json`.

### Per-PR flow

```
# 1. wipe stale "new" results so the sweep measures THIS PR's code
rm -rf results/parity/new/rel-f1

# 2. run the sweep (devkyaw is cached after first PR; only "new"
#    re-runs)
bash scripts/parity_sweep.sh 5 5

# 3. aggregate + compare
python3 scripts/aggregate_parity.py results/parity
```

### Pass / fail criterion

For each task `t` in `{driver-position, driver-top3}`:

- Compute `μ_dev, σ_dev` from the 5 dev-kyaw seeds.
- Compute `μ_new, σ_new` from the 5 PR-branch seeds.
- **PASS** iff `|μ_new − μ_dev| ≤ max(σ_dev, σ_new)`.
- **FAIL** halts the automation and reports per-seed numbers.

### Tracking

After each PR's sweep, append a row to `docs/parity_results.md` with
the PR id, both means, both stds, both gaps, both thresholds, and the
verdict. The doc becomes the audit log of every refactor's metric
impact.

### Exemptions

A PR is exempt from the parity sweep iff it touches **none** of:

- `model.py`, `encoders.py`, `codebook.py`, `local_module.py`
- `gfm_data/` (sampler, collate, task_tokens, graph_cache, tf_store,
  multi_task_dataset, stypes)
- `heads/multi_task_head.py`, `losses/multi_task_loss.py`
- `train_multi_task.py`, `main_node_ddp.py`

PR 1.4 (compute_dataset_stats tooling-only) is the only Phase-1 PR
expected to qualify; all others must run the sweep.

---

## Existing tests we depend on (do not duplicate)

- `tests/test_centroid_sync.py` — VQ-EMA DDP sync. PR 1.1's DDP
  centroid-count test extends this rather than replacing.
- `tests/test_emb_dims_discovery.py`, `test_embedding_encoder.py`,
  `test_type_encoder.py`, `test_column_semantic_embedding.py` —
  encoder shape contracts. PRs 1.2 / 1.3 must not break these; if
  they need updating, that's a signal of an API change to call out
  in the PR description.
- `tests/test_collate_single_task.py` — collate shape invariants.
  PR 1.3 (lazy GloVe via type-name strings in batch dict) likely
  needs an update here.
- `tests/test_multi_task_dataset.py` — task token construction.
  Phase 1 changes don't touch this; Phase 2 checkpoint plumbing
  might if it changes how task_tokens get persisted.
- `tests/test_ml_sanity.py` — existing baseline ML sanity. Run
  before each Phase 1 PR merges to confirm no regression beyond the
  per-PR expected delta.

## Running the tests

- **Unit tests** (CI-fast): `pytest tests/ -m "not sanity"`. Target
  total wall: < 1 min.
- **ML sanity** (manual gate per PR): `pytest tests/ -m sanity`.
  Target wall: 5-15 min on the laptop GPU. Required-passing before
  merge.
- **Intuition checks** (researcher-owned): scripts under
  `tools/intuition_*.py` or notebooks in `expts/intuition/`. Results
  pasted into the relevant PR description; not required-passing but
  the PR shouldn't merge if intuition contradicts the design intent.
