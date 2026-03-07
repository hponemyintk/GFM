# Overall Execution Plan

## Summary

7 tasks across 4 phases to transform RelGT into a DDP-safe foundation model with multi-dataset pretraining and zero-shot evaluation.

## Task List

| Task | File | Description |
|------|------|-------------|
| 1 | `task1_table_name_namespacing.md` | Namespace tables, precompute GloVe embeddings |
| 2 | `task2_backbone_head_separation.md` | Add encode(), optional head, MaskedTokenHead |
| 3 | `task3_multi_dataset_data_loading.md` | Union mappings, namespace HeteroData, multi-dataset RelGTTokens |
| 4 | `task4_foundation_training_script.md` | main_foundation.py with interleaved sampling + DDP sync |
| 5 | `task5_masked_token_pretraining.md` | Masking strategy, reconstruction loss |
| 6 | `task6_zero_shot_evaluation.md` | Target-as-masked-token zero-shot prediction |
| 7 | `task7_ddp_global_attention_fixes.md` | Eliminate c_idx, sync VQ EMA, verify SyncBatchNorm |

## Dependency Graph

```
Phase 1 ─── Task 1 (namespacing + GloVe precompute)  ─── encoders.py, utils.py
         └── Task 2 (backbone/head separation)         ─── model.py
                │
Phase 2 ─── Task 3 (multi-dataset loading)  ─────────── new data_loading.py, utils.py
         ├── Task 5 (masked token pretraining) ───────── new masking.py, model.py
         └── Task 7 (DDP global attention fixes) ─────── model.py, codebook.py
                │
Phase 3 ─── Task 4 (foundation training script) ─────── new main_foundation.py
                │
Phase 4 ─── Task 6 (zero-shot evaluation) ───────────── new zero_shot.py
```

## Phase 1: Foundation Building Blocks

**Tasks 1 and 2 in parallel. No dependencies between them.**

### Task 1: Table Name Namespacing + GloVe Precomputation — PARTIALLY DONE
- Files: `encoders.py`, `utils.py`
- **DONE (commit 58cdc8d)**:
  - `NeighborNodeTypeEncoder`: precompute GloVe at init as buffer, forward is buffer index + Linear
  - `SharedEmbeddingEncoder`: discovers EMB_DIM from col_stats_dict, per-dim Linear projectors
  - SentenceTransformer removed as submodule (fixes DDP unused params)
- **REMAINING (data pipeline, not encoder)**:
  - `RelGTTokens`: add `union_node_type_map` and `dataset_name` parameters
  - New: `namespace_hetero_data()` utility function
  - Table name namespacing in col_stats_dict, col_names_dict, node_type_map
- DDP impact: GloVe bottleneck eliminated
- Tests: T1.1–T1.8

### Task 2: Backbone / Head Separation
- Files: `model.py`
- Changes:
  - Add `encode(return_full_sequence=bool)` method
  - Refactor `forward()` to call `encode()`
  - Make `self.head` optional (None when no out_channels)
  - Add `MaskedTokenHead` class
- DDP impact: None directly, but enables Task 5 and Task 7
- Tests: T2.1–T2.7

**Gate: Run regression test R1 (main_node_ddp.py still works)**

---

## Phase 2: Core Components

**Tasks 3, 5, and 7 in parallel. Each depends on Phase 1 but not on each other.**

### Task 3: Multi-Dataset Data Loading
- Files: new `data_loading.py`, modify `utils.py`
- Depends on: Task 1 (namespacing)
- Changes:
  - `build_union_mappings()`: load all datasets, namespace, merge
  - `create_all_relgt_tokens()`: create tokens for all dataset/task/split combos
  - `RelGTTokens`: use union mappings, pass through dataset_name metadata
- DDP impact: Each GPU loads same data independently (fine)
- Tests: T3.1–T3.7

### Task 5: Masked Token Pretraining
- Files: new `masking.py`, modify `model.py`
- Depends on: Task 2 (encode() method)
- Changes:
  - `apply_masking()`: random masking of K-1 neighbor positions
  - `compute_reconstruction_loss()`: type CE + feature MSE
  - Feature target generation (two-pass with detach)
- DDP impact: Masking is per-sample, no DDP issues
- Tests: T5.1–T5.7

### Task 7: DDP Global Attention Fixes
- Files: `model.py`, verify `codebook.py`
- Depends on: Task 2 (both modify model.py)
- Changes:
  - Eliminate `c_idx` buffer entirely
  - Replace log-count bias with `_ema_cluster_size`
  - Remove `num_nodes` parameter from RelGTLayer and RelGT
  - Add `sync_vq_buffers()` function (used by Task 4)
  - Verify SyncBatchNorm covers VQ's self.bn
- DDP impact: This IS the DDP fix — makes global attention correct under multi-GPU
- Tests: T7.1–T7.8

**Gate: Run regression tests R1, R2, R3**

---

## Phase 3: Training Script

**Sequential — depends on all of Phase 2.**

### Task 4: Foundation Training Script
- Files: new `main_foundation.py`
- Depends on: Tasks 1, 2, 3, 5, 7
- Changes:
  - CLI with --held_out_dataset, --sampling_alpha, --mask_ratio
  - Load all datasets via Task 3
  - InterleavedIterator with temperature-scaled sampling
  - Training loop calling apply_masking + encode + reconstruction loss
  - sync_vq_buffers() after each step
  - SyncBatchNorm verification
  - Checkpoint save/load
  - Wandb logging with dataset provenance
- DDP specifics:
  - DistributedSampler per loader, set_epoch each epoch
  - Rank-dependent random seed in InterleavedIterator
  - sync_vq_buffers() is collective (all ranks must call)
  - dist.barrier() before evaluation
  - Only rank 0 logs and saves
- Tests: T4.1–T4.10

**Gate: Multi-GPU training runs for 1 epoch without crash or divergence**

---

## Phase 4: Evaluation

**Depends on Phase 3.**

### Task 6: Zero-Shot Evaluation
- Files: new `zero_shot.py`, integrate with `main_foundation.py`
- Depends on: Tasks 2, 3, 4, 5
- Changes:
  - `mask_target_column()`: targeted masking of the task's target feature
  - `zero_shot_evaluate()`: forward with masked target, reconstruct prediction
  - Map reconstructed features to task label space
  - Integration into main_foundation.py evaluation loop
- DDP specifics:
  - Evaluation in model.eval() mode — no VQ updates, no sync needed
  - Gather predictions across GPUs (same pattern as current test() function)
- Tests: T6.1–T6.6

**Gate: Zero-shot predictions are finite and better than random**

---

## Files Changed / Created Summary

| File | Status | Modified By Tasks |
|------|--------|-------------------|
| `encoders.py` | Modified | Task 1 |
| `model.py` | Modified | Tasks 2, 7 |
| `utils.py` | Modified | Tasks 1, 3 |
| `codebook.py` | Unchanged | (verified by Task 7) |
| `local_module.py` | Unchanged | — |
| `main_node_ddp.py` | Unchanged | (backward compat verified) |
| `data_loading.py` | **New** | Task 3 |
| `masking.py` | **New** | Task 5 |
| `main_foundation.py` | **New** | Task 4 |
| `zero_shot.py` | **New** | Task 6 |

## Estimated Parallelism

```
Timeline:
─────────────────────────────────────────────────────────────────
Phase 1:  [Task 1] ████████          [Task 2] ████████
Phase 2:  [Task 3] ████████  [Task 5] ████████  [Task 7] ██████
Phase 3:  [Task 4] ████████████████
Phase 4:  [Task 6] ████████████
Testing:  ─── T ──── T ──── T ──────── T ──────── T ────────── T
─────────────────────────────────────────────────────────────────
          (T = testing gate between phases)
```

Tasks 1+2 can be assigned to different agents simultaneously.
Tasks 3+5+7 can be assigned to three different agents simultaneously.
Tasks 4 and 6 are sequential.

## Critical Path

Task 1 → Task 3 → Task 4 → Task 6

This is the longest dependency chain. Task 7 (DDP fixes) must also complete before Task 4 but is smaller in scope.
