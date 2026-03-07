# Testing Plan

## Implemented Tests (95 tests, all passing)

Tests for current encoder changes are implemented in `tests/`:
- `test_type_encoder.py` (15): GloVe precompute, buffer/param checks, index validation
- `test_embedding_encoder.py` (18): dim-aware projectors, 2D reshape fix, ambiguity detection
- `test_zscore.py` (26): buffer registration, Z-score math, NaN imputation, serialization guard, save/load roundtrip
- `test_emb_dims_discovery.py` (9): emb_dims plumbing from col_stats_dict
- `test_ml_sanity.py` (27): representation quality, gradient health, overfit, Z-score effectiveness

Run: `source relgt_env/bin/activate && python -m pytest tests/ -v`

## Future Task Tests (not yet implemented)

All tests below are organized by phase matching the task dependency graph. Each phase's tests must pass before proceeding to the next phase.

---

## Phase 1: Independent Changes (Tasks 1 & 2)

### Task 1: Table Name Namespacing + GloVe Precomputation

**Unit Tests (no GPU required):**

```
T1.1 — _name_to_glove_text conversion
    Input:  "rel-f1__drivers"    -> Expected: "rel f1 drivers"
    Input:  "rel-hm__articles"   -> Expected: "rel hm articles"
    Input:  "drivers"            -> Expected: "drivers"
    Input:  "mask"               -> Expected: "mask"
    Input:  "rel-trial__study"   -> Expected: "rel trial  study"  (double space ok, stripped)
    Input:  ""                   -> Expected: ""

T1.2 — GloVe precomputation at init
    Create NeighborNodeTypeEncoder with node_type_map={"rel-f1__drivers": 0, "rel-hm__users": 1}
    Assert: model.glove_embeddings has shape [3, 300]  (2 types + 1 mask)
    Assert: model.glove_embeddings is a buffer (not parameter)
    Assert: no self.embedder attribute exists on the model
    Assert: no self.st_model attribute exists on the model

T1.3 — Forward produces correct shapes
    Create encoder with 3 table types, embedding_dim=64
    Input: type_indices of shape [4, 100] with values in {0, 1, 2}
    Assert: output shape is [4, 100, 64]

T1.4 — Different namespaced names produce different embeddings
    Embed "rel-f1__drivers" vs "rel-hm__drivers" vs "drivers"
    Assert: all three GloVe vectors are different (cosine similarity < 1.0)

T1.5 — namespace_hetero_data utility
    Create a small HeteroData with node_types ["users", "items"]
    Apply namespace_hetero_data(data, "rel-hm")
    Assert: data.node_types == ["rel-hm__users", "rel-hm__items"]
    Assert: edge types are updated accordingly
    Assert: .tf, .time, .num_nodes attributes transferred correctly

T1.6 — Z-score buffer names don't collide
    Create NeighborTfsEncoder with col_names_dict containing:
        "rel-f1__drivers": {numerical: ["speed"]},
        "rel-hm__drivers": {numerical: ["age"]}
    Assert: both buffers registered with distinct names
    Assert: _num_mean_rel_f1__drivers != _num_mean_rel_hm__drivers

T1.7 — Backward compatibility
    Create NeighborNodeTypeEncoder with un-namespaced node_type_map={"drivers": 0}
    Assert: forward() works without errors
    Assert: output shape correct
```

**Integration Test (single GPU):**

```
T1.8 — Single-dataset training still works
    Run main_node_ddp.py with original un-namespaced data for 2 steps
    Assert: no errors, loss decreases or is finite
    (This is the backward compatibility smoke test)
```

---

### Task 2: Backbone / Head Separation

**Unit Tests:**

```
T2.1 — encode() returns correct shape
    Create RelGT with channels=64, K=100
    Call encode(..., return_full_sequence=True)
    Assert: output shape [B, K, 64]
    Call encode(..., return_full_sequence=False) or encode() default
    Assert: output shape [B, 64]

T2.2 — forward() backward compatibility
    Create RelGT with out_channels=1
    Call forward() with same inputs as current code
    Assert: output shape matches current behavior
    Assert: output values match (numerically identical to old code)

T2.3 — forward() with no head raises error
    Create RelGT with out_channels=None
    Call forward()
    Assert: raises RuntimeError or AttributeError with clear message

T2.4 — encode() and forward() consistency
    x_enc = model.encode(...)
    x_fwd = model.forward(...)
    Assert: model.head(x_enc) == x_fwd  (numerically)

T2.5 — MaskedTokenHead shapes
    Create MaskedTokenHead(channels=64, num_node_types=10)
    Input: x_set of shape [4, 100, 64]
    Assert: output["type_pred"] shape [4, 100, 10]
    Assert: output["feature_pred"] shape [4, 100, 64]

T2.6 — Gradient flow through encode()
    Call encode(), apply a dummy loss on output, call backward()
    Assert: all encoder parameters have non-None gradients
    Assert: in_mixture parameters have non-None gradients
```

**Integration Test:**

```
T2.7 — Existing main_node_ddp.py still works
    Run main_node_ddp.py for 2 steps (uses forward(), not encode())
    Assert: identical behavior to pre-change code
```

---

## Phase 2: Parallel Changes (Tasks 3, 5, 7)

### Task 3: Multi-Dataset Data Loading

**Unit Tests:**

```
T3.1 — Union node_type_map contains all tables
    Load 2 small datasets (e.g., rel-f1 and one other)
    Build union mappings
    Assert: all table names from both datasets present (namespaced)
    Assert: no duplicate indices
    Assert: indices are contiguous 0..N-1

T3.2 — col_stats_dict has no key collisions
    Build union col_stats_dict from 2 datasets
    Assert: len(union) == len(dataset1_stats) + len(dataset2_stats)
    Assert: all keys are namespaced

T3.3 — RelGTTokens with union_node_type_map
    Create RelGTTokens with union_node_type_map for one dataset
    Call __getitem__(0)
    Assert: sample["types"] values are all valid indices in union map
    Assert: sample["types"] values match the namespaced types (not the original local indices)

T3.4 — Collation with union mapping
    Create batch of 4 samples via collate()
    Assert: grouped_tfs keys match union type indices
    Assert: grouped_indices are correct

T3.5 — dataset_name metadata passes through
    Create RelGTTokens with dataset_name="rel-f1"
    sample, label = dataset[0]
    Assert: sample contains dataset_name field

T3.6 — HDF5 regeneration check
    If precomputed HDF5 exists from single-dataset run (old type indices):
    Assert: code detects stale HDF5 or forces regeneration when union map differs
    (Important: old HDF5 files have WRONG type indices under union mapping)
```

**Integration Test:**

```
T3.7 — Load 2 datasets end-to-end
    Load rel-f1 and one small dataset
    Build union mappings
    Create RelGTTokens for both
    Create DataLoaders
    Iterate one batch from each
    Assert: batches have correct shapes and no NaN values
```

---

### Task 5: Masked Token Pretraining

**Unit Tests:**

```
T5.1 — apply_masking never masks position 0
    Create batch with B=8, K=100
    Apply masking with mask_ratio=0.5
    Assert: mask_positions[:, 0] is all False

T5.2 — Mask ratio is approximately correct
    Apply masking with mask_ratio=0.15, B=100, K=300
    Expected masked per sample: ~0.15 * 299 ≈ 45
    Assert: mean masked count is within [40, 50]

T5.3 — Masked values are replaced correctly
    Apply masking
    Assert: masked_batch["neighbor_types"][mask] == mask_type_idx (all same)
    Assert: masked_batch["neighbor_times"][mask] == -1.0 (all same)
    Assert: masked_batch["neighbor_hops"][mask] == -1 (all same)

T5.4 — Original values preserved in targets
    Assert: mask_targets["original_types"] matches the unmasked batch at masked positions
    Assert: lengths match: len(original_types) == mask.sum()

T5.5 — compute_reconstruction_loss returns finite scalar
    Create dummy predictions and targets with correct shapes
    loss = compute_reconstruction_loss(pred, targets)
    Assert: loss.dim() == 0 (scalar)
    Assert: loss.isfinite()
    Assert: loss.requires_grad

T5.6 — Feature targets are detached
    Generate feature targets (two-pass: encode originals, detach)
    Assert: original_feature_targets.requires_grad == False

T5.7 — Loss decreases over multiple steps
    Create small model + masking head
    Run 20 optimizer steps on a fixed batch
    Assert: loss at step 20 < loss at step 0
```

---

### Task 7: DDP Global Attention Fixes

**Unit Tests (single GPU, verify logic):**

```
T7.1 — c_idx removed from RelGTLayer
    Create RelGTLayer with conv_type="full"
    Assert: not hasattr(layer, 'c_idx')

T7.2 — num_nodes no longer required
    Create RelGTLayer without num_nodes parameter
    Assert: no error

T7.3 — Log-count bias uses _ema_cluster_size
    Create RelGTLayer
    Manually set vq._ema_cluster_size to known values
    Run global_forward()
    Verify: dots bias matches log(_ema_cluster_size.clamp(min=1))

T7.4 — global_forward runs without c_idx
    Create RelGTLayer
    Run global_forward with random input
    Assert: output has correct shape, no error

T7.5 — Backward compatibility: main_node_ddp.py still works
    Run main_node_ddp.py for 2 steps with conv_type="full"
    Assert: no errors (num_nodes param removed/optional)
```

**Multi-GPU Tests (2 GPUs minimum):**

```
T7.6 — VQ buffers synced after sync_vq_buffers()
    Launch 2 processes
    Each process updates VQ with different random data (10 steps)
    Call sync_vq_buffers()
    Assert: vq._ema_cluster_size is identical on both GPUs (torch.allclose)
    Assert: vq._ema_w is identical on both GPUs
    Assert: vq._embedding_output is identical on both GPUs

T7.7 — Without sync, buffers diverge (control test)
    Launch 2 processes
    Each process updates VQ with different random data (10 steps)
    Do NOT call sync_vq_buffers()
    Assert: vq._ema_cluster_size is DIFFERENT on the two GPUs

T7.8 — SyncBatchNorm covers VQ's self.bn
    Create full RelGT model
    Apply SyncBatchNorm.convert_sync_batchnorm()
    Walk all modules
    Assert: zero instances of nn.BatchNorm1d remaining
    Assert: VQ's bn is instance of SyncBatchNorm
```

---

## Phase 3: Foundation Training Script (Task 4)

**Single-GPU Smoke Tests:**

```
T4.1 — Script launches without crash
    Run main_foundation.py with --held_out_dataset rel-f1
    Load only 2 small datasets (mock or subset)
    Run for 3 steps
    Assert: no crash, loss is finite

T4.2 — Interleaved iterator samples from all datasets
    Create InterleavedIterator with 3 mock loaders
    Sample 100 batches
    Assert: all 3 dataset keys appear at least once

T4.3 — Temperature-scaled sampling distribution
    Create InterleavedIterator with alpha=0.5
    Dataset sizes: {"A": 10000, "B": 100, "C": 1000}
    Sample 10000 batches, count per dataset
    Assert: A sampled most, C second, B least
    Assert: B sampled > 0 (not starved)
    Assert: ratio roughly matches sqrt(size) distribution

T4.4 — Checkpoint save and load
    Train for 2 steps, save checkpoint
    Load checkpoint into fresh model
    Assert: model.state_dict() matches saved state
    Assert: config and union_node_type_map preserved

T4.5 — DistributedSampler epoch update
    Call interleaved_iter.set_epoch(1) then set_epoch(2)
    Assert: sampler.epoch updated for all loaders
```

**Multi-GPU Tests (2+ GPUs):**

```
T4.6 — DDP training runs for 1 epoch
    torchrun --nproc_per_node=2 main_foundation.py --held_out_dataset rel-f1
    Run for 10 steps
    Assert: no hang, no crash, loss is finite on rank 0

T4.7 — Model parameters synced after training
    After 10 steps, compare model.parameters() on rank 0 and rank 1
    Assert: all parameters identical (DDP guarantee)

T4.8 — VQ buffers synced during training
    After 10 steps, compare VQ buffers on rank 0 and rank 1
    Assert: _ema_cluster_size identical (within floating point tolerance)
    Assert: _embedding_output identical

T4.9 — Different GPUs process different batches
    Log dataset_task_key per step per rank
    Assert: rank 0 and rank 1 sometimes get different dataset keys (expected)

T4.10 — Gradient sync across GPUs
    After one step, compare gradients on rank 0 vs rank 1
    Assert: gradients are identical (DDP syncs after backward)
```

---

## Phase 4: Zero-Shot Evaluation (Task 6)

**Unit Tests:**

```
T6.1 — mask_target_column masks only the target
    Create batch, mask the target column for seed nodes
    Assert: target column at position 0 is zeroed
    Assert: all other positions and columns unchanged

T6.2 — Predictions have correct shape
    Run zero_shot_evaluate on small held-out loader
    For binary classification: Assert predictions shape [N] with values in [0, 1]
    For regression: Assert predictions shape [N] with finite values
    For multilabel: Assert predictions shape [N, num_labels]

T6.3 — No data leakage
    Check that during evaluation, the target column is NOT accessible
    Assert: model never sees unmasked target values

T6.4 — task.evaluate() runs without error
    Pass zero-shot predictions to RelBench's task.evaluate()
    Assert: returns valid metrics dict
    Assert: metric values are not NaN
```

**End-to-End Test:**

```
T6.5 — Zero-shot better than random
    Train foundation model for a few epochs on all-but-one datasets
    Evaluate zero-shot on held-out dataset
    Compare against random prediction baseline
    Assert: zero-shot metrics > random baseline
    (This is a sanity check, not a performance target)

T6.6 — Zero-shot on unseen dataset
    Hold out rel-f1, train on everything else
    Run zero-shot on rel-f1's tasks
    Assert: predictions are finite and in valid range
    Assert: evaluation pipeline completes without error
```

---

## Regression Tests (Run After Every Phase)

```
R1 — main_node_ddp.py single-dataset training
    Run for 5 steps on rel-f1/driver-top3 with conv_type="full"
    Assert: no errors, loss decreases
    This catches any backward-compatibility breaks

R2 — Model save/load roundtrip
    Save model state_dict, load into fresh model
    Assert: all keys match, no missing/unexpected keys
    Run one forward pass, assert same output

R3 — Precomputed HDF5 compatibility
    If HDF5 files exist from pre-change runs:
    Assert: code either loads them correctly OR regenerates them
    (With union_node_type_map, old HDF5 type indices are WRONG — must regenerate)
```

---

## Test Infrastructure

**How to run:**
```bash
# Unit tests (no GPU)
python -m pytest tests/test_namespacing.py
python -m pytest tests/test_backbone.py
python -m pytest tests/test_masking.py
python -m pytest tests/test_ddp_fixes.py

# Single-GPU integration
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tests/test_integration_single.py

# Multi-GPU DDP
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tests/test_integration_ddp.py

# Regression
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 main_node_ddp.py \
    --dataset rel-f1 --task driver-top3 --epochs 1 --max_steps_per_epoch 5

# End-to-end foundation
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main_foundation.py \
    --held_out_dataset rel-f1 --epochs 1 --max_steps_per_epoch 10
```

**Test data:** For fast iteration, use only 2-3 small RelBench datasets (e.g., rel-f1 and rel-trial) with reduced K (e.g., K=50) and small batch sizes.

**Pass criteria per phase:**
- Phase 1: All T1.x and T2.x pass + R1, R2
- Phase 2: All T3.x, T5.x, T7.x pass + R1, R2, R3
- Phase 3: All T4.x pass + R1, R2
- Phase 4: All T6.x pass + full regression suite
