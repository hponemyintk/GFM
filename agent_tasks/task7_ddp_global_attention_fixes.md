# Task 7: DDP Fixes for Global Attention

## Objective

Fix three DDP correctness issues in the global attention / VQ codebook path that cause silent divergence across GPUs. These bugs exist in the current single-dataset code and will become critical failures in multi-dataset training.

## Context

### Bug 1: VQ EMA Buffers Not Synced

`VectorQuantizerEMA` (`codebook.py`) stores codebook state as **buffers**, not parameters:
- `_embedding` — codebook vectors
- `_embedding_output` — denormalized codebook (used as keys/values in attention)
- `_ema_cluster_size` — EMA of cluster sizes
- `_ema_w` — EMA of weighted embeddings

DDP only syncs gradients of `nn.Parameter`. These buffers are updated independently on each GPU via the EMA update in `codebook.py:87-104`. After a few steps, each GPU has a **different codebook**, meaning each GPU computes global attention against different keys/values.

### Bug 2: `c_idx` Per-Node Tracking Diverges

`RelGTLayer.c_idx` (shape `[num_nodes]`) stores which centroid each node was last assigned to. Updated at `model.py:149`: `self.c_idx[batch_idx] = x_idx.squeeze()`. Each GPU only sees a subset of nodes per step, so `c_idx` diverges across GPUs. Used for log-count attention bias at `model.py:133-138`.

With multi-dataset training, `c_idx` would need to be `[total_nodes_across_all_datasets]` — potentially millions of entries, wasting memory on every GPU.

### Bug 3: VQ BatchNorm May Not Be SyncBatchNorm

`VectorQuantizerEMA.bn` is `BatchNorm1d`. The `SyncBatchNorm.convert_sync_batchnorm()` call should convert it, but this must be verified — if it's missed, the BN running stats diverge.

## Files to Modify

### 1. `model.py` — `RelGTLayer.global_forward()`

**Eliminate `c_idx`, use `_ema_cluster_size` for log-count bias:**

Replace lines 133-138:
```python
# CURRENT (broken under DDP):
c, c_count = self.c_idx.unique(return_counts=True)
centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
centroid_count[c.to(torch.long)] = c_count
dots = dots + torch.log(centroid_count.view(1, 1, -1))
```

With:
```python
# NEW (DDP-safe):
# _ema_cluster_size is synced via all-reduce (see sync_vq_buffers in Task 4)
dots = dots + torch.log(self.vq._ema_cluster_size.clamp(min=1).view(1, 1, -1))
```

**Remove `c_idx` buffer and its update:**

Remove from `__init__` (line 65-66):
```python
# DELETE:
c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.long)
self.register_buffer("c_idx", c)
```

Remove from `global_forward` (line 149):
```python
# DELETE:
self.c_idx[batch_idx] = x_idx.squeeze().to(torch.long)
```

**Remove `num_nodes` parameter from `RelGTLayer.__init__`:**

`num_nodes` was only used to size `c_idx`. With `c_idx` gone, it's no longer needed. This also removes `num_nodes` from `RelGT.__init__` (which passes it to `RelGTLayer`).

**Impact on backward compatibility:** `main_node_ddp.py` line 226 passes `num_nodes=data["train"].data.num_nodes`. This parameter should be removed or made optional (ignored with a deprecation warning).

### 2. `codebook.py` — No direct changes

The VQ codebook code itself doesn't change. The sync is handled externally by `sync_vq_buffers()` in the training script (Task 4). The codebook continues to do local EMA updates; the training script periodically all-reduces the buffers.

### 3. `model.py` — `RelGT.__init__()`

**Remove `num_nodes` from constructor or make it optional:**

```python
def __init__(
    self,
    num_nodes: int = 0,  # DEPRECATED: no longer used, kept for backward compat
    ...
):
```

Remove from the `RelGTLayer` constructor call (line 234):
```python
# Before:
RelGTLayer(..., num_nodes=num_nodes, ...)
# After:
RelGTLayer(...) # num_nodes removed
```

### 4. Verification: SyncBatchNorm coverage

Add an assertion in the training script (Task 4) after `convert_sync_batchnorm`:

```python
# Verify VQ BatchNorm was converted
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm1d):
        raise RuntimeError(
            f"BatchNorm1d found at {name} after SyncBatchNorm conversion. "
            f"This will cause DDP divergence."
        )
```

## What This Fixes

| Before | After |
|--------|-------|
| Each GPU has different VQ codebook | All GPUs share synced codebook via all-reduce |
| `c_idx` diverges, log-count bias is wrong | Log-count from `_ema_cluster_size` (smooth, synced) |
| `c_idx` memory: O(total_nodes) per GPU | Zero — buffer eliminated |
| `num_nodes` required at model init | Not required — model is node-count agnostic |
| VQ BatchNorm maybe not SyncBatchNorm | Verified with assertion |

## What This Changes for Inference

At eval on unseen datasets:
- No `c_idx` to worry about (it's gone)
- Centroids are in `_embedding_output` (a buffer, saved in state_dict)
- Unseen nodes query centroids via attention — works without any per-node state
- `_ema_cluster_size` provides the log-count bias from training — no update needed

## Testing

See `agent_tasks/testing_plan.md` — Phase 2, Task 7 tests.

## Dependencies

- Task 2 (backbone separation) — both modify `model.py`, coordinate to avoid conflicts
- Task 4 (training script) — implements `sync_vq_buffers()` which calls into the cleaned-up VQ
- Must be done BEFORE Task 4 (training script relies on DDP-safe VQ)
