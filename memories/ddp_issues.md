# DDP Multi-GPU Issues

## Identified Bugs (3)

### Bug 1: c_idx Buffer Divergence (CRITICAL)
- `RelGTLayer` stores `c_idx` as a buffer (shape `[num_nodes]`)
- DDP does NOT sync buffers across GPUs
- Each GPU sees different batches -> different VQ assignments -> c_idx diverges
- Used for log-count attention bias in `RelGTLayer.global_forward()`
- **Fix (Task 7)**: Eliminate c_idx entirely, use `_ema_cluster_size` for log-count bias instead

#### Why the author originally used c_idx (not _ema_cluster_size)
In the **single-dataset** setting where GOAT/RelGT was designed:
- `c_idx` gives **exact counts** (histogram of latest known assignments for ALL nodes), not smoothed estimates
- Nodes recur frequently (one graph, same nodes across batches) — entries stay fresh
- `num_nodes` is small and fixed — buffer is cheap
- DDP drift is benign — all GPUs see same distribution, counts are statistically similar
- `_ema_cluster_size` is a **smoothed approximation** — less precise than exact counts when data is fresh

In the **multi-dataset foundation model** setting, all advantages reverse:
- Millions of nodes → most entries stale (rarely seen)
- Memory scales to 10M+ nodes × 8 bytes
- Different GPUs may see different datasets → real divergence, not just noise
- `_ema_cluster_size` is better: smooth, small fixed shape [num_centroids], DDP-syncable

**Alternatives considered**:
1. Batch-local counts (zero state, noisy) — viable ablation
2. Remove log-count bias entirely (simplest, test empirically)
3. Keep c_idx + fix via hash map + all-reduce (complex, still stale, worse than EMA)

### Bug 2: VQ EMA Buffers Not Synced
- `_embedding`, `_ema_cluster_size`, `_ema_w` are all buffers, not parameters
- EMA updates happen per-GPU with only local batch statistics
- Codebooks diverge silently across GPUs
- **Fix (Task 7)**: Add `sync_vq_buffers()` function using `dist.all_reduce()` after each training step

### Bug 3: num_nodes Dependency
- `RelGTLayer` takes `num_nodes` parameter, used to allocate `c_idx`
- In multi-dataset setting, total nodes vary per dataset
- **Fix (Task 7)**: Remove `num_nodes` parameter entirely (consequence of eliminating c_idx)

## Identified Bottlenecks (4)

1. **GloVe CPU inference** — SentenceTransformer runs on CPU every forward pass (Fix: precompute at init, Task 1)
2. **Laplacian PE** — Sparse eigenvector computation on CPU (already fixed: compute on CPU, transfer to device)
3. **find_unused_parameters=True** — Extra DDP overhead, but needed if not all heads used
4. **BatchNorm in FeedForwardNetwork** — SyncBatchNorm needed for correctness under DDP (already handled with convert_sync_batchnorm)

## DDP Checklist for Foundation Training (Task 4)
- DistributedSampler per loader, set_epoch each epoch
- Rank-dependent random seed in InterleavedIterator
- sync_vq_buffers() after every training step (collective - all ranks must call)
- dist.barrier() before evaluation
- Only rank 0 logs to wandb and saves checkpoints
- SyncBatchNorm conversion on CUDA
