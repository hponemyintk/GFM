# Task 4: Foundation Model Training Script

## Objective

Create `main_foundation.py` — the multi-dataset, self-supervised pretraining script with DDP support, interleaved dataset sampling, leave-one-out structure, and DDP-safe global attention.

## Context

Currently `main_node_ddp.py` trains on one dataset/task with supervised labels. The foundation model script replaces this with:
- Loading all RelBench datasets (via Task 3's `data_loading.py`)
- Self-supervised masked token prediction (via Task 5's masking strategy)
- Temperature-scaled interleaved batch sampling across datasets
- Leave-one-out: one dataset is held out for zero-shot evaluation
- DDP-safe VQ codebook synchronization (via Task 7's fixes)

## File to Create

### `main_foundation.py`

#### CLI Arguments

```python
parser.add_argument("--held_out_dataset", type=str, required=True,
                    help="Dataset to hold out for zero-shot evaluation")
parser.add_argument("--sampling_alpha", type=float, default=0.5,
                    help="Temperature for dataset sampling. 0=uniform, 1=proportional")
parser.add_argument("--mask_ratio", type=float, default=0.15,
                    help="Fraction of neighbor tokens to mask during pretraining")
parser.add_argument("--vq_sync_interval", type=int, default=1,
                    help="Sync VQ codebook across GPUs every N steps (1=every step)")
# Keep existing args: lr, epochs, batch_size, channels, num_layers, num_heads,
# gt_conv_type, num_neighbors, num_centroids, ff_dropout, attn_dropout, etc.
```

#### Script Structure

```
1.  Parse arguments
2.  Initialize DDP
3.  Load all datasets and build union mappings (Task 3)
4.  Create RelGTTokens for all dataset/task/split combos
5.  Build DataLoaders with DistributedSampler per dataset
6.  Build model (backbone only, no task head) + masking head
7.  DDP wrapping with SyncBatchNorm + VQ sync hooks
8.  Build interleaved dataset iterator
9.  Training loop (masked token prediction + VQ sync)
10. Zero-shot evaluation on held-out dataset (Task 6)
11. Save checkpoint
12. Cleanup
```

#### Step 5: DataLoaders

Create one DataLoader per (dataset, task, "train") combination:

```python
train_loaders = {}
for (dataset_name, task_name), tokens in all_tokens["train"].items():
    sampler = DistributedSampler(tokens, shuffle=True, seed=args.seed)
    loader = DataLoader(
        tokens,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=tokens.collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    train_loaders[(dataset_name, task_name)] = loader
```

#### Step 7: DDP Wrapping (CRITICAL)

```python
# 1. SyncBatchNorm — converts ALL BatchNorm layers including VQ's self.bn
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

# 2. Verify VQ BatchNorm was converted
if hasattr(model, 'convs'):
    for layer in model.convs:
        if hasattr(layer, 'vq'):
            assert isinstance(layer.vq.bn, torch.nn.SyncBatchNorm), \
                "VQ BatchNorm not converted to SyncBatchNorm!"

# 3. DDP wrapping
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# 4. Also wrap masking head in DDP
masking_head = DDP(masking_head, device_ids=[local_rank])
```

#### Step 8: Interleaved Dataset Iterator

```python
class InterleavedIterator:
    """Samples batches from multiple DataLoaders with temperature-scaled probabilities.

    DDP note: Each GPU runs its own InterleavedIterator independently.
    Different GPUs may sample different datasets on the same step.
    This is fine — DDP syncs parameter gradients regardless of input.
    The VQ buffers are synced explicitly (see sync_vq_buffers).
    """

    def __init__(self, loaders: Dict[str, DataLoader], alpha: float = 0.5,
                 seed: int = 42, rank: int = 0):
        self.loaders = loaders
        self.iterators = {k: iter(v) for k, v in loaders.items()}
        self.alpha = alpha
        # Use rank-dependent seed so GPUs sample differently
        self.rng = random.Random(seed + rank)

        sizes = {k: len(v.dataset) for k, v in loaders.items()}
        raw_weights = {k: s ** alpha for k, s in sizes.items()}
        total = sum(raw_weights.values())
        self.probs = {k: w / total for k, w in raw_weights.items()}
        self.keys = list(self.probs.keys())
        self.weights = [self.probs[k] for k in self.keys]

    def set_epoch(self, epoch):
        """Update all DistributedSamplers for the new epoch."""
        for loader in self.loaders.values():
            if hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)
        # Re-create iterators after sampler update
        self.iterators = {k: iter(v) for k, v in self.loaders.items()}

    def __iter__(self):
        return self

    def __next__(self):
        key = self.rng.choices(self.keys, weights=self.weights, k=1)[0]
        try:
            batch = next(self.iterators[key])
        except StopIteration:
            self.iterators[key] = iter(self.loaders[key])
            batch = next(self.iterators[key])
        return batch, key
```

#### Step 9: Training Loop with DDP-Safe VQ Sync

```python
def sync_vq_buffers(model):
    """Explicitly all-reduce VQ EMA buffers across GPUs.
    Called after each training step (or every N steps).
    Must be called by ALL ranks (all-reduce is collective).
    """
    world_size = dist.get_world_size()
    # Access underlying model if wrapped in DDP
    base_model = model.module if hasattr(model, 'module') else model
    for layer in base_model.convs:
        if not hasattr(layer, 'vq'):
            continue
        vq = layer.vq
        # Average the EMA stats across GPUs
        dist.all_reduce(vq._ema_cluster_size, op=dist.ReduceOp.SUM)
        vq._ema_cluster_size /= world_size
        dist.all_reduce(vq._ema_w, op=dist.ReduceOp.SUM)
        vq._ema_w /= world_size
        # Recompute embedding from synced EMA
        vq._embedding.data = vq._ema_w / vq._ema_cluster_size.unsqueeze(1).clamp(min=1e-5)
        # Recompute denormalized output
        running_std = torch.sqrt(vq.bn.running_var + 1e-5).unsqueeze(0)
        running_mean = vq.bn.running_mean.unsqueeze(0)
        vq._embedding_output.data = vq._embedding * running_std + running_mean


def train_epoch(epoch, interleaved_iter, model, masking_head, optimizer, args):
    model.train()
    masking_head.train()
    loss_accum = 0
    count = 0

    interleaved_iter.set_epoch(epoch)

    for step in range(args.max_steps_per_epoch):
        batch, dataset_task_key = next(interleaved_iter)

        # Move batch to device (same pattern as main_node_ddp.py:291-303)
        neighbor_types = batch["neighbor_types"].to(device)
        node_indices = batch["node_indices"].to(device)
        neighbor_hops = batch["neighbor_hops"].to(device)
        neighbor_times = batch["neighbor_times"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_vec = batch["batch"].to(device)
        grouped_tf_dict = {
            'grouped_tfs': batch['grouped_tfs'],
            'grouped_indices': batch['grouped_indices'],
            'flat_batch_idx': batch['flat_batch_idx'],
            'flat_nbr_idx': batch['flat_nbr_idx']
        }

        # Apply masking (Task 5)
        masked_batch, mask_targets = apply_masking(batch, mask_ratio=args.mask_ratio)

        optimizer.zero_grad()

        # Forward through backbone
        representations = model(
            masked_batch["neighbor_types"],
            masked_batch["node_indices"],
            masked_batch["neighbor_hops"],
            masked_batch["neighbor_times"],
            masked_batch["grouped_tf_dict"],
            edge_index=masked_batch["edge_index"],
            batch=masked_batch["batch"],
            return_full_sequence=True,  # for masking head
        )

        # Reconstruction loss
        pred = masking_head(representations)
        loss = compute_reconstruction_loss(pred, mask_targets)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # DDP: Sync VQ codebook buffers across GPUs
        if step % args.vq_sync_interval == 0:
            sync_vq_buffers(model)

        loss_accum += loss.item()
        count += 1

        if local_rank == 0:
            wandb.log({
                "train_loss": loss.item(),
                "dataset_task": str(dataset_task_key),
                "global_step": global_step,
            })

    return loss_accum / count
```

#### Step 10: Checkpointing

```python
torch.save({
    "backbone": model.module.state_dict(),
    "masking_head": masking_head.module.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": vars(args),
    "union_node_type_map": union_node_type_map,
    "union_col_stats_dict": union_col_stats_dict,
    "epoch": epoch,
}, os.path.join(args.out_dir, f"foundation_ckpt_epoch{epoch}.pt"))
```

## DDP Checklist for This Script

- [ ] `SyncBatchNorm.convert_sync_batchnorm()` called BEFORE DDP wrapping
- [ ] VQ's `self.bn` verified as SyncBatchNorm after conversion
- [ ] `sync_vq_buffers()` called after each training step (collective — all ranks)
- [ ] `DistributedSampler.set_epoch()` called each epoch for all loaders
- [ ] `InterleavedIterator` uses rank-dependent seed for different sampling per GPU
- [ ] `dist.barrier()` before evaluation
- [ ] Only rank 0 logs to wandb and saves checkpoints
- [ ] `dist.broadcast` model state from rank 0 after loading best checkpoint
- [ ] `find_unused_parameters=True` on DDP (some encode paths skip head/VQ)

## Testing

See `agent_tasks/testing_plan.md` — Phase 3 tests.

## Dependencies

- Task 1 (namespacing) — table names must be namespaced
- Task 2 (backbone separation) — needs `model.encode()` method
- Task 3 (data loading) — needs `build_union_mappings()` and `create_all_relgt_tokens()`
- Task 5 (masking) — needs `apply_masking()` and `compute_reconstruction_loss()`
- Task 7 (DDP fixes) — `c_idx` elimination and VQ sync must be in place
