# Plan: Resume-from-Checkpoint for DDP Multi-Task Pretraining

## Status

**IMPLEMENTED** — 2026-05-11. Changes in `train_multi_task.py` (helpers +
resume-load before the epoch loop + per-epoch save/prune), `main_node_ddp.py`
(`--resume` / `--keep_checkpoints` flags), `scripts/pretrain_p4d.sh`
(`RESUME` / `KEEP_CHECKPOINTS` env passthrough). Tests in
`tests/test_resume_checkpoint.py` (9 CPU unit tests + a `@pytest.mark.slow`
2-process gloo DDP smoke). Verified end-to-end on `rel-f1.driver-top3`: fresh
2-epoch run → `RESUME=auto` to 4 epochs → resumes at epoch 3, `global_step`
continues, `checkpoint_epoch_0001.pt` pruned (`KEEP_CHECKPOINTS=3`),
`per_epoch_macro` carries `{1,2,3,4}` across the resume.

Note: RNG is split into per-rank epoch-indexed sidecars
(`rng_state_epoch{N:04d}_rank{R}.pt`) rather than a single blob in the main
checkpoint — model dropout draws from the per-rank torch global RNG, which
diverges across ranks within an epoch.

## Motivation

Training the GFM backbone takes ~8 days (20 epochs x ~9.5 hours/epoch
on 8x A100 GPUs). Jobs frequently die mid-run (CUDA OOM during
validation, pod preemption, network issues). Currently there is no
resume mechanism — training restarts from scratch every time, wasting
days of GPU compute.

## Design

Save a full training checkpoint after each completed epoch. On restart,
load the latest checkpoint and continue from the next epoch.

### Checkpoint contents (`checkpoint_epoch_{N:04d}.pt`)

- `model.module.state_dict()`
- `optimizer.state_dict()`
- `loss_fn.state_dict()` (captures `log_sigma2` for uncertainty mode)
- `epoch`, `global_step`
- `best_macro`, `best_epoch`, `best_ckpt_written`, `per_epoch_macro`
- RNG states: torch, CUDA, numpy, python random

### CLI interface

```
--resume PATH|auto   # 'auto' finds latest checkpoint_epoch_*.pt in out_dir
--keep_checkpoints 3 # prune older checkpoints (0=keep all)
```

### Usage

```bash
# Normal run
BATCH=64 NPROC=8 EPOCHS=20 ./scripts/run_holdout_no_searchstream.sh

# Resume after crash
RESUME=auto SKIP_BUILD=1 BATCH=64 NPROC=8 EPOCHS=20 ./scripts/run_holdout_no_searchstream.sh
```

## Files to modify

| File | Change |
|------|--------|
| `main_node_ddp.py` | Add `--resume` and `--keep_checkpoints` argparse entries |
| `train_multi_task.py` | Add `_save_training_checkpoint`, `_find_latest_checkpoint`, `_load_training_checkpoint` functions; modify epoch loop to use `start_epoch` from checkpoint |
| `scripts/pretrain_p4d.sh` | Pass-through `RESUME` and `KEEP_CHECKPOINTS` env vars |

## Implementation details

### Save location

- Same directory as existing best checkpoints: `{args.out_dir}/multi_task/`
- Filename: `checkpoint_epoch_0001.pt`, `checkpoint_epoch_0002.pt`, etc.
- Coexists with existing `best_full.pt` / `best_backbone.pt` (unchanged)

### Save timing

Checkpoint is saved at end of each epoch, AFTER validation and
best-tracking logic completes. This means:
- A crash during training loses that epoch's work (~9.5 hours worst case)
- A crash during validation loses that epoch's work too
- Could optionally save before validation to reduce worst-case loss

### Resume flow

1. All ranks load checkpoint from shared filesystem (NFS/PVC)
2. Restore model, optimizer, loss_fn state
3. Restore RNG states for deterministic data ordering
4. Set `start_epoch = ckpt["epoch"] + 1`
5. Restore `best_macro`/`best_epoch` tracking state
6. `dist.barrier()` to sync all ranks
7. Training loop starts at `range(start_epoch, args.epochs + 1)`

### Determinism guarantee

The `DistributedMultiTaskSampler` seeds with `self.seed + self.epoch`,
making data order deterministic per epoch. After resume at epoch N+1,
`train_sampler.set_epoch(N+1)` produces identical batch ordering as a
continuous run. No sampler changes needed.

### Disk budget

~500 MB per checkpoint (model ~135MB + optimizer ~270MB + overhead).
With `keep_checkpoints=3`: ~1.5 GB total. Negligible on 8TB NVMe.

### Pruning

After each save, glob `checkpoint_epoch_*.pt`, sort, delete all but
the most recent `keep_n` files. Zero-padded epoch numbers ensure
correct lexicographic order.

## Edge cases

- **Architecture mismatch on resume**: `load_state_dict` raises key
  error — correct behavior (don't silently resume with wrong config)
- **Fewer epochs on resume**: `range(16, 11)` is empty — skips to
  test eval immediately
- **More epochs on resume**: continues from saved epoch through new max
- **Corrupt checkpoint**: let `torch.load` exception propagate; user
  deletes the file and falls back to an older checkpoint
