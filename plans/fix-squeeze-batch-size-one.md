# Fix: IndexError on batch-size-1 during validation

## Status

**FIXED** — 2026-05-11. One-line fix in `local_module.py:89`.

## Symptom

DDP pretraining (8x A100, batch_size=64, 31 tasks) completes epoch 1
training (~9.5 hours) but crashes on the **last batch** of validation
task 20 (batch 113/114):

```
[rank5]: IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

Stack trace points to `model.py:110`:
```python
out = torch.cat([out_local, out_global], dim=1)
```

## Root cause

In `local_module.py:89`, the `LocalModule.forward()` method uses an
**unqualified `.squeeze()`**:

```python
output = (node_tensor + neighbor_tensor).squeeze()
```

When `node_tensor + neighbor_tensor` has shape `(1, 1, D)` (batch_size=1,
seq_dim=1), `.squeeze()` removes **all** size-1 dimensions, collapsing
to shape `(D,)` — a 1D tensor. The downstream `torch.cat(..., dim=1)`
then fails because both operands are 1D.

This only triggers when the last validation batch has exactly 1 sample,
which happens when `dataset_size % (num_gpus * batch_size)` leaves a
remainder of 1 on some DDP rank. The val/test `DistributedSampler` uses
`drop_last=False` (correctly — we want full evaluation), so the final
batch can be arbitrarily small.

## Fix

```diff
- output = (node_tensor + neighbor_tensor).squeeze()
+ output = (node_tensor + neighbor_tensor).squeeze(1)
```

`.squeeze(1)` only removes the sequence dimension (dim=1, always size 1
after the sum), preserving the batch dimension regardless of batch size.

## File changed

- `local_module.py:89`

## Reproduction conditions

- `drop_last=False` on val/test DataLoader (current setting)
- Any batch_size + num_gpus combination where the last batch on some
  rank has exactly 1 sample
- `conv_type="full"` in `RelGTConv` (triggers both local and global
  forward paths that need to be concatenated)
