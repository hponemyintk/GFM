# AWS p4d.24xlarge runbook

Operational notes for running `scripts/pretrain_p4d.sh` on AWS p4d
(8 × A100 40 GB, 96 vCPU, ~1.1 TB RAM, ~8 TB local NVMe), including
the fully-offline recipe for pods without outbound internet.

---

## TL;DR (online pod with internet + `WANDB_API_KEY` set)

```bash
DATASETS=rel-f1,rel-event STEPS_PER_TASK=1000 EPOCHS=20 \
    ./scripts/pretrain_p4d.sh
```

The script handles caching, parallelism, and per-phase verification.
First run downloads RelBench data + the sentence-transformers GloVe
model; subsequent runs hit local caches.

---

## Full offline AWS recipe

For pods with no outbound internet (locked-down VPC, air-gapped
account, etc.) you need to seed two caches once on a box that DOES
have internet, then rsync them over.

### 1. Seed RelBench data (once, on any internet-connected machine)

```bash
python3 - <<'PY'
from relbench.datasets import get_dataset
for d in ['rel-amazon', 'rel-avito', 'rel-event',
          'rel-f1', 'rel-hm', 'rel-stack', 'rel-trial']:
    print(f"caching {d} ...")
    get_dataset(d, download=True).get_db(upto_test_timestamp=False)
PY
```

This populates `~/.cache/relbench/<dataset>/db/`. Each dataset is a
few hundred MB to ~13 GB on disk (rel-event is the biggest); total
across all of v2 is roughly 30–50 GB.

### 2. Seed the GloVe text embedder (once, same box)

```bash
huggingface-cli download \
    sentence-transformers/average_word_embeddings_glove.6B.300d
```

This populates
`~/.cache/huggingface/hub/models--sentence-transformers--average_word_embeddings_glove.6B.300d/`.
About 500 MB.

### 3. rsync the caches to the offline pod

```bash
# from the seeding box, target the pod's $HOME:
rsync -av --progress ~/.cache/relbench/    pod:~/.cache/relbench/
rsync -av --progress ~/.cache/huggingface/ pod:~/.cache/huggingface/
```

Verify on the pod:

```bash
ls ~/.cache/relbench/                                # should list rel-* dirs
ls ~/.cache/huggingface/hub/ | grep glove            # should match
```

### 4. Disable wandb (or leave online if VPC allows)

The script defaults `WANDB_MODE=online`. For an offline pod, override:

```bash
export WANDB_MODE=offline
```

(Or unset `WANDB_API_KEY` first; the script's default is online for
users who have an API key configured.)

### 5. `pretrain_p4d.sh` auto-detects offline caches

When the GloVe model directory is present, the script automatically
sets:

```
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

so `sentence_transformers` and `huggingface_hub` never reach out to
check for revisions. RelBench's `pooch`-based loader is content-
addressed and only downloads if a file is missing — with the rsynced
cache it skips network entirely.

### 6. Launch

```bash
WANDB_MODE=offline DATASETS=rel-f1,rel-event STEPS_PER_TASK=1000 \
    EPOCHS=20 ./scripts/pretrain_p4d.sh
```

The first log lines confirm the offline state:

```
[hf] GloVe cache present -> HF_HUB_OFFLINE=1
```

If you see `[hf] GloVe cache NOT found` instead, your rsync missed
something — go back to step 3.

---

## Memory & concurrency knobs

The script exposes per-phase concurrency to keep RAM bounded on
smaller pods:

| Knob | Default | Phase | Notes |
|---|---|---|---|
| `PARALLEL_TF_BUILDS` | `NPROC` (8) | 1 (TF memmap, GPU) | One A100 per dataset; clamped to physical GPU count at runtime |
| `PARALLEL_SHARD_BUILDS` | `min(NPROC, 4)` (4) | 2 (shards, RAM) | Each builder loads HeteroData (~25 GB peak transiently for rel-event); 4 concurrent → ~100 GB transient, sustained ~few GB after TF drop |
| `NPROC` | 8 | 3 (training, DDP) | Number of GPUs; matches `torchrun --nproc_per_node` |

If your pod has less than the standard 1.1 TB RAM, drop
`PARALLEL_SHARD_BUILDS`:

```bash
PARALLEL_SHARD_BUILDS=2 ... ./scripts/pretrain_p4d.sh
```

---

## Wall-time targets

For the full RelBench v2 binary+regression task set (~25 tasks) at
the script's defaults (`channels=512`, `K=300`, `BATCH=512`,
`NUM_LAYERS=4`, 8×A100):

| Knob set | Total optimizer steps | Wall (train+eval) |
|---|---|---|
| `STEPS_PER_TASK=500 EPOCHS=10` (matches expts/run-large-base) | 125k | ~6–12 h |
| `STEPS_PER_TASK=1000 EPOCHS=20` (recommended) | 500k | ~30–70 h (~3 days) |
| `STEPS_PER_TASK=3000 EPOCHS=30` (heavy) | 2.25M | ~13–26 days |

Rough rules of thumb (eval cost grows with both EPOCHS and N_tasks):

- **Quick shake-out** (verify pipeline + see loss decrease):
  `STEPS_PER_TASK=200 EPOCHS=3` → ~1 h
- **Solid pretraining**: `STEPS_PER_TASK=1000 EPOCHS=20` → ~3 days
- **Compute-bound research run**: `STEPS_PER_TASK=2000 EPOCHS=30` → ~5–10 days

---

## Failure modes and recovery

The script is resume-safe. Each phase 1 dataset and phase 2
(dataset, task) writes a `.done` sentinel only on success; re-running
skips already-built artifacts. If a build fails:

1. Find the failing artifact:
   ```
   ERROR: phase 2 failed to produce shards for: rel-event.user-attendance
     see results/p4d_pretrain/build_shard_rel-event_user-attendance.log for details
   ```
2. Read the log; common causes are OOM during the (transient) full-
   `HeteroData` window or stypes regeneration on a corrupt
   `stypes.json`.
3. Fix the root cause (drop `PARALLEL_SHARD_BUILDS`, delete corrupt
   stypes.json so it regenerates), then re-run the script — done
   sentinels prevent rebuilding everything.

### Phase 3 OOM during dataset load

If torchrun crashes with OOM-kill while still printing
`Loading Database object from .../db...`, the cause is 8 DDP ranks all
pickle-loading the raw RelBench DB simultaneously. Each rank holds the
full DB pickle (~25 GiB on rel-event / rel-amazon) plus transient
make_pkey_fkey_graph buffers; with 8 ranks this peaks past pod RAM
even though each rank's *steady* state (after dropping tf+edge_index)
is small.

Mitigations baked in:
- `--load_concurrency 1` (default in the launcher) serializes loads:
  only 1 rank loads at a time, peak transient = 1× per-rank instead of
  8×. Set `LOAD_CONCURRENCY=2` (or 4) for faster startup if you have
  RAM headroom.
- relbench's `Dataset.get_db` is `@lru_cache`-decorated; we
  `cache_clear()` it after `make_pkey_fkey_graph` so the raw pickle
  doesn't stay pinned for the rest of training.
- Per-rank RSS is logged at `[rss r0] pre-load <ds>` and
  `[rss r0] post-load <ds>`; grep for `\[rss` in the train log to see
  which dataset's load is the peak.
- `DatasetGraphCache.all_nodes` is now a lazy ``@property``. The
  previous eager construction materialized a Python
  ``List[Tuple[str, int]]`` with one entry per node across every type
  -- on rel-event (~100M+ nodes) that was ~8 GiB of *Python tuple
  objects* per dataset, replicated on every DDP rank, and further
  duplicated by every DataLoader worker fork (CPython ref-count writes
  break COW). It is only used by the streaming sampler's fallback
  path; ``--mode precomputed_shards`` (the launcher default) never
  touches it.
