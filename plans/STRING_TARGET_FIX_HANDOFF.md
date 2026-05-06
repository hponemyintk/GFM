# Handoff: rel-trial string-target fix

**Branch:** `gfm_test_using-table_agnostic_model-branch`
**Started:** 2026-05-03
**Source briefs:** `~/Downloads/reexternalsenderrecommands/fix-rel-trial-string-targets.md`,
`~/Downloads/reexternalsenderrecommands/bugfix_wait_all_set_e.md`

## TL;DR

- ✅ Bash `set -e` fix already committed in `258cbfb` — nothing more to do.
- ⚠️ String-target fix from brief had bugs (incomplete coverage + NaN ambiguity).
  Refactored as a shared helper applied at **both** call sites.
- 🔲 **Verification still pending** — pytest hangs on M1 import chain.
  Need to verify on the CUDA cluster.

## Review findings (full notes from this session)

### Bash fix (`bugfix_wait_all_set_e.md`) — DONE
- All 6 expected `|| true` additions verified at:
  - `scripts/holdout_dataset_eval_clean.sh:588,597`
  - `scripts/holdout_dataset_eval.sh:568,577`
  - `scripts/holdout_task_dev.sh:479,488`
- Brief mentions an "ad-hoc edit not yet committed" in clean.sh — that note
  is **stale**, the commit covers all three.

### String-target fix (`fix-rel-trial-string-targets.md` Option A) — bugs found

1. **CRITICAL — incomplete coverage.** Brief patches only
   `tools/precompute_shards.py:219`. But `get_node_train_table_input` is
   also called at:
   - `gfm_data/task_tokens.py:210` (active `TaskTokens` class — pretrain,
     finetune, main DDP all hit this)
   - `utils.py:331` (legacy `RelGTTokens` — confirmed dead, no instantiations)
   - `tests/smoke_rel_f1.py:100`, `tests/smoke_pr2_modes.py:112`

   Mutating `table.df[col]` does NOT persist across processes (parquet on
   disk is the source of truth via `task.get_table()` LRU cache). So even
   after Option A, the 3 affected tasks would crash again at
   `task_tokens.py:210` when training starts.

2. **MEDIUM — NaN false-positive.** Brief checks
   `if table.df[col].isna().any()` after `.map()`, but `.map` returns NaN
   for both unmapped strings AND pre-existing NaN. Need to compare:
   `unmapped = original.notna() & mapped.isna()`.

3. **MINOR.** Proposed snippet redeclares `table = task.get_table(split)`
   redundantly with line 218. (LRU-cached so harmless but noisy.)

### What was applied

Created shared helper `coerce_string_target_to_numeric` in
`gfm_data/task_tokens.py` and called it at both relevant call sites.
Skipped `utils.py:331` since `RelGTTokens` is dead code.

## Files changed (uncommitted)

```
 M gfm_data/task_tokens.py     (+32 lines: helper + 1 call site)
 M tools/precompute_shards.py  (+2 lines: import + call)
?? tests/test_string_target_coercion.py  (new, 8 tests)
```

### Diff summary

**`gfm_data/task_tokens.py`** — added module-level helper after the
`TASK_TYPE_*` constants:

```python
_STRING_TARGET_MAP = {"t": 1, "f": 0, "yes": 1, "no": 0, "true": 1, "false": 0}

def coerce_string_target_to_numeric(table, target_col: str) -> None:
    col = table.df[target_col]
    if col.dtype != object:
        return
    mapped = col.map(_STRING_TARGET_MAP)
    unmapped = col.notna() & mapped.isna()
    if unmapped.any():
        bad = col[unmapped].unique().tolist()
        raise ValueError(
            f"Target column {target_col!r} has unmapped string values: {bad}. "
            f"Add mappings to _STRING_TARGET_MAP. "
            f"Known keys: {sorted(_STRING_TARGET_MAP.keys())}"
        )
    table.df[target_col] = mapped
```

And call it in `TaskTokens.__init__` right after `self.table = task.get_table(...)`:

```python
self.table = task.get_table(split=split)
coerce_string_target_to_numeric(self.table, task.target_col)  # NEW
self.table_input = get_node_train_table_input(self.table, task)
```

**`tools/precompute_shards.py`** — added import alongside other `gfm_data`
imports and call before `get_node_train_table_input`:

```python
from gfm_data.task_tokens import coerce_string_target_to_numeric  # NEW

# in precompute_split():
table = task.get_table(split)
coerce_string_target_to_numeric(table, task.target_col)  # NEW
table_input = get_node_train_table_input(table, task)
```

**`tests/test_string_target_coercion.py`** — 8 tests:
- numeric column noop, int column noop
- `'t'/'f'` → 1/0 mapping
- `'yes'/'no'` mapping
- pre-existing NaN preserved alongside string targets (the brief's bug)
- unmapped string raises with clear message
- idempotent on second call
- doesn't touch other columns

## What's still pending

### 1. Run the unit tests on a working environment

On M1 Mac the import chain hangs (per `tests/conftest.py` comment about
`torch_geometric` JIT compilation). On the CUDA cluster:

```bash
source relgt_env/bin/activate
python -m pytest tests/test_string_target_coercion.py -v
```

Expect 8 passing tests.

### 2. End-to-end verification: rebuild one of the 3 failed shards

```bash
python tools/precompute_shards.py \
  --dataset rel-trial --task eligibilities-child \
  --K 300 --shard_size 50000 \
  --out_dir shards/rel-trial/eligibilities-child \
  --splits train val test
```

Should complete without `ValueError: could not convert string to float: 'f'`
and produce `.done` sentinels.

### 3. Verify training-phase fix

Run a short pretrain on `rel-trial` including one of the 3 string-target
tasks. The `TaskTokens.__init__` patch should prevent the same crash in
the training process. Suggested smoke:

```bash
# Or whatever the standard rel-trial pretrain command is — point is that
# main_node_ddp.py constructs TaskTokens for one of the 3 affected tasks.
```

### 4. Commit

If all three verifications pass, commit message draft:

```
Fix rel-trial string-target tasks crashing relbench's astype(float)

Three RelBench v2 autocomplete tasks (rel-trial eligibilities-child,
eligibilities-adult, studies-has_dmc) ship binary targets as 't'/'f'
strings rather than numeric 1/0. relbench.modeling.graph.get_node_train_table_input
unconditionally calls .astype(float) and crashes.

Add coerce_string_target_to_numeric helper in gfm_data/task_tokens.py
that maps strings -> ints in place, preserving existing NaN. Apply it
at both call sites (TaskTokens.__init__ for the training path, and
tools/precompute_shards.py for shard build) since table.df mutation
does not persist across processes.

Files:
  gfm_data/task_tokens.py
  tools/precompute_shards.py
  tests/test_string_target_coercion.py (new, 8 tests)
```

## Known M1 quirk encountered

`pytest tests/test_string_target_coercion.py` hangs at "collecting ..." on
M1 Mac. Reason: `gfm_data.task_tokens` imports `gfm_data.graph_cache` which
pulls in real `torch_geometric` JIT compilation. The `tests/conftest.py`
mocks help for some test files but not when the SUT module itself imports
`graph_cache` directly.

Workaround attempts (incomplete) — both pytest invocations were killed
after several minutes with no output beyond "collecting ...". Did **not**
prove tests fail; just that the M1 environment can't run them quickly.
Run on CUDA cluster instead.

## References within repo

- `MEMORY.md` — project overview
- `memories/codebase_status.md` — broader pending work
- `memories/m1_mac_compatibility.md` — context on the M1 import hang
