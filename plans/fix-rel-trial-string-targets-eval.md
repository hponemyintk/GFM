# Fix: ValueError during evaluation on rel-trial string-target tasks

## Status

**FIXED** — 2026-05-12.

## Symptom

DDP pretraining (and the post-hoc head / TabPFN evaluators) crash during
validation/test with:

```
ValueError: pos_label=1 is not a valid label. It should be one of ['f', 't']
```

It surfaces inside `average_precision_score` (the first metric these
tasks attach), which checks `pos_label` against the observed labels and
rejects the numeric default `1` when the labels are the strings
`'f'`/`'t'`.

## Affected tasks

Three RelBench v2 autocomplete tasks ship binary targets as `'t'`/`'f'`
**strings** in their task tables (the same three already documented in
`gfm_data/task_tokens.py` and `tests/test_string_target_coercion.py`):

- `rel-trial.eligibilities-adult`  (target column `adult`)
- `rel-trial.eligibilities-child`  (target column `child`)
- `rel-trial.studies-has_dmc`      (target column `has_dmc`)

Two other rel-trial tasks were initially lumped in here but are **not**
affected — verified directly against the cached task parquets:
`study-outcome` carries `int32` targets and `study-adverse` is a
**regression** task with `float64` targets. Both already evaluate fine.

## Root cause

RelBench's `task.get_table()` returns DataFrames with `'t'`/`'f'` string
values in the target column for the three tasks above. When
`task.evaluate()` runs, sklearn's classification metrics see string
labels alongside numeric predictions and crash.

`coerce_string_target_to_numeric` in `gfm_data/task_tokens.py` already
maps `'t'`→1, `'f'`→0 (and `yes/no/true/false`) for shard building —
`TaskTokens.__init__` calls it — but it was never run before evaluation.
And because `task.get_table` is `@lru_cache`-keyed on the *exact* call
args, the entry `TaskTokens` coerces (`get_table(split="val")`,
keyword) is a different object from the one the eval call sites read
(`get_table("val")`, positional) or the one `task.evaluate(preds)` pulls
for the test split internally (`get_table("test", mask_input_cols=False)`)
— so a one-shot coercion at load time does not reach the eval path.

## Fix

Applied at every entry point that calls `task.evaluate()`:

1. **Coerce at the eval call site (load-bearing).** Right before
   `task.evaluate(preds, table)`, fetch the table that will be scored
   against, run `coerce_string_target_to_numeric(table, task.target_col)`
   on it, and pass it explicitly. For the **test** split this means
   `task.get_table("test", mask_input_cols=False)` — `get_table("test")`
   masks the target column to gate users into the official evaluator, and
   `mask_input_cols=False` is exactly the table `task.evaluate(preds)`
   would otherwise fetch internally, just made visible so we can fix it
   up first. For the **val** split `task.get_table("val")` already
   carries the target; we just coerce it.

2. **Pre-warm coercion at task load time (belt-and-suspenders).** Right
   after the existing `task.get_table(split=...)` cache pre-warm, also
   `coerce_string_target_to_numeric(...)` the returned table. No-op on
   the test split (target column masked there). This keeps the
   `get_table(split=...)` cache entries clean for shard build too; it is
   *not* sufficient on its own (different lru_cache key from the eval
   path), which is why step 1 is the actual fix.

## Files changed

| File | What changed |
|------|--------------|
| `train_multi_task.py` | Import `coerce_string_target_to_numeric`. Pre-warm-coerce all splits in `_build_caches_and_tokens` (the existing `get_table(split=...)` loop). In `_eval()`: pick `get_table("test", mask_input_cols=False)` for the test split (else `get_table(split)`), coerce it, pass it explicitly to `task.evaluate()`. |
| `main_node_ddp.py` | Import `coerce_string_target_to_numeric`. Pre-warm-coerce all splits in `_do_load_db_and_graph` (the existing `get_table(split=...)` loop). Coerce + pass the val table explicitly at both `task.evaluate(..., get_table("val"))` sites; for the final test eval, fetch `get_table("test", mask_input_cols=False)`, coerce it, and pass it instead of relying on `task.evaluate(preds)`'s internal fetch. |
| `tools/finetune_head.py` | In `_final_evaluate()`: import `coerce_string_target_to_numeric`, fetch `get_table("test", mask_input_cols=False)`, coerce it, size `full_preds` from `len(test_table.df)`, pass the table to `task.evaluate()`. Updated the helper docstring + the `main()` comment. |
| `tools/tabpfn_eval.py` | Same change as `tools/finetune_head.py`'s `_final_evaluate()`. |
| `tests/test_finetune_head.py`, `tests/test_tabpfn_eval.py` | The `_final_evaluate` mock contract changed (now `get_table(split, mask_input_cols=...)`, a table with a real `.df`, `target_col`, and `evaluate(preds, table)`). Added a `_make_fake_task(n_test, metrics)` helper whose unmasked test table carries `'t'`/`'f'` strings and whose `evaluate` asserts the caller coerced them to numeric first — turning these into regression tests for the fix. |
| `tests/smoke_rel_trial_eval.py` | New standalone smoke (see below). |

## Smoke test

To verify the fix on the affected tasks without running full training:

1. `python tests/smoke_rel_trial_eval.py`
2. Loads each of the 3 rel-trial string-target tasks (downloading the
   small task parquets if absent), coerces the train / val /
   unmasked-test target columns via `coerce_string_target_to_numeric`,
   generates random predictions, and calls `task.evaluate(preds, table)`
   on val and test — asserting the metric dict comes back with finite
   values. Skips (without failing) any task that can't be loaded offline,
   as long as ≥1 was exercised.
3. Completes in well under a minute on CPU once the task parquets are
   cached; no GPU, no DDP, no `dataset.get_db()`.
4. To confirm the bug still bites without the fix, comment out the
   `coerce_string_target_to_numeric(...)` call inside `_check_eval` —
   you'll get the original `ValueError` from the very first task.

Also covered:
- `tests/test_string_target_coercion.py` — unit tests for the helper
  (unchanged; all pass).
- `tests/test_finetune_head.py`, `tests/test_tabpfn_eval.py` — the
  updated mocks exercise the new `_final_evaluate` contract.

## Related

- `logs/fix-rel-trial-string-targets.md` — same root cause but during
  shard building (phase 2), not evaluation (phase 3).
- `logs/fix-squeeze-batch-size-one.md` — the squeeze fix that let
  training reach the evaluation phase where this bug surfaced.

## Note: pre-existing unrelated test failures

Running the full `tests/` suite shows ~28 failures (`AttributeError` from
`torch_geometric` being a `MagicMock`). These are a pre-existing
test-isolation issue — `tests/conftest.py` installs MagicMock stubs for
`torch_geometric` and they leak between test files depending on
collection order, which is why ~12 test files carry an
`if isinstance(sys.modules.get("torch_geometric"), _MagicMock): ...`
un-mock guard. The failure set is identical with and without this fix
(verified via `git stash`); not in scope here.
