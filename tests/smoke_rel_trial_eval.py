"""Smoke: ``task.evaluate()`` survives rel-trial string-target tasks.

Three RelBench v2 autocomplete tasks ship binary targets as ``'t'``/``'f'``
strings rather than numeric 1/0:

  * ``rel-trial.eligibilities-adult``  (target column ``adult``)
  * ``rel-trial.eligibilities-child``  (target column ``child``)
  * ``rel-trial.studies-has_dmc``      (target column ``has_dmc``)

Without coercion, ``task.evaluate(pred)`` blows up inside sklearn with
``ValueError: pos_label=1 is not a valid label. It should be one of
['f', 't']`` -- ``average_precision_score`` (the first metric for these
tasks) checks ``pos_label`` against the observed labels and rejects the
numeric default. Coercing ``'t'->1, 'f'->0`` before evaluation fixes it.

(The two other rel-trial autocomplete-ish tasks sometimes lumped in here
-- ``study-outcome`` (int32) and ``study-adverse`` (float64 regression)
-- already carry numeric targets and are *not* part of this bug.)

This script:
  1. loads each affected task (downloading the small task parquets if
     absent);
  2. coerces train / val / unmasked-test target columns to numeric via
     ``gfm_data.task_tokens.coerce_string_target_to_numeric`` -- exactly
     what the training / eval / tools entry points now do;
  3. generates uniform-random predictions and calls ``task.evaluate``
     on val and test;
  4. asserts the metric dict comes back with finite values.

To confirm the bug still bites without the fix, comment out the two
``coerce_string_target_to_numeric(...)`` calls below -- you'll get the
original ``ValueError`` from the very first task.

Run as: ``python tests/smoke_rel_trial_eval.py``. Takes well under a
minute on CPU once the rel-trial task parquets are cached; no GPU, no
DDP, no graph materialization (we never touch ``dataset.get_db()``).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from relbench.tasks import get_task

from gfm_data.task_tokens import coerce_string_target_to_numeric

# (task name, expected target column) for the three string-target tasks.
_STRING_TARGET_TASKS = [
    ("eligibilities-adult", "adult"),
    ("eligibilities-child", "child"),
    ("studies-has_dmc", "has_dmc"),
]

# Metrics RelBench attaches to these binary tasks (order matters: the
# first one to choke on string labels is average_precision).
_EXPECTED_METRIC_KEYS = {"average_precision", "accuracy", "f1", "roc_auc"}


def _check_eval(task, split: str, rng: np.random.Generator) -> dict:
    """Coerce ``split``'s target column, run a random-prediction
    ``task.evaluate`` against it, and sanity-check the result."""
    if split == "test":
        # get_table("test") masks the target column to gate users into
        # the official evaluator; mask_input_cols=False is the unmasked
        # table task.evaluate(pred) would otherwise pull internally.
        table = task.get_table("test", mask_input_cols=False)
    else:
        table = task.get_table(split)

    assert task.target_col in table.df.columns, (
        f"{split}: expected target column {task.target_col!r} present in "
        f"the unmasked table, got columns {list(table.df.columns)}"
    )
    coerce_string_target_to_numeric(table, task.target_col)

    # Post-coercion the target must be numeric 0/1 (NaNs allowed).
    tgt = table.df[task.target_col]
    nonnull = tgt.dropna()
    assert set(np.unique(nonnull.to_numpy())).issubset({0, 1}), (
        f"{split}: target column still non-binary after coercion: "
        f"{sorted(set(np.unique(nonnull.to_numpy())))}"
    )

    preds = rng.random(len(table.df))
    metrics = task.evaluate(preds, table)

    assert set(metrics.keys()) == _EXPECTED_METRIC_KEYS, (
        f"{split}: unexpected metric keys {set(metrics.keys())}"
    )
    for k, v in metrics.items():
        assert math.isfinite(float(v)), f"{split}: metric {k} is not finite: {v}"
    return metrics


def main() -> None:
    rng = np.random.default_rng(0)
    tested = 0
    skipped = []
    for tk_name, expected_target in _STRING_TARGET_TASKS:
        full_name = f"rel-trial.{tk_name}"
        print(f"--- {full_name} ---")
        try:
            task = get_task("rel-trial", tk_name, download=True)
        except Exception as e:  # offline + not cached, mostly
            print(f"  SKIP (could not load: {type(e).__name__}: {e})")
            skipped.append(full_name)
            continue

        assert task.target_col == expected_target, (
            f"{full_name}: expected target_col {expected_target!r}, "
            f"got {task.target_col!r}"
        )

        val_metrics = _check_eval(task, "val", rng)
        print(f"  val : {val_metrics}")
        test_metrics = _check_eval(task, "test", rng)
        print(f"  test: {test_metrics}")
        tested += 1

    print()
    if skipped:
        print(f"skipped {len(skipped)} task(s) (no cache / no network): "
              f"{', '.join(skipped)}")
    assert tested > 0, (
        "no rel-trial string-target task could be loaded -- expected at "
        "least one of " + ", ".join(n for n, _ in _STRING_TARGET_TASKS)
    )
    print(f"ALL REL-TRIAL EVAL SMOKE CHECKS PASSED ({tested} task(s) exercised)")


if __name__ == "__main__":
    main()
