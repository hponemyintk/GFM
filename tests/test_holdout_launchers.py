"""Sanity tests for the Phase-4 / Phase-5 holdout launcher scripts.

Heavy: actually running the scripts kicks off real pretraining + the
adoption tooling, which takes 10s of minutes per dataset. That's
left to the manual smoke (the script's own first run on an unseen
TF store).

What we DO check here: the scripts parse cleanly under bash, and the
defaults satisfy the safety invariants (HOLDOUT not in TASKS_KEEP,
script exits non-zero if the user violates that, etc.).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _bash():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    return bash


def test_holdout_task_dev_bash_syntax():
    """The launcher must pass `bash -n` (syntax-only check)."""
    bash = _bash()
    p = SCRIPTS / "holdout_task_dev.sh"
    assert p.exists(), f"missing {p}"
    rc = subprocess.run([bash, "-n", str(p)], check=False)
    assert rc.returncode == 0


def test_holdout_task_dev_rejects_holdout_in_pretrain_csv(tmp_path):
    """If PRETRAIN_TASKS_CSV (the union of all-but-holdout tasks)
    accidentally includes a holdout task, the script must reject it
    BEFORE pretraining starts. Catches the easiest user mistake."""
    bash = _bash()
    p = SCRIPTS / "holdout_task_dev.sh"
    env = os.environ.copy()
    env["DATASETS"] = "rel-f1 rel-hm"
    env["HOLDOUTS"] = "rel-f1:driver-top3 rel-hm:user-churn"
    # Deliberately INCLUDE driver-top3 in the pretrain CSV.
    env["PRETRAIN_TASKS_CSV"] = "rel-f1.driver-top3:1.0,rel-f1.driver-position:1.0,rel-hm.item-sales:1.0"
    env["OUT_DIR"] = str(tmp_path / "out")
    proc = subprocess.run(
        [bash, str(p)], env=env, check=False,
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "holdout" in (proc.stderr + proc.stdout).lower()


def test_holdout_task_dev_rejects_unknown_dataset_in_holdouts(tmp_path):
    """Unknown dataset in HOLDOUTS must exit non-zero (defaults are
    keyed by dataset name)."""
    bash = _bash()
    p = SCRIPTS / "holdout_task_dev.sh"
    env = os.environ.copy()
    env["DATASETS"] = "rel-totally-fake"
    env["HOLDOUTS"] = "rel-totally-fake:anything"
    env.pop("PRETRAIN_TASKS_CSV", None)
    env["OUT_DIR"] = str(tmp_path / "out")
    proc = subprocess.run(
        [bash, str(p)], env=env, check=False,
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "no default task list" in (proc.stderr + proc.stdout)


def test_holdout_task_dev_rejects_holdout_not_in_full_tasks(tmp_path):
    """If a HOLDOUT task isn't in the dataset's known full task list,
    the script must reject it -- otherwise the pretrain CSV would be
    ill-formed (subtracting a non-existent element)."""
    bash = _bash()
    p = SCRIPTS / "holdout_task_dev.sh"
    env = os.environ.copy()
    env["DATASETS"] = "rel-f1"
    env["HOLDOUTS"] = "rel-f1:not-a-real-task"
    env.pop("PRETRAIN_TASKS_CSV", None)
    env["OUT_DIR"] = str(tmp_path / "out")
    proc = subprocess.run(
        [bash, str(p)], env=env, check=False,
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "not in known tasks" in (proc.stderr + proc.stdout)


def test_holdout_task_dev_default_paper_safe_subset():
    """The launcher's defaults must restrict to the paper-benchmarked
    subset (stable-seed tasks) per docs/truncated_graph_caveat.md.
    Defaults pin to rel-f1 + rel-hm (both holdouts binary -> AUROC,
    consistent metric across datasets). rel-event has only one
    paper-safe task (user-ignore), so it's excluded from defaults --
    "all-but-one" pretrain has nothing left after holding it out."""
    src = (SCRIPTS / "holdout_task_dev.sh").read_text()
    # Default DATASETS -- rel-event excluded (only one paper-safe
    # task, see docs/truncated_graph_caveat.md).
    assert 'DATASETS="${DATASETS:-rel-f1 rel-hm}"' in src, (
        "default DATASETS should be the paper-safe 2-dataset subset"
    )
    # Default HOLDOUTS -- both binary, AUROC.
    assert "rel-f1:driver-top3" in src   # paper expts/run-large-base-experiments
    assert "rel-hm:user-churn" in src    # paper expts/run-large-base-experiments
    # rel-event holdout must NOT be in the default HOLDOUTS string.
    # Match against the assignment line strictly so the explanatory
    # comments naming rel-event don't false-positive.
    assert (
        'HOLDOUTS="${HOLDOUTS:-rel-f1:driver-top3 rel-hm:user-churn}"'
        in src
    ), "default HOLDOUTS should be the paper-safe 2-dataset subset"
    # rel-f1 default task list -- paper-safe stable-seed tasks only.
    assert (
        '[rel-f1]="driver-position driver-dnf driver-top3"'
    ) in src, (
        "rel-f1 default task list must be the paper-safe subset; "
        "results-position / qualifying-position / "
        "driver-circuit-compete excluded"
    )
    # rel-hm default task list -- paper-safe stable-seed tasks only.
    assert (
        '[rel-hm]="user-churn item-sales"'
    ) in src, (
        "rel-hm default task list must be the paper-safe subset; "
        "transactions-price excluded"
    )
    # rel-event entry exists (for explicit opt-in) but limited to
    # the one paper-safe task.
    assert (
        '[rel-event]="user-ignore"'
    ) in src, (
        "rel-event default must be limited to user-ignore "
        "(only paper-safe task on this dataset)"
    )


def test_holdout_task_dev_rejects_malformed_holdouts_entry(tmp_path):
    """HOLDOUTS entries must be 'dataset:task' -- malformed entries
    (no colon, empty fields) must reject."""
    bash = _bash()
    p = SCRIPTS / "holdout_task_dev.sh"
    env = os.environ.copy()
    env["DATASETS"] = "rel-f1"
    env["HOLDOUTS"] = "rel-f1-driver-top3"  # missing colon
    env["OUT_DIR"] = str(tmp_path / "out")
    proc = subprocess.run(
        [bash, str(p)], env=env, check=False,
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "malformed HOLDOUTS" in (proc.stderr + proc.stdout)


def test_holdout_dataset_eval_bash_syntax():
    """Phase-5 launcher passes syntax-only check."""
    bash = _bash()
    p = SCRIPTS / "holdout_dataset_eval.sh"
    assert p.exists(), f"missing {p}"
    rc = subprocess.run([bash, "-n", str(p)], check=False)
    assert rc.returncode == 0


def test_holdout_dataset_eval_rejects_same_source_target(tmp_path):
    """SOURCE == TARGET violates Phase-5 semantics (would be a Phase-4
    holdout-task run); the script must redirect the user."""
    bash = _bash()
    p = SCRIPTS / "holdout_dataset_eval.sh"
    env = os.environ.copy()
    env["SOURCE"] = "rel-f1"
    env["TARGET"] = "rel-f1"
    env["OUT_DIR"] = str(tmp_path / "out")
    proc = subprocess.run(
        [bash, str(p)], env=env, check=False,
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    err = proc.stderr + proc.stdout
    assert "SOURCE and TARGET must differ" in err
    assert "holdout_task_dev.sh" in err  # redirect the user


def test_holdout_dataset_eval_rejects_unknown_source(tmp_path):
    bash = _bash()
    p = SCRIPTS / "holdout_dataset_eval.sh"
    env = os.environ.copy()
    env["SOURCE"] = "rel-totally-fake"
    env["TARGET"] = "rel-hm"
    env.pop("SOURCE_TASKS_CSV", None)
    env["OUT_DIR"] = str(tmp_path / "out")
    proc = subprocess.run(
        [bash, str(p)], env=env, check=False,
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "no default SOURCE_TASKS" in (proc.stderr + proc.stdout)
