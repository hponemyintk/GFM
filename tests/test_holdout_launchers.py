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


def test_holdout_task_dev_rejects_holdout_in_tasks_keep(tmp_path):
    """If TASKS_KEEP_CSV includes the HOLDOUT task, the script must
    exit non-zero before launching pretraining (catches the easiest
    user mistake)."""
    bash = _bash()
    p = SCRIPTS / "holdout_task_dev.sh"
    env = os.environ.copy()
    env["DATASET"] = "rel-f1"
    env["HOLDOUT"] = "driver-top3"
    # Deliberately INCLUDE driver-top3 -- script should reject this.
    env["TASKS_KEEP_CSV"] = "rel-f1.driver-top3:1.0,rel-f1.driver-position:1.0"
    env["OUT_DIR"] = str(tmp_path / "out")
    # Fast-fail: the script reaches the holdout-vs-keep check before
    # any expensive setup.
    proc = subprocess.run(
        [bash, str(p)], env=env, check=False,
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "HOLDOUT task" in (proc.stderr + proc.stdout)


def test_holdout_task_dev_rejects_unknown_dataset(tmp_path):
    """Unknown DATASET with no explicit TASKS_KEEP_CSV must exit non-zero."""
    bash = _bash()
    p = SCRIPTS / "holdout_task_dev.sh"
    env = os.environ.copy()
    env["DATASET"] = "rel-totally-fake"
    env["HOLDOUT"] = "anything"
    env.pop("TASKS_KEEP_CSV", None)
    env["OUT_DIR"] = str(tmp_path / "out")
    proc = subprocess.run(
        [bash, str(p)], env=env, check=False,
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "no default TASKS_KEEP" in (proc.stderr + proc.stdout)


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
