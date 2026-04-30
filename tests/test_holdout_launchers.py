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

    Defaults pin to rel-f1 + rel-event with one binary holdout
    (rel-f1:driver-top3) + one regression holdout
    (rel-event:user-attendance), so the frozen-backbone GFM claim
    is tested on both head types simultaneously."""
    src = (SCRIPTS / "holdout_task_dev.sh").read_text()
    # Default DATASETS -- rel-f1 + rel-event for mixed-metric holdouts.
    assert 'DATASETS="${DATASETS:-rel-f1 rel-event}"' in src, (
        "default DATASETS should be rel-f1 + rel-event"
    )
    # Default HOLDOUTS -- one binary + one regression.
    assert "rel-f1:driver-top3" in src   # paper expts/run-large-base-experiments
    assert "rel-event:user-attendance" in src  # paper Table 1a (MAE 0.2502)
    # Match against the assignment line strictly so the explanatory
    # comments don't false-positive.
    assert (
        'HOLDOUTS="${HOLDOUTS:-rel-f1:driver-top3 '
        'rel-event:user-attendance}"'
    ) in src, "default HOLDOUTS must be the binary+regression pair"
    # rel-f1 default task list -- 3 stable-seed paper tasks PLUS the
    # 2 autocomplete regression tasks (results-position,
    # qualifying-position) that need FULL_GRAPH=1 to build cleanly.
    # driver-circuit-compete is link-prediction, kept out of defaults.
    assert (
        '[rel-f1]="driver-position driver-dnf driver-top3 '
        'results-position qualifying-position"'
    ) in src, (
        "rel-f1 default task list must include both stable-seed and "
        "autocomplete tasks; FULL_GRAPH=1 default makes the "
        "autocomplete tasks build cleanly"
    )
    # FULL_GRAPH default must be 1 (enabled).
    assert 'FULL_GRAPH="${FULL_GRAPH:-1}"' in src, (
        "FULL_GRAPH must default to 1 so autocomplete tasks "
        "(results-position, qualifying-position) build cleanly"
    )
    # rel-event default task list -- 3 paper-benchmarked tasks
    # (user-attendance regression, user-repeat / user-ignore binary)
    # PLUS 3 autocomplete tasks that need FULL_GRAPH=1
    # (event_interest-{interested,not_interested} binary,
    # users-birthyear regression). Confirmed runnable end-to-end on
    # p4d in commit e6aa7a3 (~8h wall-time for the 11-task pretrain).
    assert (
        '[rel-event]="user-attendance user-repeat user-ignore '
        'event_interest-interested event_interest-not_interested '
        'users-birthyear"'
    ) in src, (
        "rel-event default task list must include all 6 task variants "
        "(3 paper-safe + 3 autocomplete enabled by FULL_GRAPH=1)"
    )
    # rel-hm task list still available for opt-in via DATASETS override.
    assert (
        '[rel-hm]="user-churn item-sales"'
    ) in src, (
        "rel-hm default task list must be the paper-safe subset; "
        "transactions-price excluded"
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


def test_holdout_dataset_eval_default_multi_source_to_relarxiv():
    """Defaults: pretrain on rel-f1 + rel-event multi-source, adopt to
    rel-arxiv. FULL_GRAPH=1 by default. Pins the p4d-ready config."""
    src = (SCRIPTS / "holdout_dataset_eval.sh").read_text()
    assert 'SOURCE="${SOURCE:-rel-f1 rel-event}"' in src, (
        "default SOURCE should be the multi-source rel-f1 + rel-event"
    )
    assert 'TARGET="${TARGET:-rel-arxiv}"' in src, (
        "default TARGET should be rel-arxiv"
    )
    assert 'FULL_GRAPH="${FULL_GRAPH:-1}"' in src, (
        "FULL_GRAPH default must be 1 so rel-event autocomplete tasks "
        "(users-birthyear / event_interest-*) build cleanly"
    )
    # rel-event lookup includes all 6 entity tasks (3 paper-safe + 3
    # autocomplete unlocked by FULL_GRAPH=1).
    for tn in ("user-attendance", "user-repeat", "user-ignore",
               "event_interest-interested",
               "event_interest-not_interested", "users-birthyear"):
        assert f'"rel-event.{tn}:1.0"' in src, (
            f"rel-event default tasks must include {tn!r}"
        )
    # rel-arxiv lookup defined (paper-citation + author-publication).
    assert '"rel-arxiv.paper-citation:1.0"' in src
    assert '"rel-arxiv.author-publication:1.0"' in src


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
