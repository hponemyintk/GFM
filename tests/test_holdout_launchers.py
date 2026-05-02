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
    # rel-hm task list available for opt-in via DATASETS override.
    # transactions-price is the RelBench v2 autocomplete regression
    # and joins under FULL_GRAPH=1 (default).
    assert (
        '[rel-hm]="user-churn item-sales transactions-price"'
    ) in src, (
        "rel-hm default task list must include all 3 entity binary/"
        "regression tasks (transactions-price unlocked by FULL_GRAPH=1)"
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


def test_holdout_dataset_eval_default_loo_to_relevent():
    """Defaults: leave-one-dataset-out across all 9 supported RelBench v2
    datasets, with TARGET=rel-event (6 tasks for richer adoption-side
    eval). FULL_GRAPH=1 by default. Pins the p4d-ready config."""
    src = (SCRIPTS / "holdout_dataset_eval.sh").read_text()
    assert 'TARGET="${TARGET:-rel-event}"' in src, (
        "default TARGET should be rel-event for the 6-task LOO holdout"
    )
    # SUPPORTED_DATASETS lists every dataset whose entity binary/
    # regression tasks the adoption pipeline supports today.
    assert (
        'SUPPORTED_DATASETS="${SUPPORTED_DATASETS:-rel-amazon rel-avito '
        'rel-event rel-f1 rel-hm rel-stack rel-trial rel-arxiv '
        'rel-ratebeer}"'
    ) in src, (
        "SUPPORTED_DATASETS must list all 9 RelBench v2 datasets with "
        "supported entity binary/regression tasks (rel-mimic and "
        "rel-salt have only multiclass / autocomplete-cls / rec tasks)"
    )
    # SOURCE auto-fills to "all SUPPORTED_DATASETS minus TARGET" when
    # the user doesn't set it explicitly.
    assert 'if [ -z "${SOURCE:-}" ]; then' in src, (
        "SOURCE must auto-fill to all-but-TARGET when unset (LOO default)"
    )
    assert 'if [ "$_ds" != "$TARGET" ]; then' in src, (
        "LOO loop must skip TARGET when building auto SOURCE list"
    )
    assert 'FULL_GRAPH="${FULL_GRAPH:-1}"' in src, (
        "FULL_GRAPH default must be 1 so autocomplete tasks across the "
        "LOO source set (transactions-price, results-position, etc.) "
        "build cleanly"
    )
    # rel-event (default TARGET) lookup includes all 6 entity tasks.
    for tn in ("user-attendance", "user-repeat", "user-ignore",
               "event_interest-interested",
               "event_interest-not_interested", "users-birthyear"):
        assert f'"rel-event.{tn}:1.0"' in src, (
            f"rel-event default tasks must include {tn!r}"
        )
    # Per-dataset task lookups must cover every supported entity
    # binary/regression task per relbench. Pinned so future relbench
    # additions / removals are caught at test time.
    EXPECTED = {
        "rel-f1": [
            "driver-position", "driver-dnf", "driver-top3",
            "results-position", "qualifying-position",
        ],
        "rel-hm": ["user-churn", "item-sales", "transactions-price"],
        "rel-arxiv": ["paper-citation", "author-publication"],
        "rel-amazon": [
            "user-churn", "item-churn", "user-ltv", "item-ltv",
        ],
        "rel-avito": [
            "ad-ctr", "user-visits", "user-clicks",
            "searchstream-click", "searchinfo-isuserloggedon",
        ],
        "rel-stack": ["user-engagement", "user-badge", "post-votes"],
        "rel-trial": [
            "study-outcome", "study-adverse", "site-success",
            "studies-enrollment", "studies-has_dmc",
            "eligibilities-adult", "eligibilities-child",
        ],
        "rel-ratebeer": [
            "beer-churn", "user-churn", "brewer-dormant",
            "user-count", "beer_ratings-total_score",
        ],
    }
    for ds, tasks in EXPECTED.items():
        for tn in tasks:
            assert f'"{ds}.{tn}:1.0"' in src, (
                f"{ds} default tasks must include {tn!r}"
            )


def test_holdout_dataset_eval_runs_extracts_in_parallel():
    """Adoption phase must dispatch up to NPROC concurrent extract jobs,
    each on its own GPU and with its own --precomputed_dir so the
    per-row HDF5 caches don't race. Pinned via source-grep so a
    revert to the sequential path fails the test."""
    src = (SCRIPTS / "holdout_dataset_eval.sh").read_text()
    # Phase markers.
    assert "Phase A: parallel extracts" in src, (
        "adoption phase must announce a parallel extract phase"
    )
    assert "Phase B: finetune_head + tabpfn_eval" in src, (
        "adoption phase must run finetune + tabpfn after extracts"
    )
    # GPU dispatcher primitives.
    assert "_acquire_gpu" in src and "_reap_finished" in src, (
        "GPU pool dispatcher must be defined"
    )
    assert 'CUDA_VISIBLE_DEVICES="$GPU"' in src, (
        "each parallel extract must pin to a specific GPU"
    )
    # Per-(task, seed) precomputed_dir override prevents shared HDF5 races.
    assert '--precomputed_dir "$PRECOMP_DIR"' in src, (
        "extract_embeddings must be invoked with a per-seed "
        "--precomputed_dir so concurrent jobs don't race on the "
        "shared HDF5 cache"
    )
    assert 'PRECOMP_DIR="$SEED_DIR/precomputed"' in src, (
        "PRECOMP_DIR must scope under the per-seed dir"
    )
    # Regression guard: _acquire_gpu MUST run in the parent shell, not
    # in a $() subshell. A subshell would mutate its own copy of
    # GPU_POOL / PID_GPU and the parent's pool would never drain --
    # every job would land on gpu=0. The function therefore returns
    # the acquired index via the global ACQUIRED_GPU.
    #
    # Strip comment lines before grep-checking so the cautionary note
    # in the function's own docstring doesn't false-positive.
    code_only = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "$(_acquire_gpu)" not in code_only, (
        "_acquire_gpu must NOT be called via $() -- the subshell would "
        "make every job land on gpu=0. Call directly and read "
        "ACQUIRED_GPU."
    )
    assert 'GPU="$ACQUIRED_GPU"' in code_only, (
        "caller must read the acquired index from the global "
        "ACQUIRED_GPU set by _acquire_gpu"
    )


def test_holdout_task_dev_runs_extracts_in_parallel():
    """Phase-4 launcher must dispatch extracts to NPROC GPUs in
    parallel (mirrors the holdout_dataset_eval.sh design). Different
    (ds, holdout) pairs already have isolated cache dirs so we don't
    need a per-job --precomputed_dir override here."""
    src = (SCRIPTS / "holdout_task_dev.sh").read_text()
    code_only = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "Phase A: parallel extracts" in src
    assert "Phase B: finetune_head" in src
    assert "_acquire_gpu" in src and "_reap_finished" in src
    assert 'CUDA_VISIBLE_DEVICES="$GPU"' in src, (
        "each parallel extract must pin to a specific GPU"
    )
    # Same subshell-trap regression check as the dataset launcher.
    assert "$(_acquire_gpu)" not in code_only, (
        "_acquire_gpu must NOT be wrapped in $() -- subshell would "
        "make every job land on gpu=0"
    )
    assert 'GPU="$ACQUIRED_GPU"' in code_only


def test_finetune_head_defaults_to_mlp2_in_both_launchers():
    """Both launchers must default --head to mlp2 (2-layer MLP). Linear
    leaves predictive headroom on 128/512-d frozen embeddings; mlp2
    fits a richer decision boundary at negligible cost. Caller can
    still flip back to FT_HEAD=linear for ablations."""
    for fn in ("holdout_task_dev.sh", "holdout_dataset_eval.sh"):
        src = (SCRIPTS / fn).read_text()
        assert 'FT_HEAD="${FT_HEAD:-mlp2}"' in src, (
            f"{fn} must default FT_HEAD to mlp2"
        )
        assert '--head "$FT_HEAD"' in src, (
            f"{fn} must pass --head \"$FT_HEAD\" so the env override "
            f"actually flows into finetune_head"
        )


def test_extract_embeddings_supports_precomputed_dir_flag():
    """The --precomputed_dir override is wired in argparse so the
    launcher can run several extracts in parallel without sharing
    HDF5 cache directories."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ee", str(SCRIPTS.parent / "tools" / "extract_embeddings.py"),
    )
    src = (SCRIPTS.parent / "tools" / "extract_embeddings.py").read_text()
    assert '--precomputed_dir' in src
    assert "_resolve_precomputed_dir" in src, (
        "_resolve_precomputed_dir helper must exist so callers can "
        "verify the override path is honored"
    )


def test_holdout_dataset_eval_clean_excludes_low_quality_tasks():
    """The 'clean' variant must drop the 5 RelBench v2 tasks whose
    supervised single-task GNN baseline is at-or-below random per the
    paper. The exclusions are kept as commented-out lines in the
    arrays (with baseline rationale) so an ablation is one-line; the
    test enforces that the lines are commented out, not deleted."""
    p = SCRIPTS / "holdout_dataset_eval_clean.sh"
    assert p.exists(), f"missing {p}"
    src = p.read_text()
    # Strip lines whose first non-space char is '#' so we only check
    # the live (uncommented) array entries.
    code_only = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    EXCLUDED = [
        "rel-event.event_interest-interested",
        "rel-event.event_interest-not_interested",
        "rel-event.users-birthyear",
        "rel-trial.site-success",
        "rel-amazon.item-ltv",
    ]
    for tn in EXCLUDED:
        assert f'"{tn}:1.0"' not in code_only, (
            f"clean variant must NOT include {tn!r} as a live array "
            "entry -- paper baseline is at or below random. Comment "
            "the line out (with the paper rationale inline) instead "
            "of leaving it active."
        )
        # And: the line must STILL exist in the file as a comment, so
        # the rationale stays visible and re-enabling is one tweak.
        assert f'"{tn}:1.0"' in src, (
            f"{tn!r} should remain in the file as a commented-out "
            "line so the exclusion rationale is visible inline"
        )
    # Sanity: the kept rel-event tasks (3 user-* forecasting) and
    # kept rel-trial tasks must still be present.
    KEPT = [
        "rel-event.user-attendance",
        "rel-event.user-repeat",
        "rel-event.user-ignore",
        "rel-trial.study-outcome",
        "rel-trial.study-adverse",
        "rel-amazon.user-churn",
        "rel-amazon.item-churn",
        "rel-amazon.user-ltv",
    ]
    for tn in KEPT:
        assert f'"{tn}:1.0"' in src, (
            f"clean variant should keep {tn!r}"
        )


def test_holdout_dataset_eval_pretrain_sweep_wraps_inner_launcher():
    """The pretrain-seed sweep wrapper must (a) exist and pass bash
    syntax, (b) loop over PRETRAIN_SEEDS, (c) pass each seed via
    SEED= to the inner launcher, (d) namespace OUT_DIR per pretrain
    seed so the inner launcher's artifacts don't collide, and (e)
    aggregate per-task metrics into <out>/aggregate.json."""
    bash = _bash()
    p = SCRIPTS / "holdout_dataset_eval_pretrain_sweep.sh"
    assert p.exists(), f"missing {p}"
    rc = subprocess.run([bash, "-n", str(p)], check=False)
    assert rc.returncode == 0
    src = p.read_text()
    assert 'PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-0 1 2}"' in src, (
        "default PRETRAIN_SEEDS must be 0 1 2"
    )
    # Inner launcher invocation must forward the pretrain seed via
    # SEED= and namespace OUT_DIR= per seed.
    assert "for PSEED in $PRETRAIN_SEEDS" in src
    assert 'SEED="$PSEED"' in src, (
        "wrapper must forward each pretrain seed as SEED= to inner"
    )
    assert 'OUT_DIR="$PSEED_DIR"' in src, (
        "wrapper must namespace OUT_DIR per pretrain seed"
    )
    # Aggregate output.
    assert 'AGG="$SWEEP_DIR/aggregate.json"' in src, (
        "aggregate path must be <sweep>/aggregate.json"
    )
    # Default INNER must point at the "clean" launcher so the sweep
    # uses the high-signal task subset by default. Override INNER to
    # the unfiltered launcher for the all-tasks ablation.
    assert 'INNER="${INNER:-scripts/holdout_dataset_eval_clean.sh}"' in src, (
        "default INNER must be holdout_dataset_eval_clean.sh"
    )
    # Every per-trial RNG must be tied to the pretrain seed:
    #   SEED  -> pretrain --seed (model init / shuffle / sampler)
    #   PYTHONHASHSEED -> hash() inside the per-row seed_val
    #                    derivation, set->list iteration order
    #   SHARDS_SUBDIR  -> namespaces precomputed shards under
    #                    $CACHE/shards[_full]/pretrain_seedN/ so
    #                    each trial rebuilds its neighbor list
    #                    from scratch (rather than reusing trial 0's)
    assert 'PYTHONHASHSEED="$PSEED"' in src, (
        "wrapper must set PYTHONHASHSEED per pretrain seed so the "
        "per-row seed_val + set->list ordering inside the sampler "
        "is deterministically tied to the trial seed"
    )
    assert 'SHARDS_SUBDIR="pretrain_seed${PSEED}"' in src, (
        "wrapper must namespace the precomputed shard tree per "
        "pretrain seed so each trial actually rebuilds neighbor "
        "lists from scratch"
    )


def test_pretrain_only_seed_sweep_no_adoption():
    """The pretrain-only seed sweep wrapper must (a) exist and pass
    bash syntax, (b) loop over PRETRAIN_SEEDS, (c) per-trial set
    SEED + PYTHONHASHSEED + SHARDS_SUBDIR derived from the trial
    seed, (d) call pretrain_p4d.sh directly (no adoption launcher
    -- this is pretrain-only). Aggregate path is
    <sweep>/aggregate.json."""
    bash = _bash()
    p = SCRIPTS / "pretrain_only_seed_sweep.sh"
    assert p.exists(), f"missing {p}"
    rc = subprocess.run([bash, "-n", str(p)], check=False)
    assert rc.returncode == 0
    src = p.read_text()
    code_only = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert 'PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-0 1 2}"' in src
    assert "for PSEED in $PRETRAIN_SEEDS" in code_only
    assert 'PYTHONHASHSEED="$PSEED"' in code_only
    assert 'SEED="$PSEED"' in code_only
    assert 'SHARDS_SUBDIR="pretrain_seed${PSEED}"' in code_only
    # Must invoke pretrain_p4d.sh directly, not a Phase-5 wrapper --
    # that's the whole point of "pretrain-only".
    assert 'bash "$REPO_ROOT/scripts/pretrain_p4d.sh"' in code_only, (
        "pretrain-only sweep must call pretrain_p4d.sh directly, "
        "not a Phase-5 adoption launcher"
    )
    # No adoption / TARGET / extract_embeddings invocations.
    assert "extract_embeddings" not in code_only
    assert "holdout_dataset_eval" not in code_only
    assert 'AGG="$SWEEP_DIR/aggregate.json"' in code_only


def test_pretrain_p4d_supports_shards_subdir_namespace():
    """pretrain_p4d.sh must accept a SHARDS_SUBDIR env knob that
    appends a per-trial subdir under $CACHE/shards[_full]/. Used by
    the backbone-variance sweep so each trial's neighbor list is
    rebuilt from scratch instead of reusing the first trial's cache."""
    src = (SCRIPTS / "pretrain_p4d.sh").read_text()
    assert 'SHARDS_SUBDIR="${SHARDS_SUBDIR:-}"' in src
    assert 'SHARDS="$SHARDS/$SHARDS_SUBDIR"' in src, (
        "pretrain_p4d.sh must concatenate SHARDS_SUBDIR onto SHARDS "
        "when the env var is set"
    )


def test_holdout_dataset_eval_forwards_shards_subdir_to_pretrain_p4d():
    """Both Phase-5 launchers must forward SHARDS_SUBDIR through the
    env block to pretrain_p4d.sh, otherwise the sweep wrapper's
    per-trial namespace gets dropped on the way down."""
    for fn in ("holdout_dataset_eval.sh", "holdout_dataset_eval_clean.sh"):
        src = (SCRIPTS / fn).read_text()
        assert 'SHARDS_SUBDIR="${SHARDS_SUBDIR:-}" \\' in src, (
            f"{fn} must forward SHARDS_SUBDIR to pretrain_p4d.sh "
            "so the sweep wrapper's per-trial shard namespace is "
            "actually applied"
        )


def test_pretrain_p4d_forwards_seed_to_torchrun():
    """pretrain_p4d.sh must pass --seed to main_node_ddp.py so the
    backbone-variance sweep actually varies the pretrain RNG."""
    src = (SCRIPTS / "pretrain_p4d.sh").read_text()
    assert 'SEED="${SEED:-' in src, (
        "pretrain_p4d.sh must accept SEED as an env var"
    )
    assert '--seed "$SEED"' in src, (
        "pretrain_p4d.sh must thread SEED into the torchrun --seed flag"
    )


def test_holdout_dataset_eval_forwards_seed_to_pretrain_p4d():
    """The Phase-5 launcher's p4d branch delegates pretrain to
    pretrain_p4d.sh via env. SEED must be in that env block,
    otherwise the backbone always trains with main_node_ddp.py's
    default seed and the pretrain-sweep wrapper has no effect."""
    for fn in ("holdout_dataset_eval.sh", "holdout_dataset_eval_clean.sh"):
        src = (SCRIPTS / fn).read_text()
        assert 'SEED="$SEED" \\' in src, (
            f"{fn} must forward SEED into the pretrain_p4d.sh env "
            "block (not just inline laptop torchrun)"
        )


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
