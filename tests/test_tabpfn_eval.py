"""Unit tests for tools/tabpfn_eval.py.

Tabpfn isn't necessarily installed in CI; the fit/predict step is
mocked. The IO contract -- load embeddings, optional projection,
mock fit + predict, scatter through task.evaluate, save JSON -- is
what these tests guard.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock as _MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if isinstance(sys.modules.get("torch_geometric"), _MagicMock):
    for _k in [k for k in sys.modules if k.startswith("torch_geometric")]:
        del sys.modules[_k]
    import torch_geometric  # noqa: F401


def _make_fake_task(n_test: int, metrics: dict):
    """Plain-object stand-in for a RelBench ``EntityTask`` whose
    *unmasked* test table carries ``'t'``/``'f'`` string targets
    (rel-trial autocomplete style). ``get_table`` accepts -- and
    ignores -- ``mask_input_cols``; ``evaluate`` asserts the caller
    coerced the target column to numeric first (i.e. ``_final_evaluate``
    ran ``coerce_string_target_to_numeric``) and then returns
    ``metrics``."""
    import pandas as _pd

    tbl = type("Tbl", (), {})()
    tbl.df = _pd.DataFrame(
        {"y": np.resize(np.array(["t", "f"], dtype=object), n_test)}
    )

    def _evaluate(preds, target_table=None):
        col = (target_table if target_table is not None else tbl).df["y"]
        assert col.dtype != object, (
            "task.evaluate() got un-coerced 't'/'f' string targets -- "
            "_final_evaluate must coerce them first"
        )
        assert len(preds) == n_test, f"pred length {len(preds)} != {n_test}"
        return dict(metrics)

    task = type("T", (), {})()
    task.target_col = "y"
    task.get_table = lambda split, mask_input_cols=None: tbl
    task.evaluate = _evaluate
    return task


def test_infer_task_kind_int_labels():
    from tools.tabpfn_eval import _infer_task_kind
    assert _infer_task_kind(np.array([0, 1, 0, 1, 0], dtype=np.int64)) == "binary"


def test_infer_task_kind_float_binary():
    from tools.tabpfn_eval import _infer_task_kind
    assert _infer_task_kind(np.array([0.0, 1.0, 0.0])) == "binary"


def test_infer_task_kind_regression():
    from tools.tabpfn_eval import _infer_task_kind
    assert _infer_task_kind(np.array([1.5, 2.7, 3.3])) == "regression"


def test_maybe_project_none_passthrough():
    """projector='none' returns the inputs unchanged."""
    from tools.tabpfn_eval import _maybe_project
    train = np.random.randn(8, 32).astype(np.float32)
    val = np.random.randn(2, 32).astype(np.float32)
    test = np.random.randn(2, 32).astype(np.float32)
    t, v, te = _maybe_project(train, val, test, kind="none")
    np.testing.assert_array_equal(t, train)
    np.testing.assert_array_equal(v, val)
    np.testing.assert_array_equal(te, test)


def test_maybe_project_pca64_dim_reduction():
    """PCA projector reduces channel dim to min(64, channels)."""
    from tools.tabpfn_eval import _maybe_project
    rng = np.random.default_rng(0)
    train = rng.standard_normal((128, 256)).astype(np.float32)
    val = rng.standard_normal((32, 256)).astype(np.float32)
    test = rng.standard_normal((32, 256)).astype(np.float32)
    t, v, te = _maybe_project(train, val, test, kind="pca64")
    assert t.shape == (128, 64)
    assert v.shape == (32, 64)
    assert te.shape == (32, 64)


def test_maybe_project_pca_no_leakage():
    """PCA must be fit on TRAIN ONLY -- val/test transform should
    use the same components. Differences between fitting on train
    vs (train + val) would indicate leakage.
    """
    from tools.tabpfn_eval import _maybe_project
    rng = np.random.default_rng(1)
    train = rng.standard_normal((64, 32)).astype(np.float32)
    val = rng.standard_normal((16, 32)).astype(np.float32) * 100  # outlier scale
    test = rng.standard_normal((16, 32)).astype(np.float32)

    t1, _, te1 = _maybe_project(train, val, test, kind="pca64")
    # Re-running with train alone (val replaced by zeros) should NOT
    # change the train-side projection -- proves PCA fit is
    # train-only.
    val_zero = np.zeros_like(val)
    t2, _, te2 = _maybe_project(train, val_zero, test, kind="pca64")
    np.testing.assert_allclose(t1, t2, rtol=1e-5)
    np.testing.assert_allclose(te1, te2, rtol=1e-5)


def test_maybe_project_unknown_raises():
    from tools.tabpfn_eval import _maybe_project
    train = np.zeros((4, 4))
    with pytest.raises(ValueError, match="unknown projector"):
        _maybe_project(train, train, train, kind="learned256")


def test_maybe_project_auto_passes_raw_below_cap():
    """auto: channels<=TABPFN_MAX_FEATURES -> identity passthrough."""
    from tools.tabpfn_eval import _maybe_project, TABPFN_MAX_FEATURES
    n, d = 32, 128
    assert d <= TABPFN_MAX_FEATURES
    train = np.random.randn(n, d)
    val = np.random.randn(n // 2, d)
    test = np.random.randn(n // 2, d)
    t, v, te = _maybe_project(train, val, test, kind="auto")
    # identity, not just same shape
    np.testing.assert_array_equal(t, train)
    np.testing.assert_array_equal(v, val)
    np.testing.assert_array_equal(te, test)


def test_maybe_project_auto_pca_caps_above_cap():
    """auto: channels>TABPFN_MAX_FEATURES -> PCA at exactly the cap."""
    from tools.tabpfn_eval import _maybe_project, TABPFN_MAX_FEATURES
    n = TABPFN_MAX_FEATURES + 256       # need >cap rows so PCA can hit it
    d = TABPFN_MAX_FEATURES + 100
    train = np.random.randn(n, d)
    val = np.random.randn(n // 4, d)
    test = np.random.randn(n // 4, d)
    t, v, te = _maybe_project(train, val, test, kind="auto")
    assert t.shape == (n, TABPFN_MAX_FEATURES)
    assert v.shape == (n // 4, TABPFN_MAX_FEATURES)
    assert te.shape == (n // 4, TABPFN_MAX_FEATURES)


def test_maybe_subsample_passthrough_below_cap():
    """If n <= cap, the subsample is the identity."""
    from tools.tabpfn_eval import _maybe_subsample
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((100, 8)).astype(np.float32)
    lab = (rng.standard_normal(100) > 0).astype(np.float32)
    e2, l2 = _maybe_subsample(emb, lab, n_max=10000, task_kind="binary")
    np.testing.assert_array_equal(e2, emb)
    np.testing.assert_array_equal(l2, lab)


def test_maybe_subsample_binary_preserves_class_share():
    """Stratified subsample keeps positive-class share within ~1pp of
    the source. Random sampling on a 5%-positive set would frequently
    miss the share by several pp -- this guards the AUC computation."""
    from tools.tabpfn_eval import _maybe_subsample
    rng = np.random.default_rng(7)
    n = 50000
    emb = rng.standard_normal((n, 4)).astype(np.float32)
    # 5% positive (matches rel-arxiv.paper-citation prevalence).
    lab = (rng.random(n) < 0.05).astype(np.float32)
    src_share = lab.mean()
    e2, l2 = _maybe_subsample(emb, lab, n_max=10000, task_kind="binary")
    assert e2.shape == (10000, 4)
    assert l2.shape == (10000,)
    new_share = l2.mean()
    assert abs(new_share - src_share) < 0.01, (
        f"share drift {src_share:.4f}->{new_share:.4f} too large"
    )


def test_maybe_subsample_regression_uniform():
    """Regression: uniform sample, returns shape (n_max, d)."""
    from tools.tabpfn_eval import _maybe_subsample
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((20000, 16)).astype(np.float32)
    lab = rng.standard_normal(20000).astype(np.float32)
    e2, l2 = _maybe_subsample(emb, lab, n_max=10000, task_kind="regression")
    assert e2.shape == (10000, 16)
    assert l2.shape == (10000,)


def test_main_end_to_end_with_mocked_tabpfn(tmp_path):
    """End-to-end main() against synthetic embeddings, with TabPFN
    fit + predict mocked. Verifies the IO contract."""
    from tools.tabpfn_eval import main as tabpfn_main

    channels = 16
    n_train, n_test = 64, 16

    # Synthetic binary task.
    rng = np.random.default_rng(42)
    train_emb = rng.standard_normal((n_train, channels)).astype(np.float32)
    train_labels = (train_emb[:, 0] > 0).astype(np.float32)
    test_emb = rng.standard_normal((n_test, channels)).astype(np.float32)
    test_labels = (test_emb[:, 0] > 0).astype(np.float32)

    for split, emb, lab, n in (("train", train_emb, train_labels, n_train),
                                ("val", test_emb, test_labels, n_test),
                                ("test", test_emb, test_labels, n_test)):
        torch.save({
            "embeddings": torch.from_numpy(emb),
            "labels": torch.from_numpy(lab),
            "global_idx": torch.arange(n, dtype=torch.long),
            "split": split,
            "task": "fake-task",
            "dataset": "fake-ds",
            "channels": channels,
        }, tmp_path / f"{split}.pt")

    out_path = tmp_path / "tabpfn.json"

    # Mock the TabPFN classifier so we don't pull the real package.
    class _StubClassifier:
        def __init__(self, **kwargs):
            # Accept arbitrary kwargs (device, n_estimators,
            # memory_saving_mode, ...) -- the real TabPFNClassifier
            # uses kwargs and our wrapper passes a few; tests should
            # not care which.
            self._kwargs = kwargs
        def fit(self, X, y): pass
        def predict_proba(self, X):
            # Deterministic positive-class prob in [0, 1].
            return np.column_stack([1 - X[:, 0] > 0, X[:, 0] > 0]).astype(float)

    fake_task = _make_fake_task(n_test, {"roc_auc": 0.88})

    fake_tabpfn_module = type(sys)("tabpfn")
    fake_tabpfn_module.TabPFNClassifier = _StubClassifier
    fake_tabpfn_module.TabPFNRegressor = _StubClassifier

    with patch.dict(sys.modules, {"tabpfn": fake_tabpfn_module}), \
         patch("relbench.tasks.get_task", return_value=fake_task):
        rc = tabpfn_main([
            "--embeddings_dir", str(tmp_path),
            "--dataset", "fake-ds", "--task", "fake-task",
            "--projector", "none",
            "--out", str(out_path),
        ])
    assert rc == 0
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["task_kind"] == "binary"
    assert saved["channels_in"] == channels
    assert saved["channels_post_projector"] == channels
    assert saved["test_metrics"]["roc_auc"] == 0.88


def test_no_backbone_loaded_in_tabpfn_path(tmp_path):
    """tabpfn_eval should not import RelGT or torch.nn.Module
    constructor for the backbone -- it's pure post-hoc on the .pt
    files. Soft check via a module-level grep, not a hard one."""
    src = (Path(__file__).resolve().parent.parent / "tools" / "tabpfn_eval.py").read_text()
    # The script must not directly import RelGT or load backbone weights.
    assert "from model import RelGT" not in src
    assert "RelGT.load_backbone" not in src
    assert "best_backbone.pt" not in src
