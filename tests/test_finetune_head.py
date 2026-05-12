"""Unit tests for tools/finetune_head.py.

Synthetic embeddings with a known label relationship, asserts the
head learns it. Heavy components (RelBench task.evaluate, real
embeddings extraction) mocked.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock as _MagicMock, patch

import pytest
import torch
import torch.nn as nn

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
    import numpy as _np
    import pandas as _pd

    tbl = type("Tbl", (), {})()
    tbl.df = _pd.DataFrame(
        {"y": _np.resize(_np.array(["t", "f"], dtype=object), n_test)}
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


def test_make_head_linear_and_mlp2():
    """Both head kinds produce a module with the right input/output dims."""
    from tools.finetune_head import _make_head

    h_lin = _make_head("linear", channels=16, out_dim=1)
    assert isinstance(h_lin, nn.Linear)
    assert h_lin.in_features == 16
    assert h_lin.out_features == 1

    h_mlp = _make_head("mlp2", channels=16, out_dim=1)
    assert h_mlp(torch.zeros(2, 16)).shape == (2, 1)


def test_make_head_unknown_raises():
    from tools.finetune_head import _make_head
    with pytest.raises(ValueError, match="unknown head kind"):
        _make_head("transformer", channels=16)


def test_infer_task_kind_binary_from_dtype():
    from tools.finetune_head import _infer_task_kind
    labels = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
    assert _infer_task_kind(labels, "x", "y") == "binary"


def test_infer_task_kind_binary_from_values():
    """Float labels in {0, 1} should be detected as binary."""
    from tools.finetune_head import _infer_task_kind
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert _infer_task_kind(labels, "x", "y") == "binary"


def test_infer_task_kind_regression():
    from tools.finetune_head import _infer_task_kind
    labels = torch.tensor([1.5, 2.7, 3.3, 4.0, 12.5])
    assert _infer_task_kind(labels, "x", "y") == "regression"


def test_train_head_converges_synthetic_binary():
    """A small linear head should learn a y = sign(<w, x>) labeling
    from random embeddings within ~50 epochs."""
    from tools.finetune_head import _make_head, _train_head_on_embeddings

    torch.manual_seed(0)
    channels = 16
    n_train, n_val = 256, 64
    w_true = torch.randn(channels)

    train_emb = torch.randn(n_train, channels)
    train_labels = ((train_emb @ w_true) > 0).float()
    val_emb = torch.randn(n_val, channels)
    val_labels = ((val_emb @ w_true) > 0).float()

    head = _make_head("linear", channels=channels, out_dim=1)
    head, best_val_loss, best_epoch = _train_head_on_embeddings(
        head, train_emb, train_labels, val_emb, val_labels,
        task_kind="binary",
        epochs=50, lr=1e-2, batch_size=32, weight_decay=0.0,
        device="cpu",
    )
    # BCE on a linearly-separable labeling should drop well under 0.4
    # within 50 epochs at lr=1e-2.
    assert best_val_loss < 0.4, (
        f"binary head failed to converge: best_val_loss={best_val_loss:.4f}"
    )
    assert best_epoch >= 1


def test_train_head_converges_synthetic_regression():
    """A linear head should learn ``y = <w, x>`` (regression) and
    drive MAE down within 100 epochs at lr=1e-2."""
    from tools.finetune_head import _make_head, _train_head_on_embeddings

    torch.manual_seed(1)
    channels = 16
    n_train, n_val = 256, 64
    w_true = torch.randn(channels) * 0.5

    train_emb = torch.randn(n_train, channels)
    train_labels = (train_emb @ w_true).float()
    val_emb = torch.randn(n_val, channels)
    val_labels = (val_emb @ w_true).float()

    head = _make_head("linear", channels=channels, out_dim=1)
    head, best_val_loss, best_epoch = _train_head_on_embeddings(
        head, train_emb, train_labels, val_emb, val_labels,
        task_kind="regression",
        epochs=100, lr=1e-2, batch_size=32, weight_decay=0.0,
        device="cpu",
    )
    # Random init MAE on |<w, x>| with std~3 is about 2-3.
    # Converged head should drive that under ~0.5.
    assert best_val_loss < 0.5, (
        f"regression head failed to converge: best_val_loss={best_val_loss:.4f}"
    )


def test_unfreeze_after_epoch_not_implemented():
    """The warm-then-unfreeze path is reserved but not implemented;
    the CLI must reject it loudly so callers know to use the frozen
    flow until then."""
    from tools.finetune_head import main as ft_main
    with pytest.raises(NotImplementedError, match="unfreeze_after_epoch"):
        ft_main([
            "--embeddings_dir", "/tmp/no",
            "--dataset", "rel-f1", "--task", "driver-top3",
            "--unfreeze_after_epoch", "5",
        ])


def test_main_end_to_end_with_synthetic_embeddings(tmp_path):
    """End-to-end main() against synthetic embeddings + mocked
    task.evaluate. Smoke for the IO contract: load three .pt files,
    train, evaluate, save head."""
    from tools.finetune_head import main as ft_main

    channels = 8
    n = 64

    # Build a learnable binary labeling from random embeddings, save
    # the three .pt files in the format extract_embeddings.py emits.
    torch.manual_seed(42)
    w = torch.randn(channels)
    for split, n_split in (("train", n), ("val", n // 4), ("test", n // 4)):
        emb = torch.randn(n_split, channels)
        labels = ((emb @ w) > 0).float()
        torch.save({
            "embeddings": emb,
            "labels": labels,
            "global_idx": torch.arange(n_split, dtype=torch.long),
            "split": split,
            "task": "fake-task",
            "dataset": "fake-ds",
            "channels": channels,
        }, tmp_path / f"{split}.pt")

    out_path = tmp_path / "head.pt"

    # Mock RelBench task.evaluate to avoid network/registry calls.
    fake_task = _make_fake_task(n // 4, {"roc_auc": 0.95, "f1": 0.9})

    with patch("relbench.tasks.get_task", return_value=fake_task):
        rc = ft_main([
            "--embeddings_dir", str(tmp_path),
            "--dataset", "fake-ds", "--task", "fake-task",
            "--head", "linear", "--epochs", "30", "--lr", "1e-2",
            "--out", str(out_path),
            "--device", "cpu",
        ])
    assert rc == 0
    assert out_path.exists()
    saved = torch.load(out_path, map_location="cpu", weights_only=False)
    for k in ("head_state_dict", "head_kind", "channels", "task_kind",
              "best_epoch", "best_val_loss", "test_metrics"):
        assert k in saved, f"missing key {k} in saved head .pt"
    assert saved["test_metrics"]["roc_auc"] == 0.95


def test_main_evaluates_test_even_when_pt_lacks_labels(tmp_path):
    """RelBench masks the target column on get_table('test') by default
    (binary leaderboard tasks like rel-f1.driver-top3), so
    extract_embeddings emits a test.pt without 'labels'. finetune_head
    must STILL produce test metrics -- _final_evaluate pulls the
    unmasked labels itself via get_table(test, mask_input_cols=False).
    Without this, frozen-backbone evaluation is silently skipped and
    summary.json reports test_metrics=null."""
    from tools.finetune_head import main as ft_main

    channels = 8
    n = 64
    torch.manual_seed(7)
    w = torch.randn(channels)

    # train + val have labels, test does NOT (mirrors the
    # extract_embeddings output for masked-target tasks).
    for split, n_split in (("train", n), ("val", n // 4)):
        emb = torch.randn(n_split, channels)
        labels = ((emb @ w) > 0).float()
        torch.save({
            "embeddings": emb, "labels": labels,
            "global_idx": torch.arange(n_split, dtype=torch.long),
            "split": split, "task": "fake-task", "dataset": "fake-ds",
            "channels": channels,
        }, tmp_path / f"{split}.pt")
    n_test = n // 4
    torch.save({
        "embeddings": torch.randn(n_test, channels),
        "global_idx": torch.arange(n_test, dtype=torch.long),
        # NO 'labels' key -- the masked-test-target case.
        "split": "test", "task": "fake-task", "dataset": "fake-ds",
        "channels": channels,
    }, tmp_path / "test.pt")

    out_path = tmp_path / "head.pt"
    fake_task = _make_fake_task(n_test, {"roc_auc": 0.88, "f1": 0.7})

    with patch("relbench.tasks.get_task", return_value=fake_task):
        rc = ft_main([
            "--embeddings_dir", str(tmp_path),
            "--dataset", "fake-ds", "--task", "fake-task",
            "--head", "linear", "--epochs", "20", "--lr", "1e-2",
            "--out", str(out_path), "--device", "cpu",
        ])
    assert rc == 0
    saved = torch.load(out_path, map_location="cpu", weights_only=False)
    assert saved["test_metrics"] is not None, (
        "test_metrics must be populated even when test.pt lacks labels; "
        "_final_evaluate fetches the unmasked test labels itself"
    )
    assert saved["test_metrics"]["roc_auc"] == 0.88
