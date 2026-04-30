"""Pure post-hoc TabPFN evaluator on pre-extracted embeddings.

Loads the .pt files produced by ``tools/extract_embeddings.py``
(PR 3.1), fits TabPFN on (train_emb, train_labels), predicts on
test_emb, and reports metrics via the RelBench task's
``evaluate(...)``. No backbone gradient anywhere.

TabPFN v2 declines inputs above ~500 features, so backbone
embeddings of channels=128 (laptop) or 512 (paper-config) fit raw,
but channels=1024 etc. need a cap. The ``--projector`` flag:

  * ``auto``   -- DEFAULT. Pass raw if channels<=500; otherwise PCA
                  to exactly 500 (fit on TRAIN only, no leakage).
                  Adapts to whatever channels the backbone produces.
  * ``none``   -- always pass raw, even above the cap. Useful as a
                  baseline / forcing flag.
  * ``pca64``  -- fixed PCA-64. Legacy default for v1 (~100-feature
                  cap); kept for ablation. Empirical sweep on
                  rel-f1.driver-top3 (laptop, channels=128) showed
                  raw beats PCA-64 on every metric.

Note: ``tabpfn>=2.0,<3`` is the recommended pin -- v7.x line gates
model-weight downloads behind a TABPFN_TOKEN, which doesn't fit a
hands-off pipeline.

Usage::

    python -m tools.tabpfn_eval \\
        --embeddings_dir <run>/embeddings/ \\
        --dataset rel-f1 --task driver-top3 \\
        --projector auto \\
        --out <run>/tabpfn_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load_split(embeddings_dir: str, split: str) -> dict:
    p = Path(embeddings_dir) / f"{split}.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {split}.pt under {embeddings_dir}; "
            f"run tools/extract_embeddings.py with --split all first"
        )
    return torch.load(p, map_location="cpu", weights_only=False)


def _infer_task_kind(labels) -> str:
    """Same heuristic as finetune_head: int dtype or float-in-{0,1} ->
    binary, else regression. Kept local to avoid a cross-tool import."""
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if labels.dtype in (torch.int32, torch.int64):
        return "binary"
    uniq = torch.unique(labels)
    if uniq.numel() <= 2 and uniq.min() >= 0 and uniq.max() <= 1:
        return "binary"
    return "regression"


# TabPFN v2 declines inputs above ~500 features. Auto-projector caps at
# this width; raise it only if the upstream model actually supports more.
TABPFN_MAX_FEATURES = 500


def _maybe_project(
    train_emb: np.ndarray,
    val_emb: np.ndarray,
    test_emb: np.ndarray,
    *,
    kind: str,
):
    """Apply optional dimensionality reduction. PCA is fit on TRAIN
    only (no val/test leakage)."""
    if kind == "none":
        return train_emb, val_emb, test_emb
    if kind == "auto":
        # Pass raw when channels fit; otherwise cap at TABPFN_MAX_FEATURES
        # via PCA. No fixed knob to tune -- adapts to whatever channels
        # the backbone produces.
        d = train_emb.shape[1]
        if d <= TABPFN_MAX_FEATURES:
            return train_emb, val_emb, test_emb
        from sklearn.decomposition import PCA
        n_components = min(TABPFN_MAX_FEATURES, train_emb.shape[0], d)
        pca = PCA(n_components=n_components)
        pca.fit(train_emb)
        return (
            pca.transform(train_emb),
            pca.transform(val_emb),
            pca.transform(test_emb),
        )
    if kind == "pca64":
        from sklearn.decomposition import PCA
        n_components = min(64, train_emb.shape[1], train_emb.shape[0])
        pca = PCA(n_components=n_components)
        pca.fit(train_emb)
        return (
            pca.transform(train_emb),
            pca.transform(val_emb),
            pca.transform(test_emb),
        )
    raise ValueError(f"unknown projector kind: {kind!r}")


def _fit_predict(train_emb, train_labels, test_emb, *, task_kind: str):
    """Fit TabPFN on train, predict on test. Returns predictions
    (probabilities for binary, scalar for regression).

    Imports tabpfn lazily so the test suite can mock it without the
    real package installed in CI.
    """
    if task_kind == "binary":
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier()
        clf.fit(train_emb, train_labels.astype(int))
        proba = clf.predict_proba(test_emb)
        # Probability of the positive class.
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    else:
        from tabpfn import TabPFNRegressor
        reg = TabPFNRegressor()
        reg.fit(train_emb, train_labels)
        return reg.predict(test_emb)


def _final_evaluate(
    test_pred: np.ndarray,
    test_global_idx: np.ndarray,
    *,
    dataset: str,
    task: str,
):
    """Scatter predictions to a [num_test_rows] array and call
    ``task.evaluate(...)``. Same layout as finetune_head's helper."""
    from relbench.tasks import get_task
    task_obj = get_task(dataset, task, download=True)

    n = len(task_obj.get_table("test"))
    full = np.full((n,), -100.0)
    for i, idx in enumerate(test_global_idx.tolist()):
        if 0 <= idx < n:
            full[idx] = float(test_pred[i])
    return task_obj.evaluate(full)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--embeddings_dir", required=True, type=str)
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--task", required=True, type=str)
    p.add_argument(
        "--projector", default="auto",
        choices=["none", "auto", "pca64"],
        help="'auto' (default): pass raw embeddings if channels<=500, "
             "else PCA-cap at 500. 'none': always raw. 'pca64': fixed "
             "PCA-64 (legacy / ablation only).",
    )
    p.add_argument("--out", type=str, default=None,
                   help="If set, save the metrics dict as JSON here.")
    args = p.parse_args(argv)

    embeddings_dir = os.path.expanduser(args.embeddings_dir)
    train = _load_split(embeddings_dir, "train")
    test = _load_split(embeddings_dir, "test")
    val = _load_split(embeddings_dir, "val") if (Path(embeddings_dir) / "val.pt").exists() else None

    if "labels" not in train:
        raise ValueError("train.pt must contain 'labels'")

    train_emb = train["embeddings"].numpy()
    train_lab = train["labels"].numpy()
    test_emb = test["embeddings"].numpy()
    val_emb = val["embeddings"].numpy() if val is not None else test_emb[:1]

    task_kind = _infer_task_kind(train_lab)
    print(
        f"[tabpfn] task_kind={task_kind} train={train_emb.shape} "
        f"test={test_emb.shape} projector={args.projector}"
    )
    train_emb, val_emb, test_emb = _maybe_project(
        train_emb, val_emb, test_emb, kind=args.projector,
    )
    print(f"[tabpfn] post-projection train={train_emb.shape}")

    test_pred = _fit_predict(
        train_emb, train_lab, test_emb, task_kind=task_kind,
    )

    metrics = _final_evaluate(
        test_pred, test["global_idx"].numpy(),
        dataset=args.dataset, task=args.task,
    )
    print(f"[tabpfn] test metrics: {metrics}")

    if args.out:
        out_path = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "task_kind": task_kind,
                "projector": args.projector,
                "channels_in": int(train["channels"]),
                "channels_post_projector": int(train_emb.shape[1]),
                "test_metrics": metrics,
            }, f, indent=2)
        print(f"[tabpfn] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
