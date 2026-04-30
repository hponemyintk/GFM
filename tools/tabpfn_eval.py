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

# TabPFN v2's in-context budget is 10k rows. Above this the package raises
# ValueError unless ``ignore_pretraining_limits=True`` is set, and even
# then the inference cost grows quadratically. We subsample on the
# Python side rather than override the package guardrail.
TABPFN_MAX_TRAIN_SAMPLES = 10000


def _maybe_subsample(
    emb: np.ndarray,
    labels: np.ndarray,
    *,
    n_max: int,
    task_kind: str,
    seed: int = 0,
):
    """Cap train support set at ``n_max`` rows.

    Binary: stratified so positive-class share survives -- uniform
    sampling would distort ROC-AUC on imbalanced labels (rel-arxiv
    paper-citation is ~5% positive). Regression: uniform.
    """
    n = emb.shape[0]
    if n <= n_max:
        return emb, labels
    rng = np.random.default_rng(seed)
    if task_kind == "binary":
        pos_idx = np.where(labels > 0.5)[0]
        neg_idx = np.where(labels <= 0.5)[0]
        share = len(pos_idx) / n
        n_pos = int(round(share * n_max))
        n_neg = n_max - n_pos
        sel_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        sel_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        sel = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(sel)
    else:
        sel = rng.choice(n, size=n_max, replace=False)
    return emb[sel], labels[sel]


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


def _fit_predict(
    train_emb, train_labels, test_emb, *, task_kind: str,
    test_chunk_size: int = 10000,
    device: str = "auto",
    n_estimators: int = 4,
):
    """Fit TabPFN on train, predict on test. Returns predictions
    (probabilities for binary, scalar for regression).

    Test is chunked: TabPFN's attention spans support_set x test_batch
    so a 193k-row test set against a 10k support set can OOM a 12 GB
    GPU in one shot. Chunking is exact (no approximation), it just
    splits the in-context inference into independent passes.

    ``device='cpu'`` is the safe fallback when even chunked GPU
    inference OOMs (e.g. 12 GB laptop GPU). ``memory_saving_mode=True``
    lets TabPFN auto-tune its own batch sizes on top.

    Imports tabpfn lazily so the test suite can mock it without the
    real package installed in CI.
    """
    if task_kind == "binary":
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(
            device=device, n_estimators=n_estimators,
            memory_saving_mode=True,
        )
        clf.fit(train_emb, train_labels.astype(int))
        preds = []
        for i in range(0, len(test_emb), test_chunk_size):
            proba = clf.predict_proba(test_emb[i:i + test_chunk_size])
            if proba.ndim == 2 and proba.shape[1] >= 2:
                preds.append(proba[:, 1])
            else:
                preds.append(proba.ravel())
        return np.concatenate(preds)
    else:
        from tabpfn import TabPFNRegressor
        reg = TabPFNRegressor(
            device=device, n_estimators=n_estimators,
            memory_saving_mode=True,
        )
        reg.fit(train_emb, train_labels)
        preds = []
        for i in range(0, len(test_emb), test_chunk_size):
            preds.append(reg.predict(test_emb[i:i + test_chunk_size]))
        return np.concatenate(preds)


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
    p.add_argument(
        "--max_train_samples", type=int, default=TABPFN_MAX_TRAIN_SAMPLES,
        help="Cap train support set at N rows (TabPFN v2 limit is 10000). "
             "Binary tasks subsample stratified, regression uniform.",
    )
    p.add_argument(
        "--device", default="auto",
        help="'auto' (default): GPU if available. 'cpu': force CPU. "
             "Use 'cpu' on small GPUs (~12 GB) where chunked GPU "
             "inference still OOMs.",
    )
    p.add_argument(
        "--n_estimators", type=int, default=4,
        help="TabPFN ensemble size (default 4). Lower reduces memory "
             "and runtime, at small accuracy cost.",
    )
    p.add_argument(
        "--test_chunk_size", type=int, default=10000,
        help="Test rows per TabPFN inference pass. Drop this if GPU "
             "OOMs even at chunk_size=10k.",
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

    n_train_full = train_emb.shape[0]
    train_emb, train_lab = _maybe_subsample(
        train_emb, train_lab,
        n_max=args.max_train_samples,
        task_kind=task_kind,
    )
    if train_emb.shape[0] != n_train_full:
        print(
            f"[tabpfn] subsampled train {n_train_full} -> "
            f"{train_emb.shape[0]} (cap={args.max_train_samples}, "
            f"task_kind={task_kind})"
        )

    test_pred = _fit_predict(
        train_emb, train_lab, test_emb, task_kind=task_kind,
        test_chunk_size=args.test_chunk_size,
        device=args.device, n_estimators=args.n_estimators,
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
                "n_train_full": int(n_train_full),
                "n_train_used": int(train_emb.shape[0]),
                "test_metrics": metrics,
            }, f, indent=2)
        print(f"[tabpfn] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
