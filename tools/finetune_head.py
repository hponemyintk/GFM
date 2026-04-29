"""Train a fresh head on top of pre-extracted backbone embeddings.

Consumes the .pt files produced by ``tools/extract_embeddings.py``
(PR 3.1) and trains a small head -- ``Linear(channels, 1)`` or a
2-layer MLP -- on the train split. Picks the best-val epoch and
reports the test metric via the RelBench task's own ``evaluate(...)``.

Two regimes:

  * **frozen-backbone** (default, fast, RAM-light): only the head
    sees gradients. The pre-extracted embeddings are inputs.
  * **warm-then-unfreeze** (``--unfreeze_after_epoch K``): trains
    the head for K epochs frozen, then loads the backbone via
    ``RelGT.load_backbone(freeze=False)`` and continues from there.
    NOT YET IMPLEMENTED -- see TODO at the bottom; for Phase 4
    holdout-task we only exercise the frozen mode. The warm-unfreeze
    path requires re-running the sampling/forward pipeline alongside
    a head with backbone gradients flowing through, which is a much
    bigger surface (essentially main_node_ddp.py with a different
    head). Land in a follow-up.

CLI::

    python -m tools.finetune_head \\
        --embeddings_dir <run>/embeddings/ \\
        --dataset rel-f1 --task driver-top3 \\
        --head linear --epochs 50 --lr 1e-3 \\
        --out <run>/head_finetuned/finetuned.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load_split(embeddings_dir: str, split: str) -> dict:
    p = Path(embeddings_dir) / f"{split}.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {split}.pt under {embeddings_dir}; "
            f"run tools/extract_embeddings.py with --split all first"
        )
    return torch.load(p, map_location="cpu", weights_only=False)


def _make_head(kind: str, channels: int, out_dim: int = 1) -> nn.Module:
    """Construct a Linear or 2-layer MLP head."""
    if kind == "linear":
        return nn.Linear(channels, out_dim)
    if kind == "mlp2":
        return nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, out_dim),
        )
    raise ValueError(f"unknown head kind: {kind!r}")


def _infer_task_kind(labels: torch.Tensor, dataset: str, task: str) -> str:
    """Decide whether the task is binary classification or regression
    from the label dtype + values. Used to pick loss + final activation
    when the RelBench task object isn't loaded."""
    if labels.dtype in (torch.int32, torch.int64):
        return "binary"
    uniq = torch.unique(labels)
    if uniq.numel() <= 2 and uniq.min() >= 0 and uniq.max() <= 1:
        return "binary"
    return "regression"


def _train_head_on_embeddings(
    head: nn.Module,
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    val_emb: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    task_kind: str,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    device: str,
) -> Tuple[nn.Module, float, int]:
    """Train head; return (head, best_val_metric, best_epoch).

    Best-val metric is loss on val split (lower is better) -- a
    cheap proxy that doesn't require RelBench's task.evaluate at
    every epoch. Final test eval uses task.evaluate.
    """
    head = head.to(device)
    train_emb = train_emb.to(device)
    train_labels = train_labels.to(device).float()
    val_emb = val_emb.to(device)
    val_labels = val_labels.to(device).float()

    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    if task_kind == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.L1Loss()  # MAE -- matches the multi-task regression loss

    # Build train loader (shuffles between epochs).
    ds = TensorDataset(train_emb, train_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val = math.inf
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        head.train()
        for x, y in loader:
            optim.zero_grad()
            pred = head(x).squeeze(-1)
            loss = loss_fn(pred.float(), y)
            loss.backward()
            optim.step()
        head.eval()
        with torch.no_grad():
            val_pred = head(val_emb).squeeze(-1)
            val_loss = float(loss_fn(val_pred.float(), val_labels).item())
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, best_val, best_epoch


def _final_evaluate(
    head: nn.Module,
    test_emb: torch.Tensor,
    test_global_idx: torch.Tensor,
    *,
    task_kind: str,
    dataset: str,
    task: str,
    device: str,
):
    """Run head on test embeddings, post-process per task type, and
    call ``task.evaluate(predictions)``.

    Predictions are scattered into a [num_test_rows] array indexed by
    global_idx so RelBench's evaluator can match them to the test
    table's row order."""
    from relbench.tasks import get_task
    task_obj = get_task(dataset, task, download=True)

    head.eval()
    with torch.no_grad():
        test_pred = head(test_emb.to(device)).squeeze(-1).cpu()

    if task_kind == "binary":
        test_pred = torch.sigmoid(test_pred)

    test_table_size = len(task_obj.get_table("test"))
    full_preds = np.full((test_table_size,), -100.0)
    for i, idx in enumerate(test_global_idx.tolist()):
        if 0 <= idx < test_table_size:
            full_preds[idx] = float(test_pred[i].item())

    metrics = task_obj.evaluate(full_preds)
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--embeddings_dir", required=True, type=str)
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--task", required=True, type=str)
    p.add_argument("--head", default="linear", choices=["linear", "mlp2"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--out", type=str, default=None,
                   help="If set, saves the trained head's state_dict here.")
    p.add_argument("--unfreeze_after_epoch", type=int, default=0,
                   help="NOT YET IMPLEMENTED. Reserved for warm-then-unfreeze.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    if args.unfreeze_after_epoch > 0:
        raise NotImplementedError(
            "--unfreeze_after_epoch is not yet implemented; the "
            "warm-then-unfreeze path requires the full sampling + "
            "backbone forward pipeline (see docstring TODO). For "
            "frozen-backbone fine-tuning leave this at 0."
        )

    embeddings_dir = os.path.expanduser(args.embeddings_dir)
    train = _load_split(embeddings_dir, "train")
    val = _load_split(embeddings_dir, "val")
    test = _load_split(embeddings_dir, "test")

    if "labels" not in train or "labels" not in val:
        raise ValueError(
            "embeddings_dir's train/val .pt must contain 'labels'; "
            "extract_embeddings.py only emits labels when the input "
            "task has them"
        )

    task_kind = _infer_task_kind(train["labels"], args.dataset, args.task)
    channels = int(train["channels"])

    head = _make_head(args.head, channels=channels, out_dim=1)
    print(f"[finetune] task_kind={task_kind} head={args.head} channels={channels}")

    head, best_val_loss, best_epoch = _train_head_on_embeddings(
        head,
        train_emb=train["embeddings"], train_labels=train["labels"],
        val_emb=val["embeddings"], val_labels=val["labels"],
        task_kind=task_kind,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, weight_decay=args.weight_decay,
        device=args.device,
    )
    print(
        f"[finetune] best epoch {best_epoch} val_loss={best_val_loss:.4f}"
    )

    if "labels" in test:
        metrics = _final_evaluate(
            head,
            test_emb=test["embeddings"], test_global_idx=test["global_idx"],
            task_kind=task_kind, dataset=args.dataset, task=args.task,
            device=args.device,
        )
        print(f"[finetune] test metrics: {metrics}")
    else:
        print(f"[finetune] no test labels; skipping final evaluation")
        metrics = None

    if args.out:
        out_path = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save({
            "head_state_dict": head.state_dict(),
            "head_kind": args.head,
            "channels": channels,
            "task_kind": task_kind,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_metrics": metrics,
        }, out_path)
        print(f"[finetune] saved head -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
