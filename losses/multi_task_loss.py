"""Multi-task loss aggregator (RT formula, plus optional rebalance modes).

Default (RT § 3.3, Ranjan et al. 2026):

    L = (1 / B) * (Σ_{i : reg}    HuberLoss(r_i, r'_i)
                  + Σ_{i : bin}   BCE(1{r_i > 0}, r'_i))

where ``r_i`` is the (z-score-normalized) target for row ``i`` and
``r'_i`` is the head output. Plain batch mean -- works because Huber on
normalized targets is O(1) and BCE is O(0.7), same order of magnitude.

Optional ``--loss_balance`` modes (kept as a flag for ablations):

* ``"none"``   -- the RT default above.
* ``"per_task_mean"`` -- Σ_t (1/T) * mean_i in task_t (...).
* ``"fixed:w1,w2,..."`` -- Σ_t w_t * mean_i in task_t (...).
* ``"uncertainty"`` -- Kendall et al. 2018, learnable log σ² per task.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gfm_data.task_tokens import TASK_TYPE_BINARY, TASK_TYPE_REGRESSION


class MultiTaskLoss(nn.Module):
    """Per-row Huber/BCE dispatch + aggregation.

    Parameters
    ----------
    num_tasks : int
        Number of distinct task ids in the run. Used by ``per_task_mean``
        / ``uncertainty`` / ``fixed`` aggregation modes.
    aggregation : str
        ``"none"`` (default, RT formula), ``"per_task_mean"``,
        ``"fixed:..."`` or ``"uncertainty"``.
    huber_delta : float
        Default 1.0 (PyTorch HuberLoss default; RT paper does not state
        a value).
    """

    def __init__(
        self,
        num_tasks: int,
        aggregation: str = "none",
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.aggregation = aggregation
        self.huber_delta = float(huber_delta)
        if aggregation.startswith("fixed:"):
            ws = [float(x) for x in aggregation[len("fixed:"):].split(",")]
            if len(ws) != num_tasks:
                raise ValueError(
                    f"fixed weights have {len(ws)} entries; need {num_tasks}"
                )
            self.register_buffer(
                "fixed_weights", torch.tensor(ws, dtype=torch.float32)
            )
        if aggregation == "uncertainty":
            # Kendall et al. log σ²; init at 0.
            self.log_sigma2 = nn.Parameter(torch.zeros(num_tasks))

    # ------------------------------------------------------------------
    def per_row_loss(
        self,
        pred: Tensor,
        labels: Tensor,
        task_type_id: Tensor,
    ) -> Tensor:
        """Return [B] per-row losses (no aggregation)."""
        if pred.shape != labels.shape:
            raise ValueError(
                f"pred shape {tuple(pred.shape)} != labels {tuple(labels.shape)}"
            )
        is_reg = task_type_id == TASK_TYPE_REGRESSION
        is_bin = task_type_id == TASK_TYPE_BINARY
        # F.huber_loss with reduction='none' yields elementwise.
        huber_per_row = F.huber_loss(
            pred, labels, reduction="none", delta=self.huber_delta,
        )
        # BCE expects float labels in {0, 1}.
        bce_per_row = F.binary_cross_entropy_with_logits(
            pred, labels.float(), reduction="none",
        )
        zero = torch.zeros_like(huber_per_row)
        per_row = torch.where(is_reg, huber_per_row,
                              torch.where(is_bin, bce_per_row, zero))
        return per_row

    # ------------------------------------------------------------------
    def forward(
        self,
        pred: Tensor,
        labels: Tensor,
        task_id: Tensor,
        task_type_id: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Aggregate per-row losses to a single scalar.

        Returns ``(loss, info)`` where ``info`` is a dict of telemetry
        useful for logging (per-task mean loss, row counts, etc.).
        """
        per_row = self.per_row_loss(pred, labels, task_type_id)

        info: Dict[str, Tensor] = {"per_row_loss_mean": per_row.detach().mean()}
        # Per-task means.
        per_task_means: List[Tensor] = []
        per_task_counts: List[int] = []
        for ti in range(self.num_tasks):
            mask = task_id == ti
            n = int(mask.sum().item())
            per_task_counts.append(n)
            if n > 0:
                per_task_means.append(per_row[mask].mean())
            else:
                per_task_means.append(per_row.new_zeros(()))
            info[f"task_{ti}_loss"] = per_task_means[-1].detach()
            info[f"task_{ti}_count"] = torch.tensor(n)

        if self.aggregation == "none":
            # RT formula: plain batch mean over all rows.
            loss = per_row.mean()
        elif self.aggregation == "per_task_mean":
            present = [m for m, c in zip(per_task_means, per_task_counts) if c > 0]
            if not present:
                loss = per_row.mean()  # degenerate
            else:
                loss = torch.stack(present).mean()
        elif self.aggregation.startswith("fixed:"):
            stacked = torch.stack(per_task_means)
            w = self.fixed_weights
            loss = (w * stacked).sum() / w.sum()
        elif self.aggregation == "uncertainty":
            # L = Σ_t  (1/(2σ²)) L_t + log σ
            stacked = torch.stack(per_task_means)
            log_sigma2 = self.log_sigma2
            inv_sigma2 = torch.exp(-log_sigma2)
            loss = ((0.5 * inv_sigma2) * stacked + 0.5 * log_sigma2).sum()
        else:
            raise ValueError(f"unknown aggregation: {self.aggregation}")

        return loss, info
