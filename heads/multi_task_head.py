"""Per-task heads + RelGT wrapper for multi-task pretraining.

Earlier this module had two shared per-datatype heads (one
``Linear(channels, 1)`` for all regression tasks, one for all binary
tasks) following RT (Ranjan et al. 2026, §3.3). That worked at training
time but has two flaws:

  * The shared boolean head learns ONE decision boundary across binary
    tasks with very different class balance (e.g., rel-event
    ``not_interested`` is 99% negative, ``user-repeat`` is 50/50).
    Result: F1=0 on every binary task at test time -- the head defaults
    to majority-class for all of them.
  * The shared numeric head similarly forces a single linear projection
    across regression tasks with different signal structures.

Per-task heads (one ``Linear(channels, 1)`` per task) cost negligible
parameters (11 tasks * 513 params = ~5.6K vs 34M backbone) but give each
task its own decision boundary. The collate enforces single-task-per-
batch, so we just index the right head by ``task_id[0]`` -- no
``torch.where`` dispatch, no per-row branching.

Embedding extraction: passing ``task_id=None`` to ``MultiTaskRelGT.forward``
returns the raw backbone embedding ``[B, channels]`` for downstream use
(TabPFN / fine-tuning a new head / offline feature store).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from gfm_data.task_tokens import TASK_TYPE_BINARY, TASK_TYPE_REGRESSION


class MultiTaskHead(nn.Module):
    """One ``Linear(channels, 1)`` per task, indexed by ``task_id``.

    Parameters
    ----------
    channels : int
        Input feature width (the backbone's output dim).
    num_tasks : int
        Total number of pretraining tasks (== len(tasks_spec)).
    task_type_ids : List[int]
        Per-task type id (``TASK_TYPE_REGRESSION`` or ``TASK_TYPE_BINARY``).
        Stored as a buffer for callers inspecting the head; the head
        itself doesn't dispatch on type, the loss does.
    """

    def __init__(
        self,
        channels: int,
        num_tasks: int,
        task_type_ids: List[int],
    ):
        super().__init__()
        if len(task_type_ids) != num_tasks:
            raise ValueError(
                f"task_type_ids length {len(task_type_ids)} != "
                f"num_tasks {num_tasks}"
            )
        self.num_tasks = num_tasks
        # One Linear(channels, 1) per task.
        self.task_heads = nn.ModuleList(
            [nn.Linear(channels, 1) for _ in range(num_tasks)]
        )
        # Stored for downstream introspection; the loss already
        # dispatches by task_type_id from the batch.
        self.register_buffer(
            "task_type_ids",
            torch.tensor(task_type_ids, dtype=torch.long),
        )

    def reset_parameters(self):
        for head in self.task_heads:
            head.reset_parameters()

    def forward(self, h: Tensor, task_id: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h : [B, C] embedding from backbone
        task_id : [B] int64, global task index in [0, num_tasks)

        Returns
        -------
        [B] real-valued (regression task) or logit (binary task). The
        caller (loss / metric) decides interpretation by looking at
        ``task_type_id`` from the batch.

        Single-task-per-batch invariant (enforced by collate) means
        all rows share the same task_id. We index the right head
        directly -- no torch.where dispatch needed.
        """
        if h.dim() != 2:
            raise ValueError(f"expected [B, C], got shape {tuple(h.shape)}")
        if task_id.shape != (h.shape[0],):
            raise ValueError(
                f"task_id shape {tuple(task_id.shape)} "
                f"!= ({h.shape[0]},)"
            )
        # All rows share the same task_id (collate invariant). Cheap
        # in-batch sanity to catch a busted collate at the cost of
        # one min/max kernel.
        ti_first = int(task_id[0].item())
        if not (0 <= ti_first < self.num_tasks):
            raise ValueError(
                f"task_id {ti_first} out of range [0, {self.num_tasks})"
            )
        return self.task_heads[ti_first](h).squeeze(-1)


class MultiTaskRelGT(nn.Module):
    """Wrap a RelGT backbone (out_channels=channels) with per-task heads.

    Forward signature::

        forward(neighbor_types, node_indices, neighbor_hops,
                neighbor_times, grouped_tf_dict,
                edge_index=None, batch=None,
                task_id: Optional[Tensor] = None) -> Tensor

    With ``task_id`` provided: returns ``[B]`` predictions for the
    indexed task's head. With ``task_id=None``: returns ``[B, channels]``
    raw backbone embeddings -- the extraction path for TabPFN /
    downstream fine-tuning / offline feature stores.
    """

    def __init__(
        self,
        backbone: nn.Module,
        channels: int,
        num_tasks: int,
        task_type_ids: List[int],
    ):
        super().__init__()
        self.backbone = backbone
        self.head = MultiTaskHead(channels, num_tasks, task_type_ids)

    def forward(
        self,
        neighbor_types,
        node_indices,
        neighbor_hops,
        neighbor_times,
        grouped_tf_dict,
        edge_index=None,
        batch=None,
        task_id: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.backbone(
            neighbor_types,
            node_indices,
            neighbor_hops,
            neighbor_times,
            grouped_tf_dict,
            edge_index=edge_index,
            batch=batch,
        )
        # If the backbone was constructed with out_channels=channels, h is
        # [B, channels]. Otherwise (e.g. dev-kyaw with out_channels=1), the
        # caller is misusing this wrapper.
        if task_id is None:
            # Embedding-only: caller wants the [B, channels] backbone
            # output for TabPFN / a new fine-tune head / a feature store.
            return h
        return self.head(h, task_id.long())
