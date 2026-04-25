"""Per-datatype shared heads + RelGT wrapper.

Mirrors RT (Ranjan et al. 2026, §3.3): one numeric head and one boolean
head, dispatched per row by the row's ``task_type_id``. Per-task heads
would also work but per-datatype is simpler, mirrors the paper, and makes
adding new tasks zero-cost (they just pick whichever head matches their
type).

The wrapper ``MultiTaskRelGT`` does **not** modify ``model.py``. It
constructs RelGT with ``out_channels=channels`` so the model's output
``[B, channels]`` is treated as the embedding feeding our heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from gfm_data.task_tokens import TASK_TYPE_BINARY, TASK_TYPE_REGRESSION


class MultiTaskHead(nn.Module):
    """Two linear heads dispatched by ``task_type_id`` per row.

    Parameters
    ----------
    channels : int
        Input feature width (the backbone's output dim).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.numeric_head = nn.Linear(channels, 1)
        self.boolean_head = nn.Linear(channels, 1)

    def reset_parameters(self):
        self.numeric_head.reset_parameters()
        self.boolean_head.reset_parameters()

    def forward(self, h: Tensor, task_type_id: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h : [B, C] embedding
        task_type_id : [B] int64, ``TASK_TYPE_REGRESSION`` or ``TASK_TYPE_BINARY``

        Returns
        -------
        [B] real-valued (regression rows) or logit (binary rows). Single
        scalar per row -- both heads have ``out_features=1``. The caller
        (loss / metric) decides what to do with each row by looking at
        ``task_type_id``.

        Both heads run on every row. Gradient through the "wrong" head is
        zeroed via the per-row mask in the loss (see
        ``losses.multi_task_loss``); doing it here would require either
        per-row sub-batching (slow) or a where-style branch (still
        backprops through both). Mask-in-loss is cleaner.
        """
        if h.dim() != 2:
            raise ValueError(f"expected [B, C], got shape {tuple(h.shape)}")
        if task_type_id.shape != (h.shape[0],):
            raise ValueError(
                f"task_type_id shape {tuple(task_type_id.shape)} "
                f"!= ({h.shape[0]},)"
            )
        is_reg = task_type_id == TASK_TYPE_REGRESSION
        is_bin = task_type_id == TASK_TYPE_BINARY
        # Both heads always run; per-row dispatch picks the result.
        reg_out = self.numeric_head(h).squeeze(-1)
        bin_out = self.boolean_head(h).squeeze(-1)
        out = torch.where(is_reg, reg_out, torch.where(is_bin, bin_out,
                                                       torch.zeros_like(reg_out)))
        return out


class MultiTaskRelGT(nn.Module):
    """Wrap a RelGT backbone (out_channels=channels) with multi-task heads.

    The wrapper exposes the same forward signature the existing training
    loop uses, plus ``task_type_id`` for per-row dispatch:

        forward(neighbor_types, node_indices, neighbor_hops,
                neighbor_times, grouped_tf_dict,
                edge_index, batch, task_type_id) -> [B] tensor

    Both the wrapped backbone and the heads receive gradients normally.
    """

    def __init__(self, backbone: nn.Module, channels: int):
        super().__init__()
        self.backbone = backbone
        self.head = MultiTaskHead(channels=channels)

    def forward(
        self,
        neighbor_types,
        node_indices,
        neighbor_hops,
        neighbor_times,
        grouped_tf_dict,
        edge_index=None,
        batch=None,
        task_type_id: Optional[Tensor] = None,
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
        if task_type_id is None:
            raise ValueError(
                "MultiTaskRelGT.forward requires task_type_id (per-row int64)"
            )
        return self.head(h, task_type_id.long())
