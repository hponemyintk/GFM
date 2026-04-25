"""Single-task batch collation.

A direct port of ``utils.RelGTTokens.collate`` (commit ``0359e08``) lifted
into a free function so a future multi-task collate (PR3) can reuse the
same per-row stacking + edge-offset logic.

PR1 adds two forward-compat keys to the batch dict:

* ``task_id`` -- ``int64[B]`` (constant within a single-task batch)
* ``task_type_id`` -- ``int64[B]`` (constant within a single-task batch)

All other keys match dev-kyaw exactly. Test ``Co5`` asserts this in
``tests/test_collate_single_task.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from gfm_data.task_tokens import TaskTokens


def collate_single_task(
    dataset: TaskTokens,
    batch: List[Tuple[dict, Optional[Tensor]]],
) -> Dict[str, object]:
    """Stack a list of ``(sample, label)`` into the batch dict consumed by RelGT."""
    samples, labels = zip(*batch)

    neighbor_types = torch.stack([s["types"] for s in samples], dim=0)  # [B, K]
    neighbor_indices = torch.stack([s["indices"] for s in samples], dim=0)
    neighbor_hops = torch.stack([s["hops"] for s in samples], dim=0)
    neighbor_times = torch.stack([s["times"] for s in samples], dim=0)

    out: Dict[str, object] = {
        "neighbor_types": neighbor_types,
        "neighbor_indices": neighbor_indices,
        "neighbor_hops": neighbor_hops,
        "neighbor_times": neighbor_times,
    }

    if dataset.target is not None:
        out["labels"] = torch.stack(labels, dim=0)
    else:
        out["labels"] = None

    first_types = [s["first_type"] for s in samples]
    first_indices = [s["first_index"] for s in samples]
    out["node_indices"] = torch.tensor(
        dataset.get_global_index(first_types, first_indices),
        dtype=torch.long,
    )

    B, K = neighbor_types.shape
    grouped_tfs: Dict[int, object] = {}
    grouped_positions: Dict[int, List[int]] = {}
    for t_id in range(len(dataset.node_types)):
        mask = neighbor_types == t_id
        if not mask.any():
            continue
        local_idxs = neighbor_indices[mask]
        prefixed = dataset.index_to_node_type[t_id]
        raw = dataset.cache.prefixed_to_raw[prefixed]
        positions_2d = torch.nonzero(mask, as_tuple=False)
        offsets_list = [int(b) * K + int(k) for (b, k) in positions_2d.tolist()]
        grouped_tfs[t_id] = dataset.cache.data[raw].tf[local_idxs]
        grouped_positions[t_id] = offsets_list

    flat_batch_idx = torch.arange(B).unsqueeze(1).expand(B, K).reshape(-1).tolist()
    flat_nbr_idx = torch.arange(K).repeat(B).tolist()
    global_idxs = torch.tensor([s["global_idx"] for s in samples], dtype=torch.long)

    out.update({
        "grouped_tfs": grouped_tfs,
        "grouped_indices": grouped_positions,
        "flat_batch_idx": flat_batch_idx,
        "flat_nbr_idx": flat_nbr_idx,
        "global_idx": global_idxs,
    })

    # Per-sample edge concatenation with K-offset.
    batched_edges: List[Tensor] = []
    batch_vec: List[Tensor] = []
    node_offset = 0
    for i, sample in enumerate(samples):
        eidx = sample["edge_index"]
        K_i = sample["types"].size(0)
        batched_edges.append(eidx + node_offset)
        batch_vec.append(torch.full((K_i,), i, dtype=torch.long))
        node_offset += K_i

    edge_index = (
        torch.cat(batched_edges, dim=1) if batched_edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    batch_out = (
        torch.cat(batch_vec, dim=0) if batch_vec
        else torch.zeros((0,), dtype=torch.long)
    )

    out.update({
        "edge_index": edge_index,
        "batch": batch_out,
    })

    # Forward-compat: per-row task identifiers (constant in PR1).
    out["task_id"] = torch.tensor(
        [s["task_id"] for s in samples], dtype=torch.long,
    )
    out["task_type_id"] = torch.tensor(
        [s["task_type_id"] for s in samples], dtype=torch.long,
    )
    return out
