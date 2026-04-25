"""Data layer for GFM.

PR1 introduces the CSR-backed graph cache and sampler that replaces the
``dict[node_type] -> list[set]`` adjacency built by ``utils.build_adjacency_hetero``.

Behavior is preserved bit-for-bit relative to ``dev-kyaw`` under fixed
``random.seed`` (see tests/test_csr_sampler.py for equivalence proofs).
"""

from gfm_data.collate import collate_multi_task, collate_single_task
from gfm_data.graph_cache import DatasetGraphCache
from gfm_data.multi_task_dataset import (
    DistributedMultiTaskSampler,
    MultiTaskConcat,
)
from gfm_data.sampler import gather_1_and_2_hop, sample_local_subgraph, sample_seeds
from gfm_data.task_tokens import TaskTokens

__all__ = [
    "DatasetGraphCache",
    "DistributedMultiTaskSampler",
    "MultiTaskConcat",
    "TaskTokens",
    "collate_multi_task",
    "collate_single_task",
    "gather_1_and_2_hop",
    "sample_local_subgraph",
    "sample_seeds",
]
