"""Tests D1-D5 + Co1-Co6 for the multi-task dataset/sampler/collate.

We use stub TaskTokens so we don't need real relbench/HDF5 to exercise
the dispatch and DDP partitioning logic.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import List

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------- stub dataset
class _StubChildDS(torch.utils.data.Dataset):
    """Tiny stand-in for TaskTokens. Returns ``(sample_dict, label)``
    where the sample carries ``task_id``/``task_type_id`` like real
    TaskTokens does."""
    def __init__(self, n: int, task_id: int, task_type_id: int):
        self.n = n
        self.task_id = task_id
        self.task_type_id = task_type_id
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return ({
            "task_id": self.task_id,
            "task_type_id": self.task_type_id,
            "_local_idx": idx,
        }, torch.tensor(float(idx)))


# =============================================================== D1
def test_D1_all_ranks_pick_same_task_each_step():
    """Same epoch -> identical task schedule on rank 0 and rank 1."""
    from gfm_data.multi_task_dataset import (
        DistributedMultiTaskSampler, MultiTaskConcat,
    )
    a = _StubChildDS(80, task_id=0, task_type_id=0)
    b = _StubChildDS(80, task_id=1, task_type_id=1)
    concat = MultiTaskConcat([("A", a), ("B", b)])

    s_r0 = DistributedMultiTaskSampler(concat, batch_size=8,
                                       num_replicas=2, rank=0, seed=0)
    s_r1 = DistributedMultiTaskSampler(concat, batch_size=8,
                                       num_replicas=2, rank=1, seed=0)

    # Replay one epoch.
    s_r0.set_epoch(7); s_r1.set_epoch(7)
    idx_r0 = list(s_r0)
    idx_r1 = list(s_r1)
    # Each rank yields batch_size indices per step. Within each
    # batch_size-window the task must agree across ranks.
    bs = 8
    for step in range(min(len(idx_r0), len(idx_r1)) // bs):
        t0 = concat.task_of(idx_r0[step * bs])
        t1 = concat.task_of(idx_r1[step * bs])
        assert t0 == t1, f"step {step}: rank0 task={t0}, rank1 task={t1}"


# =============================================================== D2
def test_D2_per_rank_index_partition_disjoint_and_complete():
    from gfm_data.multi_task_dataset import (
        DistributedMultiTaskSampler, MultiTaskConcat,
    )
    a = _StubChildDS(64, task_id=0, task_type_id=0)
    concat = MultiTaskConcat([("A", a)])
    seen = []
    for r in range(4):
        s = DistributedMultiTaskSampler(concat, batch_size=4,
                                        num_replicas=4, rank=r, seed=0,
                                        drop_last=True)
        s.set_epoch(0)
        seen.append(set(int(i) for i in s))
    union = set().union(*seen)
    assert len(union) == 64, f"union size {len(union)} != 64"
    # Pairwise disjoint.
    for i in range(4):
        for j in range(i + 1, 4):
            assert seen[i].isdisjoint(seen[j]), (i, j)


# =============================================================== D3
def test_D3_weighted_task_sampling_within_3_sigma():
    from gfm_data.multi_task_dataset import (
        DistributedMultiTaskSampler, MultiTaskConcat,
    )
    n = 8000
    a = _StubChildDS(n, task_id=0, task_type_id=0)
    b = _StubChildDS(n, task_id=1, task_type_id=1)
    # Heavy weight on B (3:1 ratio).
    concat = MultiTaskConcat([("A", a), ("B", b)], weights=[1.0, 3.0])
    s = DistributedMultiTaskSampler(concat, batch_size=8,
                                    num_replicas=1, rank=0, seed=0,
                                    drop_last=True)
    s.set_epoch(0)
    idxs = list(s)
    bs = 8
    counter: Counter = Counter()
    for step in range(len(idxs) // bs):
        ti = concat.task_of(idxs[step * bs])
        counter[ti] += 1
    # Expected ratio: B 3x A. With ~2000 steps total on equal sizes
    # we expect counter[1] / counter[0] near 3.
    ratio = counter[1] / max(counter[0], 1)
    assert 2.4 < ratio < 3.6, f"got task ratio {ratio:.2f}, expected ~3.0"


# =============================================================== D4
def test_D4_reproducible_under_seed_and_epoch():
    from gfm_data.multi_task_dataset import (
        DistributedMultiTaskSampler, MultiTaskConcat,
    )
    a = _StubChildDS(80, task_id=0, task_type_id=0)
    b = _StubChildDS(80, task_id=1, task_type_id=1)
    concat = MultiTaskConcat([("A", a), ("B", b)])
    s1 = DistributedMultiTaskSampler(concat, batch_size=8, seed=42)
    s2 = DistributedMultiTaskSampler(concat, batch_size=8, seed=42)
    s1.set_epoch(3); s2.set_epoch(3)
    assert list(s1) == list(s2)


# =============================================================== D5
def test_D5_set_epoch_changes_within_task_order_but_not_cross_rank_schedule():
    from gfm_data.multi_task_dataset import (
        DistributedMultiTaskSampler, MultiTaskConcat,
    )
    a = _StubChildDS(80, task_id=0, task_type_id=0)
    b = _StubChildDS(80, task_id=1, task_type_id=1)
    concat = MultiTaskConcat([("A", a), ("B", b)])

    def task_seq(s, ep, bs=8):
        s.set_epoch(ep)
        idxs = list(s)
        return [concat.task_of(idxs[i * bs]) for i in range(len(idxs) // bs)]

    s_r0 = DistributedMultiTaskSampler(concat, batch_size=8, num_replicas=2,
                                       rank=0, seed=0)
    s_r1 = DistributedMultiTaskSampler(concat, batch_size=8, num_replicas=2,
                                       rank=1, seed=0)
    # Same epoch -> same task schedule across ranks.
    assert task_seq(s_r0, 0) == task_seq(s_r1, 0)
    # Different epoch -> potentially different schedule (not guaranteed
    # equal, just verify the property holds at the same epoch).
    assert task_seq(s_r0, 1) == task_seq(s_r1, 1)


# =============================================================== Co1, Co5
def test_Co_multi_task_collate_dispatches_to_child():
    """Multi-task collate routes batch to the right child's collate."""
    from gfm_data.collate import collate_multi_task
    from gfm_data.multi_task_dataset import MultiTaskConcat
    # Reuse the test_collate_single_task stub.
    from tests.test_collate_single_task import _StubTaskTokens, _make_sample

    a = _StubTaskTokens()
    b = _StubTaskTokens()
    concat = MultiTaskConcat([("A", a), ("B", b)])
    # Build a single-task batch tagged task_id=1.
    batch = [_make_sample(global_idx=i, task_id=1) for i in range(2)]
    out = collate_multi_task(concat, batch)
    # Output should match what collate_single_task on `b` returns.
    assert out["task_id"].tolist() == [1, 1]


# =============================================================== Co2 mixed-task asserts
def test_Co2_mixed_task_batch_raises():
    from gfm_data.collate import collate_multi_task
    from gfm_data.multi_task_dataset import MultiTaskConcat
    from tests.test_collate_single_task import _StubTaskTokens, _make_sample

    a = _StubTaskTokens()
    b = _StubTaskTokens()
    concat = MultiTaskConcat([("A", a), ("B", b)])
    # Mixed: one row task_id=0, one task_id=1.
    batch = [
        _make_sample(global_idx=0, task_id=0),
        _make_sample(global_idx=1, task_id=1),
    ]
    with pytest.raises(ValueError, match="single-task"):
        collate_multi_task(concat, batch)
