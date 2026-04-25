"""Multi-task / multi-dataset training plumbing.

Two pieces:

1. ``MultiTaskConcat`` -- a ``torch.utils.data.Dataset`` that concatenates
   N ``TaskTokens`` instances. Index ``i`` is mapped to the underlying
   task and its local index. ``__getitem__`` returns a sample dict that
   already carries ``task_id`` / ``task_type_id`` (set in PR1) so the
   collate can dispatch per-row in PR3 mixed batches.

2. ``DistributedMultiTaskSampler`` -- a DDP-aware sampler that:

   * draws **one task per step** with probability proportional to
     ``w_t * |D_t|``, using a global RNG seeded only by ``epoch`` so all
     ranks pick the **same task at the same step** (avoids DDP
     gradient-shape mismatch);
   * within the chosen task, partitions per-task indices across ranks
     like ``DistributedSampler``;
   * reshuffles within each task on ``set_epoch``.

The sampler yields global indices into ``MultiTaskConcat``, batched into
chunks of ``batch_size`` per step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler


@dataclass
class _TaskInfo:
    """Bookkeeping for one (dataset, task) participant."""
    name: str  # e.g. "rel-f1.driver-position"
    weight: float
    offset: int  # global index where this task's range starts in MultiTaskConcat
    size: int  # local size


class MultiTaskConcat(Dataset):
    """Concatenates a list of TaskTokens-like datasets.

    Each child dataset must implement ``__len__`` and
    ``__getitem__(i) -> (sample_dict, label)`` and tag samples with
    ``task_id`` and ``task_type_id``. The PR1 ``TaskTokens`` already does.

    Parameters
    ----------
    tasks : Sequence[Tuple[str, Dataset]]
        ``(name, ds)`` pairs. ``name`` is used in metric logging and as
        the key for ``--task_weights``.
    weights : Optional[Sequence[float]]
        Per-task draw weight. Default = uniform (1.0 each).
    """

    def __init__(
        self,
        tasks: Sequence[Tuple[str, Dataset]],
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if len(tasks) == 0:
            raise ValueError("MultiTaskConcat requires at least one task")

        self.tasks: List[_TaskInfo] = []
        self.datasets: List[Dataset] = []
        if weights is None:
            weights = [1.0] * len(tasks)
        if len(weights) != len(tasks):
            raise ValueError(f"weights length {len(weights)} != tasks {len(tasks)}")

        offset = 0
        for (name, ds), w in zip(tasks, weights):
            n = len(ds)
            self.datasets.append(ds)
            self.tasks.append(_TaskInfo(name=name, weight=float(w),
                                        offset=offset, size=n))
            offset += n
        self.total = offset

    def __len__(self) -> int:
        return self.total

    def task_of(self, idx: int) -> int:
        """Return which task id ``idx`` belongs to (linear scan; tasks are few)."""
        for i, t in enumerate(self.tasks):
            if t.offset <= idx < t.offset + t.size:
                return i
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        ti = self.task_of(idx)
        local = idx - self.tasks[ti].offset
        sample, label = self.datasets[ti][local]
        # Sanity: child dataset already populates task_id / task_type_id.
        # We don't override; we trust the tagging set up at TaskTokens
        # construction so per-task normalization stats etc. travel with
        # the row.
        return sample, label


class DistributedMultiTaskSampler(Sampler[int]):
    """Yields global indices into MultiTaskConcat for one rank.

    The sequence is built per-epoch as follows:

    1. Decide a deterministic **task schedule** for the whole epoch using
       a global RNG seeded only by ``epoch``. This guarantees every rank
       picks the same task at the same step. The number of "task slots"
       per epoch is ``num_steps_per_epoch * batch_size`` (one task id
       per row, but the whole batch comes from the same task so the
       step-level sequence length is ``num_steps_per_epoch``).
    2. For each task, compute a **per-rank index list** by partitioning
       a shuffled view of that task's local indices across world_size,
       like ``DistributedSampler``.
    3. Walk the task schedule. For each step, take the next
       ``batch_size`` indices from the chosen task's per-rank list.

    Parameters
    ----------
    dataset : MultiTaskConcat
    batch_size : int
        Per-rank batch size. The sampler yields indices in flat form;
        the DataLoader's batch_sampler / batch_size pairs them up. We
        emit them in batch_size-sized contiguous runs from the same
        task to give the collate a homogeneous-task batch by default.
        (Mixed-task batches across the *whole* sweep are achieved by
        scheduling many different tasks across the steps; within a
        single batch a single task dominates -- a deliberate choice
        for simpler grouped_tfs handling and reproducible DDP shapes.)
    num_replicas, rank : int
        Standard DDP world params. Defaults read from
        ``torch.distributed`` when initialized.
    shuffle : bool
        Reshuffle within-task indices each epoch.
    seed : int
        Base seed; combined with ``epoch`` for reproducibility.
    drop_last : bool
        Drop trailing partial batches of a task if its remaining count
        within an epoch is < batch_size.

    Notes
    -----
    Implementation choice: each batch is **single-task** (rows from one
    task only). This keeps the existing collate logic simple (TFs from
    one cache, one prefixed-type namespace) while still letting the
    *epoch* mix tasks freely. PR3.5 may revisit if we want true
    intra-batch mixing.
    """

    def __init__(
        self,
        dataset: MultiTaskConcat,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas is None or rank is None:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
                    rank = rank if rank is not None else dist.get_rank()
            except Exception:
                pass
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        if not (0 <= rank < num_replicas):
            raise ValueError(f"rank={rank} not in [0, {num_replicas})")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    # The number of optimizer-steps per epoch is sum over tasks of:
    #   ceil(per_task_per_rank / batch_size) (or floor if drop_last)
    # weighted by w_t. We keep a simple definition: each task contributes
    # all of its per-rank rows across the epoch, scaled by its weight.
    def __len__(self) -> int:
        return self._planned_total_indices()

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    # ------------------------------------------------------------------
    def _planned_total_indices(self) -> int:
        return sum(self._steps_per_task()) * self.batch_size

    def _steps_per_task(self) -> List[int]:
        """Steps allocated per task this epoch, scaled by w_t.

        Step budget is anchored at the per-rank size of task 0 (or the
        max if task 0 is small): ``steps_t = ceil(B0 * w_t / w_0)``,
        where ``B0 = per_rank_size_0 // batch_size``. Tasks with weight
        > 1 cycle through their indices multiple times within the epoch.
        """
        per_task = self._per_rank_size()
        # Use the largest task as the anchor so smaller-but-heavier
        # tasks are allowed to oversample their pool.
        anchor_idx = int(max(range(len(per_task)),
                             key=lambda i: per_task[i] / max(self.dataset.tasks[i].weight, 1e-9)))
        anchor_size = per_task[anchor_idx]
        anchor_w = self.dataset.tasks[anchor_idx].weight
        if anchor_size < self.batch_size or anchor_w <= 0:
            return [0] * len(per_task)
        anchor_steps = (
            anchor_size // self.batch_size if self.drop_last
            else math.ceil(anchor_size / self.batch_size)
        )
        out = []
        for t in self.dataset.tasks:
            ratio = t.weight / anchor_w
            out.append(max(0, int(round(anchor_steps * ratio))))
        return out

    def _per_rank_size(self) -> List[int]:
        """How many rows of each task end up on this rank per epoch."""
        out = []
        for t in self.dataset.tasks:
            base = t.size // self.num_replicas
            rem = 1 if self.rank < (t.size % self.num_replicas) else 0
            out.append(base + rem)
        return out

    # ------------------------------------------------------------------
    def _per_task_per_rank_indices(self, gen: torch.Generator) -> List[List[int]]:
        """Shuffled per-rank slice of each task's local indices.

        We use the **same** generator state across ranks here -- the
        partition into rank chunks is then deterministic from rank id
        alone, so all ranks agree on which indices each rank holds.
        """
        out: List[List[int]] = []
        for t in self.dataset.tasks:
            if self.shuffle:
                perm = torch.randperm(t.size, generator=gen).tolist()
            else:
                perm = list(range(t.size))
            # Round-robin partition.
            mine = perm[self.rank::self.num_replicas]
            out.append(mine)
        return out

    def _task_schedule(self, gen: torch.Generator) -> List[int]:
        """Per-step list of task ids for this epoch.

        Each task contributes ``_steps_per_task()[ti]`` steps; the
        sequence is a uniformly-shuffled mix so tasks interleave rather
        than running in long blocks.
        """
        steps_per_task = self._steps_per_task()
        sched: List[int] = []
        for ti, n in enumerate(steps_per_task):
            sched.extend([ti] * n)
        # Shuffle the schedule so tasks interleave each epoch.
        if sched:
            perm = torch.randperm(len(sched), generator=gen).tolist()
            sched = [sched[p] for p in perm]
        return sched

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[int]:
        # Shared (cross-rank) generator -- same seed on every rank so the
        # task schedule and the within-task permutations agree.
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)
        per_task_idx = self._per_task_per_rank_indices(gen)
        # The schedule consumes from gen too; same seed -> same schedule.
        schedule = self._task_schedule(gen)

        # Walk schedule; pop batch_size indices from chosen task each step.
        # When a task is over-weighted (steps_t > |pool|/batch_size), we
        # wrap the cursor around the pool so its indices cycle through
        # multiple times within one epoch. This keeps the per-task
        # oversampling we promised in the docstring while still giving
        # every batch a contiguous-pool sample.
        cursors = [0] * len(self.dataset.tasks)
        for ti in schedule:
            pool = per_task_idx[ti]
            if len(pool) == 0:
                continue
            start = cursors[ti] % len(pool)
            end = start + self.batch_size
            if end <= len(pool):
                slice_ = pool[start:end]
            else:
                # Wrap.
                slice_ = pool[start:] + pool[: end - len(pool)]
            cursors[ti] += self.batch_size
            if len(slice_) < self.batch_size:
                if self.drop_last:
                    continue
                pad_n = self.batch_size - len(slice_)
                slice_ = slice_ + pool[:pad_n]
            offset = self.dataset.tasks[ti].offset
            for local in slice_:
                yield offset + local
