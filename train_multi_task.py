"""Multi-task training entry point (PR3).

Imported by ``main_node_ddp.py`` when ``--tasks`` is provided. Keeps the
single-task path in main_node_ddp.py untouched so ML3.5 parity stays
apples-to-apples.

Pipeline:
  1. parse ``--tasks "ds.task[:w], ..."`` into a list of (dataset, task,
     weight) triples.
  2. build per-dataset DatasetGraphCache (one per unique dataset, shared
     by all tasks of that dataset).
  3. build per-(dataset, task, split) TaskTokens; adopt train target
     stats on val/test for regression z-score.
  4. wrap into MultiTaskConcat per split and use
     DistributedMultiTaskSampler for train; per-task DistributedSampler
     for val/test (so we evaluate each task on its own subset).
  5. wrap RelGT(out_channels=channels) backbone in MultiTaskRelGT (two
     shared heads).
  6. MultiTaskLoss with the configured aggregation.
  7. train + eval per-task; log per-task metrics + macro mean.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from tqdm import tqdm

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from gfm_data import (
    DatasetGraphCache,
    DistributedMultiTaskSampler,
    MultiTaskConcat,
    TaskTokens,
    collate_multi_task,
    collate_single_task,
)
from gfm_data.task_tokens import (
    TASK_TYPE_BINARY,
    TASK_TYPE_REGRESSION,
)
from heads.multi_task_head import MultiTaskRelGT
from losses.multi_task_loss import MultiTaskLoss
from model import RelGT
from utils import GloveTextEmbedding


# --------------------------------------------------------- arg parsing
def parse_tasks(spec: str) -> List[Tuple[str, str, float]]:
    """``"rel-f1.driver-position:1,rel-f1.driver-top3:1.5"`` -> list."""
    out: List[Tuple[str, str, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            head, w = chunk.rsplit(":", 1)
            weight = float(w)
        else:
            head, weight = chunk, 1.0
        if "." not in head:
            raise ValueError(f"task spec '{chunk}' must be 'dataset.task[:w]'")
        ds, tk = head.split(".", 1)
        out.append((ds, tk, weight))
    return out


# --------------------------------------------------------------- data
def _rss_gb() -> float:
    """Resident set size of the current process in GiB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except Exception:
        pass
    return float("nan")


def _load_dataset(name: str, cache_dir: str, device: str):
    """Load one dataset's HeteroData + col_stats.

    Routes stypes loading through gfm_data.stypes.load_or_generate_stypes
    so we get the same defensive behavior as the offline tools (validates
    JSON, regenerates on corrupt NaN/float entries, writes clean .value
    serialization). Then filters the stypes dict against the DB's actual
    columns -- RelBench tasks like results-position strip leakage-risk
    columns; without filtering, torch_frame.Dataset.__init__ raises.
    """
    from gfm_data.stypes import (
        filter_to_db_columns as _filter_stypes,
        load_or_generate_stypes as _load_stypes,
    )
    import gc
    dset = get_dataset(name, download=True)
    stypes_path = Path(cache_dir) / name / "stypes.json"
    # IMPORTANT: pass upto_test_timestamp=False here. Otherwise the
    # entity tables stop at train cutoff and TEST seeds (which reference
    # entities created between train and test cutoffs) index out of
    # bounds in the CSR adjacency. Temporal leakage is still prevented
    # by the per-row seed_time filter in the sampler -- the materialized
    # graph just needs to *contain* all entities the task tables reference.
    cs = _load_stypes(stypes_path, dset, upto_test_timestamp=False)
    db = dset.get_db(upto_test_timestamp=False)
    cs = _filter_stypes(cs, db)
    data, col_stats = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=cs,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256,
        ),
        # Distinct cache dir from the train-cutoff materialization so the
        # two layouts don't collide.
        cache_dir=f"{cache_dir}/{name}/materialized_full",
    )
    # OOM mitigation: relbench's Dataset.get_db is decorated with
    # @lru_cache(maxsize=None), so the raw pickle (~25 GiB on
    # rel-event/rel-amazon) is pinned to the bound method's cache --
    # NOT to a plain instance attribute. Clear the lru_cache to drop
    # the strong reference, then delete the dset itself.
    try:
        dset.get_db.cache_clear()
    except Exception:
        pass
    del db
    del dset
    gc.collect()
    return data, col_stats


def _build_caches_and_tokens(args, local_rank: int, tasks_spec, device: str):
    """Build {dataset: DatasetGraphCache} and {split: MultiTaskConcat}."""
    # Group tasks by dataset; build one cache per dataset.
    by_ds: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for ds, tk, w in tasks_spec:
        by_ds[ds].append((tk, w))

    caches: Dict[str, DatasetGraphCache] = {}
    col_stats_per_ds: Dict[str, dict] = {}

    # OOM mitigation: stagger dataset loads across DDP ranks. With 8 GPUs
    # all ranks would otherwise call make_pkey_fkey_graph + get_db
    # simultaneously, peaking at 8x the per-rank transient (raw db pickle,
    # tf tensors, text-embed buffers). After the load each rank drops
    # tf+edge_index, so steady-state is small -- but the transient was
    # blowing past the pod limit. Serializing into chunks of size
    # `args.load_concurrency` (default 1) bounds peak transient memory.
    ddp_world = dist.get_world_size() if dist.is_initialized() else 1
    ddp_rank = dist.get_rank() if dist.is_initialized() else 0
    chunk = max(1, int(getattr(args, "load_concurrency", 1)))

    def _log_rss(tag: str, ds: str = ""):
        # Each rank prints to its own line; tag includes rank so logs
        # interleave readably.
        print(f"[rss r{ddp_rank}] {tag}{(' ' + ds) if ds else ''}: "
              f"{_rss_gb():.2f} GiB", flush=True)

    for ds_name in by_ds:
        if local_rank == 0:
            print(f"[multi-task] loading dataset '{ds_name}' ...", flush=True)
        # Serialize: only ranks in the current "slot" load at once.
        # Slot = ddp_rank // chunk. We barrier between slots.
        if dist.is_initialized() and ddp_world > chunk:
            for slot in range(0, ddp_world, chunk):
                if slot <= ddp_rank < slot + chunk:
                    _log_rss("pre-load", ds_name)
                    data, col_stats = _load_dataset(
                        ds_name, args.cache_dir, device,
                    )
                    _log_rss("post-load", ds_name)
                dist.barrier()
        else:
            _log_rss("pre-load", ds_name)
            data, col_stats = _load_dataset(ds_name, args.cache_dir, device)
            _log_rss("post-load", ds_name)
        cache_root = (
            os.path.join(args.tf_store_dir, ds_name)
            if args.tf_store_dir else None
        )
        caches[ds_name] = DatasetGraphCache(
            data=data,
            undirected=True,
            name_prefix=ds_name,  # PR3: prefix types so multiple ds don't collide
            tf_store_root=cache_root,
        )
        col_stats_per_ds[ds_name] = col_stats

        # Capture col_names_dict per (prefixed_type) BEFORE dropping
        # tf. _build_model needs it to construct RelGT but doesn't
        # need the actual TF tensors.
        caches[ds_name]._captured_col_names_dict = {}
        for _prefixed in caches[ds_name].node_types:
            _raw = caches[ds_name].prefixed_to_raw[_prefixed]
            _store = caches[ds_name].data[_raw]
            if hasattr(_store, "tf"):
                caches[ds_name]._captured_col_names_dict[_prefixed] = \
                    _store.tf.col_names_dict

        # OOM mitigation: now that the cache holds the CSR adjacency
        # AND we've captured col_names_dict, the raw HeteroData's
        # edge_index tensors and (when tf_store_root is set) tf
        # tensors are dead weight. With 8 DDP ranks each holding
        # rel-event's ~25 GB of in-RAM tensors, peak RAM blew past
        # pod limits and got OOM-killed in phase 3.
        # Pin num_nodes BEFORE dropping tf -- NodeStorage.num_nodes
        # is a property that infers from tf and returns None once tf
        # is gone unless we pin it (same gotcha as
        # tools/precompute_shards.py).
        import gc as _gc
        for _nt in list(data.node_types):
            store = data[_nt]
            try:
                n = store.num_nodes
                if n is not None:
                    store.num_nodes = int(n)
            except Exception:
                pass
            if cache_root is not None and hasattr(store, "tf"):
                try:
                    del store["tf"]
                except Exception:
                    try:
                        delattr(store, "tf")
                    except Exception:
                        pass
        for _et in list(data.edge_types):
            estore = data[_et]
            if "edge_index" in estore:
                try:
                    del estore["edge_index"]
                except Exception:
                    try:
                        delattr(estore, "edge_index")
                    except Exception:
                        pass
        _gc.collect()

    # PR4: pre-compute the unified type map ONCE here so every TaskTokens
    # gets the same global vocabulary at construction. This makes HDF5 /
    # precomputed-shards correct under multi-dataset training (otherwise
    # they'd write per-cache local ids and the model would mis-embed).
    unified_type_map: Dict[str, int] = {}
    for ds_name in sorted(caches.keys()):
        for prefixed in caches[ds_name].node_types:
            if prefixed not in unified_type_map:
                unified_type_map[prefixed] = len(unified_type_map)

    # OOM mitigation: pre-load EntityTask objects with the same chunked
    # DDP-barrier serialization used for _load_dataset. Without this,
    # all 8 ranks call ``get_task(ds, tk, download=True)`` simultaneously
    # and each call internally invokes ``get_dataset(ds).get_db()`` --
    # which constructs a NEW Dataset instance separate from the one
    # _load_dataset already freed, reloading the raw pickle into memory.
    # For rel-event that's ~25 GiB per rank; 8 ranks x 25 GiB = ~200 GiB
    # transient spike on top of the ~176 GiB steady-state HeteroData.
    # Serializing into chunks of ``args.load_concurrency`` bounds the
    # transient to 1 rank x 25 GiB at a time, and we ``cache_clear()``
    # the lru-cached raw DB after each slot so it doesn't accumulate.
    import gc as _gc_tasks
    task_objs_by_key: Dict[Tuple[str, str], EntityTask] = {}

    def _load_tasks_for_dataset(ds_name: str) -> None:
        for (tk_name, _w) in by_ds[ds_name]:
            task_obj = get_task(ds_name, tk_name, download=True)
            # Pre-warm the per-split parquet caches INSIDE the serialized
            # slot. ``task.get_table(split)`` reads {cache_dir}/{split}.parquet
            # if present; if missing it falls into ``_get_table`` ->
            # ``dataset.get_db()`` which would otherwise fire on all 8
            # ranks simultaneously the first time TaskTokens.__init__
            # runs. Doing it here ensures any DB pickle load is bounded
            # by ``args.load_concurrency``. After this call,
            # task.get_table is lru-cached on the task object; later
            # invocations in TaskTokens / eval are pure dict lookups.
            for _split in ("train", "val", "test"):
                try:
                    task_obj.get_table(_split)
                except Exception as e:
                    # Don't let a single split failure blow up startup;
                    # TaskTokens construction below will surface a real
                    # error if this turns out to be load-blocking.
                    print(
                        f"[multi-task] WARN: pre-warm get_table({_split}) "
                        f"failed for {ds_name}.{tk_name}: {e}",
                        flush=True,
                    )
            task_objs_by_key[(ds_name, tk_name)] = task_obj
        # Free the raw DB pickle that get_task's internal get_dataset()
        # populated via lru_cache. The Dataset instance itself is
        # process-local cheap metadata; we just want the multi-GiB
        # pickle reference dropped before the next rank loads.
        try:
            _dset = get_dataset(ds_name, download=False)
            _dset.get_db.cache_clear()
            del _dset
        except Exception:
            pass
        _gc_tasks.collect()

    for ds_name in by_ds:
        if local_rank == 0:
            print(
                f"[multi-task] loading task objects for '{ds_name}' "
                f"({len(by_ds[ds_name])} tasks) ...",
                flush=True,
            )
        if dist.is_initialized() and ddp_world > chunk:
            for slot in range(0, ddp_world, chunk):
                if slot <= ddp_rank < slot + chunk:
                    _log_rss("pre-task-load", ds_name)
                    _load_tasks_for_dataset(ds_name)
                    _log_rss("post-task-load", ds_name)
                dist.barrier()
        else:
            _log_rss("pre-task-load", ds_name)
            _load_tasks_for_dataset(ds_name)
            _log_rss("post-task-load", ds_name)

    # Build TaskTokens per (dataset, task, split). task_id is the global
    # index of the (ds, tk) pair in tasks_spec order.
    task_tokens: Dict[str, List[TaskTokens]] = {"train": [], "val": [], "test": []}
    task_objs: List[EntityTask] = []
    task_names: List[str] = []
    weights: List[float] = []

    for ti, (ds_name, tk_name, w) in enumerate(tasks_spec):
        task = task_objs_by_key[(ds_name, tk_name)]
        task_objs.append(task)
        task_names.append(f"{ds_name}.{tk_name}")
        weights.append(w)
        # Three splits.
        train_tokens = TaskTokens(
            cache=caches[ds_name], task=task, K=args.num_neighbors,
            split="train", mode=args.mode, precompute=args.precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{ds_name}/{tk_name}",
            shards_dir=(
                os.path.join(args.shards_dir, ds_name, tk_name)
                if args.shards_dir else None
            ),
            train_stage=args.train_stage,
            task_id=ti, unified_type_map=unified_type_map,
        )
        # Adopt train stats on val/test for regression z-score.
        val_tokens = TaskTokens(
            cache=caches[ds_name], task=task, K=args.num_neighbors,
            split="val", mode=args.mode, precompute=args.precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{ds_name}/{tk_name}",
            shards_dir=(
                os.path.join(args.shards_dir, ds_name, tk_name)
                if args.shards_dir else None
            ),
            train_stage=args.train_stage,
            task_id=ti, unified_type_map=unified_type_map,
        )
        test_tokens = TaskTokens(
            cache=caches[ds_name], task=task, K=args.num_neighbors,
            split="test", mode=args.mode, precompute=args.precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{ds_name}/{tk_name}",
            shards_dir=(
                os.path.join(args.shards_dir, ds_name, tk_name)
                if args.shards_dir else None
            ),
            train_stage=args.train_stage,
            task_id=ti, unified_type_map=unified_type_map,
        )
        val_tokens.adopt_target_stats(
            train_tokens.target_mean, train_tokens.target_std,
        )
        test_tokens.adopt_target_stats(
            train_tokens.target_mean, train_tokens.target_std,
        )
        # Optional row cap for laptop runs. Train only -- val/test stay
        # full because task.evaluate expects the full RelBench table size.
        if args.max_rows_per_task > 0:
            n = min(args.max_rows_per_task, len(train_tokens.node_idxs))
            train_tokens.node_idxs = train_tokens.node_idxs[:n]
            if train_tokens.time is not None:
                train_tokens.time = train_tokens.time[:n]
            if train_tokens.target is not None:
                train_tokens.target = train_tokens.target[:n]
        task_tokens["train"].append(train_tokens)
        task_tokens["val"].append(val_tokens)
        task_tokens["test"].append(test_tokens)

    concats: Dict[str, MultiTaskConcat] = {}
    for split in ("train", "val", "test"):
        concats[split] = MultiTaskConcat(
            tasks=[(name, tok) for name, tok in zip(task_names, task_tokens[split])],
            weights=weights,
        )

    # All caches share the same prefixed type universe? They DON'T -- each
    # cache has its own prefixed types. To build one model that handles
    # all of them, we union the type maps.
    return caches, concats, task_objs, task_names, col_stats_per_ds, task_tokens


def _release_cache_data(args, caches, task_tokens, local_rank: int) -> None:
    """Free per-cache HeteroData refs after _build_model.

    OOM mitigation (final pass): in shards+tf_store mode no remaining
    code path reads ``cache.data``. The HeteroData object still holds
    per-type ``time`` tensors (int64 per node -- up to several GiB on
    rel-event), per-type ``x`` if any, and stale edge metadata. All
    of this gets COW-duplicated on every DataLoader worker fork
    (CPython ref-count writes break COW). Drop the reference now so
    the workers fork with minimal anon memory.

    Conditions for the drop to be safe:
      * args.mode in {precomputed_shards, hdf5}: streaming mode
        calls cache.data inside the per-batch sampler.
      * args.tf_store_dir set: cache.tf_view falls back to
        ``data[type].tf`` only when tf_store_root is None.
    Must be called AFTER _build_model -- _build_model has a fallback
    path that reads cache.data when ``_captured_col_names_dict`` is
    missing a prefix (test paths only; the multi-task flow always
    captures, but the fallback would crash if data is None).
    """
    if (
        args.mode not in ("precomputed_shards", "hdf5")
        or args.tf_store_dir is None
    ):
        return
    import gc as _gc_final
    for ds_name, cache in caches.items():
        cache.data = None
    # TaskTokens.data captured a reference to HeteroData at __init__
    # ("exposed for back-compat with main_node_ddp.py"); even with
    # cache.data nulled, those copies keep the HeteroData alive (and
    # COW-duplicated on every worker fork). Null them too.
    for split_toks in task_tokens.values():
        for tok in split_toks:
            tok.data = None
    _gc_final.collect()
    if local_rank == 0:
        print(
            f"[multi-task] released cache.data on {len(caches)} caches "
            f"(shards mode + tf_store_dir; HeteroData no longer needed)",
            flush=True,
        )
        print(f"[rss r{local_rank}] post-release: {_rss_gb():.2f} GiB",
              flush=True)
    # Align ranks before the next DDP collective. ``gc.collect()`` is
    # synchronous and can take several seconds on a heap that just
    # held multi-GiB HeteroData refs; without an explicit barrier the
    # slowest rank could lag into DDP init's first all_reduce while
    # faster ranks have already started, eating into NCCL's 30-min
    # timeout for no good reason. Cheap insurance.
    if dist.is_initialized():
        dist.barrier()


# --------------------------------------------------------- model build
def _build_model(args, caches, col_stats_per_ds, type_to_index, num_nodes_total, device):
    """Build RelGT(out_channels=channels) so its output is an embedding."""
    # Union col_names_dict over all datasets/tables.
    # NOTE: we use cache._captured_col_names_dict (stashed at cache
    # construction time in _build_caches_and_tokens) so this still
    # works after the OOM mitigation drops the tf tensors.
    col_names_dict: Dict[str, dict] = {}
    col_stats_unified: Dict[str, dict] = {}
    for ds_name, cache in caches.items():
        captured = getattr(cache, "_captured_col_names_dict", {})
        for prefixed in cache.node_types:
            if prefixed in captured:
                col_names_dict[prefixed] = captured[prefixed]
            else:
                # Fallback for code paths that didn't capture (single-task,
                # tests, etc.) -- read directly from data while available.
                raw = cache.prefixed_to_raw[prefixed]
                store = cache.data[raw]
                if hasattr(store, "tf"):
                    col_names_dict[prefixed] = store.tf.col_names_dict
            raw = cache.prefixed_to_raw[prefixed]
            if raw in col_stats_per_ds[ds_name]:
                col_stats_unified[prefixed] = col_stats_per_ds[ds_name][raw]
    backbone = RelGT(
        num_nodes=num_nodes_total,
        max_neighbor_hop=3,  # 0,1,2 + fallback (3)
        node_type_map=type_to_index,
        col_names_dict=col_names_dict,
        col_stats_dict=col_stats_unified,
        local_num_layers=args.num_layers,
        channels=args.channels,
        out_channels=args.channels,  # KEY: use channels so output is embedding
        global_dim=args.channels // 2,
        heads=args.num_heads,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        conv_type=args.gt_conv_type,
        ablate=args.ablate,
        gnn_pe_dim=args.gnn_pe_dim,
        num_centroids=args.num_centroids,
        sample_node_len=args.num_neighbors,
        args=args,
    ).to(device)
    wrapper = MultiTaskRelGT(backbone=backbone, channels=args.channels).to(device)
    return wrapper


# --------------------------------------------------------- training
def _gpu_stats(handle, device):
    import pynvml
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu, torch.cuda.memory_allocated(device) / 1024**2, \
        torch.cuda.memory_reserved(device) / 1024**2


def _macro_score(per_task_metrics: Dict[int, dict], task_objs: List["EntityTask"]) -> float:
    """Average a per-task val metric to a comparable scalar.

    Returns the macro-mean over tasks, where each task contributes:
      - AUROC (already in (0, 1], higher = better) for binary classification
      - 1/(1+MAE) (in (0, 1], higher = better) for regression
    Tasks with naturally easier MAE will dominate the regression term;
    that is acceptable here since the alternative -- trying to compute
    a 'normalized' MAE on z-score scale -- requires extra plumbing and
    macro-best-epoch selection is robust to scale at the relative
    ranking level.
    """
    scores = []
    for ti, m in per_task_metrics.items():
        tt = task_objs[ti].task_type
        if tt == TaskType.BINARY_CLASSIFICATION:
            s = float(m["roc_auc"])
        elif tt == TaskType.REGRESSION:
            s = 1.0 / (1.0 + float(m["mae"]))
        else:
            continue
        scores.append(s)
    return sum(scores) / len(scores) if scores else float("-inf")


def run(args, local_rank: int, device, gpu_handle):
    tasks_spec = parse_tasks(args.tasks)
    if local_rank == 0:
        print(f"[multi-task] tasks: {tasks_spec}")

    caches, concats, task_objs, task_names, col_stats_per_ds, task_tokens = (
        _build_caches_and_tokens(args, local_rank, tasks_spec, f"cuda:{local_rank}")
    )

    # The unified type map was already wired into each TaskTokens at
    # construction (see _build_caches_and_tokens). Pull it back out here
    # for the model construction.
    type_to_index = task_tokens["train"][0].node_type_to_index
    num_nodes_total = sum(c.num_nodes_total() for c in caches.values())

    # DataLoaders
    train_sampler = DistributedMultiTaskSampler(
        concats["train"], batch_size=args.batch_size, seed=args.seed,
    )
    loader_train = DataLoader(
        concats["train"], batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=lambda b: collate_multi_task(concats["train"], b),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    # For val/test we run each task separately with a standard sampler on
    # its TaskTokens, so the eval is straightforwardly per-task.
    loader_val: Dict[int, DataLoader] = {}
    loader_test: Dict[int, DataLoader] = {}
    for ti, name in enumerate(task_names):
        for split, dst in (("val", loader_val), ("test", loader_test)):
            tok = task_tokens[split][ti]
            samp = DistributedSampler(tok, shuffle=False, seed=args.seed,
                                      drop_last=False)
            dst[ti] = DataLoader(
                tok, batch_size=args.batch_size, sampler=samp,
                collate_fn=lambda b, t=tok: collate_single_task(t, b),
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                pin_memory=True,
            )

    # Model + heads.
    model = _build_model(args, caches, col_stats_per_ds, type_to_index,
                         num_nodes_total, device)
    # OOM mitigation: free cache.data NOW that _build_model has read
    # what it needs. Workers fork from the parent rank below; if we
    # don't drop here, every fork inherits per-type time tensors and
    # stale HeteroData metadata that COW-duplicates on first ref-count
    # write (Python objects in workers can't COW-share long).
    _release_cache_data(args, caches, task_tokens, local_rank)
    # col_stats_per_ds is no longer needed -- _build_model copied what
    # it needed into the backbone. Free it pre-fork too.
    col_stats_per_ds.clear()
    for n, p in model.named_parameters():
        if p.dtype == torch.int16:
            p.data = p.data.to(torch.int64)
    for n, b in model.named_buffers():
        if b.dtype == torch.int16:
            b.data = b.data.to(torch.int64)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True,
    )

    if local_rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[multi-task] total params: {n_params}")

    loss_fn = MultiTaskLoss(num_tasks=len(tasks_spec),
                            aggregation=args.loss_balance).to(device)

    world_size = dist.get_world_size()
    optim = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr * world_size, weight_decay=args.weight_decay,
    )

    if local_rank == 0:
        wandb.init(project="rel-gt-expts", name=args.run_name, config=vars(args))
    output_path = os.path.join(args.out_dir, "multi_task")
    os.makedirs(output_path, exist_ok=True)

    global_step = 0

    def _train_epoch(epoch):
        nonlocal global_step
        model.train()
        train_sampler.set_epoch(epoch)
        loss_accum = 0.0
        count = 0
        total_steps = min(
            len(loader_train) // 1, args.max_steps_per_epoch,
        )
        for step, batch in enumerate(
            tqdm(loader_train, total=total_steps, desc=f"Train e{epoch}",
                 disable=(local_rank != 0)), start=1
        ):
            n_types = batch["neighbor_types"].to(device)
            n_idx = batch["node_indices"].to(device)
            n_hop = batch["neighbor_hops"].to(device)
            n_t = batch["neighbor_times"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_v = batch["batch"].to(device)
            grouped = {
                "grouped_tfs": batch["grouped_tfs"],
                "grouped_indices": batch["grouped_indices"],
                "flat_batch_idx": batch["flat_batch_idx"],
                "flat_nbr_idx": batch["flat_nbr_idx"],
            }
            labels = batch["labels"].to(device).float()
            task_id = batch["task_id"].to(device)
            task_type = batch["task_type_id"].to(device)

            optim.zero_grad()
            pred = model(n_types, n_idx, n_hop, n_t, grouped,
                         edge_index=edge_index, batch=batch_v,
                         task_type_id=task_type)
            loss, info = loss_fn(pred.float(), labels, task_id, task_type)
            loss.backward()

            # ML6 (plan §6.3.2): per-datatype head grad norms. Log them
            # so head-starvation surfaces immediately (one head receiving
            # ~zero grad while the other dominates).
            head_module = model.module.head
            num_g = head_module.numeric_head.weight.grad
            bool_g = head_module.boolean_head.weight.grad
            num_norm = float(num_g.norm().item()) if num_g is not None else 0.0
            bool_norm = float(bool_g.norm().item()) if bool_g is not None else 0.0

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            v = float(loss.detach().item())
            loss_accum += v * pred.size(0); count += pred.size(0)
            global_step += 1
            if local_rank == 0:
                gpu_util, mem_a, _ = _gpu_stats(gpu_handle, device)
                payload = {"train_loss": v, "global_step": global_step,
                           "lr": optim.param_groups[0]["lr"],
                           "gpu_util_percent": gpu_util,
                           "gpu_mem_allocated_MB": mem_a,
                           "head_grad_norm_numeric": num_norm,
                           "head_grad_norm_boolean": bool_norm}
                for k, t in info.items():
                    if k.startswith("task_"):
                        payload[k] = float(t.item() if isinstance(t, torch.Tensor) else t)
                wandb.log(payload)
            if step >= args.max_steps_per_epoch:
                break
        return (loss_accum / count) if count > 0 else float("inf")

    @torch.no_grad()
    def _eval(loader_dict, split, epoch):
        model.eval()
        per_task_metrics: Dict[int, dict] = {}
        for ti, loader in loader_dict.items():
            if loader.sampler is not None and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            preds_local = []
            idxs_local = []
            for batch in tqdm(loader, desc=f"{split} task{ti}",
                              disable=(local_rank != 0)):
                n_types = batch["neighbor_types"].to(device)
                n_idx = batch["node_indices"].to(device)
                n_hop = batch["neighbor_hops"].to(device)
                n_t = batch["neighbor_times"].to(device)
                edge_index = batch["edge_index"].to(device)
                batch_v = batch["batch"].to(device)
                grouped = {
                    "grouped_tfs": batch["grouped_tfs"],
                    "grouped_indices": batch["grouped_indices"],
                    "flat_batch_idx": batch["flat_batch_idx"],
                    "flat_nbr_idx": batch["flat_nbr_idx"],
                }
                task_type = batch["task_type_id"].to(device)
                pred = model.module(n_types, n_idx, n_hop, n_t, grouped,
                                    edge_index=edge_index, batch=batch_v,
                                    task_type_id=task_type)
                # Denormalize regression preds; sigmoid binary preds.
                tok = task_tokens[split][ti]
                if tok.task_type_id == TASK_TYPE_REGRESSION:
                    pred = tok.denormalize_pred(pred)
                else:
                    pred = torch.sigmoid(pred)
                preds_local.append(pred.detach().cpu().numpy())
                idxs_local.append(batch["global_idx"].cpu().numpy())
            local_preds = np.concatenate(preds_local) if preds_local else np.array([])
            local_idxs = np.concatenate(idxs_local) if idxs_local else np.array([])
            gathered = [None] * world_size if local_rank == 0 else None
            dist.gather_object((local_idxs, local_preds),
                               object_gather_list=gathered, dst=0)
            if local_rank == 0:
                full = np.full((len(loader.dataset),), -100.0)
                for g in gathered:
                    g_i, g_p = g
                    for i, p in zip(g_i, g_p):
                        full[i] = p
                metrics = task_objs[ti].evaluate(
                    full, task_objs[ti].get_table(split)
                ) if split == "val" else task_objs[ti].evaluate(full)
                per_task_metrics[ti] = metrics
        return per_task_metrics

    if args.train_stage == "finetune":
        # Best-macro checkpointing. Each epoch we score per-task val
        # metric on a comparable scale (AUROC for binary, 1/(1+MAE) for
        # regression -- both in (0, 1], higher is better) and average.
        # The single checkpoint at the best-macro epoch is loaded before
        # the final test eval. This is the standard single-model
        # multi-task convention (T5 / RT pretraining); per-task fine-
        # tuning would beat it but needs a separate run per task.
        best_macro = -math.inf
        best_state = None
        best_epoch = 0
        per_epoch_macro: Dict[int, float] = {}

        for epoch in range(1, args.epochs + 1):
            tr_loss = _train_epoch(epoch)
            dist.barrier()
            val_metrics = _eval(loader_val, "val", epoch)
            if local_rank == 0:
                print(f"Epoch {epoch:02d} train_loss={tr_loss:.4f}")
                for ti, m in val_metrics.items():
                    print(f"  val[{task_names[ti]}]: {m}")
                macro = _macro_score(val_metrics, task_objs)
                per_epoch_macro[epoch] = macro
                wandb.log({"epoch": epoch, "epoch_train_loss": tr_loss,
                           "val_macro": macro,
                           **{f"val_{task_names[ti]}_{k}": float(v)
                              for ti, m in val_metrics.items()
                              for k, v in m.items()}})
                if macro > best_macro:
                    best_macro = macro
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.module.state_dict())
                    print(f"  [best] macro={macro:.4f} @ epoch {epoch}; checkpoint cached in memory")
                else:
                    print(f"  macro={macro:.4f} (best={best_macro:.4f} @ epoch {best_epoch})")
            dist.barrier()

        # Load best-macro checkpoint on rank 0, broadcast to all ranks.
        if local_rank == 0 and best_state is not None:
            print(f"\nLoading best-macro checkpoint (epoch {best_epoch}, macro={best_macro:.4f}) "
                  f"before test eval.")
            model.module.load_state_dict(best_state)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        for buf in model.buffers():
            dist.broadcast(buf.data, src=0)
        dist.barrier()

        # Final test on the loaded best-macro model.
        test_metrics = _eval(loader_test, "test", 0)
        if local_rank == 0:
            print("=== test ===")
            for ti, m in test_metrics.items():
                print(f"  test[{task_names[ti]}]: {m}")
            with open(os.path.join(output_path, f"{args.seed}.json"), "w") as f:
                json.dump({
                    "test_metrics": {task_names[ti]: m for ti, m in test_metrics.items()},
                    "best_epoch": int(best_epoch),
                    "best_val_macro": float(best_macro),
                    "per_epoch_macro": {str(k): float(v) for k, v in per_epoch_macro.items()},
                }, f, indent=2)
