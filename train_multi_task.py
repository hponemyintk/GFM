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
def _load_dataset(name: str, cache_dir: str, device: str):
    dset = get_dataset(name, download=True)
    stypes_path = Path(cache_dir) / name / "stypes.json"
    try:
        with open(stypes_path) as f:
            cs = json.load(f)
    except FileNotFoundError:
        cs = get_stype_proposal(dset.get_db())
        stypes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_path, "w") as f:
            json.dump(cs, f, indent=2, default=str)
    for tab, c2s in cs.items():
        for col, st in c2s.items():
            c2s[col] = stype(st) if isinstance(st, str) else st
    data, col_stats = make_pkey_fkey_graph(
        dset.get_db(),
        col_to_stype_dict=cs,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256,
        ),
        cache_dir=f"{cache_dir}/{name}/materialized",
    )
    return data, col_stats


def _build_caches_and_tokens(args, local_rank: int, tasks_spec, device: str):
    """Build {dataset: DatasetGraphCache} and {split: MultiTaskConcat}."""
    # Group tasks by dataset; build one cache per dataset.
    by_ds: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for ds, tk, w in tasks_spec:
        by_ds[ds].append((tk, w))

    caches: Dict[str, DatasetGraphCache] = {}
    col_stats_per_ds: Dict[str, dict] = {}
    for ds_name in by_ds:
        if local_rank == 0:
            print(f"[multi-task] loading dataset '{ds_name}' ...")
        data, col_stats = _load_dataset(ds_name, args.cache_dir, device)
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

    # Build TaskTokens per (dataset, task, split). task_id is the global
    # index of the (ds, tk) pair in tasks_spec order.
    task_tokens: Dict[str, List[TaskTokens]] = {"train": [], "val": [], "test": []}
    task_objs: List[EntityTask] = []
    task_names: List[str] = []
    weights: List[float] = []

    for ti, (ds_name, tk_name, w) in enumerate(tasks_spec):
        task = get_task(ds_name, tk_name, download=True)
        task_objs.append(task)
        task_names.append(f"{ds_name}.{tk_name}")
        weights.append(w)
        # Three splits.
        train_tokens = TaskTokens(
            cache=caches[ds_name], task=task, K=args.num_neighbors,
            split="train", mode=args.mode, precompute=args.precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{ds_name}/{tk_name}",
            shards_dir=args.shards_dir, train_stage=args.train_stage,
            task_id=ti,
        )
        # Adopt train stats on val/test for regression z-score.
        val_tokens = TaskTokens(
            cache=caches[ds_name], task=task, K=args.num_neighbors,
            split="val", mode=args.mode, precompute=args.precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{ds_name}/{tk_name}",
            shards_dir=args.shards_dir, train_stage=args.train_stage,
            task_id=ti,
        )
        test_tokens = TaskTokens(
            cache=caches[ds_name], task=task, K=args.num_neighbors,
            split="test", mode=args.mode, precompute=args.precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{ds_name}/{tk_name}",
            shards_dir=args.shards_dir, train_stage=args.train_stage,
            task_id=ti,
        )
        val_tokens.adopt_target_stats(
            train_tokens.target_mean, train_tokens.target_std,
        )
        test_tokens.adopt_target_stats(
            train_tokens.target_mean, train_tokens.target_std,
        )
        # Optional row cap for laptop runs.
        if args.max_rows_per_task > 0:
            for tok in (train_tokens, val_tokens, test_tokens):
                n = min(args.max_rows_per_task, len(tok.node_idxs))
                tok.node_idxs = tok.node_idxs[:n]
                if tok.time is not None:
                    tok.time = tok.time[:n]
                if tok.target is not None:
                    tok.target = tok.target[:n]
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


# --------------------------------------------------------- model build
def _build_unified_type_map(caches: Dict[str, DatasetGraphCache]):
    """Concatenate per-dataset prefixed type lists into a single global map."""
    all_types: List[str] = []
    for ds_name in sorted(caches.keys()):
        all_types.extend(caches[ds_name].node_types)
    type_to_index = {t: i for i, t in enumerate(all_types)}
    index_to_type = {i: t for i, t in enumerate(all_types)}
    return type_to_index, index_to_type, all_types


def _build_model(args, caches, col_stats_per_ds, type_to_index, num_nodes_total, device):
    """Build RelGT(out_channels=channels) so its output is an embedding."""
    # Union col_names_dict over all datasets/tables (raw type -> col_names_dict).
    # The encoder iterates raw type names; PR1 prefixing means cache keys are
    # prefixed, so we need a flat mapping that the encoder will see.
    col_names_dict: Dict[str, dict] = {}
    col_stats_unified: Dict[str, dict] = {}
    for ds_name, cache in caches.items():
        for prefixed in cache.node_types:
            raw = cache.prefixed_to_raw[prefixed]
            tf = cache.data[raw].tf
            col_names_dict[prefixed] = tf.col_names_dict
            # col_stats from each dataset use raw type names.
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


def run(args, local_rank: int, device, gpu_handle):
    tasks_spec = parse_tasks(args.tasks)
    if local_rank == 0:
        print(f"[multi-task] tasks: {tasks_spec}")

    caches, concats, task_objs, task_names, col_stats_per_ds, task_tokens = (
        _build_caches_and_tokens(args, local_rank, tasks_spec, f"cuda:{local_rank}")
    )

    # Build unified node-type map across all caches.
    type_to_index, _, _ = _build_unified_type_map(caches)
    # Patch each TaskTokens' index map so its collate uses the unified one.
    for split in ("train", "val", "test"):
        for tok in task_tokens[split]:
            tok.node_type_to_index = type_to_index
            tok.index_to_node_type = {i: t for t, i in type_to_index.items()}
            tok.node_types = list(type_to_index.keys())
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
                           "gpu_mem_allocated_MB": mem_a}
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
        for epoch in range(1, args.epochs + 1):
            tr_loss = _train_epoch(epoch)
            dist.barrier()
            val_metrics = _eval(loader_val, "val", epoch)
            if local_rank == 0:
                print(f"Epoch {epoch:02d} train_loss={tr_loss:.4f}")
                for ti, m in val_metrics.items():
                    print(f"  val[{task_names[ti]}]: {m}")
                wandb.log({"epoch": epoch, "epoch_train_loss": tr_loss,
                           **{f"val_{task_names[ti]}_{k}": float(v)
                              for ti, m in val_metrics.items()
                              for k, v in m.items()}})
            dist.barrier()

        # Final test
        test_metrics = _eval(loader_test, "test", 0)
        if local_rank == 0:
            print("=== test ===")
            for ti, m in test_metrics.items():
                print(f"  test[{task_names[ti]}]: {m}")
            with open(os.path.join(output_path, f"{args.seed}.json"), "w") as f:
                json.dump({"test_metrics": {task_names[ti]: m
                                            for ti, m in test_metrics.items()}},
                          f, indent=2)
