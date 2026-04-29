import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict
import wandb

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import pynvml

from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

# within this project
from model import RelGT
from utils import GloveTextEmbedding
from gfm_data import (
    DatasetGraphCache,
    DistributedMultiTaskSampler,
    MultiTaskConcat,
    TaskTokens,
    collate_multi_task,
    collate_single_task,
)
from gfm_data.task_tokens import TASK_TYPE_BINARY, TASK_TYPE_REGRESSION
from heads.multi_task_head import MultiTaskRelGT
from losses.multi_task_loss import MultiTaskLoss

torch.autograd.set_detect_anomaly(True)

############################
# 1. Parse arguments
############################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--precompute", action="store_true", default=True)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=512)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--gt_conv_type", type=str, default="full")
parser.add_argument("--ablate", type=str, default="none")
parser.add_argument("--gnn_pe_dim", type=int, default=0)
parser.add_argument("--num_neighbors", type=int, default=300)
parser.add_argument("--num_centroids", type=int, default=4096)
parser.add_argument("--ff_dropout", type=float, default=0.1)
parser.add_argument("--attn_dropout", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.00001)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--pos_enc", type=str, default="none")
parser.add_argument("--max_degree", type=int, default=10000)
parser.add_argument("--pos_enc_dim", type=int, default=128)
parser.add_argument("--max_steps_per_epoch", type=int, default=3000)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=str, default="results/debug")
parser.add_argument("--run_name", type=str, default="debug")
parser.add_argument('--model_parameters', type=int, default=0, help='Number of model parameters')
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--train_stage", type=str, default="finetune", choices=["finetune"])
# PR2 data-layer mode flags. Default keeps dev-kyaw / PR1 behavior intact.
parser.add_argument(
    "--mode",
    type=str,
    default="hdf5",
    choices=["hdf5", "streaming", "precomputed_shards"],
    help="Sample materialization strategy. 'hdf5' (default) matches dev-kyaw. "
         "'streaming' samples in DataLoader workers. 'precomputed_shards' "
         "reads memmap shards built by tools/precompute_shards.py.",
)
parser.add_argument(
    "--shards_dir",
    type=str,
    default=None,
    help="Directory of precomputed shards (required for --mode=precomputed_shards).",
)
parser.add_argument(
    "--tf_store_dir",
    type=str,
    default=None,
    help="Directory of memmap-backed TensorFrame columns. If set, TF reads "
         "go through gfm_data.tf_store.TFStoreReader instead of the in-RAM "
         "data[type].tf -- needed for datasets too big to fit TFs in RAM.",
)
parser.add_argument(
    "--max_rows_per_task",
    type=int,
    default=0,
    help="If > 0, cap the number of seed rows used per (split, task). "
         "Combined with --tf_store_dir this bounds resident memory for "
         "laptop-scale runs on big datasets.",
)
# PR3 multi-task flags. When --tasks is provided, --dataset/--task are ignored.
parser.add_argument(
    "--tasks",
    type=str,
    default=None,
    help="Comma-separated list of '<dataset>.<task>[:weight]' pairs, e.g. "
         "'rel-f1.driver-position:1,rel-f1.driver-top3:1'. When set, the "
         "single-task --dataset/--task path is replaced by multi-task "
         "training over a MultiTaskConcat with DistributedMultiTaskSampler.",
)
parser.add_argument(
    "--loss_balance",
    type=str,
    default="none",
    help="Multi-task loss aggregation: 'none' (RT batch-mean, default), "
         "'per_task_mean', 'fixed:w1,w2,...', or 'uncertainty'.",
)
parser.add_argument(
    "--load_concurrency",
    type=int,
    default=1,
    help="Phase-3 OOM mitigation: how many DDP ranks may load a dataset "
         "simultaneously (via make_pkey_fkey_graph + get_db). Default 1 "
         "serializes loads across the 8 GPUs so peak transient RAM is "
         "bounded by one rank's load instead of WORLD_SIZE x. Bump to 2-4 "
         "for faster startup if pod RAM headroom allows.",
)
parser.add_argument(
    "--full_graph",
    action="store_true",
    default=False,
    help="Use upto_test_timestamp=False so the entity tables include "
         "rows created after train_cutoff. Required for autocomplete "
         "tasks (users-birthyear, results-position, qualifying-position, "
         "transactions-price) whose val/test seeds reference entities "
         "added after the cutoff. Temporal leakage is still prevented "
         "by the per-neighbor seed_time filter at gfm_data/sampler.py:69. "
         "See docs/truncated_graph_caveat.md.",
)

args = parser.parse_args()
MULTI_TASK = args.tasks is not None

############################
# 2. Initialize DDP and set device
############################
dist.init_process_group(backend="nccl")
# local_rank = args.local_rank
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

# Only the main process (rank 0) initializes wandb and prints logs.
if local_rank == 0:
    args.run_name = f"{args.dataset}-{args.task}-{args.run_name}"

def init_gpu_utilization(device_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return handle

def get_gpu_stats(handle, device):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu
    mem_allocated = torch.cuda.memory_allocated(device) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
    return gpu_util, mem_allocated, mem_reserved

print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

gpu_handle = init_gpu_utilization(local_rank)

# PR3 dispatch: if --tasks is set, run the multi-task pipeline and exit.
# The single-task code below is unchanged (preserves dev-kyaw parity).
if MULTI_TASK:
    from train_multi_task import run as run_multi_task
    if local_rank == 0:
        args.run_name = f"multi_task-{args.run_name}"
    run_multi_task(args, local_rank, device, gpu_handle)
    dist.destroy_process_group()
    import sys; sys.exit(0)

############################
# 3. Load dataset, task, and prepare data
############################
dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")

# Route stypes through the safe helper rather than reading json
# inline. The helper validates the file and regenerates on corrupt
# NaN content (the AWS bug fixed by gfm_data.stypes). It only
# touches dataset.get_db() WHEN regeneration is needed -- and that
# get_db call is folded into the chunked load slot below, so even
# the cold-cache regen path is bounded by args.load_concurrency.
from gfm_data.stypes import (
    filter_to_db_columns as _filter_stypes,
    load_or_generate_stypes as _load_stypes,
)

# OOM mitigation: serialize get_db + make_pkey_fkey_graph across ranks
# (chunks of args.load_concurrency, default 1). Otherwise 8 GPUs all
# pickle-load the raw DB simultaneously and peak past pod RAM.
_world = dist.get_world_size() if dist.is_initialized() else 1
_rank = dist.get_rank() if dist.is_initialized() else 0
_chunk = max(1, int(args.load_concurrency))


def _do_load_db_and_graph():
    # stypes load+regen lives INSIDE the slot so the cold-cache
    # regen path's get_db() doesn't fan out across ranks.
    # upto_test_timestamp controls entity table truncation:
    #   True  (default)        -- matches dev-kyaw / RelGT paper guardrail.
    #   False (--full_graph)   -- required for autocomplete tasks whose
    #                             val/test seeds reference entities added
    #                             after train_cutoff. The per-neighbor
    #                             seed_time filter at sampler.py:69 is
    #                             the actual leakage barrier in both modes.
    upto = not args.full_graph
    cs_loaded = _load_stypes(stypes_cache_path, dataset, upto_test_timestamp=upto)
    db_local = dataset.get_db(upto_test_timestamp=upto)
    cs_local = _filter_stypes(cs_loaded, db_local)
    mat_suffix = "materialized_full" if args.full_graph else "materialized"
    d_local, cs_dict_local = make_pkey_fkey_graph(
        db_local,
        col_to_stype_dict=cs_local,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=f"cuda:{local_rank}"),
            batch_size=256,
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/{mat_suffix}",
    )
    # Pre-warm the task parquet caches inside the slot. If a parquet
    # is missing, ``task.get_table`` falls into ``_get_table`` which
    # calls ``self.dataset.get_db()`` -- doing it here keeps that
    # cache-miss bounded to one rank, not all 8.
    # IMPORTANT: pass split as a KEYWORD arg so the @lru_cache key
    # matches downstream callers (TaskTokens / eval) which use the
    # keyword form -- functools keys positional and keyword args
    # separately and a positional pre-warm would miss the cache.
    for _split in ("train", "val", "test"):
        try:
            task.get_table(split=_split)
        except Exception as e:
            print(
                f"[single-task] WARN: pre-warm get_table({_split}) "
                f"failed: {e}",
                flush=True,
            )
    return db_local, d_local, cs_dict_local


if dist.is_initialized() and _world > _chunk:
    for _slot in range(0, _world, _chunk):
        if _slot <= _rank < _slot + _chunk:
            _db, data, col_stats_dict = _do_load_db_and_graph()
        dist.barrier()
else:
    _db, data, col_stats_dict = _do_load_db_and_graph()
# OOM mitigation (single-task DDP path): relbench's Dataset.get_db is
# @lru_cache(maxsize=None), so the raw pickle (~25 GiB on rel-event /
# rel-amazon) is pinned to the bound-method cache, not an instance attr.
# Clear the cache to release the reference now that we have HeteroData.
try:
    dataset.get_db.cache_clear()
except Exception:
    pass
del _db
import gc as _gc_db
_gc_db.collect()

# Build the CSR graph cache once for this dataset; the three splits share it.
graph_cache = DatasetGraphCache(
    data=data,
    undirected=True,
    name_prefix=None,
    tf_store_root=args.tf_store_dir,  # None = in-RAM TFs (default)
)

# Capture col_names_dict per node type BEFORE dropping tf -- the
# model construction below needs it but doesn't need the actual TF
# tensors.
_captured_col_names = {
    nt: data[nt].tf.col_names_dict
    for nt in data.node_types
    if hasattr(data[nt], "tf")
}

# OOM mitigation: edge_index tensors are dead weight after CSR is
# built; if tf_store_dir is set, in-RAM tf is also dead weight (reads
# go through TFStoreReader). Pin num_nodes before dropping tf so the
# NodeStorage property doesn't start returning None.
import gc as _gc
for _nt in list(data.node_types):
    _store = data[_nt]
    try:
        _n = _store.num_nodes
        if _n is not None:
            _store.num_nodes = int(_n)
    except Exception:
        pass
    if args.tf_store_dir is not None and hasattr(_store, "tf"):
        try:
            del _store["tf"]
        except Exception:
            try:
                delattr(_store, "tf")
            except Exception:
                pass
for _et in list(data.edge_types):
    _es = data[_et]
    if "edge_index" in _es:
        try:
            del _es["edge_index"]
        except Exception:
            try:
                delattr(_es, "edge_index")
            except Exception:
                pass
_gc.collect()

data = {
    split: TaskTokens(
        cache=graph_cache,
        task=task,
        K=args.num_neighbors,
        split=split,
        mode=args.mode,
        precompute=args.precompute,
        precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
        shards_dir=args.shards_dir,
        train_stage=args.train_stage,
    )
    for split in ["train", "val", "test"]
}

# Adopt train target stats on val/test for regression z-score. Without
# this each split fits its own mean/std, and the model -- trained on
# the train distribution -- would have its predictions denormalized by
# val/test stats (which differ from train), producing systematic
# offsets at eval time. Mirrors train_multi_task.py:395-400.
if data["train"].target_mean is not None:
    data["val"].adopt_target_stats(
        data["train"].target_mean, data["train"].target_std,
    )
    data["test"].adopt_target_stats(
        data["train"].target_mean, data["train"].target_std,
    )

# Optional seed-row cap for memory-bounded laptop runs (plan §6.3.6).
# Only cap the *train* split; val/test stay full because task.evaluate()
# expects the full RelBench table size.
if args.max_rows_per_task > 0:
    ds = data["train"]
    n = min(args.max_rows_per_task, len(ds.node_idxs))
    ds.node_idxs = ds.node_idxs[:n]
    if ds.time is not None:
        ds.time = ds.time[:n]
    if ds.target is not None:
        ds.target = ds.target[:n]
    if local_rank == 0:
        print(f"[train] capped to {n} seed rows via --max_rows_per_task")

############################
# 4. Create DataLoaders (with a DistributedSampler for training)
############################
def _collate_for(ds: TaskTokens):
    return lambda batch: collate_single_task(ds, batch)

train_sampler = DistributedSampler(data["train"], shuffle=True, seed=args.seed)
loader_train = DataLoader(
    data["train"],
    batch_size=args.batch_size,
    sampler=train_sampler,
    collate_fn=_collate_for(data["train"]),
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
    pin_memory=True)

val_sampler = DistributedSampler(data["val"], shuffle=False, seed=args.seed, drop_last=False)
loader_val = DataLoader(
    data["val"],
    batch_size=args.batch_size,
    sampler=val_sampler,
    collate_fn=_collate_for(data["val"]),
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
    pin_memory=True
)

test_sampler = DistributedSampler(data["test"], shuffle=False, seed=args.seed, drop_last=False)
loader_test = DataLoader(
    data["test"],
    batch_size=args.batch_size,
    sampler=test_sampler,
    collate_fn=_collate_for(data["test"]),
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
    pin_memory=True
)


loader_dict: Dict[str, DataLoader] = {"train": loader_train, "val": loader_val, "test": loader_test}

############################
# 5. Set up the task-specific settings
############################
clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

############################
# 6. Build and wrap the model in DDP
############################
model = RelGT(
    max_neighbor_hop=data["train"].max_neighbor_hop,
    node_type_map=data["train"].node_type_to_index,
    col_names_dict=_captured_col_names,
    col_stats_dict=col_stats_dict,
    local_num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    global_dim=args.channels//2,
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

# Before DDP initialization, cast problematic tensors
for name, param in model.named_parameters():
    if param.dtype == torch.int16:
        param.data = param.data.to(torch.int64)

for name, buf in model.named_buffers():
    if buf.dtype == torch.int16:
        buf.data = buf.data.to(torch.int64)

model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

if local_rank == 0:
    print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if local_rank == 0:
    print(f"Total model parameters: {total_params}")
args.model_parameters = total_params

if local_rank == 0:
    wandb.init(project="rel-gt-expts", name=args.run_name, config=vars(args))

output_path = os.path.join(args.out_dir, args.dataset, args.task)
os.makedirs(output_path, exist_ok=True)

world_size = dist.get_world_size()
base_lr = args.lr * world_size
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

global_step = 0

############################
# 7. Training and Evaluation Loops
############################
def train_supervised(epoch) -> float:
    global global_step
    model.train()
    loss_accum = count_accum = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    
    train_sampler.set_epoch(epoch)
    
    for step, batch in enumerate(tqdm(loader_dict["train"], total=total_steps, desc="Train"), start=1):
        # Move tensors to the proper device.
        neighbor_types = batch["neighbor_types"].to(device)
        node_indices = batch["node_indices"].to(device)
        neighbor_hops = batch["neighbor_hops"].to(device)
        neighbor_times = batch["neighbor_times"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_vec = batch["batch"].to(device)

        grouped_tf_dict = {
            'grouped_tfs': batch['grouped_tfs'],
            'grouped_indices': batch['grouped_indices'],
            'flat_batch_idx': batch['flat_batch_idx'],
            'flat_nbr_idx': batch['flat_nbr_idx']
        }
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        pred = model(
            neighbor_types,
            node_indices,
            neighbor_hops,
            neighbor_times,
            grouped_tf_dict,
            edge_index=edge_index,
            batch=batch_vec
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred        
        loss = loss_fn(pred.float(), labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = loss.detach().item()
        gpu_util, mem_allocated, mem_reserved = get_gpu_stats(gpu_handle, device)
        # Only rank 0 logs training metrics.
        if local_rank == 0:
            wandb.log({"train_loss": loss_value,
                       "global_step": global_step,
                       "lr": optimizer.param_groups[0]["lr"],
                       "gpu_util_percent": gpu_util,
                       "gpu_mem_allocated_MB": mem_allocated,
                       "gpu_mem_reserved_MB": mem_reserved})

        loss_accum += loss_value * pred.size(0)
        count_accum += pred.size(0)
        global_step += 1

        if step >= args.max_steps_per_epoch:
            break

    return loss_accum / count_accum if count_accum > 0 else float('inf')

@torch.no_grad()
def test(loader: DataLoader, eval_model, epoch, desc) -> np.ndarray:
    if loader.sampler is not None and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
        
    eval_model.eval()
    pred_list = []
    idx_list = []
    
    for batch in tqdm(loader, desc=desc, disable=(local_rank != 0)):
        neighbor_types = batch["neighbor_types"].to(device)
        node_indices = batch["node_indices"].to(device)
        neighbor_hops = batch["neighbor_hops"].to(device)
        neighbor_times = batch["neighbor_times"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_vec = batch["batch"].to(device)
        
        grouped_tf_dict = {
            'grouped_tfs': batch['grouped_tfs'],
            'grouped_indices': batch['grouped_indices'],
            'flat_batch_idx': batch['flat_batch_idx'],
            'flat_nbr_idx': batch['flat_nbr_idx']
        }
        pred = eval_model(
            neighbor_types,
            node_indices,
            neighbor_hops,
            neighbor_times,
            grouped_tf_dict,
            edge_index=edge_index,
            batch=batch_vec
        )
        if task.task_type == TaskType.REGRESSION:
            # Model was trained on z-scored targets (TaskTokens.__getitem__
            # at gfm_data/task_tokens.py:604), so its outputs live in
            # train z-space. Undo before clamping with raw-scale
            # clamp_min/clamp_max and before task.evaluate -- which both
            # expect raw-scale predictions. ``loader.dataset`` is the
            # val or test TaskTokens; its target_mean/std were adopted
            # from train above, so denormalize is bit-equivalent across
            # splits. Mirrors train_multi_task.py:770-775.
            pred = loader.dataset.denormalize_pred(pred)
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu().numpy())
        idx_list.append(batch["global_idx"].cpu().numpy())
    
    # Concatenate local predictions & indices
    local_preds = np.concatenate(pred_list, axis=0) if pred_list else np.array([])
    local_idxs  = np.concatenate(idx_list,  axis=0) if idx_list  else np.array([])

    # Gather on rank 0
    gathered = [None for _ in range(world_size)] if local_rank == 0 else None
    dist.gather_object((local_idxs, local_preds), object_gather_list=gathered, dst=0)

    if local_rank == 0:
        all_preds = np.full((len(loader.dataset),), -100.0)
        for i in range(world_size):
            g_idx, g_pred = gathered[i]
            for idx, pred in zip(g_idx, g_pred):
                all_preds[idx] = pred
        return all_preds
    else:
        return None

if args.train_stage == "finetune":
    # Supervised Finetuning Stage:
    best_val_metric = -math.inf if higher_is_better else math.inf
    state_dict = None

    for epoch in range(1, args.epochs + 1):
        # use supervised training loop.
        train_loss = train_supervised(epoch)
        # scheduler.step()
        
        dist.barrier()
        eval_model = model.module  # get the underlying model
        
        # Run evaluation on the validation set.
        val_pred = test(loader_dict["val"], eval_model=eval_model, epoch=epoch, desc="Val")
        if local_rank == 0:
            val_metrics = task.evaluate(val_pred, task.get_table("val"))
            print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")
            wandb.log({
                "epoch": epoch,
                "epoch_train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })
            
            if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
                not higher_is_better and val_metrics[tune_metric] <= best_val_metric
            ):
                best_val_metric = val_metrics[tune_metric]
                state_dict = copy.deepcopy(model.module.state_dict())
                torch.save(state_dict, os.path.join(output_path, "finetuned.pt"))
        dist.barrier()

    if local_rank == 0 and state_dict is not None:
        model.module.load_state_dict(state_dict)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=0)
    dist.barrier()

    # Final evaluation after finetuning:
    final_val_preds = test(loader_dict["val"], eval_model=model.module, epoch=0, desc="Val")
    final_test_preds = test(loader_dict["test"], eval_model=model.module, epoch=0, desc="Test")

    if local_rank == 0:
        val_metrics = task.evaluate(final_val_preds, task.get_table("val"))
        print(f"Best Val metrics: {val_metrics}")

        test_metrics = task.evaluate(final_test_preds)
        print(f"Best Test metrics: {test_metrics}")

        best_metrics_dict = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }
        file_path = os.path.join(output_path, str(args.seed) + ".json")
        with open(file_path, "w") as f:
            json.dump(best_metrics_dict, f, indent=4)
    
    if local_rank == 0:
        print(f"[{args.train_stage.capitalize()} Stage] Training complete. No supervised evaluation performed.")


############################
# 8. Cleanup
############################
dist.destroy_process_group()