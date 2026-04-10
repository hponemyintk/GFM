import argparse
import copy
from contextlib import nullcontext
import json
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict
if os.environ.get('WANDB_API_KEY'):
    from c1_aiml_aem import wandb
else:
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
from utils import (
    GloveTextEmbedding, RelGTTokens, RelGTTokensOnline,
    build_adjacency_csr, build_batch_edge_index_from_selected,
)
from pass_sampler import PASSHeteroSampler

torch.autograd.set_detect_anomaly(False)

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
parser.add_argument("--sampling_workers", type=int, default=None,
                    help="Workers for precompute sampling (default: min(32, cpu_count()-1))")
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
parser.add_argument("--amp", action="store_true", default=False,
                    help="Enable BF16 mixed precision training")
parser.add_argument("--sampler", type=str, default="pass", choices=["random", "pass"],
                    help="Sampling strategy: 'random' (default precomputed) or 'pass' (learned PASS-GNN)")
parser.add_argument("--sample_scope", type=int, default=1200,
                    help="PASS: candidate scope size per seed node")
parser.add_argument("--pass_hidden_dim", type=int, default=32,
                    help="PASS: hidden dim for attention projection")

args = parser.parse_args()
if args.sampling_workers is None:
    args.sampling_workers = max(1, min(32, os.cpu_count() - 1))

############################
# 2. Initialize DDP and set device
############################
dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
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

############################
# 3. Load dataset, task, and prepare data
############################
# Only rank 0 downloads/caches to avoid race conditions that cause segfaults.
if local_rank == 0:
    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: EntityTask = get_task(args.dataset, args.task, download=True)

    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=f"cuda:{local_rank}"), batch_size=256
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
    )

dist.barrier()

# Non-rank-0 processes load from cache after rank 0 has finished downloading.
if local_rank != 0:
    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: EntityTask = get_task(args.dataset, args.task, download=True)

    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)

    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=f"cuda:{local_rank}"), batch_size=256
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
    )

if args.sampler == "pass":
    # PASS sampler: use scope-based online dataset
    data = {
        split: RelGTTokensOnline(
            data=data,
            task=task,
            K=args.num_neighbors,
            sample_scope=args.sample_scope,
            split=split,
            undirected=True,
            precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
            num_workers=args.sampling_workers,
            train_stage=args.train_stage)
            for split in ["train", "val", "test"]
    }
    # Build CSR adjacency for online edge_index construction
    data_cpu = data["train"].data.to("cpu")
    csr_adj = build_adjacency_csr(data_cpu, undirected=True)
    idx_to_type = data["train"].index_to_node_type
    type_to_idx = data["train"].node_type_to_index
else:
    data = {
        split: RelGTTokens(
            data=data,
            task=task,
            K=args.num_neighbors,
            split=split,
            undirected=True,
            precompute=args.precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
            num_workers=args.sampling_workers,
            train_stage=args.train_stage)
            for split in ["train", "val", "test"]
    }

############################
# 4. Create DataLoaders (with a DistributedSampler for training)
############################
train_sampler = DistributedSampler(data["train"], shuffle=True, seed=args.seed)
loader_train = DataLoader(
    data["train"], 
    batch_size=args.batch_size, 
    sampler=train_sampler,
    collate_fn=data["train"].collate,
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
    pin_memory=True)

val_sampler = DistributedSampler(data["val"], shuffle=False, seed=args.seed, drop_last=False)
loader_val = DataLoader(
    data["val"],
    batch_size=args.batch_size,
    sampler=val_sampler,
    # shuffle=False,
    collate_fn=data["val"].collate,
    num_workers=args.num_workers,
    persistent_workers=False,     # val/test are called only once per epoch, the speed up here isn't worth the ram overhead
    pin_memory=True
)

test_sampler = DistributedSampler(data["test"], shuffle=False, seed=args.seed, drop_last=False)
loader_test = DataLoader(
    data["test"],
    batch_size=args.batch_size,
    sampler=test_sampler,
    # shuffle=False,
    collate_fn=data["test"].collate,
    num_workers=args.num_workers,
    persistent_workers=False,     # val/test are called only once per epoch, the speed up here isn't worth the ram overhead
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
    num_nodes=data["train"].data.num_nodes,
    max_neighbor_hop=data["train"].max_neighbor_hop,
    node_type_map=data["train"].node_type_to_index,
    col_names_dict={node_type: data["train"].data[node_type].tf.col_names_dict 
                    for node_type in data["train"].data.node_types},
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

# PASS sampler setup
pass_sampler = None
if args.sampler == "pass":
    pass_sampler = PASSHeteroSampler(
        tfs_encoder=model.module.tfs_encoder,
        num_types=len(data["train"].node_types),
        embed_dim=args.channels,
        hidden_dim=args.pass_hidden_dim,
    ).to(device)
    # Combined optimizer: model params + sampler's own params
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pass_sampler.own_parameters()),
        lr=base_lr, weight_decay=args.weight_decay
    )
    # Keep reference to the underlying data for encoding
    hetero_data = data["train"].data
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

global_step = 0

############################
# 7. Training and Evaluation Loops
############################
def train_supervised(epoch) -> float:
    global global_step
    model.train()
    loss_accum = count_accum = 0
    adjusted_max_steps = max(1, args.max_steps_per_epoch // world_size)
    total_steps = min(len(loader_dict["train"]), adjusted_max_steps)
    
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
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext()
        with amp_ctx:
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

        if step >= adjusted_max_steps:
            break

    return loss_accum / count_accum if count_accum > 0 else float('inf')


def train_pass(epoch) -> float:
    """Training loop with PASS learned sampling + REINFORCE."""
    global global_step
    model.train()
    pass_sampler.train()
    loss_accum = count_accum = 0
    adjusted_max_steps = max(1, args.max_steps_per_epoch // world_size)
    total_steps = min(len(loader_dict["train"]), adjusted_max_steps)
    K = args.num_neighbors

    train_sampler.set_epoch(epoch)

    for step, scope_batch in enumerate(tqdm(loader_dict["train"], total=total_steps, desc="Train (PASS)"), start=1):
        labels = scope_batch["labels"].to(device)

        # 1. Encode candidates and seeds through shared tfs_encoder
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext()
        with amp_ctx:
            candidate_embeds = pass_sampler.encode_candidates(scope_batch, hetero_data, device)
            seed_embeds = pass_sampler.encode_seeds(scope_batch, hetero_data, device)

        # 2. PASS attention-based selection
        scope_counts = scope_batch["scope_count"].to(device)
        selected_idx, dist = pass_sampler(seed_embeds, candidate_embeds, scope_counts, K)

        # 3. Gather selected node info from scope
        scope_types = scope_batch["scope_types"].to(device)
        scope_indices = scope_batch["scope_indices"].to(device)
        scope_hops = scope_batch["scope_hops"].to(device)
        scope_times = scope_batch["scope_times"].to(device)
        seed_type = scope_batch["seed_type"].to(device)
        seed_index = scope_batch["seed_index"].to(device)

        selected_types = torch.gather(scope_types, 1, selected_idx)
        selected_indices = torch.gather(scope_indices, 1, selected_idx)
        selected_hops = torch.gather(scope_hops, 1, selected_idx)
        selected_times = torch.gather(scope_times, 1, selected_idx)

        B = seed_type.shape[0]

        # 4. Prepend seed → [B, K]
        neighbor_types = torch.cat([seed_type.unsqueeze(1), selected_types], dim=1)
        neighbor_indices = torch.cat([seed_index.unsqueeze(1), selected_indices], dim=1)
        neighbor_hops = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device), selected_hops], dim=1)
        neighbor_times = torch.cat([torch.zeros(B, 1, dtype=torch.float, device=device), selected_times], dim=1)

        # 5. Reuse selected K embeddings (avoid re-encoding)
        selected_embeds = pass_sampler.selected_embeds  # [B, K-1, D]
        seed_embed_for_tfs = seed_embeds.unsqueeze(1)    # [B, 1, D]
        preencoded_tfs = torch.cat([seed_embed_for_tfs, selected_embeds], dim=1)  # [B, K, D]

        # 6. Build edge_index on CPU
        edge_index, batch_vec = build_batch_edge_index_from_selected(
            neighbor_types, neighbor_indices, csr_adj, idx_to_type, type_to_idx
        )
        edge_index = edge_index.to(device)
        batch_vec = batch_vec.to(device)

        # 7. Compute node_indices for global module
        first_types = neighbor_types[:, 0].cpu().tolist()
        first_indices = neighbor_indices[:, 0].cpu().tolist()
        node_indices = torch.tensor(
            data["train"].get_global_index(first_types, first_indices),
            dtype=torch.long, device=device
        )

        # 8. Forward through RelGT with pre-encoded tfs
        # Note: uses model.module directly because DDP's forward would mark
        # tfs_encoder as "unused" (it was called outside the model forward),
        # which would break gradient sync. We manually all_reduce all gradients instead.
        optimizer.zero_grad()
        with amp_ctx:
            x_set = model.module.forward_with_preencoded_tfs(
                neighbor_types, node_indices, neighbor_hops, neighbor_times,
                preencoded_tfs,
                edge_index=edge_index, batch=batch_vec
            )
            x_set.retain_grad()  # capture gradient for REINFORCE
            pred = model.module.head(x_set)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            task_loss = loss_fn(pred.float(), labels)

        # 9. Task loss backward — captures x_set.grad for REINFORCE
        task_loss.backward()

        # 10. REINFORCE loss for sampling policy
        if x_set.grad is not None:
            chain_grad = x_set.grad.detach()
            sample_loss = pass_sampler.reinforce_loss(chain_grad)
            sample_loss.backward()

        # 11. Synchronize all gradients across ranks (model.module bypasses DDP hooks)
        if world_size > 1:
            for p in list(model.parameters()) + list(pass_sampler.own_parameters()):
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        clip_grad_norm_(
            list(model.parameters()) + list(pass_sampler.own_parameters()),
            max_norm=1.0
        )
        optimizer.step()

        loss_value = task_loss.detach().item()
        gpu_util, mem_allocated, mem_reserved = get_gpu_stats(gpu_handle, device)
        if local_rank == 0:
            log_dict = {
                "train_loss": loss_value,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "gpu_util_percent": gpu_util,
                "gpu_mem_allocated_MB": mem_allocated,
                "gpu_mem_reserved_MB": mem_reserved,
            }
            if x_set.grad is not None:
                log_dict["sample_loss"] = sample_loss.detach().item()
            wandb.log(log_dict)

        loss_accum += loss_value * pred.size(0)
        count_accum += pred.size(0)
        global_step += 1

        if step >= adjusted_max_steps:
            break

    return loss_accum / count_accum if count_accum > 0 else float('inf')


@torch.no_grad()
def test_pass(loader: DataLoader, eval_model, epoch, desc) -> np.ndarray:
    """Evaluation with PASS sampler (uses learned policy, no REINFORCE)."""
    if loader.sampler is not None and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    eval_model.eval()
    pass_sampler.eval()
    pred_list = []
    idx_list = []
    K = args.num_neighbors
    eval_data = loader.dataset

    for scope_batch in tqdm(loader, desc=desc, disable=(local_rank != 0)):
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext()
        with amp_ctx:
            candidate_embeds = pass_sampler.encode_candidates(scope_batch, hetero_data, device)
            seed_embeds = pass_sampler.encode_seeds(scope_batch, hetero_data, device)

        scope_counts = scope_batch["scope_count"].to(device)
        selected_idx, _ = pass_sampler(seed_embeds, candidate_embeds, scope_counts, K)

        scope_types = scope_batch["scope_types"].to(device)
        scope_indices = scope_batch["scope_indices"].to(device)
        scope_hops = scope_batch["scope_hops"].to(device)
        scope_times = scope_batch["scope_times"].to(device)
        seed_type = scope_batch["seed_type"].to(device)
        seed_index = scope_batch["seed_index"].to(device)
        B = seed_type.shape[0]

        selected_types = torch.gather(scope_types, 1, selected_idx)
        selected_indices = torch.gather(scope_indices, 1, selected_idx)
        selected_hops = torch.gather(scope_hops, 1, selected_idx)
        selected_times = torch.gather(scope_times, 1, selected_idx)

        neighbor_types = torch.cat([seed_type.unsqueeze(1), selected_types], dim=1)
        neighbor_indices = torch.cat([seed_index.unsqueeze(1), selected_indices], dim=1)
        neighbor_hops = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device), selected_hops], dim=1)
        neighbor_times = torch.cat([torch.zeros(B, 1, dtype=torch.float, device=device), selected_times], dim=1)

        preencoded_tfs = torch.cat([seed_embeds.unsqueeze(1), pass_sampler.selected_embeds], dim=1)

        edge_index, batch_vec = build_batch_edge_index_from_selected(
            neighbor_types, neighbor_indices, csr_adj, idx_to_type, type_to_idx
        )
        edge_index = edge_index.to(device)
        batch_vec = batch_vec.to(device)

        first_types = neighbor_types[:, 0].cpu().tolist()
        first_indices = neighbor_indices[:, 0].cpu().tolist()
        node_indices = torch.tensor(
            eval_data.get_global_index(first_types, first_indices),
            dtype=torch.long, device=device
        )

        with amp_ctx:
            x_set = eval_model.forward_with_preencoded_tfs(
                neighbor_types, node_indices, neighbor_hops, neighbor_times,
                preencoded_tfs, edge_index=edge_index, batch=batch_vec
            )
            pred = eval_model.head(x_set)

        if task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().float().cpu().numpy())
        idx_list.append(scope_batch["global_idx"].cpu().numpy())

    local_preds = np.concatenate(pred_list, axis=0) if pred_list else np.array([])
    local_idxs = np.concatenate(idx_list, axis=0) if idx_list else np.array([])

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
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext()
        with amp_ctx:
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
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().float().cpu().numpy())
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

    # Select training and eval functions based on sampler
    _train_fn = train_pass if args.sampler == "pass" else train_supervised
    _test_fn = test_pass if args.sampler == "pass" else test

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_fn(epoch)

        dist.barrier()
        eval_model = model.module  # get the underlying model

        # Run evaluation on the validation set.
        val_pred = _test_fn(loader_dict["val"], eval_model=eval_model, epoch=epoch, desc="Val")
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
                if args.sampler == "pass":
                    torch.save(pass_sampler.state_dict(), os.path.join(output_path, "pass_sampler.pt"))
        dist.barrier()

    if local_rank == 0 and state_dict is not None:
        model.module.load_state_dict(state_dict)
        if args.sampler == "pass":
            # Only load sampler-specific keys (tfs_encoder already loaded via model)
            sampler_sd = torch.load(os.path.join(output_path, "pass_sampler.pt"), map_location=device)
            own_keys = {k: v for k, v in sampler_sd.items() if not k.startswith("tfs_encoder.")}
            pass_sampler.load_state_dict(own_keys, strict=False)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=0)
    # Broadcast PASS sampler own params (not covered by model broadcast)
    if args.sampler == "pass" and pass_sampler is not None:
        for p in pass_sampler.own_parameters():
            dist.broadcast(p.data, src=0)
    dist.barrier()

    # Final evaluation after finetuning:
    final_val_preds = _test_fn(loader_dict["val"], eval_model=model.module, epoch=0, desc="Val")
    final_test_preds = _test_fn(loader_dict["test"], eval_model=model.module, epoch=0, desc="Test")

    if local_rank == 0:
        val_metrics = task.evaluate(final_val_preds, task.get_table("val"))
        print(f"Best Val metrics: {val_metrics}")

        test_metrics = task.evaluate(final_test_preds)
        print(f"Best Test metrics: {test_metrics}")

        best_metrics_dict = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }
        wandb.log({
            **{f"best_val_{k}": v for k, v in val_metrics.items()},
            **{f"best_test_{k}": v for k, v in test_metrics.items()}
        })
        file_path = os.path.join(output_path, str(args.seed) + ".json")
        with open(file_path, "w") as f:
            json.dump(best_metrics_dict, f, indent=4)
    
    if local_rank == 0:
        print(f"[{args.train_stage.capitalize()} Stage] Training complete. No supervised evaluation performed.")


############################
# 8. Cleanup
############################
dist.destroy_process_group()
