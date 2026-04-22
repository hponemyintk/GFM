import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, List

# Check if WANDB_API_KEY exists and is not an empty string
if os.environ.get("WANDB_API_KEY"):
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
from distill_sampler import DistillSampler, gumbel_top_k
import utils as _utils
from utils import (
    GloveTextEmbedding,
    RelGTTokens,
    RelGTScopeTokens,
    build_edge_index_for_selection,
    build_adjacency_hetero,
    write_curated_hdf5,
)

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

# Distilled-sampler CLI
parser.add_argument("--run_mode", type=str, default="teacher", choices=["teacher", "distill", "joint"])
parser.add_argument("--sample_scope", type=int, default=3000)
parser.add_argument("--sample_temp", type=float, default=1.0)
parser.add_argument("--teacher_ckpt", type=str, default=None,
                    help="Path to phase-1 teacher checkpoint. Auto-derived from out_dir if None.")
parser.add_argument("--sampler_ckpt", type=str, default=None,
                    help="Path to phase-2 sampler checkpoint. Auto-derived from out_dir if None.")
parser.add_argument("--curate_stochastic", action="store_true", default=False,
                    help="Use Gumbel-Top-K at curate time for the train split (ablation).")

args = parser.parse_args()

############################
# 2. Initialize DDP and set device
############################
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

if local_rank == 0:
    args.run_name = f"{args.dataset}-{args.task}-{args.run_mode}-{args.run_name}"

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
dataset: Dataset = get_dataset(args.dataset, download=False)
task: EntityTask = get_task(args.dataset, args.task, download=False)

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

raw_data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=f"cuda:{local_rank}"), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

############################
# 3b. Dataset construction per run_mode
############################
precomputed_dir = f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}"
curated_dir = os.path.join(precomputed_dir, f"curated_{args.num_neighbors}")

def build_relgt_tokens(precomputed_dir_):
    return {
        split: RelGTTokens(
            data=raw_data,
            task=task,
            K=args.num_neighbors,
            split=split,
            undirected=True,
            precompute=args.precompute,
            precomputed_dir=precomputed_dir_,
            num_workers=args.num_workers,
            train_stage=args.train_stage,
        )
        for split in ["train", "val", "test"]
    }

if args.run_mode == "joint":
    # For joint mode, training data are the *curated* subgraphs, written by the
    # curate step below. If they don't exist yet, rank 0 writes them.
    data = None
else:
    # teacher and distill both train on the standard random-K HDF5.
    data = build_relgt_tokens(precomputed_dir)


############################
# 4. Task-specific loss/metric setup
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
# 5. Helper: build RelGT model
############################
def build_relgt(num_nodes, node_type_map, col_names_dict):
    m = RelGT(
        num_nodes=num_nodes,
        max_neighbor_hop=2 + 1,
        node_type_map=node_type_map,
        col_names_dict=col_names_dict,
        col_stats_dict=col_stats_dict,
        local_num_layers=args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
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
    # Cast problematic tensor dtypes before DDP.
    for name, param in m.named_parameters():
        if param.dtype == torch.int16:
            param.data = param.data.to(torch.int64)
    for name, buf in m.named_buffers():
        if buf.dtype == torch.int16:
            buf.data = buf.data.to(torch.int64)
    m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
    return m

output_path = os.path.join(args.out_dir, args.dataset, args.task)
os.makedirs(output_path, exist_ok=True)
world_size = dist.get_world_size()
base_lr = args.lr * world_size

def default_teacher_ckpt():
    return args.teacher_ckpt or os.path.join(output_path, "phase1.pt")

def default_sampler_ckpt():
    return args.sampler_ckpt or os.path.join(output_path, "sampler.pt")


############################
# 6. Shared helpers
############################
def move_batch(batch):
    out = {
        "neighbor_types":   batch["neighbor_types"].to(device),
        "node_indices":     batch["node_indices"].to(device),
        "neighbor_hops":    batch["neighbor_hops"].to(device),
        "neighbor_times":   batch["neighbor_times"].to(device),
        "grouped_tf_dict": {
            'grouped_tfs':    batch['grouped_tfs'],
            'grouped_indices': batch['grouped_indices'],
            'flat_batch_idx':  batch['flat_batch_idx'],
            'flat_nbr_idx':    batch['flat_nbr_idx'],
        },
    }
    # edge_index/batch only exist on the K-sized (RelGTTokens) loader.
    if "edge_index" in batch:
        out["edge_index"] = batch["edge_index"].to(device)
        out["batch_vec"]  = batch["batch"].to(device)
    if batch.get("labels") is not None:
        out["labels"] = batch["labels"].to(device)
    out["global_idx"] = batch["global_idx"]
    return out


############################
# 7. Phase 1 / 3b: supervised training (teacher or joint-finetune)
############################
def run_supervised_training(model_ddp, loader_dict, sampler_obj, ckpt_out,
                            frozen_modules=None):
    """Shared supervised training loop. `frozen_modules` is a list of submodules
    that must stay in eval() mode through training to freeze BN running stats.
    """
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_ddp.parameters()),
        lr=base_lr, weight_decay=args.weight_decay,
    )

    def reapply_frozen_eval():
        if frozen_modules is None:
            return
        # model_ddp wraps the real model in .module
        for m in frozen_modules:
            m.eval()

    global_step = 0

    def train_epoch(epoch):
        nonlocal global_step
        model_ddp.train()
        reapply_frozen_eval()
        loss_accum = count_accum = 0
        total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
        sampler_obj.set_epoch(epoch)

        for step, batch in enumerate(tqdm(loader_dict["train"], total=total_steps, desc="Train"), start=1):
            b = move_batch(batch)
            optimizer.zero_grad()
            pred = model_ddp(
                b["neighbor_types"], b["node_indices"], b["neighbor_hops"],
                b["neighbor_times"], b["grouped_tf_dict"],
                edge_index=b["edge_index"], batch=b["batch_vec"],
            )
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            loss = loss_fn(pred.float(), b["labels"])
            loss.backward()
            clip_grad_norm_(model_ddp.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = loss.detach().item()
            gpu_util, mem_allocated, mem_reserved = get_gpu_stats(gpu_handle, device)
            if local_rank == 0:
                wandb.log({
                    "train_loss": loss_value, "global_step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "gpu_util_percent": gpu_util,
                    "gpu_mem_allocated_MB": mem_allocated,
                    "gpu_mem_reserved_MB": mem_reserved,
                })
            loss_accum += loss_value * pred.size(0)
            count_accum += pred.size(0)
            global_step += 1
            if step >= args.max_steps_per_epoch:
                break
        return loss_accum / count_accum if count_accum > 0 else float('inf')

    @torch.no_grad()
    def evaluate(loader, eval_model, epoch, desc):
        if loader.sampler is not None and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        eval_model.eval()
        pred_list, idx_list = [], []
        for batch in tqdm(loader, desc=desc, disable=(local_rank != 0)):
            b = move_batch(batch)
            pred = eval_model(
                b["neighbor_types"], b["node_indices"], b["neighbor_hops"],
                b["neighbor_times"], b["grouped_tf_dict"],
                edge_index=b["edge_index"], batch=b["batch_vec"],
            )
            if task.task_type == TaskType.REGRESSION:
                pred = torch.clamp(pred, clamp_min, clamp_max)
            if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
                pred = torch.sigmoid(pred)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu().numpy())
            idx_list.append(b["global_idx"].cpu().numpy())

        local_preds = np.concatenate(pred_list, axis=0) if pred_list else np.array([])
        local_idxs  = np.concatenate(idx_list,  axis=0) if idx_list  else np.array([])
        gathered = [None for _ in range(world_size)] if local_rank == 0 else None
        dist.gather_object((local_idxs, local_preds), object_gather_list=gathered, dst=0)

        if local_rank == 0:
            all_preds = np.full((len(loader.dataset),), -100.0)
            for i in range(world_size):
                g_idx, g_pred = gathered[i]
                for idx, pred in zip(g_idx, g_pred):
                    all_preds[idx] = pred
            return all_preds
        return None

    best_val_metric = -math.inf if higher_is_better else math.inf
    state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(epoch)
        dist.barrier()
        eval_model = model_ddp.module
        val_pred = evaluate(loader_dict["val"], eval_model, epoch, "Val")
        if local_rank == 0:
            val_metrics = task.evaluate(val_pred, task.get_table("val"))
            print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")
            wandb.log({
                "epoch": epoch, "epoch_train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })
            if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
                not higher_is_better and val_metrics[tune_metric] <= best_val_metric
            ):
                best_val_metric = val_metrics[tune_metric]
                state_dict = copy.deepcopy(model_ddp.module.state_dict())
                torch.save(state_dict, ckpt_out)
        dist.barrier()

    if local_rank == 0 and state_dict is not None:
        model_ddp.module.load_state_dict(state_dict)
    for param in model_ddp.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model_ddp.buffers():
        dist.broadcast(buf.data, src=0)
    dist.barrier()

    final_val_preds  = evaluate(loader_dict["val"],  model_ddp.module, 0, "Val")
    final_test_preds = evaluate(loader_dict["test"], model_ddp.module, 0, "Test")
    if local_rank == 0:
        val_metrics  = task.evaluate(final_val_preds,  task.get_table("val"))
        test_metrics = task.evaluate(final_test_preds)
        print(f"Best Val metrics: {val_metrics}")
        print(f"Best Test metrics: {test_metrics}")
        wandb.log({
            **{f"best_val_{k}": v for k, v in val_metrics.items()},
            **{f"best_test_{k}": v for k, v in test_metrics.items()},
        })
        file_path = os.path.join(output_path, str(args.seed) + f"_{args.run_mode}.json")
        with open(file_path, "w") as f:
            json.dump({"val_metrics": val_metrics, "test_metrics": test_metrics}, f, indent=4)


############################
# 8. Mode: teacher
############################
def mode_teacher():
    model = build_relgt(
        num_nodes=data["train"].data.num_nodes,
        node_type_map=data["train"].node_type_to_index,
        col_names_dict={nt: data["train"].data[nt].tf.col_names_dict
                        for nt in data["train"].data.node_types},
    )
    model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if local_rank == 0:
        print(model_ddp)
        total_params = sum(p.numel() for p in model_ddp.parameters() if p.requires_grad)
        print(f"Total trainable params: {total_params}")
        args.model_parameters = total_params
        wandb.init(project="rel-gt-expts", name=args.run_name, config=vars(args))

    loaders = _make_loaders(data)
    run_supervised_training(
        model_ddp, loaders, loaders["train"].sampler,
        ckpt_out=default_teacher_ckpt(),
        frozen_modules=None,
    )


############################
# 9. Mode: distill
############################
def mode_distill():
    teacher = build_relgt(
        num_nodes=data["train"].data.num_nodes,
        node_type_map=data["train"].node_type_to_index,
        col_names_dict={nt: data["train"].data[nt].tf.col_names_dict
                        for nt in data["train"].data.node_types},
    )
    tpath = default_teacher_ckpt()
    if not os.path.exists(tpath):
        raise FileNotFoundError(f"Teacher checkpoint not found at {tpath} — run phase 1 first.")
    teacher.load_state_dict(torch.load(tpath, map_location=device))
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    # No DDP wrap on teacher — no gradients to sync.

    sampler = DistillSampler(
        embed_dim=4 * args.channels,
        hidden_dim=args.channels,
        num_node_types=len(data["train"].node_types),
        num_heads=args.num_heads,
    ).to(device)
    sampler_ddp = DDP(sampler, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(sampler_ddp.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    loaders = _make_loaders(data)
    train_sampler = loaders["train"].sampler

    if local_rank == 0:
        total_params = sum(p.numel() for p in sampler_ddp.parameters() if p.requires_grad)
        print(f"Sampler trainable params: {total_params}")
        wandb.init(project="rel-gt-expts", name=args.run_name, config=vars(args))

    global_step = 0
    best_val_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        sampler_ddp.train()
        train_sampler.set_epoch(epoch)
        loss_accum = count_accum = 0
        total_steps = min(len(loaders["train"]), args.max_steps_per_epoch)

        for step, batch in enumerate(tqdm(loaders["train"], total=total_steps, desc="Distill"), start=1):
            b = move_batch(batch)
            with torch.no_grad():
                _, extras = teacher(
                    b["neighbor_types"], b["node_indices"], b["neighbor_hops"],
                    b["neighbor_times"], b["grouped_tf_dict"],
                    edge_index=b["edge_index"], batch=b["batch_vec"],
                    extract_seed_logits=True, return_base_concat=True,
                )
            teacher_logits = extras["seed_logits"]          # [B, H, K]
            base_concat    = extras["base_concat"]          # [B, K, 4*C]

            q_imp = sampler_ddp(base_concat, b["neighbor_types"])
            loss = DistillSampler.distillation_loss(q_imp, teacher_logits)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(sampler_ddp.parameters(), max_norm=1.0)
            optimizer.step()

            lv = loss.detach().item()
            if local_rank == 0:
                wandb.log({"distill_loss": lv, "global_step": global_step,
                           "lr": optimizer.param_groups[0]["lr"]})
            loss_accum += lv * base_concat.size(0)
            count_accum += base_concat.size(0)
            global_step += 1
            if step >= args.max_steps_per_epoch:
                break

        # Val loss
        sampler_ddp.eval()
        val_loss_sum = val_count = 0.0
        with torch.no_grad():
            for batch in tqdm(loaders["val"], desc="DistillVal", disable=(local_rank != 0)):
                b = move_batch(batch)
                _, extras = teacher(
                    b["neighbor_types"], b["node_indices"], b["neighbor_hops"],
                    b["neighbor_times"], b["grouped_tf_dict"],
                    edge_index=b["edge_index"], batch=b["batch_vec"],
                    extract_seed_logits=True, return_base_concat=True,
                )
                q_imp = sampler_ddp(extras["base_concat"], b["neighbor_types"])
                vl = DistillSampler.distillation_loss(q_imp, extras["seed_logits"]).item()
                val_loss_sum += vl * extras["base_concat"].size(0)
                val_count += extras["base_concat"].size(0)
        val_loss = val_loss_sum / max(val_count, 1)
        dist.barrier()

        if local_rank == 0:
            train_loss = loss_accum / max(count_accum, 1)
            print(f"Epoch {epoch:02d} distill_train={train_loss:.5f} distill_val={val_loss:.5f}")
            wandb.log({"epoch": epoch, "epoch_distill_train": train_loss,
                       "epoch_distill_val": val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(sampler_ddp.module.state_dict(), default_sampler_ckpt())

    dist.barrier()


############################
# 10. Mode: joint = 3a curate + 3b retrain
############################
def _curate_one_split(teacher, sampler, scope_dataset, out_path, stochastic: bool, split: str):
    """Rank-0-only. Produces a phase-3 HDF5 in RelGTTokens format."""
    N = len(scope_dataset)
    K = args.num_neighbors

    types_out   = np.zeros((N, K), dtype=np.int16)
    indices_out = np.zeros((N, K), dtype=np.int32)
    hops_out    = np.zeros((N, K), dtype=np.int8)
    times_out   = np.zeros((N, K), dtype=np.float32)
    edges_out   : List[np.ndarray] = [None] * N

    loader = DataLoader(
        scope_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=scope_dataset.collate,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True,
    )

    teacher.eval()
    sampler.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Curate[{split}]"):
            nt   = batch["neighbor_types"].to(device)
            nidx = batch["node_indices"].to(device)
            nh   = batch["neighbor_hops"].to(device)
            ntm  = batch["neighbor_times"].to(device)
            gtf  = {
                'grouped_tfs':    batch['grouped_tfs'],
                'grouped_indices': batch['grouped_indices'],
                'flat_batch_idx':  batch['flat_batch_idx'],
                'flat_nbr_idx':    batch['flat_nbr_idx'],
            }

            # Lightweight path: only the frozen base encoders + LNs. O(S) memory,
            # not O(S^2) — avoids running the transformer over scope=3000 tokens.
            base_concat = teacher.base_concat_forward(nt, nh, ntm, gtf)     # [B, S, 4C]
            q_imp = sampler(base_concat, nt)                    # [B, H, S]
            score = q_imp.mean(dim=1)                           # [B, S] — reduce heads

            sel = gumbel_top_k(score, k=K - 1, temperature=args.sample_temp,
                               stochastic=stochastic)            # [B, K-1] in [1, S)

            nt_cpu   = batch["neighbor_types"].numpy()
            nidx_cpu = batch["neighbor_indices"].numpy()
            nh_cpu   = batch["neighbor_hops"].numpy()
            ntm_cpu  = batch["neighbor_times"].numpy()
            sel_cpu  = sel.cpu().numpy()
            global_idxs = batch["global_idx"].tolist()
            B = nt.shape[0]

            for b in range(B):
                g = global_idxs[b]
                picks = np.concatenate([[0], sel_cpu[b]])       # seed at 0, then selected
                types_out[g]   = nt_cpu[b, picks]
                indices_out[g] = nidx_cpu[b, picks]
                hops_out[g]    = nh_cpu[b, picks]
                times_out[g]   = ntm_cpu[b, picks]
                eidx = build_edge_index_for_selection(
                    types_out[g], indices_out[g],
                    scope_dataset.index_to_node_type,
                )
                edges_out[g] = eidx

    write_curated_hdf5(out_path, types_out, indices_out, hops_out, times_out, edges_out)


def mode_joint():
    # ---- 3a: ensure curated HDF5s exist ----
    splits = ["train", "val", "test"]
    # RelGTTokens reader expects {precomputed_dir}/{K}/{split}.h5, so we mirror
    # that layout here when writing curated HDF5s.
    curated_paths = {s: os.path.join(curated_dir, str(args.num_neighbors), f"{s}.h5")
                     for s in splits}

    if local_rank == 0:
        missing = [s for s in splits if not os.path.exists(curated_paths[s])]
        if missing:
            print(f"[joint] Curating missing splits: {missing}")
            # Ensure GLOBAL_ADJ is populated on this rank regardless of whether
            # the scope HDF5 is cached — edge_index rebuild needs it.
            if _utils.GLOBAL_ADJ is None:
                print("[joint] Building adjacency for edge_index reconstruction...")
                _utils.GLOBAL_ADJ = build_adjacency_hetero(raw_data, undirected=True)
            # Build scope datasets (precompute if needed).
            scope_sets = {
                s: RelGTScopeTokens(
                    data=raw_data, task=task,
                    sample_scope=args.sample_scope, split=s,
                    undirected=True, precompute=True,
                    precomputed_dir=precomputed_dir,
                    num_workers=args.num_workers,
                )
                for s in missing
            }
            # Load teacher + sampler.
            teacher = build_relgt(
                num_nodes=list(scope_sets.values())[0].data.num_nodes,
                node_type_map=list(scope_sets.values())[0].node_type_to_index,
                col_names_dict={nt: list(scope_sets.values())[0].data[nt].tf.col_names_dict
                                for nt in list(scope_sets.values())[0].data.node_types},
            )
            teacher.load_state_dict(torch.load(default_teacher_ckpt(), map_location=device))
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher.eval()

            sampler_m = DistillSampler(
                embed_dim=4 * args.channels,
                hidden_dim=args.channels,
                num_node_types=len(raw_data.node_types),
                num_heads=args.num_heads,
            ).to(device)
            sampler_m.load_state_dict(torch.load(default_sampler_ckpt(), map_location=device))
            for p in sampler_m.parameters():
                p.requires_grad_(False)
            sampler_m.eval()

            for s in missing:
                # Stochastic only for train split if flag is set; val/test always deterministic.
                stoch = args.curate_stochastic and (s == "train")
                os.makedirs(os.path.dirname(curated_paths[s]), exist_ok=True)
                _curate_one_split(teacher, sampler_m, scope_sets[s], curated_paths[s], stoch, s)
            del teacher, sampler_m
            torch.cuda.empty_cache()
        else:
            print(f"[joint] All curated HDF5s exist at {curated_dir}. Skipping curate step.")
    dist.barrier()

    # ---- 3b: train RelGT on curated data with partial freeze ----
    curated_data = build_relgt_tokens(curated_dir)
    model = build_relgt(
        num_nodes=curated_data["train"].data.num_nodes,
        node_type_map=curated_data["train"].node_type_to_index,
        col_names_dict={nt: curated_data["train"].data[nt].tf.col_names_dict
                        for nt in curated_data["train"].data.node_types},
    )
    # Load phase-1 weights.
    model.load_state_dict(torch.load(default_teacher_ckpt(), map_location=device))

    frozen_modules = [
        model.type_encoder, model.hop_encoder, model.time_encoder, model.tfs_encoder,
        model.layer_norm_type, model.layer_norm_hop, model.layer_norm_time, model.layer_norm_tfs,
        model.pe_encoder, model.layer_norm_pe,
    ]
    for m in frozen_modules:
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model_ddp.parameters() if p.requires_grad)
        print(f"Phase-3 trainable params: {total_params}")
        wandb.init(project="rel-gt-expts", name=args.run_name, config=vars(args))

    loaders = _make_loaders(curated_data)
    run_supervised_training(
        model_ddp, loaders, loaders["train"].sampler,
        ckpt_out=os.path.join(output_path, "phase3.pt"),
        frozen_modules=frozen_modules,
    )


############################
# 11. DataLoader construction
############################
def _make_loaders(datasets):
    train_sampler = DistributedSampler(datasets["train"], shuffle=True, seed=args.seed)
    loader_train = DataLoader(
        datasets["train"], batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=datasets["train"].collate, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, pin_memory=True,
    )
    val_sampler = DistributedSampler(datasets["val"], shuffle=False, seed=args.seed, drop_last=False)
    loader_val = DataLoader(
        datasets["val"], batch_size=args.batch_size, sampler=val_sampler,
        collate_fn=datasets["val"].collate, num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0), pin_memory=True,
    )
    test_sampler = DistributedSampler(datasets["test"], shuffle=False, seed=args.seed, drop_last=False)
    loader_test = DataLoader(
        datasets["test"], batch_size=args.batch_size, sampler=test_sampler,
        collate_fn=datasets["test"].collate, num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0), pin_memory=True,
    )
    return {"train": loader_train, "val": loader_val, "test": loader_test}


############################
# 12. Dispatch
############################
if args.run_mode == "teacher":
    mode_teacher()
elif args.run_mode == "distill":
    mode_distill()
elif args.run_mode == "joint":
    mode_joint()
else:
    raise ValueError(f"Unknown run_mode: {args.run_mode}")

dist.destroy_process_group()
