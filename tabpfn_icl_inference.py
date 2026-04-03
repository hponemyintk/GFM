import argparse
import json
import math
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except ImportError:
    raise ImportError("TabPFN not installed. Install with: pip install tabpfn")

from model import RelGT
from utils import GloveTextEmbedding, RelGTTokens

torch.autograd.set_detect_anomaly(False)

############################
# 1. Parse arguments
############################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--precompute", action="store_true", default=True)
parser.add_argument("--channels", type=int, default=168)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--gt_conv_type", type=str, default="full")
parser.add_argument("--ablate", type=str, default="none")
parser.add_argument("--gnn_pe_dim", type=int, default=0)
parser.add_argument("--num_neighbors", type=int, default=300)
parser.add_argument("--num_centroids", type=int, default=4096)
parser.add_argument("--ff_dropout", type=float, default=0.1)
parser.add_argument("--attn_dropout", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--sampling_workers", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=str, default="results/debug")
parser.add_argument("--run_name", type=str, default="tabpfn-icl")
parser.add_argument("--cache_dir", type=str,
                    default=os.path.expanduser("~/.cache/relbench_examples"))
parser.add_argument("--amp", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to finetuned.pt (default: {out_dir}/{dataset}/{task}/finetuned.pt)")
parser.add_argument("--tabpfn_context_size", type=int, default=10000,
                    help="Max training samples for TabPFN context")

args = parser.parse_args()
if args.sampling_workers is None:
    args.sampling_workers = max(1, min(32, os.cpu_count() - 1))
if args.checkpoint is None:
    args.checkpoint = os.path.join(args.out_dir, args.dataset, args.task, "finetuned.pt")

############################
# 2. Initialize DDP and set device
############################
dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
world_size = dist.get_world_size()

print(f"[Rank {local_rank}] Using device: {device}")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

############################
# 3. Load dataset, task, and prepare data
############################
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
        train_stage="finetune")
    for split in ["train", "val", "test"]
}

############################
# 4. Create DataLoaders
############################
loaders = {}
for split in ["train", "val", "test"]:
    sampler = DistributedSampler(data[split], shuffle=False, seed=args.seed, drop_last=False)
    loaders[split] = DataLoader(
        data[split],
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=data[split].collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

############################
# 5. Task-specific settings
############################
clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    tune_metric = "mae"
    higher_is_better = False
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

############################
# 6. Build model and load checkpoint
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

if local_rank == 0:
    print(f"Loading checkpoint from: {args.checkpoint}")
state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

if local_rank == 0:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}")

############################
# 7. Feature extraction
############################
@torch.no_grad()
def extract_all_features(model, loader, device, args, desc="Extracting"):
    model.eval()
    features_list = []
    labels_list = []
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
            feats = model.extract_features(
                neighbor_types, node_indices, neighbor_hops, neighbor_times,
                grouped_tf_dict, edge_index=edge_index, batch=batch_vec
            )
        features_list.append(feats.float().cpu().numpy())
        labels_list.append(batch["labels"].cpu().numpy())
        idx_list.append(batch["global_idx"].cpu().numpy())

    local_feats = np.concatenate(features_list, axis=0)
    local_labels = np.concatenate(labels_list, axis=0)
    local_idxs = np.concatenate(idx_list, axis=0)

    # Gather on rank 0
    gathered = [None for _ in range(world_size)] if local_rank == 0 else None
    dist.gather_object((local_idxs, local_feats, local_labels), object_gather_list=gathered, dst=0)

    if local_rank == 0:
        n_samples = len(loader.dataset)
        feat_dim = local_feats.shape[1]
        label_shape = local_labels.shape[1:] if local_labels.ndim > 1 else ()

        all_feats = np.zeros((n_samples, feat_dim), dtype=np.float32)
        all_labels = np.zeros((n_samples, *label_shape), dtype=np.float32)

        for rank_data in gathered:
            g_idx, g_feats, g_labels = rank_data
            for i, idx in enumerate(g_idx):
                all_feats[idx] = g_feats[i]
                all_labels[idx] = g_labels[i]

        return all_feats, all_labels
    return None, None


if local_rank == 0:
    print("Extracting features from all splits...")

train_feats, train_labels = extract_all_features(model, loaders["train"], device, args, desc="Train features")
val_feats, val_labels = extract_all_features(model, loaders["val"], device, args, desc="Val features")
test_feats, test_labels = extract_all_features(model, loaders["test"], device, args, desc="Test features")

if local_rank == 0:
    print(f"Train features: {train_feats.shape}, Val features: {val_feats.shape}, Test features: {test_feats.shape}")

############################
# 8. Prepare TabPFN context
############################
# Rank 0 prepares context, then broadcasts to all ranks
ctx_data = [None]
if local_rank == 0:
    if len(train_feats) > args.tabpfn_context_size:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(train_feats), args.tabpfn_context_size, replace=False)
        ctx_features = train_feats[indices]
        ctx_labels = train_labels[indices]
        print(f"Subsampled training context: {len(train_feats)} -> {args.tabpfn_context_size}")
    else:
        ctx_features = train_feats
        ctx_labels = train_labels
    ctx_data[0] = (ctx_features, ctx_labels)

dist.broadcast_object_list(ctx_data, src=0)
ctx_features, ctx_labels = ctx_data[0]

if local_rank == 0:
    print(f"TabPFN context: {ctx_features.shape[0]} samples, {ctx_features.shape[1]} features")

############################
# 9. DDP TabPFN inference
############################
n_estimators = world_size

def tabpfn_predict_shard(ctx_features, ctx_labels, test_features, task_type):
    if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
        if task_type == TaskType.MULTILABEL_CLASSIFICATION:
            predictions = []
            for col in range(ctx_labels.shape[1]):
                clf = TabPFNClassifier(device=str(device), n_estimators=n_estimators)
                clf.fit(ctx_features, ctx_labels[:, col])
                pred = clf.predict_proba(test_features)
                predictions.append(pred[:, 1] if pred.shape[1] == 2 else pred)
            return np.stack(predictions, axis=1)
        else:
            clf = TabPFNClassifier(device=str(device), n_estimators=n_estimators)
            clf.fit(ctx_features, ctx_labels.ravel())
            pred = clf.predict_proba(test_features)
            return pred[:, 1]
    elif task_type == TaskType.REGRESSION:
        reg = TabPFNRegressor(device=str(device), n_estimators=n_estimators)
        reg.fit(ctx_features, ctx_labels.ravel())
        pred = reg.predict(test_features)
        if clamp_min is not None:
            pred = np.clip(pred, clamp_min, clamp_max)
        return pred


def run_tabpfn_ddp(ctx_features, ctx_labels, all_test_features, task_type, desc="TabPFN"):
    # Rank 0 broadcasts test features, all ranks get a shard
    test_data = [None]
    if local_rank == 0:
        test_data[0] = all_test_features
    dist.broadcast_object_list(test_data, src=0)
    all_test = test_data[0]

    # Shard test data across ranks
    n_total = len(all_test)
    chunk_size = math.ceil(n_total / world_size)
    start = local_rank * chunk_size
    end = min(start + chunk_size, n_total)
    local_test = all_test[start:end]

    if local_rank == 0:
        print(f"  {desc}: {n_total} samples, {chunk_size} per rank, n_estimators={n_estimators}")

    # Each rank runs TabPFN on its shard
    if len(local_test) > 0:
        local_preds = tabpfn_predict_shard(ctx_features, ctx_labels, local_test, task_type)
    else:
        local_preds = np.array([])

    # Gather predictions on rank 0
    gathered = [None for _ in range(world_size)] if local_rank == 0 else None
    dist.gather_object((start, end, local_preds), object_gather_list=gathered, dst=0)

    if local_rank == 0:
        if local_preds.ndim > 1:
            all_preds = np.zeros((n_total, local_preds.shape[1]), dtype=np.float32)
        else:
            all_preds = np.zeros(n_total, dtype=np.float32)
        for s, e, preds in gathered:
            if len(preds) > 0:
                all_preds[s:e] = preds
        return all_preds
    return None


if local_rank == 0:
    print("\nRunning TabPFN ICL inference...")

val_preds = run_tabpfn_ddp(ctx_features, ctx_labels, val_feats if local_rank == 0 else None, task.task_type, desc="Val")
test_preds = run_tabpfn_ddp(ctx_features, ctx_labels, test_feats if local_rank == 0 else None, task.task_type, desc="Test")

############################
# 10. Evaluation
############################
if local_rank == 0:
    val_metrics = task.evaluate(val_preds, task.get_table("val"))
    print(f"\nVal metrics: {val_metrics}")

    test_metrics = task.evaluate(test_preds)
    print(f"Test metrics: {test_metrics}")

    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "tabpfn_context_size": len(ctx_features),
        "tabpfn_n_estimators": n_estimators,
        "feature_dim": ctx_features.shape[1],
    }

    output_path = os.path.join(args.out_dir, args.dataset, args.task)
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"tabpfn_icl_{args.seed}.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {file_path}")

############################
# 11. Cleanup
############################
dist.destroy_process_group()
