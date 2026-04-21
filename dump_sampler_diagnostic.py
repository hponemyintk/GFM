"""Standalone diagnostic: dump (teacher_logits, q_imp) on one val batch.

This script reuses utils.RelGTTokens, model.RelGT, distill_sampler.DistillSampler
but does NOT use DDP, and does not modify any training code. Intended to be run
after `--run_mode distill` has produced phase1.pt + sampler.pt.

Usage:
    python dump_sampler_diagnostic.py \
        --dataset rel-event --task user-repeat \
        --out_dir results/<run> --num_neighbors 128 --channels 128 \
        --num_layers 2 --batch_size 16 --cache_dir ~/.cache/relbench_examples
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything

from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from model import RelGT
from distill_sampler import DistillSampler
from utils import GloveTextEmbedding, RelGTTokens


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir",
                   default=os.path.expanduser("~/.cache/relbench_examples"))
    # model
    p.add_argument("--num_neighbors", type=int, default=300)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--channels", type=int, default=512)
    p.add_argument("--ff_dropout", type=float, default=0.3)
    p.add_argument("--attn_dropout", type=float, default=0.3)
    p.add_argument("--gt_conv_type", default="full")
    p.add_argument("--ablate", default="none")
    p.add_argument("--gnn_pe_dim", type=int, default=0)
    p.add_argument("--num_centroids", type=int, default=4096)
    # runtime
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    return p.parse_args()


def main():
    args = build_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(args.dataset, download=False)
    task = get_task(args.dataset, args.task, download=False)

    stypes_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    with open(stypes_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)

    raw_data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=str(device)), batch_size=256),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
    )

    precomputed_dir = f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}"
    ds = RelGTTokens(
        data=raw_data, task=task, K=args.num_neighbors, split=args.split,
        undirected=True, precompute=True, precomputed_dir=precomputed_dir,
        num_workers=args.num_workers, train_stage="finetune",
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=ds.collate, num_workers=args.num_workers,
    )

    # --- Determine task output shape (matches build_relgt in main_node_ddp.py) ---
    from relbench.base import TaskType
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels
    else:
        raise ValueError(f"Unsupported task type {task.task_type}")

    model = RelGT(
        num_nodes=ds.data.num_nodes,
        max_neighbor_hop=2 + 1,
        node_type_map=ds.node_type_to_index,
        col_names_dict={nt: ds.data[nt].tf.col_names_dict for nt in ds.data.node_types},
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
    for name, p in model.named_parameters():
        if p.dtype == torch.int16:
            p.data = p.data.to(torch.int64)

    # main_node_ddp.py nests checkpoints under {out_dir}/{dataset}/{task}/
    ckpt_dir = os.path.join(args.out_dir, args.dataset, args.task)
    teacher_ckpt = os.path.join(ckpt_dir, "phase1.pt")
    sampler_ckpt = os.path.join(ckpt_dir, "sampler.pt")
    model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sampler = DistillSampler(embed_dim=4 * args.channels, hidden_dim=args.channels).to(device)
    sampler.load_state_dict(torch.load(sampler_ckpt, map_location=device))
    sampler.eval()

    batch = next(iter(loader))
    for k, v in list(batch.items()):
        if hasattr(v, "to"):
            batch[k] = v.to(device)
    grouped_tf_dict = {
        "grouped_tfs": batch["grouped_tfs"],
        "grouped_indices": batch["grouped_indices"],
        "flat_batch_idx": batch["flat_batch_idx"],
        "flat_nbr_idx": batch["flat_nbr_idx"],
    }

    with torch.no_grad():
        _, extras = model(
            batch["neighbor_types"], batch["node_indices"], batch["neighbor_hops"],
            batch["neighbor_times"], grouped_tf_dict,
            edge_index=batch["edge_index"], batch=batch["batch"],
            extract_seed_logits=True, return_base_concat=True,
        )
        teacher_logits = extras["seed_logits"].cpu().numpy()
        base_concat = extras["base_concat"]
        q_imp = sampler(base_concat).cpu().numpy()

    out_path = os.path.join(ckpt_dir, "distill_diagnostic.npz")
    np.savez(out_path, teacher_logits=teacher_logits, q_imp=q_imp)
    print(f"Wrote {out_path}  shape={teacher_logits.shape}")


if __name__ == "__main__":
    main()
