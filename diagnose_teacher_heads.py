"""Inspect per-head attention structure of the teacher's last encoder layer.

Read-only diagnostic: uses forward hooks on q_proj/k_proj of the final encoder
to capture per-head Q, K without modifying local_module.py.

Reports per-head pre-softmax stats, pairwise correlation between heads,
and correlation / top-K overlap with a 'gold' ranking (sum over heads of
post-softmax attention — what the teacher actually aggregates).
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame import stype
from torch_geometric import seed_everything

from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.tasks import get_task

from model import RelGT
from utils import GloveTextEmbedding, RelGTTokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir",
                   default=os.path.expanduser("~/.cache/relbench_examples"))
    p.add_argument("--num_neighbors", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--ff_dropout", type=float, default=0.3)
    p.add_argument("--attn_dropout", type=float, default=0.3)
    p.add_argument("--gt_conv_type", default="full")
    p.add_argument("--ablate", default="none")
    p.add_argument("--gnn_pe_dim", type=int, default=0)
    p.add_argument("--num_centroids", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="val")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(args.dataset, download=False)
    task = get_task(args.dataset, args.task, download=False)

    stypes_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    with open(stypes_path) as f:
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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=ds.collate, num_workers=args.num_workers)

    if task.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
        out_channels = 1
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels
    else:
        raise ValueError(task.task_type)

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
    for p in model.parameters():
        if p.dtype == torch.int16:
            p.data = p.data.to(torch.int64)

    ckpt_dir = os.path.join(args.out_dir, args.dataset, args.task)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "phase1.pt"),
                                     map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Last encoder layer of the last RelGTLayer's local_module.
    last_enc = model.convs[-1].local_module.layers[-1]

    captured = {}
    def mk_hook(name):
        def hook(_m, _inp, out):
            captured[name] = out.detach()
        return hook
    h_q = last_enc.q_proj.register_forward_hook(mk_hook("Q"))
    h_k = last_enc.k_proj.register_forward_hook(mk_hook("K"))

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
        _ = model(
            batch["neighbor_types"], batch["node_indices"], batch["neighbor_hops"],
            batch["neighbor_times"], grouped_tf_dict,
            edge_index=batch["edge_index"], batch=batch["batch"],
        )
    h_q.remove(); h_k.remove()

    Q = captured["Q"]                       # [B, L, D]
    K = captured["K"]
    B, L, D = Q.shape
    H = args.num_heads
    dh = D // H
    assert D % H == 0

    Q = Q.view(B, L, H, dh).transpose(1, 2)
    K = K.view(B, L, H, dh).transpose(1, 2)

    q_seed = Q[:, :, 0:1, :]
    dots = torch.matmul(q_seed, K.transpose(-2, -1)).squeeze(2) / math.sqrt(dh)
    attn = F.softmax(dots, dim=-1)
    gold = attn.sum(dim=1)

    # Drop self-entry (col 0).
    dots = dots[:, :, 1:]
    attn = attn[:, :, 1:]
    gold = gold[:, 1:]
    target_mean_pre  = dots.mean(dim=1)
    target_mean_post = attn.mean(dim=1)

    d = dots.cpu().numpy()
    g = gold.cpu().numpy()
    tt_pre  = target_mean_pre.cpu().numpy()
    tt_post = target_mean_post.cpu().numpy()

    print(f"\nBatch={B}  Heads={H}  d_head={dh}  candidates={d.shape[-1]}")

    print("\n--- per-head pre-softmax stats ---")
    for h in range(H):
        print(f"  head {h}: mean={d[:, h].mean():+.3f}  std={d[:, h].std():.3f}  "
              f"range=[{d[:, h].min():+.2f}, {d[:, h].max():+.2f}]")
    print(f"  MEAN (current target): mean={tt_pre.mean():+.3f}  std={tt_pre.std():.3f}")
    print(f"  GOLD (sum post-softmax): mean={g.mean():+.3f}  std={g.std():.3f}  "
          f"range=[{g.min():+.2f}, {g.max():+.2f}]")

    print("\n--- pairwise Pearson between heads (pre-softmax, seed-avg) ---")
    pp = np.zeros((H, H))
    for h1 in range(H):
        for h2 in range(H):
            pp[h1, h2] = np.mean([pearsonr(d[b, h1], d[b, h2]).statistic for b in range(B)])
    print(np.round(pp, 2))

    print("\n--- each head vs gold (sum post-softmax) ---")
    for h in range(H):
        pear = np.mean([pearsonr(d[b, h], g[b]).statistic for b in range(B)])
        spear = np.mean([spearmanr(d[b, h], g[b]).statistic for b in range(B)])
        def ov(k):
            return np.mean([
                len(set(np.argsort(d[b, h])[-k:]) & set(np.argsort(g[b])[-k:])) / k
                for b in range(B)
            ])
        print(f"  head {h}: Pearson={pear:+.3f}  Spearman={spear:+.3f}  "
              f"top-16 ov={ov(16):.3f}  top-32 ov={ov(32):.3f}")

    def metrics(arr, name):
        pear = np.mean([pearsonr(arr[b], g[b]).statistic for b in range(B)])
        spear = np.mean([spearmanr(arr[b], g[b]).statistic for b in range(B)])
        def ov(k):
            return np.mean([
                len(set(np.argsort(arr[b])[-k:]) & set(np.argsort(g[b])[-k:])) / k
                for b in range(B)
            ])
        print(f"  {name}: Pearson={pear:+.3f}  Spearman={spear:+.3f}  "
              f"top-16 ov={ov(16):.3f}  top-32 ov={ov(32):.3f}")

    print("\n--- candidate targets vs gold ---")
    metrics(tt_pre,  "mean pre-softmax (CURRENT)")
    metrics(tt_post, "mean post-softmax        ")


if __name__ == "__main__":
    main()
