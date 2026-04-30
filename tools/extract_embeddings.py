"""Extract per-seed embeddings from a pretrained backbone.

Loads the backbone from a Phase-2 checkpoint (best_full.pt /
best_backbone.pt + backbone_meta.json + backbone_schema.pt), runs
forward(task_id=None) over the requested split's seeds, and dumps
``[N, channels]`` embeddings + labels + global_idx as a ``.pt``
file per split.

The output is consumable by:
  * ``tools/finetune_head.py`` (PR 3.2) -- trains a fresh head on
    these embeddings
  * ``tools/tabpfn_eval.py`` (PR 3.3) -- pure post-hoc TabPFN

Usage::

    python -m tools.extract_embeddings \\
        --backbone_meta   <run>/backbone_meta.json \\
        --backbone_weights <run>/best_backbone.pt \\
        --backbone_schema  <run>/backbone_schema.pt \\
        --dataset rel-f1 --task driver-top3 \\
        --split all \\
        --out_dir <run>/embeddings/

For Phase 4 holdout-task: same dataset as training, --task is the
held-out one. For Phase 5 cross-dataset: the dataset is held-out
(typically not in the saved backbone_meta's registered datasets);
pass ``--register_new_dataset`` so this script computes stats via
``tools.compute_dataset_stats`` and registers them on the encoder
before forward.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _resolve_precomputed_dir(args) -> str:
    """Pick the directory the per-row neighbor HDF5 cache lives under.

    ``--precomputed_dir`` overrides everything; when unset we fall back
    to ``<cache_dir>/precomputed/<dataset>/<task>``. The override lets
    the launcher run several extractions in parallel (one per seed /
    one per task) without them racing on the same shared HDF5 files.
    """
    if getattr(args, "precomputed_dir", None):
        return args.precomputed_dir
    return os.path.join(
        args.cache_dir, "precomputed", args.dataset, args.task,
    )


def _build_loader(args, split: str, cache, task) -> Tuple[object, object]:
    """Construct (dataset, loader) for one split. Mirrors the per-task
    val/test loader construction in train_multi_task.py:606-619 so the
    embedding extraction has identical batch shape semantics.
    """
    from gfm_data import TaskTokens, collate_single_task

    precomputed_dir = _resolve_precomputed_dir(args)
    tok = TaskTokens(
        cache=cache, task=task, K=args.num_neighbors,
        split=split, mode=args.mode,
        precompute=args.precompute,
        precomputed_dir=precomputed_dir,
        shards_dir=None,
        train_stage="finetune",
    )
    # Adopt train target stats on val/test for regression z-score
    # symmetry. Train/val/test all read the same target_mean/std at
    # __getitem__; any divergence here would shift the labels we
    # save out for downstream heads.
    if split != "train":
        train_tok = TaskTokens(
            cache=cache, task=task, K=args.num_neighbors,
            split="train", mode=args.mode,
            precompute=args.precompute,
            precomputed_dir=precomputed_dir,
            shards_dir=None,
            train_stage="finetune",
        )
        if train_tok.target_mean is not None:
            tok.adopt_target_stats(train_tok.target_mean, train_tok.target_std)

    sampler = DistributedSampler(
        tok, shuffle=False, seed=args.seed, drop_last=False,
    ) if torch.distributed.is_initialized() else None
    loader = DataLoader(
        tok, batch_size=args.batch_size,
        sampler=sampler, shuffle=False,
        collate_fn=lambda b: collate_single_task(tok, b),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return tok, loader


@torch.no_grad()
def _extract_split(model, loader, device) -> Dict[str, torch.Tensor]:
    """Forward(task_id=None) over a split; collect embeddings + labels
    + global_idx. Returns CPU tensors for downstream save."""
    model.eval()
    emb_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    idx_list: List[torch.Tensor] = []

    for batch in loader:
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
        # Backbone returns [B, channels] when called directly (no head
        # wrap). Adoption-loaded RelGT outputs embeddings since
        # out_channels=channels (PR 2.2 convention).
        emb = model(
            n_types, n_idx, n_hop, n_t, grouped,
            edge_index=edge_index, batch=batch_v,
        )
        # If the loaded artifact still has the trailing MLP head from
        # training (out_channels=1 for single-task), emb is [B, 1].
        # Caller is expected to use a backbone saved with
        # out_channels=channels; but be defensive.
        if emb.dim() == 3:
            # [B, K, C] -- this happens if the model is the full
            # encoder stack without the seed-only collapse. Take the
            # seed token (position 0).
            emb = emb[:, 0, :]
        emb_list.append(emb.detach().cpu())

        labels = batch.get("labels")
        if labels is not None:
            label_list.append(labels.detach().cpu())
        idx_list.append(batch["global_idx"].detach().cpu())

    out = {
        "embeddings": torch.cat(emb_list, dim=0) if emb_list else torch.zeros(0),
        "global_idx": torch.cat(idx_list, dim=0) if idx_list else torch.zeros(0, dtype=torch.long),
    }
    if label_list:
        out["labels"] = torch.cat(label_list, dim=0)
    return out


def _maybe_register_new_dataset(backbone, args) -> None:
    """If --register_new_dataset is set, walk the dataset's TF store +
    HeteroData metadata and call backbone.tfs_encoder.register_dataset
    so adoption-time prefixed types + their stats get added to the
    encoder."""
    if not args.register_new_dataset:
        return
    from tools.compute_dataset_stats import compute_dataset_stats

    # Match build_tf_store / _build_cache: read from tf_store_full when
    # the adoption build is in full-graph mode, else tf_store.
    tf_suffix = "tf_store_full" if args.full_graph else "tf_store"
    tf_store_root = os.path.join(args.cache_dir, tf_suffix, args.dataset)
    if not os.path.exists(tf_store_root):
        raise FileNotFoundError(
            f"--register_new_dataset requires a TF store at "
            f"{tf_store_root}; run tools/build_tf_store.py "
            f"{'--full_graph ' if args.full_graph else ''}first."
        )
    col_stats = compute_dataset_stats(
        tf_store_root, name_prefix=f"{args.dataset}::",
    )
    # col_names_dict has to be discovered alongside; the easiest
    # source is the TF store's per-table meta.json. Build it from
    # col_stats's keys (each table is an entry).
    import json as _json
    col_names_dict: Dict[str, Dict] = {}
    for prefixed in col_stats.keys():
        bare = prefixed.split("::", 1)[1]
        meta_path = Path(tf_store_root) / bare / "meta.json"
        with open(meta_path) as f:
            tm = _json.load(f)
        # Re-hydrate stype keys (they're stringified in JSON).
        import torch_frame as _tf
        rev = {
            "numerical": _tf.numerical,
            "categorical": _tf.categorical,
            "multicategorical": _tf.multicategorical,
            "timestamp": _tf.timestamp,
            "embedding": _tf.embedding,
        }
        col_names_dict[prefixed] = {
            rev[s]: list(cs) for s, cs in tm["col_names_dict"].items()
            if s in rev
        }

    # The new prefixed types need integer ids that don't collide
    # with the encoder's existing ids. Append them after the existing
    # max.
    existing = backbone.tfs_encoder.node_type_map
    next_idx = (max(existing.values()) + 1) if existing else 0
    new_node_type_map = {}
    for prefixed in col_stats.keys():
        if prefixed in existing:
            new_node_type_map[prefixed] = existing[prefixed]
        else:
            new_node_type_map[prefixed] = next_idx
            next_idx += 1

    backbone.tfs_encoder.register_dataset(
        node_type_map=new_node_type_map,
        col_names_dict=col_names_dict,
        col_stats_dict=col_stats,
    )
    print(
        f"[extract] register_dataset registered "
        f"{len(new_node_type_map)} prefixed types from "
        f"{tf_store_root}"
    )


def _build_cache(args):
    """Materialize a DatasetGraphCache for the target dataset/task.

    Mirrors what train_multi_task._build_caches_and_tokens does but
    for one (dataset, task) pair. Honors ``args.full_graph`` so the
    adoption build matches the pretraining mode -- mismatched modes
    yield different seed-id indexing and produce embeddings that
    don't align with the pretrained backbone.
    """
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from relbench.modeling.graph import make_pkey_fkey_graph
    from torch_frame.config.text_embedder import TextEmbedderConfig

    from gfm_data import DatasetGraphCache
    from gfm_data.stypes import (
        load_or_generate_stypes as _load_stypes,
        filter_to_db_columns as _filter_stypes,
    )
    from utils import GloveTextEmbedding

    dset = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)
    stypes_path = Path(args.cache_dir) / args.dataset / "stypes.json"
    upto = not args.full_graph
    cs = _load_stypes(stypes_path, dset, upto_test_timestamp=upto)
    db = dset.get_db(upto_test_timestamp=upto)
    cs = _filter_stypes(cs, db)
    mat_suffix = "materialized_full" if args.full_graph else "materialized"
    tf_suffix = "tf_store_full" if args.full_graph else "tf_store"
    data, _col_stats = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=cs,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device="cpu"), batch_size=256,
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/{mat_suffix}",
    )
    cache = DatasetGraphCache(
        data=data, undirected=True, name_prefix=args.dataset,
        tf_store_root=os.path.join(args.cache_dir, tf_suffix, args.dataset)
        if args.use_tf_store else None,
    )
    return cache, task


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backbone_meta", required=True, type=str)
    p.add_argument("--backbone_weights", required=True, type=str)
    p.add_argument("--backbone_schema", required=True, type=str)
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--task", required=True, type=str)
    p.add_argument(
        "--split", default="all", choices=["train", "val", "test", "all"],
    )
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument(
        "--precomputed_dir", type=str, default=None,
        help="Override the per-row neighbor HDF5 cache location. Default "
             "is <cache_dir>/precomputed/<dataset>/<task>. Pass a unique "
             "path per parallel extraction (e.g. one per seed) so several "
             "extracts can run concurrently without racing on the same "
             "HDF5 files.",
    )
    p.add_argument("--num_neighbors", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode", default="hdf5",
        choices=["hdf5", "streaming", "precomputed_shards"],
    )
    p.add_argument("--precompute", action="store_true", default=True)
    p.add_argument(
        "--cache_dir", type=str,
        default=os.path.expanduser("~/.cache/relbench_examples"),
    )
    p.add_argument(
        "--use_tf_store", action="store_true", default=False,
        help="Use memmap TF store via DatasetGraphCache(tf_store_root=...). "
             "Required for big datasets where the in-RAM tf doesn't fit.",
    )
    p.add_argument(
        "--register_new_dataset", action="store_true", default=False,
        help="Phase-5 cross-dataset path: compute stats via "
             "tools.compute_dataset_stats and call register_dataset "
             "on the loaded encoder before forward. Required when the "
             "target dataset wasn't in the saved backbone_meta.",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--full_graph", action="store_true", default=False,
        help="Use upto_test_timestamp=False so the materialization "
             "includes entities created after train_cutoff. Must match "
             "what the backbone was pretrained with: if pretraining used "
             "--full_graph, adoption must too, otherwise the seed-id "
             "indexing differs between train and adoption. "
             "See docs/truncated_graph_caveat.md.",
    )
    args = p.parse_args(argv)

    # Expand user paths.
    args.cache_dir = os.path.expanduser(args.cache_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Seed the per-seed-row neighbor sampling. The sampler at
    # gfm_data/sampler.py uses Python's ``random.sample`` to pick K of
    # N neighbors, so the embeddings depend on ``random.seed(args.seed)``
    # being set BEFORE the precompute pass. Caller must also wipe the
    # cached HDF5 shards (~/.cache/relbench_examples/precomputed/...)
    # between seeded runs, otherwise the precompute gets reused.
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Load backbone (frozen, eval mode by default).
    from model import RelGT
    print(f"[extract] loading backbone from {args.backbone_meta}")
    backbone = RelGT.load_backbone(
        args.backbone_meta, args.backbone_weights, args.backbone_schema,
        map_location=args.device,
    ).to(args.device)

    # 2. Optional: register the target dataset if it wasn't part of training.
    _maybe_register_new_dataset(backbone, args)

    # 3. Build the dataset graph cache + task.
    print(f"[extract] building cache for {args.dataset}.{args.task}")
    cache, task = _build_cache(args)

    # 4. Extract per requested split.
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        print(f"[extract] split={split} ...")
        tok, loader = _build_loader(args, split, cache, task)
        result = _extract_split(backbone, loader, args.device)
        result["split"] = split
        result["task"] = args.task
        result["dataset"] = args.dataset
        result["channels"] = int(backbone.tfs_encoder.channels)
        out_path = os.path.join(out_dir, f"{split}.pt")
        torch.save(result, out_path)
        print(
            f"[extract]   saved {result['embeddings'].shape[0]} embeddings "
            f"({result['embeddings'].shape}) -> {out_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
