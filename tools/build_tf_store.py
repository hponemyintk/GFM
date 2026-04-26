"""Offline build of per-table TF memmap stores for one dataset.

Usage::

    python tools/build_tf_store.py \\
        --dataset rel-f1 \\
        --out_dir ~/.cache/relbench_examples/tf_store/rel-f1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph

from gfm_data.stypes import load_or_generate_stypes
from gfm_data.tf_store import build_dataset_tf_store
from utils import GloveTextEmbedding


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/relbench_examples"))
    return p.parse_args()


def main():
    args = parse_args()
    dataset = get_dataset(args.dataset, download=True)

    stypes_path = Path(args.cache_dir) / args.dataset / "stypes.json"
    cs = load_or_generate_stypes(stypes_path, dataset, upto_test_timestamp=False)

    # Use GPU for text embedding when available -- CPU GloVe on the big
    # datasets (rel-event has ~41M rows) is the difference between a
    # 5-minute and an 11-hour build.
    import torch as _torch
    embed_device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"[tf_store] text embedder device: {embed_device}")

    # upto_test_timestamp=False: entity tables must contain all rows the
    # test split references; temporal leakage is enforced at sampling
    # time via per-row seed_time filtering, not via materialization
    # cutoff. (RelBench's default True drops post-train-cutoff entities
    # which then crash test eval with IndexError.)
    data, _ = make_pkey_fkey_graph(
        dataset.get_db(upto_test_timestamp=False),
        col_to_stype_dict=cs,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=embed_device),
            batch_size=512,
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized_full",
    )

    print(f"Building TF store for {args.dataset} -> {args.out_dir}")
    build_dataset_tf_store(data, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
