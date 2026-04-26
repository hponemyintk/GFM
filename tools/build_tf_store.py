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

from gfm_data.tf_store import build_dataset_tf_store
from utils import GloveTextEmbedding


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/relbench_examples"))
    return p.parse_args()


def _load_or_generate_stypes(stypes_path: Path, dataset):
    """Load stypes.json defensively; regenerate if missing or corrupt.

    Older runs (or non-strict JSON loaders) can leave NaN / null / float
    entries in the stored stypes dict. torch_frame inside
    ``make_pkey_fkey_graph`` then calls ``.split(...)`` on those values
    and crashes with ``'float' object has no attribute 'split'``. This
    helper validates that every value is either a string or None;
    otherwise it regenerates the file from scratch and rewrites it
    using each stype's ``.value`` (clean strings, no ``default=str``
    surprises).
    """
    cs = None
    if stypes_path.exists():
        try:
            with open(stypes_path) as f:
                raw = json.load(f)
            ok = isinstance(raw, dict) and all(
                isinstance(c2s, dict)
                and all(isinstance(v, (str, type(None))) for v in c2s.values())
                for c2s in raw.values()
            )
            if ok:
                cs = raw
            else:
                print(f"[stypes] {stypes_path} has non-string entries; regenerating",
                      file=sys.stderr)
        except Exception as e:
            print(f"[stypes] {stypes_path} unreadable ({e}); regenerating",
                  file=sys.stderr)

    if cs is None:
        from relbench.modeling.utils import get_stype_proposal
        cs_raw = get_stype_proposal(dataset.get_db(upto_test_timestamp=False))
        # Clean serialization: explicit .value for stype enums.
        cs = {}
        for tab, c2s in cs_raw.items():
            cs[tab] = {}
            for col, st in c2s.items():
                if hasattr(st, "value"):
                    cs[tab][col] = st.value
                elif isinstance(st, str):
                    cs[tab][col] = st
                # silently drop unknown / NaN / None entries
        stypes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_path, "w") as f:
            json.dump(cs, f, indent=2)

    # Convert to stype enums; drop any leftover non-string values.
    out = {}
    for tab, c2s in cs.items():
        out[tab] = {}
        for col, st in c2s.items():
            if isinstance(st, str):
                try:
                    out[tab][col] = stype(st)
                except ValueError:
                    pass  # unknown stype name (drop)
            # else: None / NaN / etc. -> drop
    return out


def main():
    args = parse_args()
    dataset = get_dataset(args.dataset, download=True)

    stypes_path = Path(args.cache_dir) / args.dataset / "stypes.json"
    cs = _load_or_generate_stypes(stypes_path, dataset)

    # upto_test_timestamp=False: entity tables must contain all rows the
    # test split references; temporal leakage is enforced at sampling
    # time via per-row seed_time filtering, not via materialization
    # cutoff. (RelBench's default True drops post-train-cutoff entities
    # which then crash test eval with IndexError.)
    data, _ = make_pkey_fkey_graph(
        dataset.get_db(upto_test_timestamp=False),
        col_to_stype_dict=cs,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device="cpu"),
            batch_size=256,
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized_full",
    )

    print(f"Building TF store for {args.dataset} -> {args.out_dir}")
    build_dataset_tf_store(data, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
