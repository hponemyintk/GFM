"""Defensive ``stypes.json`` loader / writer.

Both the offline tools (``tools/build_tf_store.py``,
``tools/precompute_shards.py``) and the online entry points
(``main_node_ddp.py``, ``train_multi_task.py``) need to:

* read ``<cache>/<dataset>/stypes.json`` if it's present,
* fall back to ``relbench.modeling.utils.get_stype_proposal`` if not,
* validate that every value is a string or ``None`` (older runs / non-
  strict JSON loaders can leave ``NaN`` / float / other entries that
  later crash ``torch_frame`` inside ``make_pkey_fkey_graph`` with the
  notorious "'float' object has no attribute 'split'" error),
* write back a clean copy using each ``stype.value`` (no ``default=str``
  surprises),
* return a ``dict[str, dict[str, stype]]`` ready to feed into
  ``make_pkey_fkey_graph(col_to_stype_dict=...)``.

Keeping the logic in one place means new code paths can't drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional


def is_valid_raw_stypes(raw: object) -> bool:
    """Return True iff every leaf value in the nested dict is str or None."""
    if not isinstance(raw, dict):
        return False
    for c2s in raw.values():
        if not isinstance(c2s, dict):
            return False
        for v in c2s.values():
            if not (isinstance(v, str) or v is None):
                return False
    return True


def normalize_for_write(cs_raw: dict) -> dict:
    """Convert ``stype`` enum values to their ``.value`` strings; drop other types.

    This is what gets written to ``stypes.json``. Keeping serialization
    explicit (instead of ``json.dump(..., default=str)``) avoids
    surprises like a ``float('nan')`` being silently turned into a
    JSON ``NaN`` literal that re-loads as a Python float and then
    crashes downstream.
    """
    out: Dict[str, Dict[str, str]] = {}
    for tab, c2s in cs_raw.items():
        out[tab] = {}
        for col, st in c2s.items():
            if hasattr(st, "value"):
                out[tab][col] = st.value
            elif isinstance(st, str):
                out[tab][col] = st
            # else: silently drop NaN / None / unknown -- relbench
            # treats absent keys as "skip this column".
    return out


def to_stype_enums(cs: dict):
    """Convert a clean str-valued dict to ``stype`` enum values.

    Drops keys whose value isn't a recognized stype name. The caller
    should treat dropped columns as "torch_frame will skip these" --
    relbench's ``make_pkey_fkey_graph`` does not require every column
    in the dict; columns it doesn't see are excluded from the TF.
    """
    from torch_frame import stype  # local import: torch_frame is heavy
    out: Dict[str, Dict[str, "stype"]] = {}
    for tab, c2s in cs.items():
        out[tab] = {}
        for col, st in c2s.items():
            if isinstance(st, str):
                try:
                    out[tab][col] = stype(st)
                except ValueError:
                    pass  # unknown stype name; drop
            # else: None / NaN -> drop
    return out


def load_or_generate_stypes(
    stypes_path: Path,
    dataset,
    *,
    upto_test_timestamp: bool = False,
    log_stream=sys.stderr,
):
    """Top-level loader. Returns ``dict[str, dict[str, stype]]``.

    Parameters
    ----------
    stypes_path : Path
        Where the cached JSON lives. Created (with parents) if missing.
    dataset : relbench dataset object
        Used by ``get_stype_proposal`` only when regenerating.
    upto_test_timestamp : bool
        Forwarded to ``dataset.get_db(...)`` for the regeneration path.
        Default ``False`` because we want full entity tables for
        materialization.
    log_stream
        Where to print "regenerating" notices. Default ``sys.stderr``;
        pass ``open(os.devnull, 'w')`` to silence.
    """
    raw: Optional[dict] = None
    if stypes_path.exists():
        try:
            with open(stypes_path) as f:
                candidate = json.load(f)
            if is_valid_raw_stypes(candidate):
                raw = candidate
            else:
                print(f"[stypes] {stypes_path} has invalid entries; "
                      f"regenerating.", file=log_stream)
        except Exception as e:
            print(f"[stypes] {stypes_path} unreadable ({e}); "
                  f"regenerating.", file=log_stream)

    if raw is None:
        from relbench.modeling.utils import get_stype_proposal
        cs_raw = get_stype_proposal(
            dataset.get_db(upto_test_timestamp=upto_test_timestamp)
        )
        raw = normalize_for_write(cs_raw)
        stypes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_path, "w") as f:
            json.dump(raw, f, indent=2)

    return to_stype_enums(raw)
