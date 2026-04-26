"""Tests for ``gfm_data.stypes`` -- the defensive stypes.json helper."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfm_data.stypes import (  # noqa: E402
    is_valid_raw_stypes,
    normalize_for_write,
    load_or_generate_stypes,
)


# ---------------------------------------------------------------------------
def test_is_valid_raw_stypes_accepts_string_or_none():
    good = {"users": {"age": "numerical", "name": "text_embedded", "deleted": None}}
    assert is_valid_raw_stypes(good)


def test_is_valid_raw_stypes_rejects_float_value():
    """The exact regression we hit on AWS: NaN in stypes.json."""
    bad = {"users": {"age": float("nan")}}
    assert not is_valid_raw_stypes(bad)


def test_is_valid_raw_stypes_rejects_int_value():
    bad = {"users": {"age": 42}}
    assert not is_valid_raw_stypes(bad)


def test_is_valid_raw_stypes_rejects_top_level_non_dict():
    assert not is_valid_raw_stypes(["numerical"])
    assert not is_valid_raw_stypes("numerical")
    assert not is_valid_raw_stypes(None)


def test_is_valid_raw_stypes_rejects_per_table_non_dict():
    bad = {"users": "numerical"}  # should be {col: stype} dict
    assert not is_valid_raw_stypes(bad)


# ---------------------------------------------------------------------------
def test_normalize_for_write_uses_value_field():
    """stype enum -> .value (string), not str(enum) which depends on __str__."""
    fake_stype = MagicMock()
    fake_stype.value = "numerical"
    cs_raw = {"users": {"age": fake_stype, "name": "text_embedded"}}
    out = normalize_for_write(cs_raw)
    assert out == {"users": {"age": "numerical", "name": "text_embedded"}}


def test_normalize_for_write_drops_nan_and_none():
    cs_raw = {
        "users": {
            "age": MagicMock(value="numerical"),
            "deleted": None,
            "weird": float("nan"),
        }
    }
    out = normalize_for_write(cs_raw)
    # None and NaN values get dropped; only known stype string remains.
    assert out["users"] == {"age": "numerical"}


# ---------------------------------------------------------------------------
def _write_corrupt_json(path: Path):
    """Mimic the AWS bug: json.dump that wrote NaN literals.

    Python's json with ``allow_nan=True`` (default) writes float('nan')
    as the literal token ``NaN`` and reads it back as float('nan').
    That re-load is what blew up downstream torch_frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write deliberately invalid content: NaN literal + a string value.
    raw = '{\n  "users": {\n    "age": "numerical",\n    "weird": NaN\n  }\n}\n'
    with open(path, "w") as f:
        f.write(raw)


def _make_proposal(stype_strs):
    """Return ``(dataset_mock, proposal)`` ready for ``patch(...)`` use."""
    stype_enum = MagicMock()
    stype_enum.value = "numerical"
    proposal = {"users": {col: stype_enum for col in stype_strs}}
    dataset = MagicMock()
    dataset.get_db.return_value = MagicMock()
    return dataset, proposal


def test_load_or_generate_stypes_uses_existing_clean_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rel-foo" / "stypes.json"
        path.parent.mkdir(parents=True)
        with open(path, "w") as f:
            json.dump({"users": {"age": "numerical"}}, f)

        out = load_or_generate_stypes(path, MagicMock())
        # Top-level keys preserved, value is a torch_frame stype enum.
        from torch_frame import stype
        assert out["users"]["age"] == stype.numerical


def test_load_or_generate_stypes_regenerates_corrupt_file():
    """The actual AWS regression: stypes.json with NaN literal."""
    import io
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rel-foo" / "stypes.json"
        _write_corrupt_json(path)

        # Sanity: the corrupt file is loadable as JSON but is_valid says no.
        with open(path) as f:
            raw = json.load(f)
        assert not is_valid_raw_stypes(raw)  # confirms the test setup

        dataset, proposal = _make_proposal(stype_strs=["age"])
        log = io.StringIO()

        with patch("relbench.modeling.utils.get_stype_proposal",
                   return_value=proposal):
            out = load_or_generate_stypes(path, dataset, log_stream=log)

        # Helper logged a regeneration notice.
        assert "regenerating" in log.getvalue().lower()

        # File on disk is now clean (no NaN, valid_raw_stypes True).
        with open(path) as f:
            new_raw = json.load(f)
        assert is_valid_raw_stypes(new_raw)
        assert new_raw == {"users": {"age": "numerical"}}

        # Returned dict is keyed by stype enum.
        from torch_frame import stype
        assert out["users"]["age"] == stype.numerical


def test_load_or_generate_stypes_creates_when_missing():
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rel-foo" / "stypes.json"
        assert not path.exists()

        dataset, proposal = _make_proposal(stype_strs=["age", "salary"])

        with patch("relbench.modeling.utils.get_stype_proposal",
                   return_value=proposal):
            out = load_or_generate_stypes(path, dataset)

        # File got created.
        assert path.exists()
        with open(path) as f:
            raw = json.load(f)
        assert raw == {"users": {"age": "numerical", "salary": "numerical"}}

        # Returned dict has stype enums.
        from torch_frame import stype
        assert out["users"]["age"] == stype.numerical
