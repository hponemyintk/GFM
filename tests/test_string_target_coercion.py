"""Unit tests for ``coerce_string_target_to_numeric``.

Three RelBench v2 autocomplete tasks (rel-trial eligibilities-child,
eligibilities-adult, studies-has_dmc) ship binary targets as ``'t'``/``'f'``
strings rather than numeric 1/0. ``relbench.modeling.graph.get_node_train_table_input``
unconditionally calls ``.astype(float)`` and crashes on these.

The helper coerces strings -> ints in place before that call, preserves
existing NaN, and raises on unmapped string values so a new dataset with
some unexpected encoding fails loudly instead of silently producing NaN.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfm_data.task_tokens import coerce_string_target_to_numeric


class _FakeTable:
    """Minimal stand-in for relbench.base.Table -- only ``.df`` is touched."""

    def __init__(self, df: pd.DataFrame):
        self.df = df


def test_numeric_float_column_is_noop():
    df = pd.DataFrame({"y": [0.0, 1.0, 0.0, 1.0], "x": [10, 20, 30, 40]})
    table = _FakeTable(df.copy())
    coerce_string_target_to_numeric(table, "y")
    pd.testing.assert_series_equal(table.df["y"], df["y"])


def test_numeric_int_column_is_noop():
    df = pd.DataFrame({"y": [0, 1, 0, 1], "x": [10, 20, 30, 40]})
    table = _FakeTable(df.copy())
    coerce_string_target_to_numeric(table, "y")
    pd.testing.assert_series_equal(table.df["y"], df["y"])


def test_t_f_strings_map_to_one_zero():
    df = pd.DataFrame({"y": ["t", "f", "t", "f", "t"]})
    table = _FakeTable(df)
    coerce_string_target_to_numeric(table, "y")
    assert table.df["y"].tolist() == [1, 0, 1, 0, 1]


def test_yes_no_true_false_strings_map_correctly():
    df = pd.DataFrame({"y": ["yes", "no", "true", "false"]})
    table = _FakeTable(df)
    coerce_string_target_to_numeric(table, "y")
    assert table.df["y"].tolist() == [1, 0, 1, 0]


def test_preexisting_nan_preserved_alongside_string_targets():
    """Brief's bug: ``.map`` returns NaN for both unmapped strings AND
    pre-existing NaN. Helper must distinguish via ``original.notna() &
    mapped.isna()`` so legit NaN passes through."""
    df = pd.DataFrame({"y": ["t", None, "f", np.nan, "t"]})
    table = _FakeTable(df)
    coerce_string_target_to_numeric(table, "y")
    out = table.df["y"]
    assert out.iloc[0] == 1
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == 0
    assert pd.isna(out.iloc[3])
    assert out.iloc[4] == 1


def test_unmapped_string_raises_with_clear_message():
    df = pd.DataFrame({"y": ["t", "f", "maybe", "t"]})
    table = _FakeTable(df)
    with pytest.raises(ValueError) as exc:
        coerce_string_target_to_numeric(table, "y")
    msg = str(exc.value)
    assert "'y'" in msg
    assert "maybe" in msg
    assert "_STRING_TARGET_MAP" in msg


def test_idempotent_on_second_call():
    df = pd.DataFrame({"y": ["t", "f", "t", "f"]})
    table = _FakeTable(df)
    coerce_string_target_to_numeric(table, "y")
    first = table.df["y"].tolist()
    coerce_string_target_to_numeric(table, "y")
    second = table.df["y"].tolist()
    assert first == second == [1, 0, 1, 0]


def test_other_columns_untouched():
    df = pd.DataFrame({
        "y": ["t", "f"],
        "other_string": ["foo", "bar"],
        "other_int": [10, 20],
    })
    table = _FakeTable(df)
    coerce_string_target_to_numeric(table, "y")
    assert table.df["other_string"].tolist() == ["foo", "bar"]
    assert table.df["other_int"].tolist() == [10, 20]
    assert table.df["y"].tolist() == [1, 0]
