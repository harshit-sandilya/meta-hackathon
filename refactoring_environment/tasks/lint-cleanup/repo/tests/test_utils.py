"""Behavioural tests for utils.py.

These tests check observable behaviour only – they must remain green
whether or not any lint violations are present in the source file.
The test suite is read-only: the agent may not modify this file.
"""

import pytest
import sys
import os

# Make the repo root importable when pytest is run from inside it.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import (
    build_report,
    compute_stats,
    count_by,
    filter_above,
    normalize_score,
    parse_csv_line,
    parse_key_value,
    safe_divide,
    slugify,
)


# ---------------------------------------------------------------------------
# parse_csv_line
# ---------------------------------------------------------------------------


def test_parse_csv_line_default_separator():
    assert parse_csv_line("a, b, c") == ["a", "b", "c"]


def test_parse_csv_line_custom_separator():
    assert parse_csv_line("x|y|z", separator="|") == ["x", "y", "z"]


def test_parse_csv_line_strips_whitespace():
    assert parse_csv_line("  hello , world  ") == ["hello", "world"]


# ---------------------------------------------------------------------------
# parse_key_value
# ---------------------------------------------------------------------------


def test_parse_key_value_valid():
    assert parse_key_value("name=Alice") == {"name": "Alice"}


def test_parse_key_value_none_input():
    assert parse_key_value(None) is None


def test_parse_key_value_malformed():
    assert parse_key_value("no-equals-sign") is None


# ---------------------------------------------------------------------------
# normalize_score
# ---------------------------------------------------------------------------


def test_normalize_score_midpoint():
    assert normalize_score(50.0) == pytest.approx(0.5)


def test_normalize_score_boundaries():
    assert normalize_score(0.0) == pytest.approx(0.0)
    assert normalize_score(100.0) == pytest.approx(1.0)


def test_normalize_score_clamp_above():
    assert normalize_score(150.0, clamp=True) == pytest.approx(1.0)


def test_normalize_score_equal_bounds():
    assert normalize_score(50.0, min_val=50.0, max_val=50.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# safe_divide
# ---------------------------------------------------------------------------


def test_safe_divide_normal():
    assert safe_divide(10.0, 2.0) == pytest.approx(5.0)


def test_safe_divide_by_zero():
    assert safe_divide(1.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


def test_compute_stats_basic():
    result = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["mean"] == pytest.approx(3.0)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(5.0)


def test_compute_stats_empty():
    assert compute_stats([]) == {"mean": 0.0, "min": 0.0, "max": 0.0}


# ---------------------------------------------------------------------------
# filter_above
# ---------------------------------------------------------------------------


def test_filter_above_keeps_strictly_greater():
    records = [{"v": 0.3}, {"v": 0.7}, {"v": 0.5}]
    assert filter_above(records, "v", 0.5) == [{"v": 0.7}]


def test_filter_above_empty_input():
    assert filter_above([], "v", 0.0) == []


# ---------------------------------------------------------------------------
# count_by
# ---------------------------------------------------------------------------


def test_count_by_basic():
    records = [{"cat": "a"}, {"cat": "b"}, {"cat": "a"}]
    assert count_by(records, "cat") == {"a": 2, "b": 1}


def test_count_by_missing_key_ignored():
    records = [{"cat": "a"}, {"other": "x"}]
    assert count_by(records, "cat") == {"a": 1}


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


def test_build_report_pipe_delimited():
    records = [{"name": "Alice", "score": 9}, {"name": "Bob", "score": 7}]
    output = build_report(records, fields=["name", "score"])
    assert output == "Alice | 9\nBob | 7"


def test_build_report_empty_records():
    assert build_report([], fields=["name"]) == ""


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


def test_slugify_basic():
    assert slugify("Hello, World!") == "hello-world"


def test_slugify_strips_leading_trailing():
    assert slugify("  foo bar  ") == "foo-bar"


def test_slugify_underscores_become_hyphens():
    assert slugify("foo_bar_baz") == "foo-bar-baz"
