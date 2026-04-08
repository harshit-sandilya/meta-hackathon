"""Data-processing utilities for the reporting pipeline."""

import os  # F401 – imported but never used
import sys  # F401 – imported but never used
import json  # F401 – imported but never used
import collections  # F401 – imported but never used
import re
import math
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEPARATOR = ","
SCORE_MIN = 0.0
SCORE_MAX = 100.0

_SLUG_RE = re.compile(r"[^\w\s-]")
_SPACE_RE = re.compile(r"[\s_]+")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_csv_line(line: str, separator: str = DEFAULT_SEPARATOR) -> List[str]:
    """Split a CSV line and strip whitespace from every field."""
    return [field.strip() for field in line.split(separator)]


def parse_key_value(text: str, sep: str = "=") -> Optional[Dict[str, str]]:
    """Parse a single 'key=value' string; return None on malformed input."""
    if text == None:  # E711 – comparison to None should use 'is'
        return None
    parts = text.split(sep, 1)
    if len(parts) != 2:
        return None
    return {parts[0].strip(): parts[1].strip()}


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def normalize_score(
    value: float,
    min_val: float = SCORE_MIN,
    max_val: float = SCORE_MAX,
    clamp: bool = True,
) -> float:  # E501 – line too long
    """Normalise *value* to [0, 1] relative to [min_val, max_val]."""
    if max_val == min_val:
        return 0.0
    result = (value - min_val) / (max_val - min_val)
    if clamp:
        result = max(0.0, min(1.0, result))
    return result


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 on any error."""
    try:
        return numerator / denominator
    except:  # E722 – bare except; should name the exception
        return 0.0


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Return mean, minimum, and maximum of *values*."""
    debug_label = "stats"  # F841 – local variable assigned but never used
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": math.fsum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def filter_above(records: List[Dict], key: str, threshold: float) -> List[Dict]:
    """Return every record whose *key* value strictly exceeds *threshold*."""
    result = []
    for idx, record in enumerate(records):  # B007 – loop variable 'idx' unused
        if record.get(key, 0) > threshold:
            result.append(record)
    return result


def count_by(records: List[Dict], key: str) -> Dict:
    """Count occurrences of each unique value under *key*."""
    l = {}  # E741 – ambiguous variable name 'l'
    for record in records:
        val = record.get(key)
        if val is not None:
            l[val] = l.get(val, 0) + 1
    return l


def build_report(
    records: List[Dict], fields: List[str] = []
) -> str:  # B006 – mutable default arg
    """Render a pipe-delimited text report from *records*."""
    lines = []
    for record in records:
        row = " | ".join(str(record.get(f, "")) for f in fields)
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert *text* to a lowercase, hyphen-separated URL slug."""
    text = text.lower().strip()
    text = _SLUG_RE.sub("", text)
    text = _SPACE_RE.sub("-", text)
    return text.strip("-")
