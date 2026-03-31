"""
Lint Grader — ruff-based weighted violation reduction.

Gold standard : zero weighted violations across all non-test .py files.
Score         : proportional weighted-violation reduction from baseline.
Penalty signal: regression (more violations than baseline) → is_regression=True.

Scenario config keys (under config["graders"]["lint"]):
  weight          : float  — contribution to overall reward (default 0.3)
  min_reduction   : float  — fraction of violations that must be removed for
                             solved=True (default 1.0, i.e. zero violations)
  category_weights: dict   — per-ruff-category override, merged with defaults
  exclude_patterns: list   — file globs to skip (merged with repo-level excludes)
  ignore_codes    : list   — ruff rule codes to ignore (e.g. ["E501", "W503"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base_grader import BaseGrader, GradeResult, _clamp, already_gold, empty_grade
from .utils import (
    MetricCache,
    RuffViolation,
    run_ruff,
    weighted_violation_count,
    RUFF_CATEGORY_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Categories we never penalise — cosmetic / opinionated rules that the
# scenario explicitly opts out of by default.
_DEFAULT_IGNORE_CODES: set[str] = {
    "E501",  # line too long — handled by formatter, not the agent
    "W291",  # trailing whitespace — formatter concern
    "W293",  # whitespace before comment
}

# Category display order in feedback (most important first)
_FEEDBACK_ORDER = ["F", "E", "B", "SIM", "UP", "W", "I", "C", "N"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_test_file(rel_path: str) -> bool:
    from pathlib import PurePosixPath

    base = PurePosixPath(rel_path).name
    parts = PurePosixPath(rel_path).parts
    return (
        base.startswith("test_")
        or base.endswith("_test.py")
        or "tests" in parts
        or "test" in parts
    )


def _merge_category_weights(
    overrides: Optional[dict[str, float]],
) -> dict[str, float]:
    """Merge scenario-level category weight overrides with defaults."""
    weights = dict(RUFF_CATEGORY_WEIGHTS)
    if overrides:
        weights.update(overrides)
    return weights


def _filter_violations(
    violations: list[RuffViolation],
    ignore_codes: set[str],
    exclude_patterns: list[str],
    repo_path: Path,
) -> list[RuffViolation]:
    """
    Remove:
      - violations in test files
      - violations whose code is in ignore_codes
      - violations in files matching exclude_patterns
    """
    import fnmatch

    filtered: list[RuffViolation] = []
    for v in violations:
        # Normalise path to relative
        try:
            rel = str(Path(v.filename).relative_to(repo_path))
        except ValueError:
            rel = v.filename

        if _is_test_file(rel):
            continue
        if v.code in ignore_codes:
            continue
        if any(fnmatch.fnmatch(rel, pat) for pat in exclude_patterns):
            continue
        filtered.append(v)
    return filtered


def _weighted_count(
    violations: list[RuffViolation],
    category_weights: dict[str, float],
) -> float:
    """Sum violations using per-category weights."""
    return sum(category_weights.get(v.category, 1.0) for v in violations)


def _summarise_by_category(
    violations: list[RuffViolation],
) -> dict[str, int]:
    """Return {category: count} sorted by _FEEDBACK_ORDER."""
    counts: dict[str, int] = {}
    for v in violations:
        counts[v.category] = counts.get(v.category, 0) + 1
    return counts


def _top_remaining(
    violations: list[RuffViolation],
    top_n: int = 3,
) -> str:
    """
    Build a compact 'top remaining' string for feedback.
    e.g. "F401 ×5, E711 ×2, B006 ×1"
    """
    code_counts: dict[str, int] = {}
    for v in violations:
        code_counts[v.code] = code_counts.get(v.code, 0) + 1

    ranked = sorted(code_counts.items(), key=lambda x: -x[1])[:top_n]
    return ", ".join(f"{code} ×{cnt}" for code, cnt in ranked)


def _build_feedback(
    baseline_weighted: float,
    current_weighted: float,
    current_violations: list[RuffViolation],
    score: float,
    is_regression: bool,
) -> str:
    if is_regression:
        delta = current_weighted - baseline_weighted
        return (
            f"[Lint] Regression: weighted violations increased by "
            f"{delta:.1f} (baseline {baseline_weighted:.1f} → "
            f"now {current_weighted:.1f}). "
            f"Top issues: {_top_remaining(current_violations)}"
        )

    if score >= 1.0:
        return "[Lint] Gold standard reached — zero weighted violations."

    top = _top_remaining(current_violations)
    remaining = current_weighted
    removed = baseline_weighted - current_weighted
    return (
        f"[Lint] Removed {removed:.1f} weighted violations "
        f"({score:.0%} of baseline {baseline_weighted:.1f}). "
        f"{remaining:.1f} remain. "
        f"Top remaining: {top}"
    )


# ---------------------------------------------------------------------------
# LintGrader
# ---------------------------------------------------------------------------


class LintGrader(BaseGrader):
    """
    Scores the agent on how much it has reduced ruff-reported violations
    across all non-test Python files, weighted by violation category severity.

    Scoring formula
    ───────────────
    weighted_baseline = Σ weight(category) for v in baseline_violations
    weighted_current  = Σ weight(category) for v in current_violations
    score = clamp((weighted_baseline - weighted_current) / weighted_baseline)

    solved = True  iff  weighted_current == 0
                   OR   reduction_fraction >= min_reduction threshold
    """

    grader_id = "lint"

    # ------------------------------------------------------------------
    # compute_metrics — runs ruff, returns raw metric dict
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        repo_path: object,
        config: dict[str, Any],
        cache: object,
    ) -> dict[str, Any]:
        """
        Run ruff over the full repo (or changed files for incremental mode).

        Returns
        -------
        {
          "weighted_total"  : float   — weighted violation sum
          "raw_count"       : int     — unweighted total count
          "by_category"     : dict    — {category: count}
          "by_file"         : dict    — {rel_path: weighted_count}
          "violations"      : list    — serialisable violation dicts
          "category_weights": dict    — weights used for this computation
          "error"           : str|None
        }
        """
        repo_path = Path(repo_path)
        cache = cache  # type: MetricCache

        lint_cfg = config.get("graders", {}).get("lint", {})
        ignore_codes = set(_DEFAULT_IGNORE_CODES)
        ignore_codes.update(lint_cfg.get("ignore_codes", []))
        exclude_patterns: list[str] = lint_cfg.get("exclude_patterns", []) + config.get(
            "exclude_patterns", []
        )
        category_weights = _merge_category_weights(
            lint_cfg.get("category_weights", None)
        )

        def _run() -> list[RuffViolation]:
            return run_ruff(repo_path, timeout_sec=30)

        try:
            all_violations: list[RuffViolation] = cache.get_or_compute(
                "ruff_full", None, _run
            )
        except Exception as exc:
            return {
                "weighted_total": 0.0,
                "raw_count": 0,
                "by_category": {},
                "by_file": {},
                "violations": [],
                "category_weights": category_weights,
                "error": str(exc),
            }

        violations = _filter_violations(
            all_violations, ignore_codes, exclude_patterns, repo_path
        )

        # Per-file weighted counts
        by_file: dict[str, float] = {}
        for v in violations:
            try:
                rel = str(Path(v.filename).relative_to(repo_path))
            except ValueError:
                rel = v.filename
            by_file[rel] = by_file.get(rel, 0.0) + category_weights.get(v.category, 1.0)

        return {
            "weighted_total": _weighted_count(violations, category_weights),
            "raw_count": len(violations),
            "by_category": _summarise_by_category(violations),
            "by_file": by_file,
            # Serialisable snapshot for state() logging
            "violations": [
                {
                    "file": str(
                        Path(v.filename).relative_to(repo_path)
                        if repo_path in Path(v.filename).parents
                        else v.filename
                    ),
                    "row": v.row,
                    "col": v.col,
                    "code": v.code,
                    "message": v.message,
                }
                for v in violations
            ],
            "category_weights": category_weights,
            "error": None,
        }

    # ------------------------------------------------------------------
    # grade — pure scoring, no side effects
    # ------------------------------------------------------------------

    def grade(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
        config: dict[str, Any],
    ) -> GradeResult:
        """
        Score lint improvement from baseline to current.

        Edge cases handled:
          - baseline already at gold       → already_gold()
          - tool error in current metrics  → empty_grade() with error feedback
          - perfect elimination            → score=1.0, solved=True
          - regression                     → score=0.0, is_regression=True
        """
        lint_cfg = config.get("graders", {}).get("lint", {})
        min_reduction: float = float(lint_cfg.get("min_reduction", 1.0))

        # Errors
        if current.get("error"):
            return empty_grade("Lint", f"ruff failed: {current['error']}")

        weighted_baseline: float = baseline.get("weighted_total", 0.0)
        weighted_current: float = current.get("weighted_total", 0.0)

        # Already at gold at baseline
        if weighted_baseline <= 0:
            return already_gold("Lint", current)

        # Rebuild violation objects for feedback (from serialisable dicts)
        current_violations = [
            RuffViolation(
                filename=v["file"],
                row=v["row"],
                col=v["col"],
                code=v["code"],
                message=v["message"],
            )
            for v in current.get("violations", [])
        ]

        score, is_regression = self.delta_score(
            baseline_value=weighted_baseline,
            current_value=weighted_current,
            lower_is_better=True,
        )

        # solved: either fully clean OR met the scenario's min_reduction threshold
        reduction_fraction = (weighted_baseline - weighted_current) / weighted_baseline
        solved = (weighted_current <= 0) or (reduction_fraction >= min_reduction)

        # Per-category delta for sub_scores
        b_by_cat: dict[str, int] = baseline.get("by_category", {})
        c_by_cat: dict[str, int] = current.get("by_category", {})
        all_cats = set(b_by_cat) | set(c_by_cat)
        cat_deltas = {
            cat: b_by_cat.get(cat, 0) - c_by_cat.get(cat, 0) for cat in all_cats
        }

        # Per-file sub-scores (relative improvement per file)
        b_by_file: dict[str, float] = baseline.get("by_file", {})
        c_by_file: dict[str, float] = current.get("by_file", {})
        file_sub_scores: dict[str, float] = {}
        for fpath, b_w in b_by_file.items():
            c_w = c_by_file.get(fpath, 0.0)
            file_sub_scores[f"file:{fpath}"] = _clamp(
                (b_w - c_w) / b_w if b_w > 0 else 1.0
            )

        feedback = _build_feedback(
            baseline_weighted=weighted_baseline,
            current_weighted=weighted_current,
            current_violations=current_violations,
            score=score,
            is_regression=is_regression,
        )

        return GradeResult(
            score=score,
            gold_distance=1.0 - score,
            raw_baseline={
                "weighted_total": weighted_baseline,
                "raw_count": baseline.get("raw_count", 0),
                "by_category": b_by_cat,
            },
            raw_current={
                "weighted_total": weighted_current,
                "raw_count": current.get("raw_count", 0),
                "by_category": c_by_cat,
            },
            delta={
                "weighted_total": weighted_baseline - weighted_current,
                "raw_count": baseline.get("raw_count", 0) - current.get("raw_count", 0),
                "by_category": cat_deltas,
            },
            feedback=feedback,
            solved=solved,
            is_regression=is_regression,
            sub_scores={
                "weighted_baseline": weighted_baseline,
                "weighted_current": weighted_current,
                "reduction_fraction": _clamp(reduction_fraction),
                **file_sub_scores,
            },
        )

    # ------------------------------------------------------------------
    # gold_standard
    # ------------------------------------------------------------------

    def gold_standard(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Gold standard: zero violations in every non-test file.
        """
        lint_cfg = config.get("graders", {}).get("lint", {})
        category_weights = _merge_category_weights(
            lint_cfg.get("category_weights", None)
        )
        return {
            "weighted_total": 0.0,
            "raw_count": 0,
            "by_category": {},
            "by_file": {},
            "violations": [],
            "category_weights": category_weights,
            "error": None,
        }
