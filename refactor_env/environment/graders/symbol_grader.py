"""
Symbol Grader — dead code elimination via dependency graph analysis.

Gold standard : zero dead symbols across all non-test .py files.
               Dead symbol = any import / function / class / variable
               with in-degree 0 after BFS from the public API entry points.

Score         : proportional weighted dead-symbol reduction from baseline.

Sub-signals (weighted sum → single score):
  1. dead_imports   — unused import statements          (weight 0.40)
  2. dead_functions — unreferenced functions / methods  (weight 0.35)
  3. dead_classes   — unreferenced class definitions    (weight 0.15)
  4. dead_variables — unreferenced module-level vars    (weight 0.05)
  5. unreachable    — code after return/raise/break     (weight 0.05)

Scenario config keys (under config["graders"]["symbol"]):
  weight          : float        — contribution to overall reward (default 0.25)
  sub_weights     : dict         — per-signal weight overrides
  public_api      : list[str]    — qualified names always treated as live roots
                                   (merged with config-level public_api)
  min_reduction   : float        — fraction of dead symbols needed for solved=True
                                   (default 1.0)
  exclude_patterns: list[str]    — file globs to skip
  ignore_kinds    : list[str]    — symbol kinds to ignore, e.g. ["variable"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_grader import BaseGrader, GradeResult, _clamp, already_gold, empty_grade
from .utils import (
    MetricCache,
    DependencyGraph,
    DeadSymbol,
    parse_repository,
    detect_unreachable_blocks,
    UnreachableBlock,
)


# ---------------------------------------------------------------------------
# Default sub-signal weights
# ---------------------------------------------------------------------------

_DEFAULT_SUB_WEIGHTS: dict[str, float] = {
    "dead_imports": 0.40,
    "dead_functions": 0.35,
    "dead_classes": 0.15,
    "dead_variables": 0.05,
    "unreachable": 0.05,
}

# Cost per symbol kind — heavier kinds penalise the score more when left dead.
# These are used to compute a *weighted* dead count rather than a raw count,
# giving the agent a clearer gradient: removing a dead class scores more than
# removing a dead variable.
_KIND_COST: dict[str, float] = {
    "function": 1.5,
    "async_function": 1.5,
    "class": 2.0,
    "import": 1.0,
    "variable": 0.5,
}

_UNREACHABLE_COST = 1.2  # per unreachable block


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


def _merge_sub_weights(overrides: dict[str, float] | None) -> dict[str, float]:
    w = dict(_DEFAULT_SUB_WEIGHTS)
    if overrides:
        w.update(overrides)
    # Re-normalise so weights always sum to 1.0
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w


def _bucket_dead(
    dead: list[DeadSymbol],
) -> dict[str, list[DeadSymbol]]:
    """Partition dead symbols into kind buckets."""
    buckets: dict[str, list[DeadSymbol]] = {
        "dead_imports": [],
        "dead_functions": [],
        "dead_classes": [],
        "dead_variables": [],
    }
    for sym in dead:
        if sym.kind == "import":
            buckets["dead_imports"].append(sym)
        elif sym.kind in ("function", "async_function"):
            buckets["dead_functions"].append(sym)
        elif sym.kind == "class":
            buckets["dead_classes"].append(sym)
        elif sym.kind == "variable":
            buckets["dead_variables"].append(sym)
    return buckets


def _weighted_dead_count(dead: list[DeadSymbol]) -> float:
    """Sum dead symbols weighted by kind cost."""
    return sum(_KIND_COST.get(s.kind, 1.0) for s in dead)


def _weighted_unreachable_count(blocks: list[UnreachableBlock]) -> float:
    return len(blocks) * _UNREACHABLE_COST


def _serialise_dead(dead: list[DeadSymbol]) -> list[dict]:
    return [
        {
            "name": s.name,
            "qualified_name": s.qualified_name,
            "kind": s.kind,
            "file": s.file,
            "line": s.line,
            "reason": s.reason,
        }
        for s in dead
    ]


def _serialise_unreachable(blocks: list[UnreachableBlock]) -> list[dict]:
    return [
        {
            "file": b.file,
            "line": b.line,
            "after_kind": b.after_kind,
            "function_name": b.function_name,
        }
        for b in blocks
    ]


def _compute_sub_score(
    baseline_weighted: float,
    current_weighted: float,
) -> tuple[float, bool]:
    """Delta score for a single sub-signal bucket."""
    if baseline_weighted <= 0:
        return (1.0, False)
    improvement = baseline_weighted - current_weighted
    is_regression = current_weighted > baseline_weighted
    return (_clamp(improvement / baseline_weighted), is_regression)


def _build_feedback(
    baseline_total: float,
    current_total: float,
    score: float,
    is_regression: bool,
    buckets_current: dict[str, list[DeadSymbol]],
    unreachable_current: list[UnreachableBlock],
    cycles: list[list[str]],
) -> str:
    if is_regression:
        delta = current_total - baseline_total
        return (
            f"[Symbol] Regression: {delta:.1f} weighted dead symbols added "
            f"(baseline {baseline_total:.1f} → now {current_total:.1f}). "
            f"Review recent edits for new unused imports or dead code."
        )

    if score >= 1.0:
        return "[Symbol] Gold standard — no dead symbols or unreachable blocks."

    parts: list[str] = []

    dead_imports = len(buckets_current["dead_imports"])
    dead_functions = len(buckets_current["dead_functions"])
    dead_classes = len(buckets_current["dead_classes"])
    dead_variables = len(buckets_current["dead_variables"])
    unreachable = len(unreachable_current)

    if dead_imports:
        # Show up to 3 import names for actionability
        names = [s.name for s in buckets_current["dead_imports"][:3]]
        suffix = f" ({', '.join(names)}{'…' if dead_imports > 3 else ''})"
        parts.append(f"{dead_imports} unused import(s){suffix}")
    if dead_functions:
        names = [s.name for s in buckets_current["dead_functions"][:3]]
        suffix = f" ({', '.join(names)}{'…' if dead_functions > 3 else ''})"
        parts.append(f"{dead_functions} dead function(s){suffix}")
    if dead_classes:
        names = [s.name for s in buckets_current["dead_classes"][:2]]
        suffix = f" ({', '.join(names)}{'…' if dead_classes > 2 else ''})"
        parts.append(f"{dead_classes} dead class(es){suffix}")
    if dead_variables:
        parts.append(f"{dead_variables} unused module-level variable(s)")
    if unreachable:
        parts.append(f"{unreachable} unreachable block(s)")
    if cycles:
        parts.append(f"{len(cycles)} import cycle(s)")

    remaining_str = "; ".join(parts) if parts else "none"
    removed = baseline_total - current_total
    return (
        f"[Symbol] Removed {removed:.1f} weighted dead symbols "
        f"({score:.0%} of baseline {baseline_total:.1f}). "
        f"Remaining: {remaining_str}."
    )


# ---------------------------------------------------------------------------
# SymbolGrader
# ---------------------------------------------------------------------------


class SymbolGrader(BaseGrader):
    """
    Scores the agent on eliminating dead code — unused imports, unreferenced
    functions/classes/variables, and unreachable blocks — across all non-test
    Python files.

    Uses DependencyGraph for cross-file reachability analysis:
      1. Build the dep graph from all parsed modules.
      2. BFS from public_api roots (from scenario.yaml + config).
      3. Any node with in-degree 0 after BFS = dead.
      4. Additionally scan for unreachable blocks (code after return/raise).

    Scoring formula (weighted sub-signals)
    ──────────────────────────────────────
    For each sub-signal s in {imports, functions, classes, variables, unreachable}:
      sub_score[s] = clamp((weighted_baseline[s] - weighted_current[s])
                           / weighted_baseline[s])

    final_score = Σ sub_weight[s] * sub_score[s]

    The per-kind cost multipliers (_KIND_COST) ensure removing a dead class
    scores more than removing a dead import, providing a richer gradient.
    """

    grader_id = "symbol"

    # ------------------------------------------------------------------
    # compute_metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        repo_path: object,
        config: dict[str, Any],
        cache: object,
    ) -> dict[str, Any]:
        """
        Build dep graph, run BFS dead-symbol detection, scan unreachable blocks.

        Returns
        -------
        {
          "dead_symbols"       : list[dict]  — serialised DeadSymbol list
          "unreachable_blocks" : list[dict]  — serialised UnreachableBlock list
          "import_cycles"      : list[list]  — each inner list is a cycle (module names)
          "bucket_counts"      : dict        — {bucket_name: count}
          "bucket_weighted"    : dict        — {bucket_name: weighted_count}
          "total_weighted"     : float       — sum across all buckets
          "total_raw"          : int         — raw unweighted count
          "error"              : str|None
        }
        """
        repo_path = Path(repo_path)
        sym_cfg = config.get("graders", {}).get("symbol", {})

        # Merge public_api from scenario + grader config
        public_api: list[str] = list(config.get("public_api", []))
        public_api += sym_cfg.get("public_api", [])

        exclude_patterns: list[str] = sym_cfg.get("exclude_patterns", []) + config.get(
            "exclude_patterns", []
        )
        ignore_kinds: set[str] = set(sym_cfg.get("ignore_kinds", []))

        def _build() -> (
            tuple[list[DeadSymbol], list[UnreachableBlock], list[list[str]]]
        ):
            modules = parse_repository(repo_path, exclude_patterns)

            # Filter test files from module map
            non_test_modules = {
                p: m for p, m in modules.items() if not _is_test_file(p)
            }

            graph = DependencyGraph()
            graph.build(non_test_modules, public_api=public_api)

            dead = graph.get_dead_symbols(exclude_tests=True)

            # Scan unreachable blocks across non-test files
            unreachable: list[UnreachableBlock] = []
            for rel_path, module in non_test_modules.items():
                unreachable.extend(detect_unreachable_blocks(module, rel_path))

            cycles = graph.get_import_cycles()
            return dead, unreachable, cycles

        try:
            dead, unreachable, cycles = cache.get_or_compute(
                "symbol_analysis", None, _build
            )
        except Exception as exc:
            return {
                "dead_symbols": [],
                "unreachable_blocks": [],
                "import_cycles": [],
                "bucket_counts": {},
                "bucket_weighted": {},
                "total_weighted": 0.0,
                "total_raw": 0,
                "error": str(exc),
            }

        # Filter by ignore_kinds
        if ignore_kinds:
            dead = [s for s in dead if s.kind not in ignore_kinds]

        buckets = _bucket_dead(dead)

        bucket_counts: dict[str, int] = {k: len(v) for k, v in buckets.items()}
        bucket_counts["unreachable"] = len(unreachable)

        bucket_weighted: dict[str, float] = {
            k: _weighted_dead_count(v) for k, v in buckets.items()
        }
        bucket_weighted["unreachable"] = _weighted_unreachable_count(unreachable)

        total_weighted = sum(bucket_weighted.values())
        total_raw = sum(bucket_counts.values())

        return {
            "dead_symbols": _serialise_dead(dead),
            "unreachable_blocks": _serialise_unreachable(unreachable),
            "import_cycles": cycles,
            "bucket_counts": bucket_counts,
            "bucket_weighted": bucket_weighted,
            "total_weighted": total_weighted,
            "total_raw": total_raw,
            "error": None,
        }

    # ------------------------------------------------------------------
    # grade — pure, no side effects
    # ------------------------------------------------------------------

    def grade(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
        config: dict[str, Any],
    ) -> GradeResult:
        """
        Score dead-symbol elimination progress from baseline → current.

        Sub-signal scores are computed independently per bucket then
        combined via weighted sum, so the agent gets a gradient signal
        even when only some symbol kinds have been cleaned up.
        """
        sym_cfg = config.get("graders", {}).get("symbol", {})
        sub_weights = _merge_sub_weights(sym_cfg.get("sub_weights", None))
        min_reduction = float(sym_cfg.get("min_reduction", 1.0))

        if current.get("error"):
            return empty_grade("Symbol", f"analysis failed: {current['error']}")

        b_weighted: dict[str, float] = baseline.get("bucket_weighted", {})
        c_weighted: dict[str, float] = current.get("bucket_weighted", {})

        baseline_total = baseline.get("total_weighted", 0.0)
        current_total = current.get("total_weighted", 0.0)

        # Already at gold at baseline
        if baseline_total <= 0:
            return already_gold("Symbol", current)

        # ── Per-bucket sub-scores ──────────────────────────────────────
        sub_scores: dict[str, float] = {}
        sub_regressions: dict[str, bool] = {}

        all_buckets = set(b_weighted) | set(c_weighted)
        for bucket in all_buckets:
            b_w = b_weighted.get(bucket, 0.0)
            c_w = c_weighted.get(bucket, 0.0)
            ss, reg = _compute_sub_score(b_w, c_w)
            sub_scores[bucket] = ss
            sub_regressions[bucket] = reg

        # ── Weighted combination ───────────────────────────────────────
        final_score: float = 0.0
        for bucket, sw in sub_weights.items():
            final_score += sw * sub_scores.get(
                bucket, 1.0 if b_weighted.get(bucket, 0) == 0 else 0.0
            )
        final_score = _clamp(final_score)

        is_regression = any(sub_regressions.values())

        # Override: if overall weighted count increased, always regression
        if current_total > baseline_total:
            is_regression = True
            final_score = 0.0

        # solved
        if baseline_total > 0:
            reduction_fraction = _clamp(
                (baseline_total - current_total) / baseline_total
            )
        else:
            reduction_fraction = 1.0
        solved = (current_total <= 0) or (reduction_fraction >= min_reduction)

        # ── Cycle penalty (informational — not in score, but in sub_scores) ──
        cycle_count_baseline = len(baseline.get("import_cycles", []))
        cycle_count_current = len(current.get("import_cycles", []))
        cycle_delta = cycle_count_baseline - cycle_count_current

        # ── Rebuild buckets from serialised data for feedback ──────────
        from ..utils.dep_graph import DeadSymbol as DS

        current_dead = [
            DS(
                name=s["name"],
                qualified_name=s["qualified_name"],
                kind=s["kind"],
                file=s["file"],
                line=s["line"],
                reason=s["reason"],
            )
            for s in current.get("dead_symbols", [])
        ]
        buckets_current = _bucket_dead(current_dead)

        from ..utils.ast_utils import UnreachableBlock as UB

        unreachable_current = [
            UB(
                file=b["file"],
                line=b["line"],
                after_kind=b["after_kind"],
                function_name=b["function_name"],
            )
            for b in current.get("unreachable_blocks", [])
        ]

        feedback = _build_feedback(
            baseline_total=baseline_total,
            current_total=current_total,
            score=final_score,
            is_regression=is_regression,
            buckets_current=buckets_current,
            unreachable_current=unreachable_current,
            cycles=current.get("import_cycles", []),
        )

        # ── Delta dict ────────────────────────────────────────────────
        b_counts: dict[str, int] = baseline.get("bucket_counts", {})
        c_counts: dict[str, int] = current.get("bucket_counts", {})
        delta_counts = {
            k: b_counts.get(k, 0) - c_counts.get(k, 0)
            for k in set(b_counts) | set(c_counts)
        }

        return GradeResult(
            score=final_score,
            gold_distance=1.0 - final_score,
            raw_baseline={
                "total_weighted": baseline_total,
                "total_raw": baseline.get("total_raw", 0),
                "bucket_counts": b_counts,
                "bucket_weighted": b_weighted,
                "import_cycles": cycle_count_baseline,
            },
            raw_current={
                "total_weighted": current_total,
                "total_raw": current.get("total_raw", 0),
                "bucket_counts": c_counts,
                "bucket_weighted": c_weighted,
                "import_cycles": cycle_count_current,
            },
            delta={
                "total_weighted": baseline_total - current_total,
                "total_raw": baseline.get("total_raw", 0) - current.get("total_raw", 0),
                "bucket_counts": delta_counts,
                "import_cycles": cycle_delta,
            },
            feedback=feedback,
            solved=solved,
            is_regression=is_regression,
            sub_scores={
                **{f"sub:{k}": v for k, v in sub_scores.items()},
                "reduction_fraction": reduction_fraction,
                "cycle_delta": float(cycle_delta),
                "weighted_baseline": baseline_total,
                "weighted_current": current_total,
            },
        )

    # ------------------------------------------------------------------
    # gold_standard
    # ------------------------------------------------------------------

    def gold_standard(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Gold standard: zero dead symbols, zero unreachable blocks,
        zero import cycles.
        """
        empty_buckets = {
            "dead_imports": 0,
            "dead_functions": 0,
            "dead_classes": 0,
            "dead_variables": 0,
            "unreachable": 0,
        }
        return {
            "dead_symbols": [],
            "unreachable_blocks": [],
            "import_cycles": [],
            "bucket_counts": empty_buckets,
            "bucket_weighted": {k: 0.0 for k in empty_buckets},
            "total_weighted": 0.0,
            "total_raw": 0,
            "error": None,
        }
