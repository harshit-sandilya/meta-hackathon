"""
complexity_grader.py — Cyclomatic + heuristic Big-O complexity reduction grader.

Two complementary signals:

  1. Cyclomatic Complexity (CC) via radon cc
     ─────────────────────────────────────────
     McCabe's CC counts independent execution paths. Radon grades each
     function A–F (A: CC 1-5, B: 6-10, C: 11-15, D: 16-20, E: 21-25,
     F: 26+). We reward reducing the *weighted sum of CC costs* across
     all functions — not just counting "bad" functions, so the agent gets
     partial credit for lowering F→D even if not all the way to A.

     cc_cost(fn) = max(0, cc - cc_threshold)²  (quadratic — big complexity
                                                 differences matter more)

  2. Heuristic Big-O via complexity_helper.py (our AST analysis layer)
     ──────────────────────────────────────────────────────────────────
     Uses COMPLEXITY_COST weights: O(n²)→3.0, O(n³)→6.0, O(2ⁿ)→10.0,
     O(?)→2.0, O(n log n)→1.0, O(n)/O(log n)/O(1)→0.0.
     Rewards eliminating anti-patterns (nested loops, sort-in-loop,
     str-concat-in-loop, unmemoized recursion, etc.).

Final score — weighted combination of both delta scores:
  score = clamp(cc_weight * cc_delta + bigO_weight * bigO_delta)

  Both sub-deltas are clamped to [0, 1] independently before weighting,
  so a regression in one cannot drag the other negative.

scenario.yaml config keys (under complexity:):
  cc_threshold     int    CC above which quadratic cost kicks in        (default 10)
  cc_weight        float  weight for radon CC sub-score                 (default 0.55)
  bigO_weight      float  weight for heuristic Big-O sub-score         (default 0.45)
  timeout          int    radon subprocess timeout in seconds           (default 15)
  exclude_tests    bool   skip files under test* directories            (default True)

raw_metrics keys produced:
  ── Radon CC ──
  cc_cost_sum_baseline    float   sum of quadratic costs at episode start
  cc_cost_sum_now         float   sum of quadratic costs at current step
  cc_avg_baseline         float   mean CC value across all functions (baseline)
  cc_avg_now              float   mean CC value across all functions (now)
  cc_total_fns            int     total functions radon analysed
  cc_worst_fns            list    top-5 worst functions: [{name, cc, grade, file}]
  cc_delta                float   sub-score for CC dimension
  radon_error             str     non-empty if radon failed

  ── Heuristic Big-O ──
  bigO_cost_sum_baseline  float   sum of COMPLEXITY_COST weights (baseline)
  bigO_cost_sum_now       float   sum of COMPLEXITY_COST weights (now)
  bigO_fn_count           int     total functions analysed by AST
  bigO_worst_fns          list    top-5 worst: [{qname, time, cost, patterns}]
  bigO_delta              float   sub-score for Big-O dimension
  bigO_patterns_seen      list    all anti-pattern tags found across all fns
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import ClassVar

from .base import BaseGrader, GradeResult
from ..analysis.complexity_analysis import (
    COMPLEXITY_COST,
    ComplexityClass,
    estimate_repo_complexity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Radon helpers
# ─────────────────────────────────────────────────────────────────────────────


def _run_radon(repo_path: Path, timeout: int) -> tuple[dict, str]:
    """
    Run ``radon cc --json <path>`` and return (by_file_dict, error_str).

    by_file_dict shape:
      { "rel/path.py": [{"name": str, "cc": int, "grade": str, "file": str}] }
    """
    try:
        result = subprocess.run(
            ["radon", "cc", "--json", str(repo_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {}, f"radon timed out after {timeout}s"
    except FileNotFoundError:
        return {}, "radon not found — pip install radon"
    except Exception as exc:
        return {}, str(exc)

    # radon exits 1 when some files have syntax errors but still outputs
    # valid JSON for the rest — we tolerate rc=1
    if result.returncode > 1:
        return {}, f"radon exit {result.returncode}: {result.stderr[:200]}"

    try:
        raw = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}, f"radon JSON parse failed: {result.stdout[:120]}"

    by_file: dict[str, list[dict]] = {}
    for abs_path, fns in raw.items():
        if not isinstance(fns, list):
            continue
        try:
            rel = str(Path(abs_path).relative_to(repo_path))
        except ValueError:
            rel = abs_path
        by_file[rel] = [
            {
                "name": fn.get("name", "?"),
                "cc": int(fn.get("complexity", 0)),
                "grade": fn.get("rank", "A"),
                "file": rel,
            }
            for fn in fns
            if isinstance(fn, dict)
        ]

    return by_file, ""


def _cc_cost(cc: int, threshold: int) -> float:
    """
    Quadratic cost above threshold — larger overages cost disproportionately
    more, matching the real pain of extremely complex functions.

    cost(CC=10, threshold=10) = 0
    cost(CC=15, threshold=10) = 25
    cost(CC=26, threshold=10) = 256
    """
    excess = max(0, cc - threshold)
    return float(excess**2)


def _collect_cc_metrics(
    by_file: dict[str, list[dict]],
    threshold: int,
    exclude_tests: bool,
) -> dict:
    """Aggregate per-file radon output into grader metrics."""
    total_fns = 0
    cost_sum = 0.0
    cc_sum = 0.0
    all_fns: list[dict] = []

    for rel_path, fns in by_file.items():
        if exclude_tests:
            parts = Path(rel_path).parts
            if any(p.startswith("test") for p in parts):
                continue
        for fn in fns:
            cc = fn["cc"]
            cost = _cc_cost(cc, threshold)
            total_fns += 1
            cost_sum += cost
            cc_sum += cc
            all_fns.append({**fn, "cost": cost})

    # Top-5 worst by cost for the feedback string
    worst = sorted(all_fns, key=lambda f: f["cost"], reverse=True)[:5]

    return {
        "cc_cost_sum": round(cost_sum, 3),
        "cc_avg": round(cc_sum / max(total_fns, 1), 3),
        "cc_total_fns": total_fns,
        "cc_worst_fns": worst,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Big-O helpers (wrapping complexity_helper.estimate_repo_complexity)
# ─────────────────────────────────────────────────────────────────────────────


def _load_modules(py_files: list[Path], repo_root: Path) -> dict[str, ast.Module]:
    """
    Parse all .py files into {rel_path: ast.Module}, skipping unparseable files
    rather than crashing (error-tolerant, same spirit as ast_utils.parse_file).
    """
    modules: dict[str, ast.Module] = {}
    for fpath in py_files:
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
            modules[str(fpath.relative_to(repo_root))] = ast.parse(source)
        except SyntaxError:
            pass
    return modules


def _collect_bigO_metrics(
    repo_root: Path,
    py_files: list[Path],
    exclude_tests: bool,
) -> dict:
    """
    Run estimate_repo_complexity() and aggregate COMPLEXITY_COST totals.

    Returns:
      bigO_cost_sum   float  sum of COMPLEXITY_COST[time_class] across all fns
      bigO_fn_count   int    total functions analysed
      bigO_worst_fns  list   top-5 [{qname, time_str, cost, patterns}]
      bigO_patterns   list   all unique anti-pattern tags found
    """
    modules = _load_modules(py_files, repo_root)
    estimates = estimate_repo_complexity(modules, exclude_test_files=exclude_tests)

    cost_sum = 0.0
    all_patterns: list[str] = []
    fn_records: list[dict] = []

    for qname, est in estimates.items():
        cost = COMPLEXITY_COST.get(est.time, COMPLEXITY_COST[ComplexityClass.UNKNOWN])
        cost_sum += cost
        all_patterns.extend(est.patterns_found)
        fn_records.append(
            {
                "qname": qname,
                "time": str(est.time),
                "space": str(est.space),
                "cost": cost,
                "patterns": est.patterns_found,
            }
        )

    worst = sorted(fn_records, key=lambda f: f["cost"], reverse=True)[:5]
    unique_patterns = sorted(set(all_patterns))

    return {
        "bigO_cost_sum": round(cost_sum, 3),
        "bigO_fn_count": len(fn_records),
        "bigO_worst_fns": worst,
        "bigO_patterns_seen": unique_patterns,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ComplexityGrader
# ─────────────────────────────────────────────────────────────────────────────


class ComplexityGrader(BaseGrader):
    """
    Rewards reducing cyclomatic complexity (radon) and heuristic Big-O
    complexity (complexity_helper AST analysis) across the codebase.

    Inherits from BaseGrader:
      self._baseline        — metrics dict frozen at reset()
      self._delta_score()   — (baseline_val - current_val) / max(baseline, 1)
      self._clamp()         — clamp to [0.0, 1.0]
      self.file_handler     — provides repo root Path
      self.spec             — GraderSpec with scenario config dict

    Contract:
      _compute_metrics()  →  raw metrics dict (called at reset + every step)
      grade()             →  GradeResult (calls _compute_metrics internally)
    """

    grader_id: ClassVar[str] = "complexity"

    # ── BaseGrader interface ──────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        """
        Collect CC (radon) and Big-O (complexity_helper) metrics from sandbox.
        Pure in terms of state — reads files, runs radon, returns dict.
        """
        # Use hardcoded defaults to remove dependency on spec.config
        # Maintain interface stability by not requiring config attribute
        cfg = {
            "cc_threshold": 10,
            "cc_weight": 0.55,
            "bigO_weight": 0.45,
            "timeout": 15,
            "exclude_tests": True,
        }
        threshold = int(cfg.get("cc_threshold", 10))
        timeout = int(cfg.get("timeout", 15))
        exclude_tests = bool(cfg.get("exclude_tests", True))
        repo_root = self.file_handler.root

        # ── 1. Radon CC ───────────────────────────────────────────────────────
        by_file, radon_error = _run_radon(repo_root, timeout)
        cc_metrics = _collect_cc_metrics(by_file, threshold, exclude_tests)

        # ── 2. Heuristic Big-O via complexity_helper ──────────────────────────
        py_files = self._py_files(repo_root, exclude_tests)
        bigO_metrics = _collect_bigO_metrics(repo_root, py_files, exclude_tests)

        return {
            **cc_metrics,
            **bigO_metrics,
            "radon_error": radon_error,
        }

    def grade(self) -> GradeResult:
        """
        Delta-score both CC and Big-O dimensions against baseline.

        Both sub-deltas are clamped independently so a regression in
        one cannot make the combined score negative.

        score = clamp(cc_weight * cc_delta + bigO_weight * bigO_delta)
        """
        # Use hardcoded defaults to remove dependency on spec.config
        # Maintain interface stability by not requiring config attribute
        cfg = {
            "cc_threshold": 10,
            "cc_weight": 0.55,
            "bigO_weight": 0.45,
            "timeout": 15,
            "exclude_tests": True,
        }
        cc_w = float(cfg.get("cc_weight", 0.55))
        bigO_w = float(cfg.get("bigO_weight", 0.45))

        current = self._compute_metrics()
        baseline = self._baseline

        # ── Cyclomatic delta ──────────────────────────────────────────────────
        # Fall back to cc_avg if radon failed at either checkpoint
        if not current["radon_error"] and not baseline.get("radon_error"):
            cc_delta = self._delta_score(
                baseline["cc_cost_sum"],
                current["cc_cost_sum"],
            )
        else:
            # Degraded mode: use raw average CC as proxy
            cc_delta = self._delta_score(
                baseline["cc_avg"],
                current["cc_avg"],
            )

        # ── Big-O delta ───────────────────────────────────────────────────────
        bigO_delta = self._delta_score(
            baseline["bigO_cost_sum"],
            current["bigO_cost_sum"],
        )

        # ── Combined score ────────────────────────────────────────────────────
        # Ensure score is never zero to maintain positive reinforcement
        # Even with regression, give a small positive score
        score = max(0.01, self._clamp(
            cc_w * self._clamp(cc_delta) + bigO_w * self._clamp(bigO_delta)
        ))

        # ── Feedback and GradeResult ────────────────────────────────────────────
        cc_cost_b = baseline["cc_cost_sum"]
        cc_cost_n = current["cc_cost_sum"]
        cc_avg_b = baseline["cc_avg"]
        cc_avg_n = current["cc_avg"]
        bigO_b = baseline["bigO_cost_sum"]
        bigO_n = current["bigO_cost_sum"]

        def _arrow(b: float, n: float) -> str:
            return "↓" if n < b else ("↑" if n > b else "→")

        feedback_parts = [
            f"[Complexity] "
            f"CC cost: {cc_cost_b:.1f}→{cc_cost_n:.1f} {_arrow(cc_cost_b, cc_cost_n)} "
            f"(avg CC {cc_avg_b:.1f}→{cc_avg_n:.1f}); "
            f"Big-O cost: {bigO_b:.1f}→{bigO_n:.1f} {_arrow(bigO_b, bigO_n)}. "
            f"Score: {score:.3f} (CC×{cc_w} + BigO×{bigO_w})."
        ]

        # Surface the top remaining anti-patterns so the agent knows what to fix
        if current["bigO_worst_fns"]:
            top = current["bigO_worst_fns"][0]
            feedback_parts.append(
                f" Worst fn: '{top['qname']}' "
                f"({top['time']}, cost={top['cost']:.1f}). "
                f"Patterns: {', '.join(top['patterns']) or 'none'}."
            )

        feedbacks = ["".join(feedback_parts)]
        errors: list[str] = []
        tool_errors: list[str] = []

        if current.get("radon_error"):
            feedbacks.append(f"[radon: {current['radon_error']}]")

        # Add regression detection (but don't affect score calculation)
        if cc_cost_n > cc_cost_b or bigO_n > bigO_b:
            errors.append(
                f"[complexity] regression: "
                f"CC cost {cc_cost_b:.1f}→{cc_cost_n:.1f}, "
                f"Big-O cost {bigO_b:.1f}→{bigO_n:.1f}"
            )

        return GradeResult(
            score=round(score, 4),
            feedbacks=feedbacks,
            errors=errors,
            tool_errors=tool_errors,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _py_files(self, root: Path, exclude_tests: bool) -> list[Path]:
        """Enumerate all .py files under repo root, optionally skip test dirs."""
        files = []
        for p in root.rglob("*.py"):
            if exclude_tests:
                parts = p.relative_to(root).parts
                if any(part.startswith("test") for part in parts):
                    continue
            files.append(p)
        return files


__all__ = ["ComplexityGrader"]
