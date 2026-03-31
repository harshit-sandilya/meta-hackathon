"""
Coverage Grader — test pass rate + line coverage tracking.

Gold standard : all tests pass AND line coverage >= target_coverage.

Dual-mode operation (set via config["graders"]["coverage"]["mode"]):
  "objective"  (default) — agent is rewarded for improving coverage
                           toward target_coverage. Score reflects progress.
  "constraint"           — tests must stay passing; coverage must not drop
                           below baseline. Score is binary per dimension:
                           1.0 if constraint holds, 0.0 if violated.
                           Used in lint-cleanup / api-rename where the goal
                           is NOT to write tests but to preserve them.

Score formula (objective mode)
────────────────────────────────
  acc_score  = passed / total                          ∈ [0, 1]
  cov_score  = progress_toward_target(                 ∈ [0, 1]
                 baseline_coverage, current_coverage,
                 target_coverage)
  final      = acc_weight * acc_score + cov_weight * cov_score

Score formula (constraint mode)
────────────────────────────────
  acc_score  = 1.0 if failed==0 and errors==0 else 0.0
  cov_score  = 1.0 if current_coverage >= baseline_coverage - tolerance
               else 0.0
  final      = acc_weight * acc_score + cov_weight * cov_score

Scenario config keys (under config["graders"]["coverage"]):
  weight          : float  — contribution to overall reward (default 0.50)
  mode            : str    — "objective" | "constraint"    (default "objective")
  target_coverage : float  — coverage fraction to aim for  (default 1.0)
  acc_weight      : float  — weight of pass-rate sub-score (default 0.70)
  cov_weight      : float  — weight of coverage sub-score  (default 0.30)
  coverage_tolerance: float— how much coverage can drop before penalised
                             in constraint mode              (default 0.02)
  test_paths      : list   — pytest target dirs/files        (default ["tests/"])
  cov_source      : str    — --cov= argument for pytest-cov  (default repo root)
  min_pass_rate   : float  — minimum acc_score for solved=True (default 1.0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_grader import BaseGrader, GradeResult, _clamp, already_gold, empty_grade
from .utils import MetricCache, PytestResult, run_pytest_cov


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ACC_WEIGHT = 0.70
_DEFAULT_COV_WEIGHT = 0.30
_DEFAULT_TARGET_COV = 1.0
_DEFAULT_COV_TOL = 0.02  # constraint mode: allow 2% coverage drop


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_mode(config: dict[str, Any]) -> str:
    cov_cfg = config.get("graders", {}).get("coverage", {})
    mode = cov_cfg.get("mode", "objective")
    if mode not in ("objective", "constraint"):
        mode = "objective"
    return mode


def _parse_weights(config: dict[str, Any]) -> tuple[float, float]:
    cov_cfg = config.get("graders", {}).get("coverage", {})
    acc_weight = float(cov_cfg.get("acc_weight", _DEFAULT_ACC_WEIGHT))
    cov_weight = float(cov_cfg.get("cov_weight", _DEFAULT_COV_WEIGHT))
    # Re-normalise so they always sum to 1.0
    total = acc_weight + cov_weight
    if total <= 0:
        return (_DEFAULT_ACC_WEIGHT, _DEFAULT_COV_WEIGHT)
    return (acc_weight / total, cov_weight / total)


def _acc_score_objective(result: dict[str, Any]) -> float:
    total = result.get("total", 0)
    passed = result.get("passed", 0)
    return passed / max(total, 1)


def _acc_score_constraint(result: dict[str, Any]) -> tuple[float, bool]:
    """Returns (score, is_regression)."""
    failed = result.get("failed", 0)
    errors = result.get("errors", 0)
    if failed == 0 and errors == 0 and result.get("total", 0) > 0:
        return (1.0, False)
    return (0.0, True)


def _coverage_score_objective(
    baseline_cov: float,
    current_cov: float,
    target_cov: float,
) -> tuple[float, bool]:
    """
    Score coverage progress toward target.
    Returns (score, is_regression).
    """
    score, is_reg = BaseGrader.progress_toward_target(
        baseline_value=baseline_cov,
        current_value=current_cov,
        target_value=target_cov,
        lower_is_better=False,
    )
    return (score, is_reg)


def _coverage_score_constraint(
    baseline_cov: float,
    current_cov: float,
    tolerance: float,
) -> tuple[float, bool]:
    """
    Constraint mode: coverage must not drop below baseline - tolerance.
    Returns (score, is_regression).
    """
    floor = baseline_cov - tolerance
    if current_cov >= floor:
        return (1.0, False)
    return (0.0, True)


def _per_file_delta(
    baseline_by_file: dict[str, float],
    current_by_file: dict[str, float],
) -> dict[str, float]:
    """
    Return per-file coverage delta: positive = improved, negative = regressed.
    Only includes files present in baseline (new files are tracked separately).
    """
    delta: dict[str, float] = {}
    for fpath, b_cov in baseline_by_file.items():
        c_cov = current_by_file.get(fpath, 0.0)
        delta[fpath] = round(c_cov - b_cov, 4)
    return delta


def _regressed_files(
    baseline_by_file: dict[str, float],
    current_by_file: dict[str, float],
    tolerance: float = 0.0,
) -> list[str]:
    """Return files whose coverage dropped by more than tolerance."""
    return [
        fpath
        for fpath, b_cov in baseline_by_file.items()
        if current_by_file.get(fpath, 0.0) < b_cov - tolerance
    ]


def _build_feedback(
    mode: str,
    acc_score: float,
    cov_score: float,
    final_score: float,
    is_regression: bool,
    baseline: dict[str, Any],
    current: dict[str, Any],
    target_cov: float,
    regressed_files: list[str],
) -> str:
    b_passed = baseline.get("passed", 0)
    b_total = baseline.get("total", 0)
    b_failed = baseline.get("failed", 0)
    b_cov = baseline.get("overall_coverage", 0.0)

    c_passed = current.get("passed", 0)
    c_total = current.get("total", 0)
    c_failed = current.get("failed", 0)
    c_errors = current.get("errors", 0)
    c_cov = current.get("overall_coverage", 0.0)

    if current.get("timed_out"):
        return (
            "[Coverage] pytest timed out — reduce test runtime or increase timeout_sec."
        )

    if current.get("parse_error"):
        return "[Coverage] Could not parse pytest output — check for syntax errors."

    if is_regression and mode == "constraint":
        parts = []
        if c_failed > 0 or c_errors > 0:
            parts.append(
                f"{c_failed} test(s) failing, {c_errors} error(s) "
                f"(was {b_failed} failing at baseline)"
            )
        if regressed_files:
            parts.append(
                f"coverage dropped in: {', '.join(regressed_files[:3])}"
                f"{'…' if len(regressed_files) > 3 else ''}"
            )
        detail = "; ".join(parts) or "unknown regression"
        return f"[Coverage] Constraint violated — {detail}."

    # Objective mode feedback
    test_str = (
        f"{c_passed}/{c_total} tests passing" if c_total > 0 else "no tests found"
    )
    cov_str = f"coverage {b_cov:.0%} → {c_cov:.0%} (target {target_cov:.0%})"

    if final_score >= 1.0:
        return f"[Coverage] Gold standard — {test_str}, {cov_str}."

    if is_regression:
        return (
            f"[Coverage] Regression: {test_str}, {cov_str}. "
            f"Fix failing tests before improving coverage."
        )

    improvement = f"{final_score:.0%} toward goal"
    extra = ""
    if c_failed > 0:
        extra = f" {c_failed} test(s) still failing."
    if regressed_files:
        extra += (
            f" Coverage dropped in "
            f"{', '.join(regressed_files[:2])}"
            f"{'…' if len(regressed_files) > 2 else ''}."
        )

    return f"[Coverage] {test_str}, {cov_str} ({improvement}).{extra}"


# ---------------------------------------------------------------------------
# CoverageGrader
# ---------------------------------------------------------------------------


class CoverageGrader(BaseGrader):
    """
    Scores the agent on test correctness (pass rate) and line coverage.

    Dual-mode:
      objective  — default; reward agent for writing more tests / fixing failures
      constraint — reward agent for keeping tests green; penalise regressions

    The two sub-scores (acc_score, cov_score) are independently computed
    then combined via configurable weights, giving the agent a clear signal
    even when only one dimension improves.
    """

    grader_id = "coverage"

    # ------------------------------------------------------------------
    # is_applicable override
    # ------------------------------------------------------------------

    def is_applicable(self, config: dict[str, Any]) -> bool:
        """Also check that a tests/ directory is referenced in config."""
        if not super().is_applicable(config):
            return False
        cov_cfg = config.get("graders", {}).get("coverage", {})
        test_paths = cov_cfg.get("test_paths", ["tests/"])
        return bool(test_paths)

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
        Run pytest with coverage and return structured metrics.

        Returns
        -------
        {
          "passed"            : int
          "failed"            : int
          "errors"            : int
          "total"             : int
          "pass_rate"         : float   ∈ [0, 1]
          "overall_coverage"  : float   ∈ [0, 1]
          "coverage_by_file"  : dict    {rel_path: float}
          "timed_out"         : bool
          "parse_error"       : bool
          "error"             : str|None
        }
        """
        repo_path = Path(repo_path)
        cov_cfg = config.get("graders", {}).get("coverage", {})

        test_paths: list[str] = cov_cfg.get("test_paths", ["tests/"])
        cov_source: str = cov_cfg.get("cov_source", str(repo_path))
        timeout_sec: int = int(cov_cfg.get("timeout_sec", 120))

        # Validate test paths exist — skip gracefully if not
        existing_test_paths = [tp for tp in test_paths if (repo_path / tp).exists()]
        if not existing_test_paths:
            return {
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "total": 0,
                "pass_rate": 0.0,
                "overall_coverage": 0.0,
                "coverage_by_file": {},
                "timed_out": False,
                "parse_error": False,
                "error": f"No test paths found: {test_paths}",
            }

        def _run() -> PytestResult:
            return run_pytest_cov(
                repo_path=repo_path,
                test_paths=existing_test_paths,
                timeout_sec=timeout_sec,
                cov_source=cov_source,
            )

        try:
            result: PytestResult = cache.get_or_compute("pytest_cov", None, _run)
        except Exception as exc:
            return {
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "total": 0,
                "pass_rate": 0.0,
                "overall_coverage": 0.0,
                "coverage_by_file": {},
                "timed_out": False,
                "parse_error": False,
                "error": str(exc),
            }

        return {
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "total": result.total,
            "pass_rate": result.pass_rate,
            "overall_coverage": result.overall_coverage,
            "coverage_by_file": result.coverage_by_file,
            "timed_out": result.timed_out,
            "parse_error": result.parse_error,
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
        Score test correctness and coverage improvement.

        Handles all edge cases:
          - pytest timeout          → empty_grade with timeout feedback
          - parse error             → empty_grade with parse feedback
          - no tests found          → empty_grade
          - baseline already gold   → already_gold
          - constraint regression   → score=0.0, is_regression=True
          - objective regression    → score reflects partial progress,
                                      is_regression=True if any dimension dropped
        """
        cov_cfg = config.get("graders", {}).get("coverage", {})
        mode = _parse_mode(config)
        target_cov = float(cov_cfg.get("target_coverage", _DEFAULT_TARGET_COV))
        tolerance = float(cov_cfg.get("coverage_tolerance", _DEFAULT_COV_TOL))
        min_pass = float(cov_cfg.get("min_pass_rate", 1.0))
        acc_weight, cov_weight = _parse_weights(config)

        # ── Error guards ──────────────────────────────────────────────
        if current.get("timed_out"):
            return empty_grade("Coverage", "pytest timed out")
        if current.get("parse_error"):
            return empty_grade("Coverage", "pytest output parse error")
        if current.get("error"):
            return empty_grade("Coverage", f"pytest error: {current['error']}")
        if current.get("total", 0) == 0:
            return empty_grade("Coverage", "no tests found")

        b_pass_rate = baseline.get("pass_rate", 0.0)
        c_pass_rate = current.get("pass_rate", 0.0)
        b_coverage = baseline.get("overall_coverage", 0.0)
        c_coverage = current.get("overall_coverage", 0.0)

        # ── Already at gold at baseline ───────────────────────────────
        baseline_at_gold = b_pass_rate >= 1.0 and b_coverage >= target_cov
        if baseline_at_gold:
            return already_gold("Coverage", current)

        # ── Per-file delta (for sub_scores and feedback) ──────────────
        b_by_file = baseline.get("coverage_by_file", {})
        c_by_file = current.get("coverage_by_file", {})
        file_delta = _per_file_delta(b_by_file, c_by_file)
        reg_files = _regressed_files(b_by_file, c_by_file, tolerance=0.0)

        # ── Mode-specific scoring ─────────────────────────────────────
        is_regression = False

        if mode == "constraint":
            acc_score, acc_reg = _acc_score_constraint(current)
            cov_score, cov_reg = _coverage_score_constraint(
                b_coverage, c_coverage, tolerance
            )
            is_regression = acc_reg or cov_reg
            final_score = _clamp(acc_weight * acc_score + cov_weight * cov_score)

        else:  # objective
            acc_score = _acc_score_objective(current)
            cov_score, cov_reg = _coverage_score_objective(
                b_coverage, c_coverage, target_cov
            )
            # Treat pass-rate drop as regression even in objective mode
            acc_reg = c_pass_rate < b_pass_rate - 1e-6
            is_regression = acc_reg or cov_reg
            final_score = _clamp(acc_weight * acc_score + cov_weight * cov_score)

        # solved: tests pass at min_pass_rate AND coverage >= target
        solved = c_pass_rate >= min_pass and c_coverage >= target_cov

        feedback = _build_feedback(
            mode=mode,
            acc_score=acc_score,
            cov_score=cov_score,
            final_score=final_score,
            is_regression=is_regression,
            baseline=baseline,
            current=current,
            target_cov=target_cov,
            regressed_files=reg_files,
        )

        # ── Delta dict ────────────────────────────────────────────────
        delta = {
            "passed": current.get("passed", 0) - baseline.get("passed", 0),
            "failed": baseline.get("failed", 0) - current.get("failed", 0),
            "overall_coverage": round(c_coverage - b_coverage, 4),
            "per_file": file_delta,
        }

        return GradeResult(
            score=final_score,
            gold_distance=1.0 - final_score,
            raw_baseline={
                "pass_rate": b_pass_rate,
                "passed": baseline.get("passed", 0),
                "failed": baseline.get("failed", 0),
                "total": baseline.get("total", 0),
                "overall_coverage": b_coverage,
            },
            raw_current={
                "pass_rate": c_pass_rate,
                "passed": current.get("passed", 0),
                "failed": current.get("failed", 0),
                "errors": current.get("errors", 0),
                "total": current.get("total", 0),
                "overall_coverage": c_coverage,
            },
            delta=delta,
            feedback=feedback,
            solved=solved,
            is_regression=is_regression,
            sub_scores={
                "acc_score": acc_score,
                "cov_score": cov_score,
                "acc_weight": acc_weight,
                "cov_weight": cov_weight,
                "target_coverage": target_cov,
                "mode": float(mode == "objective"),
                "regressed_files": float(len(reg_files)),
            },
        )

    # ------------------------------------------------------------------
    # gold_standard
    # ------------------------------------------------------------------

    def gold_standard(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Gold standard: all tests pass, coverage at target_coverage.
        """
        cov_cfg = config.get("graders", {}).get("coverage", {})
        target_cov = float(cov_cfg.get("target_coverage", _DEFAULT_TARGET_COV))
        return {
            "passed": 1,  # at least one test, all passing
            "failed": 0,
            "errors": 0,
            "total": 1,
            "pass_rate": 1.0,
            "overall_coverage": target_cov,
            "coverage_by_file": {},
            "timed_out": False,
            "parse_error": False,
            "error": None,
        }
