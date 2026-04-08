"""
coverage_grader.py — Test correctness & coverage tracking.

Dual-mode grader:

  Mode A — constraint  (lint-cleanup, api-rename, style-enforce)
  ─────────────────────────────────────────────────────────────
  Tests must keep passing and coverage must not regress.
  score = pass_rate * coverage_multiplier

  Mode B — objective   (test-coverage task)
  ──────────────────────────────────────────
  Writing new tests to improve coverage is the primary goal.
  score = clamp((cov_now - cov_base) / max(target - cov_base, 0.01)) * pass_rate

Mode resolved from spec.config["coverage"]["mode"] (default: "constraint").
If scenario_type == "test-coverage", defaults to "objective".

Raw metrics keys:
  passed            int
  failed            int
  errors            int
  total             int
  pass_rate         float
  line_coverage     float
  branch_coverage   float
  per_file          dict[str, {"line": float, "branch": float}]
  timed_out         bool
  run_error         str
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import ClassVar

from ....models.actions import RunShellParams
from ....models.grader_spec import GraderSpec
from ...sandbox.files import FileHandler
from ...sandbox.runner import ShellExecutor
from .base import BaseGrader, GradeResult


_PASS_RE = re.compile(r"(\d+) passed")
_FAIL_RE = re.compile(r"(\d+) failed")
_ERROR_RE = re.compile(r"(\d+) error")


class CoverageGrader(BaseGrader):
    """
    Reports how many tests pass/fail and tracks line/branch coverage.

    _compute_metrics() — runs pytest --cov, returns raw counts + coverage.
    grade()            — delta-scores against self._baseline; auto-picks mode.
    grade_as_constraint() / grade_as_objective() — force a mode; used by reward.py.
    """

    grader_id: ClassVar[str] = "coverage"

    # ── BaseGrader interface ──────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        """
        Run pytest inside the sandbox and return raw test + coverage metrics.

        Writes all pytest/coverage artefacts to a temp dir so the sandbox
        working tree is never polluted between steps.
        """
        # Use hardcoded defaults to remove dependency on spec.config
        # Maintain interface stability by not requiring config attribute
        cfg = {
            "mode": "constraint",
            "coverage_tolerance": 0.02,
            "timeout": 60,
            "test_paths": ["tests/"],
            "source_paths": ["."],
            "branch_coverage": True,
            "target_coverage": 0.80,
        }
        repo_path = self.file_handler.root
        timeout = int(cfg.get("timeout", 60))
        test_paths = cfg.get("test_paths", ["tests/"])
        src_paths = cfg.get("source_paths", ["."])
        use_branch = cfg.get("branch_coverage", True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cov_json = tmp_path / "coverage.json"
            pytest_json = tmp_path / "pytest_report.json"

            cmd = [
                "python",
                "-m",
                "pytest",
                "--tb=no",
                "-q",
                "--json-report",
                f"--json-report-file={pytest_json}",
                f"--cov-report=json:{cov_json}",
                *[f"--cov={s}" for s in src_paths],
                *(["--cov-branch"] if use_branch else []),
                *test_paths,
            ]

            # Convert list to string
            cmd_str = " ".join(cmd)
            result = self.executor.run(
                RunShellParams(command=cmd_str, timeout_sec=timeout, workdir=".")
            )

            if result.timed_out:
                return _empty_metrics(
                    timed_out=True,
                    run_error=f"pytest timed out after {timeout}s",
                )
            if hasattr(result, 'run_error') and result.run_error and result.run_error.strip():
                return _empty_metrics(run_error=result.run_error)
            # Check return_code only if it exists and is not 0
            if hasattr(result, 'return_code') and result.return_code != 0:
                return _empty_metrics(run_error=result.stderr or "pytest failed")

            passed, failed, errors = _parse_counts(pytest_json, result.stdout)
            total = passed + failed + errors
            line_cov, branch_cov, per_file = _parse_coverage(cov_json)

        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": total,
            "pass_rate": round(passed / max(total, 1), 4),
            "line_coverage": line_cov,
            "branch_coverage": branch_cov,
            "per_file": per_file,
            "timed_out": False,
            "run_error": "",
        }

    def grade(self) -> GradeResult:
        """Compute current metrics and delta-score against baseline."""
        current = self._compute_metrics()
        return self._score(self._baseline, current)

    # ── Wrappers for reward.py ────────────────────────────────────────────────

    def grade_as_constraint(self) -> GradeResult:
        """Force constraint mode (pass-rate + coverage must not regress)."""
        current = self._compute_metrics()
        return self._score(self._baseline, current, force_mode="constraint")

    def grade_as_objective(self) -> GradeResult:
        """Force objective mode (coverage improvement toward target)."""
        current = self._compute_metrics()
        return self._score(self._baseline, current, force_mode="objective")

    # ── Internal scoring ──────────────────────────────────────────────────────

    def _score(
        self,
        baseline: dict,
        current: dict,
        force_mode: str | None = None,
    ) -> GradeResult:
        mode = force_mode or self._resolve_mode()

        # Run failed entirely
        if current.get("timed_out") or current.get("run_error"):
            msg = current.get("run_error") or "pytest timed out"
            return GradeResult(
                score=0.0,
                feedbacks=[f"[Coverage] pytest did not complete: {msg}"],
                errors=[] if not current.get("timed_out") else [f"pytest timed out after {current.get('timeout', 'unknown')}s"],
                tool_errors=[] if not current.get("run_error") else [current.get("run_error")],
                added_violations=0,
            )

        if mode == "constraint":
            return self._constraint_result(baseline, current)
        return self._objective_result(baseline, current)

    def _resolve_mode(self) -> str:
        # Check spec.config if available, default to constraint mode
        if hasattr(self.spec, 'config') and self.spec.config and 'coverage' in self.spec.config:
            return self.spec.config.get('coverage', {}).get('mode', 'constraint')
        return "constraint"

    def _constraint_result(self, baseline: dict, current: dict) -> GradeResult:
        """
        score = pass_rate * coverage_multiplier
          1.0× if coverage held within tolerance
          0.5× if coverage dropped beyond tolerance
        """
        # Use hardcoded defaults to remove dependency on spec.config
        # Maintain interface stability by not requiring config attribute
        cfg = {
            "mode": "constraint",
            "coverage_tolerance": 0.02,
            "timeout": 60,
            "test_paths": ["tests/"],
            "source_paths": ["."],
            "branch_coverage": True,
            "target_coverage": 0.80,
        }
        tolerance = float(cfg.get("coverage_tolerance", 0.02))

        pass_rate = current["pass_rate"]
        cov_now = current["line_coverage"]
        cov_base = baseline["line_coverage"]
        cov_ok = cov_now >= (cov_base - tolerance)
        score = self._clamp(pass_rate * (1.0 if cov_ok else 0.5))

        delta = round((cov_now - cov_base) * 100, 1)
        sign = "+" if delta >= 0 else ""
        p, t = current["passed"], current["total"]

        feedback = (
            f"[Coverage/constraint] {p}/{t} tests pass ({pass_rate:.0%}). "
            f"Coverage: {cov_base:.1%} → {cov_now:.1%} ({sign}{delta}pp)"
            + (
                "."
                if cov_ok
                else f" — dropped beyond {tolerance:.0%} tolerance; score halved."
            )
        )

        return GradeResult(
            score=round(score, 4),
            feedbacks=[feedback],
            errors=[],
            tool_errors=[],
            added_violations=0,
        )

    def _objective_result(self, baseline: dict, current: dict) -> GradeResult:
        """
        score = clamp((cov_now - cov_base) / max(target - cov_base, 0.01)) * pass_rate

        pass_rate multiplier prevents gaming by writing stubs that collect
        lines but never assert anything.
        """
        # Use hardcoded defaults to remove dependency on spec.config
        # Maintain interface stability by not requiring config attribute
        cfg = {
            "mode": "constraint",
            "coverage_tolerance": 0.02,
            "timeout": 60,
            "test_paths": ["tests/"],
            "source_paths": ["."],
            "branch_coverage": True,
            "target_coverage": 0.80,
        }
        # Merge spec.config if available
        if hasattr(self.spec, 'config') and self.spec.config and 'coverage' in self.spec.config:
            cfg.update(self.spec.config['coverage'])
        target = float(cfg.get("target_coverage", 0.80))

        cov_now = current["line_coverage"]
        cov_base = baseline["line_coverage"]
        pass_rate = current["pass_rate"]
        gap = max(target - cov_base, 0.01)
        score = self._clamp((cov_now - cov_base) / gap) * pass_rate

        delta = round((cov_now - cov_base) * 100, 1)
        sign = "+" if delta >= 0 else ""
        p, t = current["passed"], current["total"]
        is_reg = cov_now < cov_base

        if is_reg:
            feedback = (
                f"[Coverage/objective] Coverage regressed: "
                f"{cov_base:.1%} → {cov_now:.1%} ({sign}{delta}pp). "
                f"Target: {target:.1%}. Tests: {p}/{t}."
            )
        elif score >= 1.0:
            feedback = (
                f"[Coverage/objective] Target reached! "
                f"Coverage {cov_now:.1%} ≥ {target:.1%}. Tests: {p}/{t}."
            )
        else:
            feedback = (
                f"[Coverage/objective] Coverage: {cov_base:.1%} → {cov_now:.1%} "
                f"({sign}{delta}pp, {score:.0%} toward {target:.1%}). "
                f"Tests: {p}/{t} ({pass_rate:.0%})."
            )

        return GradeResult(
            score=round(score, 4),
            feedbacks=[feedback],
            errors=[],
            tool_errors=[],
            added_violations=0,
        )


# ── Module-level helpers (pure, no I/O) ──────────────────────────────────────


def _empty_metrics(*, timed_out: bool = False, run_error: str = "") -> dict:
    return {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "total": 0,
        "pass_rate": 0.0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "per_file": {},
        "timed_out": timed_out,
        "run_error": run_error,
    }


def _parse_counts(pytest_json: Path, stdout: str) -> tuple[int, int, int]:
    """Try pytest-json-report first, fall back to terminal regex."""
    if pytest_json.exists():
        try:
            s = json.loads(pytest_json.read_text()).get("summary", {})
            return (
                s.get("passed", 0),
                s.get("failed", 0),
                s.get("errors", 0) + s.get("error", 0),
            )
        except Exception:
            pass
    passed = int(m.group(1)) if (m := _PASS_RE.search(stdout)) else 0
    failed = int(m.group(1)) if (m := _FAIL_RE.search(stdout)) else 0
    errors = int(m.group(1)) if (m := _ERROR_RE.search(stdout)) else 0
    return passed, failed, errors


def _parse_coverage(cov_json: Path) -> tuple[float, float, dict]:
    """Parse a coverage.py JSON report. Returns (line_cov, branch_cov, per_file)."""
    if not cov_json.exists():
        return 0.0, 0.0, {}
    try:
        data = json.loads(cov_json.read_text())
    except Exception:
        return 0.0, 0.0, {}

    t = data.get("totals", {})
    line_cov = t.get("covered_lines", 0) / max(t.get("num_statements", 1), 1)
    n_br = t.get("num_branches", 0)
    branch_cov = (t.get("covered_branches", 0) / max(n_br, 1)) if n_br else 0.0

    per_file: dict = {}
    for fname, fdata in data.get("files", {}).items():
        s = fdata.get("summary", {})
        f_br = s.get("num_branches", 0)
        per_file[fname] = {
            "line": s.get("covered_lines", 0) / max(s.get("num_statements", 1), 1),
            "branch": (s.get("covered_branches", 0) / max(f_br, 1)) if f_br else 0.0,
        }

    return round(line_cov, 4), round(branch_cov, 4), per_file


__all__ = ["CoverageGrader"]
