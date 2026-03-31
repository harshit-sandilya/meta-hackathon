"""
metrics.py  —  Parallel grader dispatcher for refactor-env.

Flow
----
  1. collect_metrics(sandbox, spec, step_count) is the single public entry point.
  2. active_components = spec.reward_weights.active_components()
     → only graders for non-zero weights are spawned (skip-zero logic).
  3. Active graders run concurrently via ThreadPoolExecutor.
     Exception in any grader is caught and surfaced as a GraderError,
     never silently swallowed.
  4. Returns GraderResults — a plain dataclass with one field per component.

Grader registry
---------------
  _GRADER_REGISTRY maps grader name (from ScenarioSpec.graders) → callable.
  Adding a new grader means: implement _grader_<name>(), register it below.
  Nothing else needs changing.

Graders
-------
  pytest_grader    →  acc  component  (TestSummary: pass_ratio)
  ruff_grader      →  qual component  (LintSummary: violation_count + delta)
  step_efficiency  →  eff  component  (pure arithmetic, no subprocess)
  fmt_grader       →  fmt  component  (placeholder, weight=0 for lint-cleanup)
"""

from __future__ import annotations

import ast
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from refactor_env.environment.registry.scenario import ScenarioSpec
    from refactor_env.environment.core.sandbox import Sandbox

from refactor_env.models import TestSummary, LintSummary


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class GraderResults:
    """
    Holds the raw score [0.0, 1.0] produced by each active grader,
    plus supporting detail objects for building observations.

    Inactive components (weight=0.0) are None.
    """

    acc: Optional[float] = None  # pytest pass_ratio
    qual: Optional[float] = None  # lint delta score
    eff: Optional[float] = None  # step efficiency
    fmt: Optional[float] = None  # format invariant score

    # Detail objects carried forward into RefactorObservation
    test_summary: TestSummary = field(default_factory=TestSummary)
    lint_summary: LintSummary = field(default_factory=LintSummary)

    # Syntax validity (checked eagerly before graders run)
    has_syntax_error: bool = False
    syntax_error_files: List[str] = field(default_factory=list)

    # Broken import detection result
    broken_imports: List[str] = field(default_factory=list)

    def score_for(self, component: str) -> float:
        """Return the grader score for a component, defaulting to 0.0 if None."""
        return getattr(self, component) or 0.0


@dataclass
class GraderError:
    """Captures a grader failure without crashing the episode."""

    component: str
    grader_name: str
    error: str


# ── Public entry point ────────────────────────────────────────────────────────


def collect_metrics(
    sandbox: "Sandbox",
    spec: "ScenarioSpec",
    step_count: int,
    executor_timeout: int = 60,
) -> tuple[GraderResults, List[GraderError]]:
    """
    Run all active graders in parallel and return (GraderResults, errors).

    Parameters
    ----------
    sandbox:
        Live Sandbox instance (must be initialised).
    spec:
        ScenarioSpec for the current task.
    step_count:
        Number of steps taken so far (used by step_efficiency grader).
    executor_timeout:
        Wall-clock timeout in seconds for the entire parallel batch.

    Returns
    -------
    (GraderResults, List[GraderError])
        GraderErrors are non-fatal — the orchestrator decides how to penalise.
    """
    results = GraderResults()
    errors: List[GraderError] = []

    # ── Eager syntax check (before any grader) ───────────────────────────────
    # A syntax error is a hard signal: acc=0, qual penalised, no point running
    # pytest since it will fail immediately. We still run ruff (to count
    # violations) but flag the syntax state.
    syntax_bad, bad_files = _check_syntax(sandbox, spec)
    results.has_syntax_error = syntax_bad
    results.syntax_error_files = bad_files

    # ── Determine active components ──────────────────────────────────────────
    active = spec.reward_weights.active_components()  # only non-zero weights

    # ── Build grader task list ────────────────────────────────────────────────
    # step_efficiency is pure arithmetic — no subprocess, runs inline
    # All others run in the thread pool
    inline_components = {"eff"}
    threaded_components = [c for c in active if c not in inline_components]

    # ── Inline graders ────────────────────────────────────────────────────────
    if "eff" in active:
        results.eff = _step_efficiency(step_count, spec)

    # ── Threaded graders ──────────────────────────────────────────────────────
    if threaded_components:
        with ThreadPoolExecutor(max_workers=len(threaded_components)) as pool:
            futures: Dict[Future, str] = {}
            for component in threaded_components:
                grader_name = getattr(spec.graders, component)
                fn = _GRADER_REGISTRY.get(grader_name)
                if fn is None:
                    errors.append(
                        GraderError(
                            component=component,
                            grader_name=grader_name,
                            error=f"Grader '{grader_name}' not found in registry.",
                        )
                    )
                    continue
                future = pool.submit(fn, sandbox, spec, results)
                futures[future] = component

            for future in as_completed(futures, timeout=executor_timeout):
                component = futures[future]
                grader_name = getattr(spec.graders, component)
                try:
                    score, detail = future.result()
                    setattr(results, component, score)
                    # Merge detail objects into results
                    if isinstance(detail, TestSummary):
                        results.test_summary = detail
                    elif isinstance(detail, LintSummary):
                        results.lint_summary = detail
                except Exception as exc:
                    errors.append(
                        GraderError(
                            component=component,
                            grader_name=grader_name,
                            error=str(exc),
                        )
                    )
                    setattr(results, component, 0.0)

    return results, errors


# ── Syntax check (eager, pre-grader) ─────────────────────────────────────────


def _check_syntax(sandbox: "Sandbox", spec: "ScenarioSpec") -> tuple[bool, List[str]]:
    """
    Parse all .py files in sandbox (excluding test_paths) with ast.parse.
    Returns (has_error, list_of_bad_relative_paths).

    Uses pure Python ast — no subprocess needed, no allowlist required.
    """
    bad: List[str] = []
    test_prefixes = tuple(p.rstrip("/") for p in spec.test_paths)

    for entry in sandbox.file_tree(recursive=True):
        if entry.is_dir:
            continue
        if not entry.path.endswith(".py"):
            continue
        # Skip test files — agent cannot edit them anyway
        if any(entry.path.startswith(p) for p in test_prefixes):
            continue
        try:
            content = sandbox.read_file(entry.path)
            ast.parse(content, filename=entry.path)
        except SyntaxError:
            bad.append(entry.path)
        except Exception:
            pass  # read errors are not syntax errors

    return bool(bad), bad


# ── Individual grader implementations ────────────────────────────────────────


def _pytest_grader(
    sandbox: "Sandbox",
    spec: "ScenarioSpec",
    results: GraderResults,
) -> tuple[float, TestSummary]:
    """
    Run pytest over spec.test_paths and return (pass_ratio, TestSummary).

    If has_syntax_error is True, skip execution entirely and return 0.0
    so we don't pay subprocess cost for a guaranteed failure.
    """
    if results.has_syntax_error:
        return 0.0, TestSummary(total=0, passed=0, failed=0, errors=1)

    from refactor_env.environment.core.executor import Executor

    executor = Executor(sandbox_root=sandbox.root)

    paths = " ".join(spec.test_paths)
    cmd = (
        f"python -m pytest {paths} "
        f"--tb=no -q --no-header "
        f"--json-report --json-report-file=/tmp/pytest_report.json"
    )

    rc, stdout, stderr = executor.run(
        cmd,
        workdir=sandbox.root,
        timeout_sec=45,
    )

    summary = _parse_pytest_output(stdout, stderr, rc)
    score = summary.pass_ratio
    return score, summary


def _ruff_grader(
    sandbox: "Sandbox",
    spec: "ScenarioSpec",
    results: GraderResults,
) -> tuple[float, LintSummary]:
    """
    Run ruff over sandbox root with task-specific rule selection.
    Returns (delta_score, LintSummary).

    delta_score = clamp(1 - current_violations / baseline_violations, 0, 1)

    Special cases:
      - If baseline_violations == 0: score = 1.0 if current == 0, else 0.0
      - If current == 0: score = 1.0 (perfect)
    """
    from refactor_env.environment.core.executor import Executor

    executor = Executor(sandbox_root=sandbox.root)

    select = ",".join(spec.lint_config.select)
    ignore_flag = ""
    if spec.lint_config.ignore:
        ignore_flag = "--ignore " + ",".join(spec.lint_config.ignore)

    # Output JSON for reliable parsing
    cmd = (
        f"ruff check . "
        f"--select {select} "
        f"{ignore_flag} "
        f"--output-format json "
        f"--quiet"
    )

    rc, stdout, stderr = executor.run(
        cmd,
        workdir=sandbox.root,
        timeout_sec=30,
    )

    summary, current_violations = _parse_ruff_output(stdout, stderr, rc)

    baseline = spec.lint_config.baseline_violations
    if baseline == 0:
        score = 1.0 if current_violations == 0 else 0.0
    else:
        score = max(0.0, min(1.0, 1.0 - (current_violations / baseline)))

    return score, summary


def _step_efficiency(step_count: int, spec: "ScenarioSpec") -> float:
    """
    Pure arithmetic efficiency score.

    Formula: max(0, 1 - (steps / max_steps) * decay_rate)

    decay_rate=0.5 → reaching max_steps gives 0.5 (not 0),
    so a late completion still beats no completion.
    """
    if spec.max_steps <= 0:
        return 1.0
    ratio = step_count / spec.max_steps
    score = max(0.0, 1.0 - ratio * spec.eff_config.decay_rate)
    return round(score, 4)


def _fmt_grader(
    sandbox: "Sandbox",
    spec: "ScenarioSpec",
    results: GraderResults,
) -> tuple[float, None]:
    """
    Placeholder fmt grader.
    Returns 1.0 if all invariant file paths exist and no forbidden patterns
    are present. Since lint-cleanup has fmt weight=0.0, this never runs
    for that task. Included for completeness on other tasks.
    """
    # Check all file_exists invariants
    for inv in spec.invariants:
        if inv.type == "file_exists":
            for path in inv.paths:
                if not os.path.isfile(os.path.join(sandbox.root, path)):
                    return 0.0, None
    return 1.0, None


# ── Output parsers ────────────────────────────────────────────────────────────


def _parse_pytest_output(stdout: str, stderr: str, rc: int) -> TestSummary:
    """
    Parse pytest -q --tb=no output into TestSummary.
    Tries JSON report first, falls back to stdout regex.
    """
    # Try JSON report
    try:
        report_path = "/tmp/pytest_report.json"
        if os.path.isfile(report_path):
            with open(report_path) as f:
                data = json.load(f)
            summary = data.get("summary", {})
            return TestSummary(
                total=summary.get("total", 0),
                passed=summary.get("passed", 0),
                failed=summary.get("failed", 0),
                errors=summary.get("error", 0),
                skipped=summary.get("skipped", 0),
            )
    except Exception:
        pass

    # Fallback: regex on stdout
    # pytest -q produces: "3 passed, 1 failed, 2 errors in 0.12s"
    total = passed = failed = errors = skipped = 0
    patterns = {
        "passed": r"(\d+) passed",
        "failed": r"(\d+) failed",
        "errors": r"(\d+) error",
        "skipped": r"(\d+) skipped",
    }
    combined = stdout + stderr
    for key, pat in patterns.items():
        m = re.search(pat, combined)
        if m:
            locals()[key]  # satisfy linter — we assign below
            if key == "passed":
                passed = int(m.group(1))
            if key == "failed":
                failed = int(m.group(1))
            if key == "errors":
                errors = int(m.group(1))
            if key == "skipped":
                skipped = int(m.group(1))

    total = passed + failed + errors + skipped
    return TestSummary(
        total=total, passed=passed, failed=failed, errors=errors, skipped=skipped
    )


def _parse_ruff_output(stdout: str, stderr: str, rc: int) -> tuple[LintSummary, int]:
    """
    Parse ruff --output-format json output into (LintSummary, violation_count).

    ruff exits 1 when violations found, 0 when clean.
    rc=2 is a ruff internal error — treat as 0 violations with a warning.
    """
    violation_count = 0
    error_by_code: Dict[str, int] = {}

    if rc == 2:
        # ruff internal error (e.g. bad config) — treat as no violations
        return LintSummary(total_errors=0), 0

    try:
        violations = json.loads(stdout) if stdout.strip() else []
        for v in violations:
            code = v.get("code", "UNKNOWN")
            error_by_code[code] = error_by_code.get(code, 0) + 1
        violation_count = len(violations)
    except (json.JSONDecodeError, TypeError):
        # ruff not installed or returned non-JSON — count lines as proxy
        lines = [l for l in stdout.splitlines() if l.strip() and not l.startswith("{")]
        violation_count = len(lines)

    summary = LintSummary(
        total_errors=violation_count,
        error_by_code=error_by_code,
    )
    return summary, violation_count


# ── Grader registry ───────────────────────────────────────────────────────────
# Maps grader name (from ScenarioSpec.graders.*) → callable.
# All callables have signature: (sandbox, spec, results) -> (float, detail | None)

_GRADER_REGISTRY = {
    "pytest_grader": _pytest_grader,
    "ruff_grader": _ruff_grader,
    "step_efficiency": None,
    "fmt_grader": _fmt_grader,
}
