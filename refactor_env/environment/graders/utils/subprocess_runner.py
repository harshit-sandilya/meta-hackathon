"""
Safe, sandboxed tool execution.

All external tool calls (ruff, pytest, radon) go through this module.
Never call subprocess.run() directly from graders — use these wrappers
which enforce timeouts, resource limits, and output sanitization.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True)
class RuffViolation:
    filename: str  # relative path
    row: int
    col: int
    code: str  # e.g. "F401"
    message: str
    category: str = ""  # first letter of code: "F", "E", "W", etc.

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", self.code[0] if self.code else "")


@dataclass
class PytestResult:
    passed: int
    failed: int
    errors: int
    total: int
    coverage_by_file: dict[str, float]  # rel_path → line coverage %
    overall_coverage: float
    timed_out: bool = False
    parse_error: bool = False

    @property
    def pass_rate(self) -> float:
        return self.passed / max(self.total, 1)

    @property
    def all_pass(self) -> bool:
        return self.failed == 0 and self.errors == 0 and self.total > 0


@dataclass
class CCResult:
    """Single radon cyclomatic complexity entry."""

    name: str  # function/method name
    qualified_name: str
    file: str
    line: int
    complexity: int  # raw McCabe CC value
    grade: str  # A/B/C/D/E/F


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------

_DEFAULT_CPU_SECONDS = 60
_DEFAULT_MEM_MB = 512


def _set_resource_limits(cpu_seconds: int, mem_mb: int) -> None:
    """Pre-exec hook: set CPU time and memory limits for child process."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        mem_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, resource.error):
        pass  # Not all platforms support all limits


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

# Environment variable allowlist — strip everything else to avoid leakage
_SAFE_ENV_KEYS = {"PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV"}


def _safe_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Build a minimal, sanitized environment for subprocess calls."""
    env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
    if extra:
        env.update(extra)
    return env


def run_tool(
    cmd: list[str],
    cwd: Path,
    timeout_sec: int = 30,
    env_extra: Optional[dict[str, str]] = None,
    cpu_seconds: int = _DEFAULT_CPU_SECONDS,
    mem_mb: int = _DEFAULT_MEM_MB,
) -> ToolResult:
    """
    Run an external command in the sandbox with hard timeouts and resource limits.

    Parameters
    ----------
    cmd         : Command and arguments list.
    cwd         : Working directory (the sandbox repo path).
    timeout_sec : Wall-clock timeout; process is killed if exceeded.
    env_extra   : Additional environment variables to pass.
    cpu_seconds : CPU time limit (RLIMIT_CPU).
    mem_mb      : Memory ceiling in MB (RLIMIT_AS).
    """

    def _preexec() -> None:
        _set_resource_limits(cpu_seconds, mem_mb)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=_safe_env(env_extra),
            preexec_fn=_preexec if sys.platform != "win32" else None,
        )
        return ToolResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(stdout="", stderr="timed_out", returncode=-1, timed_out=True)
    except FileNotFoundError as exc:
        return ToolResult(stdout="", stderr=str(exc), returncode=-2, timed_out=False)
    except Exception as exc:
        return ToolResult(stdout="", stderr=str(exc), returncode=-3, timed_out=False)


# ---------------------------------------------------------------------------
# Ruff
# ---------------------------------------------------------------------------

# Severity weights for weighted violation counting in LintGrader
RUFF_CATEGORY_WEIGHTS: dict[str, float] = {
    "F": 1.5,  # pyflakes — undefined names, unused imports (real bugs)
    "E": 1.0,  # pycodestyle errors
    "B": 1.2,  # bugbear — likely bugs
    "UP": 0.7,  # pyupgrade — modernization
    "W": 0.5,  # pycodestyle warnings
    "I": 0.6,  # isort
    "C": 0.8,  # conventions
    "N": 0.6,  # pep8 naming
    "SIM": 0.9,  # simplify
}


def run_ruff(
    repo_path: Path,
    files: Optional[list[str]] = None,
    timeout_sec: int = 30,
) -> list[RuffViolation]:
    """
    Run ruff and return structured violation list.

    Parameters
    ----------
    files : If given, run ruff only on these relative paths (incremental mode).
            If None, runs on the entire repo_path directory.
    """
    targets = [str(repo_path / f) for f in files] if files else [str(repo_path)]
    cmd = ["ruff", "check", "--output-format", "json", "--no-cache"] + targets

    result = run_tool(cmd, cwd=repo_path, timeout_sec=timeout_sec)

    if result.timed_out or not result.stdout.strip():
        return []

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    violations: list[RuffViolation] = []
    for item in raw:
        code = item.get("code", "")
        loc = item.get("location", {})
        violations.append(
            RuffViolation(
                filename=item.get("filename", ""),
                row=loc.get("row", 0),
                col=loc.get("column", 0),
                code=code,
                message=item.get("message", ""),
            )
        )
    return violations


def weighted_violation_count(violations: list[RuffViolation]) -> float:
    """Sum violations weighted by category severity."""
    return sum(RUFF_CATEGORY_WEIGHTS.get(v.category, 1.0) for v in violations)


# ---------------------------------------------------------------------------
# Pytest + coverage
# ---------------------------------------------------------------------------


def run_pytest_cov(
    repo_path: Path,
    test_paths: Optional[list[str]] = None,
    timeout_sec: int = 120,
    cov_source: Optional[str] = None,
) -> PytestResult:
    """
    Run pytest with coverage and return structured results.

    Requires pytest, pytest-json-report, and pytest-cov to be installed
    in the sandbox environment.
    """
    cov_src = cov_source or str(repo_path)
    targets = test_paths or ["tests/"]

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--tb=no",
        "--quiet",
        "--json-report",
        "--json-report-file=/tmp/pytest_report.json",
        f"--cov={cov_src}",
        "--cov-report=json:/tmp/coverage.json",
        "--cov-report=",  # suppress terminal coverage output
    ] + targets

    result = run_tool(
        cmd, cwd=repo_path, timeout_sec=timeout_sec, cpu_seconds=timeout_sec
    )

    if result.timed_out:
        return PytestResult(0, 0, 0, 0, {}, 0.0, timed_out=True)

    # Parse pytest JSON report
    try:
        with open("/tmp/pytest_report.json") as f:
            report = json.load(f)
        summary = report.get("summary", {})
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        errors = summary.get("error", 0)
        total = summary.get("total", passed + failed + errors)
    except (OSError, json.JSONDecodeError, KeyError):
        return PytestResult(0, 0, 0, 0, {}, 0.0, parse_error=True)

    # Parse coverage JSON
    coverage_by_file: dict[str, float] = {}
    overall_coverage = 0.0
    try:
        with open("/tmp/coverage.json") as f:
            cov_data = json.load(f)
        totals = cov_data.get("totals", {})
        overall_coverage = totals.get("percent_covered", 0.0) / 100.0
        for fpath, fdata in cov_data.get("files", {}).items():
            pct = fdata.get("summary", {}).get("percent_covered", 0.0)
            # Make path relative to repo_path
            try:
                rel = str(Path(fpath).relative_to(repo_path))
            except ValueError:
                rel = fpath
            coverage_by_file[rel] = pct / 100.0
    except (OSError, json.JSONDecodeError):
        pass

    return PytestResult(
        passed=passed,
        failed=failed,
        errors=errors,
        total=total,
        coverage_by_file=coverage_by_file,
        overall_coverage=overall_coverage,
    )


# ---------------------------------------------------------------------------
# Radon — cyclomatic complexity
# ---------------------------------------------------------------------------

_RADON_GRADE_THRESHOLDS = {"A": 5, "B": 10, "C": 15, "D": 20, "E": 25}


def _cc_value_to_grade(cc: int) -> str:
    if cc <= 5:
        return "A"
    if cc <= 10:
        return "B"
    if cc <= 15:
        return "C"
    if cc <= 20:
        return "D"
    if cc <= 25:
        return "E"
    return "F"


def run_radon_cc(
    repo_path: Path,
    files: Optional[list[str]] = None,
    timeout_sec: int = 30,
) -> list[CCResult]:
    """
    Run radon cyclomatic complexity analysis and return structured results.
    """
    targets = [str(repo_path / f) for f in files] if files else [str(repo_path)]
    cmd = ["radon", "cc", "--json", "--min", "A"] + targets

    result = run_tool(cmd, cwd=repo_path, timeout_sec=timeout_sec)
    if result.timed_out or not result.stdout.strip():
        return []

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    cc_results: list[CCResult] = []
    for fpath, entries in raw.items():
        try:
            rel = str(Path(fpath).relative_to(repo_path))
        except ValueError:
            rel = fpath
        for entry in entries:
            cc_val = entry.get("complexity", 0)
            cc_results.append(
                CCResult(
                    name=entry.get("name", ""),
                    qualified_name=entry.get("fullname", entry.get("name", "")),
                    file=rel,
                    line=entry.get("lineno", 0),
                    complexity=cc_val,
                    grade=_cc_value_to_grade(cc_val),
                )
            )
    return cc_results


def run_radon_mi(
    repo_path: Path,
    files: Optional[list[str]] = None,
    timeout_sec: int = 30,
) -> dict[str, float]:
    """
    Run radon maintainability index.
    Returns {rel_path: mi_score} where score is 0–100 (higher = more maintainable).
    Scores below 20 are considered unmaintainable (grade F).
    """
    targets = [str(repo_path / f) for f in files] if files else [str(repo_path)]
    cmd = ["radon", "mi", "--json"] + targets

    result = run_tool(cmd, cwd=repo_path, timeout_sec=timeout_sec)
    if result.timed_out or not result.stdout.strip():
        return {}

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}

    mi_scores: dict[str, float] = {}
    for fpath, data in raw.items():
        try:
            rel = str(Path(fpath).relative_to(repo_path))
        except ValueError:
            rel = fpath
        if isinstance(data, (int, float)):
            mi_scores[rel] = float(data)
        elif isinstance(data, dict):
            mi_scores[rel] = float(data.get("mi", 0.0))
    return mi_scores
