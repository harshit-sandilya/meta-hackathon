"""
Structure Grader — scenario.yaml invariant enforcement.

Gold standard : every declared invariant passes (score = 1.0).

Invariants are declared in scenario.yaml under the top-level key
`invariants:` as a list. Each invariant is a dict with:

  type        : str   — one of the invariant kinds listed below
  weight      : float — relative importance (default 1.0); used in
                        checklist_score() so failing a critical invariant
                        tanks the score more than a minor one
  severity    : str   — "hard" | "soft" (default "soft")
                        hard invariants set is_regression=True if violated
                        regardless of overall score direction
  description : str   — optional human-readable label for feedback

Supported invariant types
─────────────────────────
  required_symbol   — a function/class must exist after refactor
    params: name (str), kind (str, optional: "function"|"class")
            file (str, optional: restrict check to one file)

  forbidden_symbol  — a name must NOT exist (e.g. old API name after rename)
    params: name (str), kind (str, optional)
            file (str, optional)

  required_file     — a file path must exist
    params: path (str)

  forbidden_file    — a file path must NOT exist
    params: path (str)

  no_edit_dirs      — no files in listed directories may be modified
    params: dirs (list[str])   e.g. ["tests/", "migrations/"]

  no_new_files      — no new .py files may be added outside allowed dirs
    params: allowed_dirs (list[str], optional)

  no_deleted_files  — no .py files in listed paths may be deleted
    params: paths (list[str])

  module_exports    — a module must export a specific set of public names
    params: module (str), exports (list[str])

  no_circular_imports — the repo must have zero import cycles
    params: (none)

  max_file_count    — a directory must contain at most N .py files
    params: dir (str), max_count (int)

  max_fanout        — a module must not import from more than N other modules
    params: module (str), max_fanout (int)

  file_size_limit   — a file must not exceed N lines
    params: path (str), max_lines (int)

  custom_ast_check  — arbitrary AST assertion (advanced)
    params: file (str), check_id (str) — registered in CUSTOM_CHECKS dict

Scenario config keys (under config["graders"]["structure"]):
  weight : float — contribution to overall reward (default 0.10)

Global scenario keys consumed directly:
  invariants : list[dict] — the invariant list described above
"""

from __future__ import annotations

import ast
import fnmatch
from pathlib import Path
from typing import Any, Callable

from .base_grader import BaseGrader, GradeResult, _clamp, already_gold, empty_grade
from .utils import (
    MetricCache,
    DependencyGraph,
    parse_repository,
    collect_definitions,
    parse_file_safe,
    get_changed_files,
    get_added_files,
    get_deleted_files,
    files_modified_in_dirs,
)


# ---------------------------------------------------------------------------
# Custom AST check registry
# Populated by tasks that need invariants beyond the built-in types.
# Keys are check_id strings referenced in scenario.yaml.
# Values are callables: (repo_path: Path, modules: dict) -> bool
# ---------------------------------------------------------------------------

CUSTOM_CHECKS: dict[str, Callable[[Path, dict[str, ast.Module]], bool]] = {}


def register_custom_check(
    check_id: str,
    fn: Callable[[Path, dict[str, ast.Module]], bool],
) -> None:
    """Register a custom AST check for use in scenario.yaml."""
    CUSTOM_CHECKS[check_id] = fn


# ---------------------------------------------------------------------------
# Per-invariant check functions
# Each returns (passed: bool, detail: str)
# ---------------------------------------------------------------------------


def _check_required_symbol(
    inv: dict[str, Any],
    modules: dict[str, ast.Module],
    repo_path: Path,
) -> tuple[bool, str]:
    name = inv.get("name", "")
    kind = inv.get("kind", None)  # None = any kind
    file_hint = inv.get("file", None)  # restrict to one file

    target_modules = {
        p: m
        for p, m in modules.items()
        if not file_hint or p == file_hint or p.endswith(file_hint)
    }

    for rel_path, module in target_modules.items():
        for defn in collect_definitions(module, rel_path):
            name_match = defn.name == name or defn.qualified_name.endswith(f".{name}")
            kind_match = (
                (kind is None)
                or (defn.kind == kind)
                or (kind == "function" and defn.kind == "async_function")
            )
            if name_match and kind_match:
                return (True, f"'{name}' found in {rel_path}:{defn.line}")

    scope = f" in {file_hint}" if file_hint else " in any file"
    kind_str = f" ({kind})" if kind else ""
    return (False, f"Required symbol '{name}'{kind_str} not found{scope}")


def _check_forbidden_symbol(
    inv: dict[str, Any],
    modules: dict[str, ast.Module],
    repo_path: Path,
) -> tuple[bool, str]:
    name = inv.get("name", "")
    kind = inv.get("kind", None)
    file_hint = inv.get("file", None)

    target_modules = {
        p: m
        for p, m in modules.items()
        if not file_hint or p == file_hint or p.endswith(file_hint)
    }

    violations: list[str] = []
    for rel_path, module in target_modules.items():
        for defn in collect_definitions(module, rel_path):
            name_match = defn.name == name or defn.qualified_name.endswith(f".{name}")
            kind_match = (
                (kind is None)
                or (defn.kind == kind)
                or (kind == "function" and defn.kind == "async_function")
            )
            if name_match and kind_match:
                violations.append(f"{rel_path}:{defn.line}")

    if violations:
        locs = ", ".join(violations[:3])
        return (False, f"Forbidden symbol '{name}' still exists at: {locs}")
    return (True, f"Forbidden symbol '{name}' correctly absent")


def _check_required_file(
    inv: dict[str, Any],
    repo_path: Path,
) -> tuple[bool, str]:
    path = inv.get("path", "")
    target = repo_path / path
    if target.exists():
        return (True, f"Required file '{path}' exists")
    return (False, f"Required file '{path}' does not exist")


def _check_forbidden_file(
    inv: dict[str, Any],
    repo_path: Path,
) -> tuple[bool, str]:
    path = inv.get("path", "")
    target = repo_path / path
    if not target.exists():
        return (True, f"Forbidden file '{path}' correctly absent")
    return (False, f"Forbidden file '{path}' still exists")


def _check_no_edit_dirs(
    inv: dict[str, Any],
    repo_path: Path,
) -> tuple[bool, str]:
    dirs: list[str] = inv.get("dirs", [])
    if not dirs:
        return (True, "no_edit_dirs: no directories specified")

    modified = files_modified_in_dirs(repo_path, dirs)
    if not modified:
        return (True, f"No protected files modified in: {', '.join(dirs)}")

    shown = modified[:5]
    suffix = f" (+{len(modified)-5} more)" if len(modified) > 5 else ""
    return (
        False,
        f"Protected files modified: {', '.join(shown)}{suffix}",
    )


def _check_no_new_files(
    inv: dict[str, Any],
    repo_path: Path,
) -> tuple[bool, str]:
    allowed_dirs: list[str] = inv.get("allowed_dirs", [])
    new_files = get_added_files(repo_path)

    if not new_files:
        return (True, "No new .py files added")

    violations: list[str] = []
    for f in new_files:
        in_allowed = any(
            f.startswith(d.rstrip("/") + "/") or f.startswith(d) for d in allowed_dirs
        )
        if not in_allowed:
            violations.append(f)

    if not violations:
        return (True, f"All new files are within allowed dirs: {allowed_dirs}")

    shown = violations[:3]
    suffix = f" (+{len(violations)-3} more)" if len(violations) > 3 else ""
    return (
        False,
        f"New files added outside allowed dirs: {', '.join(shown)}{suffix}",
    )


def _check_no_deleted_files(
    inv: dict[str, Any],
    repo_path: Path,
) -> tuple[bool, str]:
    protected_paths: list[str] = inv.get("paths", [])
    deleted = get_deleted_files(repo_path)

    if not deleted:
        return (True, "No .py files deleted")

    deleted_set = set(deleted)
    violations: list[str] = []
    for path in protected_paths:
        if path in deleted_set:
            violations.append(path)
        # Glob support: paths may be directory prefixes
        for d in deleted_set:
            if fnmatch.fnmatch(d, path):
                violations.append(d)

    violations = list(set(violations))
    if not violations:
        return (True, "No protected files deleted")

    return (False, f"Protected file(s) deleted: {', '.join(violations)}")


def _check_module_exports(
    inv: dict[str, Any],
    modules: dict[str, ast.Module],
    repo_path: Path,
) -> tuple[bool, str]:
    module_path: str = inv.get("module", "")
    required_exports: list[str] = inv.get("exports", [])

    # Find the module file
    target_module = None
    target_rel = None
    for rel_path, module in modules.items():
        if rel_path == module_path or rel_path.endswith(module_path):
            target_module = module
            target_rel = rel_path
            break

    if target_module is None:
        return (False, f"Module '{module_path}' not found in repo")

    # Collect all defined public names
    defined_names = {
        defn.name
        for defn in collect_definitions(target_module, target_rel)
        if defn.is_public
    }

    missing = [name for name in required_exports if name not in defined_names]
    if not missing:
        return (
            True,
            f"Module '{module_path}' exports all required names: "
            f"{required_exports}",
        )

    return (
        False,
        f"Module '{module_path}' missing exports: {missing}",
    )


def _check_no_circular_imports(
    graph: DependencyGraph,
) -> tuple[bool, str]:
    cycles = graph.get_import_cycles()
    if not cycles:
        return (True, "No import cycles detected")

    # Show first cycle for actionability
    first = " → ".join(cycles[0][:5])
    if len(cycles[0]) > 5:
        first += " → …"
    return (
        False,
        f"{len(cycles)} import cycle(s) detected. First: [{first}]",
    )


def _check_max_file_count(
    inv: dict[str, Any],
    repo_path: Path,
) -> tuple[bool, str]:
    directory = inv.get("dir", "")
    max_count = int(inv.get("max_count", 999))
    target_dir = repo_path / directory

    if not target_dir.exists():
        return (True, f"Directory '{directory}' does not exist (vacuously true)")

    py_files = list(target_dir.glob("*.py"))
    count = len(py_files)
    if count <= max_count:
        return (True, f"'{directory}' has {count} .py files (limit {max_count})")
    return (
        False,
        f"'{directory}' has {count} .py files, exceeds limit of {max_count}",
    )


def _check_max_fanout(
    inv: dict[str, Any],
    graph: DependencyGraph,
    repo_path: Path,
) -> tuple[bool, str]:
    module_path = inv.get("module", "")
    max_fanout = int(inv.get("max_fanout", 10))
    fanout = graph.get_module_fanout(module_path)

    if fanout <= max_fanout:
        return (
            True,
            f"'{module_path}' imports from {fanout} modules (limit {max_fanout})",
        )
    return (
        False,
        f"'{module_path}' imports from {fanout} modules, exceeds limit of {max_fanout}",
    )


def _check_file_size_limit(
    inv: dict[str, Any],
    repo_path: Path,
) -> tuple[bool, str]:
    path = inv.get("path", "")
    max_lines = int(inv.get("max_lines", 500))
    target = repo_path / path

    if not target.exists():
        return (True, f"File '{path}' does not exist (vacuously true)")

    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return (False, f"Could not read '{path}'")

    count = len(lines)
    if count <= max_lines:
        return (True, f"'{path}' has {count} lines (limit {max_lines})")
    return (
        False,
        f"'{path}' has {count} lines, exceeds limit of {max_lines}",
    )


def _check_custom_ast(
    inv: dict[str, Any],
    modules: dict[str, ast.Module],
    repo_path: Path,
) -> tuple[bool, str]:
    check_id = inv.get("check_id", "")
    fn = CUSTOM_CHECKS.get(check_id)
    if fn is None:
        return (
            False,
            f"Custom check '{check_id}' not registered in CUSTOM_CHECKS",
        )
    try:
        passed = fn(repo_path, modules)
        return (
            passed,
            f"Custom check '{check_id}' {'passed' if passed else 'failed'}",
        )
    except Exception as exc:
        return (False, f"Custom check '{check_id}' raised: {exc}")


# ---------------------------------------------------------------------------
# Invariant dispatcher
# ---------------------------------------------------------------------------


def _evaluate_invariant(
    inv: dict[str, Any],
    modules: dict[str, ast.Module],
    repo_path: Path,
    graph: DependencyGraph,
) -> tuple[bool, str]:
    """Dispatch a single invariant dict to its checker. Returns (passed, detail)."""
    inv_type = inv.get("type", "")

    if inv_type == "required_symbol":
        return _check_required_symbol(inv, modules, repo_path)

    elif inv_type == "forbidden_symbol":
        return _check_forbidden_symbol(inv, modules, repo_path)

    elif inv_type == "required_file":
        return _check_required_file(inv, repo_path)

    elif inv_type == "forbidden_file":
        return _check_forbidden_file(inv, repo_path)

    elif inv_type == "no_edit_dirs":
        return _check_no_edit_dirs(inv, repo_path)

    elif inv_type == "no_new_files":
        return _check_no_new_files(inv, repo_path)

    elif inv_type == "no_deleted_files":
        return _check_no_deleted_files(inv, repo_path)

    elif inv_type == "module_exports":
        return _check_module_exports(inv, modules, repo_path)

    elif inv_type == "no_circular_imports":
        return _check_no_circular_imports(graph)

    elif inv_type == "max_file_count":
        return _check_max_file_count(inv, repo_path)

    elif inv_type == "max_fanout":
        return _check_max_fanout(inv, graph, repo_path)

    elif inv_type == "file_size_limit":
        return _check_file_size_limit(inv, repo_path)

    elif inv_type == "custom_ast_check":
        return _check_custom_ast(inv, modules, repo_path)

    else:
        return (False, f"Unknown invariant type: '{inv_type}'")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _serialise_results(
    results: list[tuple[dict, bool, str]],
) -> list[dict]:
    """Serialise invariant evaluation results for metrics dict."""
    return [
        {
            "type": inv.get("type", ""),
            "description": inv.get("description", inv.get("type", "")),
            "weight": float(inv.get("weight", 1.0)),
            "severity": inv.get("severity", "soft"),
            "passed": passed,
            "detail": detail,
        }
        for inv, passed, detail in results
    ]


def _build_feedback(
    results: list[dict],
    score: float,
    is_regression: bool,
    baseline_score: float,
) -> str:
    if not results:
        return "[Structure] No invariants defined for this scenario."

    total = len(results)
    passing = sum(1 for r in results if r["passed"])
    failing = [r for r in results if not r["passed"]]

    if score >= 1.0:
        return f"[Structure] All {total} invariant(s) satisfied."

    if is_regression and score < baseline_score - 1e-6:
        hard_fails = [r for r in failing if r["severity"] == "hard"]
        if hard_fails:
            labels = "; ".join(r["detail"] for r in hard_fails[:2])
            return f"[Structure] Regression: hard invariant(s) violated. " f"{labels}"

    # Show up to 3 failing invariants with details
    fail_lines: list[str] = []
    for r in failing[:3]:
        sev = "❌ HARD" if r["severity"] == "hard" else "⚠ soft"
        fail_lines.append(f"{sev}: {r['detail']}")
    suffix = f" (+{len(failing)-3} more)" if len(failing) > 3 else ""

    return (
        f"[Structure] {passing}/{total} invariant(s) passing ({score:.0%}). "
        f"Failing: {'; '.join(fail_lines)}{suffix}"
    )


# ---------------------------------------------------------------------------
# StructureGrader
# ---------------------------------------------------------------------------


class StructureGrader(BaseGrader):
    """
    Scores the agent on satisfying the structural invariants declared in
    scenario.yaml.

    Unlike other graders that measure improvement from a continuous baseline,
    StructureGrader uses a weighted checklist model:
      - Each invariant either passes (1) or fails (0)
      - Each invariant has a weight (default 1.0)
      - score = Σ(weight_i * passed_i) / Σ(weight_i)

    Hard invariants set is_regression=True if violated, regardless of the
    overall score direction. This is how the environment enforces non-negotiable
    constraints like "never edit tests/" or "old function name must not exist".

    Baseline is computed at reset() but unlike other graders its value is
    not used in the scoring formula — the score is always the absolute
    fraction of invariants passing. The baseline is stored only to detect
    regressions (did we pass more at reset than we do now?).
    """

    grader_id = "structure"

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
        Evaluate every invariant in config["invariants"] against the
        current state of the repo.

        Returns
        -------
        {
          "results"         : list[dict]  — per-invariant outcome
          "passing_count"   : int
          "failing_count"   : int
          "total_count"     : int
          "weighted_score"  : float       — checklist_score output
          "failed_labels"   : list[str]
          "hard_violations" : list[str]   — details of hard-severity fails
          "error"           : str|None
        }
        """
        repo_path = Path(repo_path)
        invariants: list[dict] = config.get("invariants", [])
        exclude_patterns: list[str] = config.get("exclude_patterns", [])

        if not invariants:
            return {
                "results": [],
                "passing_count": 0,
                "failing_count": 0,
                "total_count": 0,
                "weighted_score": 1.0,  # vacuously true
                "failed_labels": [],
                "hard_violations": [],
                "error": None,
            }

        def _evaluate() -> list[tuple[dict, bool, str]]:
            modules = parse_repository(repo_path, exclude_patterns)

            # Build dep graph for cycle + fanout checks
            graph = DependencyGraph()
            graph.build(modules, public_api=config.get("public_api", []))

            evaluated: list[tuple[dict, bool, str]] = []
            for inv in invariants:
                passed, detail = _evaluate_invariant(inv, modules, repo_path, graph)
                evaluated.append((inv, passed, detail))
            return evaluated

        try:
            evaluated = cache.get_or_compute("structure_invariants", None, _evaluate)
        except Exception as exc:
            return {
                "results": [],
                "passing_count": 0,
                "failing_count": len(invariants),
                "total_count": len(invariants),
                "weighted_score": 0.0,
                "failed_labels": [],
                "hard_violations": [],
                "error": str(exc),
            }

        # Build checklist for checklist_score()
        checks = [
            (
                inv.get("description", inv.get("type", f"inv_{i}")),
                passed,
                float(inv.get("weight", 1.0)),
            )
            for i, (inv, passed, _) in enumerate(evaluated)
        ]
        weighted_score, failed_labels = self.checklist_score(checks)

        serialised = _serialise_results(evaluated)

        hard_violations = [
            r["detail"]
            for r in serialised
            if not r["passed"] and r["severity"] == "hard"
        ]

        return {
            "results": serialised,
            "passing_count": sum(1 for _, p, _ in evaluated if p),
            "failing_count": sum(1 for _, p, _ in evaluated if not p),
            "total_count": len(evaluated),
            "weighted_score": weighted_score,
            "failed_labels": failed_labels,
            "hard_violations": hard_violations,
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
        Score structural invariant compliance.

        Scoring is absolute (not delta-based):
          score = current weighted_score

        Regression is detected by comparing to baseline weighted_score
        AND by checking for any hard-severity violations.

        solved = True iff score == 1.0 (all invariants pass).
        """
        if current.get("error"):
            return empty_grade(
                "Structure", f"invariant check failed: {current['error']}"
            )

        # No invariants declared → vacuously satisfied
        if current.get("total_count", 0) == 0:
            return already_gold("Structure", current)

        current_score: float = current.get("weighted_score", 0.0)
        baseline_score: float = baseline.get("weighted_score", 0.0)

        # Hard violations always trigger regression
        hard_violations: list[str] = current.get("hard_violations", [])
        is_regression = bool(hard_violations) or current_score < baseline_score - 1e-6

        solved = current_score >= 1.0 - 1e-9

        feedback = _build_feedback(
            results=current.get("results", []),
            score=current_score,
            is_regression=is_regression,
            baseline_score=baseline_score,
        )

        # Per-invariant sub_scores (for diagnostics)
        sub_scores: dict[str, float] = {}
        for r in current.get("results", []):
            label = r.get("description", r.get("type", "inv"))
            sub_scores[f"inv:{label}"] = 1.0 if r["passed"] else 0.0

        # Delta: which invariants changed state from baseline to current?
        b_results = {r["description"]: r["passed"] for r in baseline.get("results", [])}
        c_results = {r["description"]: r["passed"] for r in current.get("results", [])}
        newly_passing = [
            k for k in c_results if c_results[k] and not b_results.get(k, True)
        ]
        newly_failing = [
            k for k in c_results if not c_results[k] and b_results.get(k, False)
        ]

        return GradeResult(
            score=current_score,
            gold_distance=1.0 - current_score,
            raw_baseline={
                "weighted_score": baseline_score,
                "passing_count": baseline.get("passing_count", 0),
                "failing_count": baseline.get("failing_count", 0),
                "total_count": baseline.get("total_count", 0),
            },
            raw_current={
                "weighted_score": current_score,
                "passing_count": current.get("passing_count", 0),
                "failing_count": current.get("failing_count", 0),
                "total_count": current.get("total_count", 0),
                "hard_violations": len(hard_violations),
            },
            delta={
                "weighted_score": current_score - baseline_score,
                "passing_count": (
                    current.get("passing_count", 0) - baseline.get("passing_count", 0)
                ),
                "newly_passing": newly_passing,
                "newly_failing": newly_failing,
            },
            feedback=feedback,
            solved=solved,
            is_regression=is_regression,
            sub_scores={
                "baseline_score": baseline_score,
                "hard_violations": float(len(hard_violations)),
                "newly_passing": float(len(newly_passing)),
                "newly_failing": float(len(newly_failing)),
                **sub_scores,
            },
        )

    # ------------------------------------------------------------------
    # gold_standard
    # ------------------------------------------------------------------

    def gold_standard(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Gold standard: every invariant passes with full weight.
        """
        invariants = config.get("invariants", [])
        results = [
            {
                "type": inv.get("type", ""),
                "description": inv.get("description", inv.get("type", "")),
                "weight": float(inv.get("weight", 1.0)),
                "severity": inv.get("severity", "soft"),
                "passed": True,
                "detail": "gold standard",
            }
            for inv in invariants
        ]
        return {
            "results": results,
            "passing_count": len(invariants),
            "failing_count": 0,
            "total_count": len(invariants),
            "weighted_score": 1.0,
            "failed_labels": [],
            "hard_violations": [],
            "error": None,
        }
