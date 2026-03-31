"""
Production Grader — production-readiness hygiene for Python source files.

"Production-ready code" in this environment is defined by a concrete,
deterministic checklist derived from Google Python Style Guide rules and
common engineering-team standards. Every check is purely static (AST +
regex), with no subprocess calls, so this grader is always fast and
side-effect free.

Checks (10 categories, each independently weighted)
─────────────────────────────────────────────────────
  1. type_annotations  — All function parameters and return types annotated
                         (excluding __init__ return, *args/**kwargs optional)
  2. docstrings        — Public functions, classes, and modules have docstrings
  3. exception_safety  — No bare `except:`, no `except Exception:` without re-raise
  4. logging_hygiene   — No `print()` calls in non-test files; logging module used
                         when stdout output is clearly intentional
  5. no_debug_code     — No `breakpoint()`, `pdb.set_trace()`, `ipdb`, `ic()`
  6. no_todos          — No TODO/FIXME/HACK/XXX/NOQA comments (counts as tech debt)
  7. exports_defined   — Public modules define `__all__` (explicit API surface)
  8. resource_safety   — File/socket opens use `with` context managers; no naked
                         `.open()` / `.connect()` calls storing result in bare var
  9. constant_naming   — Module-level constants follow UPPER_SNAKE_CASE
 10. error_messages    — `raise` statements include a non-empty message string
                         (not bare `raise SomeError()` with no context)

Scoring
───────
  Each check produces a ratio: compliant_items / total_items ∈ [0, 1].
  (Checks with no applicable items score 1.0 — vacuously compliant.)

  per_check_score_i  = compliant / max(total, 1)

  weighted_raw       = Σ (weight_i × score_i)
  weight_sum         = Σ weight_i  (only for checks that had ≥1 applicable item)
  final_score        = weighted_raw / max(weight_sum, ε)

  Delta scoring:
    score = clamp(final_score - baseline_final_score + baseline_final_score)
            ← reduces to final_score, but baseline lets us detect regression

  Gold standard: all checks score 1.0  →  final_score = 1.0

Scenario config keys (under config["graders"]["production"]):
  weight              : float        — contribution to overall reward (default 0.10)
  checks_enabled      : list[str]    — subset of checks to run (default: all)
  check_weights       : dict[str, float] — override per-check weights
  min_public_items    : int          — min items before a check is "applicable" (default 1)
  exclude_patterns    : list[str]    — file globs to skip
  strict_annotations  : bool         — if True, *args/**kwargs must also be annotated
                                       (default False)
  require_module_doc  : bool         — score module docstrings (default True)
"""

from __future__ import annotations

import ast
import re
import tokenize
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base_grader import BaseGrader, GradeResult, _clamp, already_gold, empty_grade
from .utils import MetricCache


# ---------------------------------------------------------------------------
# Weights — tuned so that annotations + docstrings dominate (style-enforce
# is the hard task; these two checks are most impactful for LLM agents)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "type_annotations": 0.22,
    "docstrings": 0.20,
    "exception_safety": 0.14,
    "logging_hygiene": 0.12,
    "no_debug_code": 0.08,
    "no_todos": 0.07,
    "exports_defined": 0.07,
    "resource_safety": 0.05,
    "constant_naming": 0.03,
    "error_messages": 0.02,
}

_ALL_CHECKS = list(_DEFAULT_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Per-file result
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome for one check in one file."""

    check: str
    compliant: int  # items passing
    total: int  # items evaluated
    violations: list[dict] = field(default_factory=list)  # {line, msg}

    @property
    def score(self) -> float:
        return self.compliant / self.total if self.total > 0 else 1.0

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "compliant": self.compliant,
            "total": self.total,
            "score": round(self.score, 4),
            "violations": self.violations[:10],  # cap for observation size
        }


@dataclass
class FileResult:
    path: str
    checks: list[CheckResult]

    def score_for(self, check: str) -> float:
        for c in self.checks:
            if c.check == check:
                return c.score
        return 1.0  # check not applicable to this file

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODO_RE = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|NOQA)\b", re.IGNORECASE)
_UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_DUNDER_RE = re.compile(r"^__[a-z_]+__$")
_DEBUG_NAMES = frozenset({"breakpoint", "set_trace", "ipdb", "ic"})
_PRINT_NAME = "print"


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _is_test_file(rel: str) -> bool:
    from pathlib import PurePosixPath

    p = PurePosixPath(rel)
    return (
        p.name.startswith("test_")
        or p.name.endswith("_test.py")
        or "tests" in p.parts
        or "test" in p.parts
    )


def _node_lineno(node: ast.AST) -> int:
    return getattr(node, "lineno", 0)


def _has_docstring(node: ast.AST) -> bool:
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr):
        val = body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return True
    return False


def _annotation_present(annotation) -> bool:
    return annotation is not None


def _is_context_managed(node: ast.Call, parent_assign: ast.stmt | None) -> bool:
    """Return True if the call appears inside a `with` statement."""
    # We check this during the walk — we tag each Call's parent
    return False  # actual logic handled in _check_resource_safety


def _collect_comments(source: str) -> list[tuple[int, str]]:
    """Return (lineno, comment_text) for all comments in source."""
    results: list[tuple[int, str]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                results.append((tok.start[0], tok.string))
    except tokenize.TokenizeError:
        pass
    return results


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_type_annotations(
    tree: ast.Module,
    strict: bool,
) -> CheckResult:
    """
    Every public function/method must have annotated parameters and return type.
    __init__ return annotation is not required.
    *args / **kwargs are optional unless strict=True.
    """
    compliant = 0
    total = 0
    violations: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_public(node.name):
            continue

        args = node.args
        is_init = node.name == "__init__"
        lineno = node.lineno
        missing: list[str] = []

        # Return annotation (skip for __init__)
        if not is_init:
            total += 1
            if _annotation_present(node.returns):
                compliant += 1
            else:
                missing.append("return")

        # Parameters (skip self/cls)
        all_args = args.posonlyargs + args.args + args.kwonlyargs
        for arg in all_args:
            if arg.arg in ("self", "cls"):
                continue
            total += 1
            if _annotation_present(arg.annotation):
                compliant += 1
            else:
                missing.append(arg.arg)

        if strict:
            for varg in [args.vararg] if args.vararg else []:
                total += 1
                if _annotation_present(varg.annotation):
                    compliant += 1
                else:
                    missing.append(f"*{varg.arg}")
            for kwarg in [args.kwarg] if args.kwarg else []:
                total += 1
                if _annotation_present(kwarg.annotation):
                    compliant += 1
                else:
                    missing.append(f"**{kwarg.arg}")

        if missing:
            violations.append(
                {
                    "line": lineno,
                    "msg": f"fn `{node.name}` missing annotations: {', '.join(missing)}",
                }
            )

    return CheckResult("type_annotations", compliant, total, violations)


def _check_docstrings(
    tree: ast.Module,
    rel_path: str,
    require_module_doc: bool,
) -> CheckResult:
    """Public functions, classes, and (optionally) modules need docstrings."""
    compliant = 0
    total = 0
    violations: list[dict] = []

    # Module-level docstring
    if require_module_doc:
        total += 1
        if _has_docstring(tree):
            compliant += 1
        else:
            violations.append(
                {"line": 1, "msg": f"module `{rel_path}` missing module docstring"}
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not _is_public(node.name):
                continue
            total += 1
            if _has_docstring(node):
                compliant += 1
            else:
                kind = "class" if isinstance(node, ast.ClassDef) else "fn"
                violations.append(
                    {
                        "line": node.lineno,
                        "msg": f"{kind} `{node.name}` missing docstring",
                    }
                )

    return CheckResult("docstrings", compliant, total, violations)


def _check_exception_safety(tree: ast.Module) -> CheckResult:
    """
    No bare `except:`.
    No `except Exception:` (or `except BaseException:`) unless the body
    re-raises (`raise` with no argument) or explicitly logs + re-raises.
    """
    compliant = 0
    total = 0
    violations: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        total += 1

        # Bare except:
        if node.type is None:
            violations.append(
                {"line": node.lineno, "msg": "bare `except:` — specify exception type"}
            )
            continue

        # except Exception / except BaseException
        broad_names = {"Exception", "BaseException"}
        if isinstance(node.type, ast.Name) and node.type.id in broad_names:
            # Check body for a bare `raise`
            has_reraise = any(
                isinstance(n, ast.Raise) and n.exc is None
                for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            )
            if not has_reraise:
                violations.append(
                    {
                        "line": node.lineno,
                        "msg": f"`except {node.type.id}:` without re-raise — swallows errors",
                    }
                )
                continue

        compliant += 1

    return CheckResult("exception_safety", compliant, total, violations)


def _check_logging_hygiene(
    tree: ast.Module,
    is_test: bool,
) -> CheckResult:
    """
    Non-test files should not use `print()` for runtime output.
    Each `print()` call is one violation item.
    """
    if is_test:
        # print() in test files is acceptable (pytest captures it)
        return CheckResult("logging_hygiene", 1, 1, [])

    compliant = 0
    total = 0
    violations: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr

        if name == _PRINT_NAME:
            total += 1
            violations.append(
                {
                    "line": node.lineno,
                    "msg": "`print()` in production code — use `logging` module",
                }
            )

    # If no print calls: single passing item so score = 1.0
    if total == 0:
        return CheckResult("logging_hygiene", 1, 1, [])

    return CheckResult("logging_hygiene", compliant, total, violations)


def _check_no_debug_code(tree: ast.Module) -> CheckResult:
    """
    No `breakpoint()`, `pdb.set_trace()`, `ipdb.*`, or `ic()` calls.
    """
    compliant = 1
    total = 1
    violations: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None

        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            # pdb.set_trace, ipdb.set_trace, etc.
            if isinstance(func.value, ast.Name):
                if func.value.id in {"pdb", "ipdb"}:
                    name = func.value.id
            name = name or func.attr

        if name and name in _DEBUG_NAMES:
            total = total or 1
            compliant = 0
            violations.append(
                {
                    "line": node.lineno,
                    "msg": f"debug call `{name}` must be removed",
                }
            )

    # Recompute: total = number of violations found + 1 "baseline passing" item
    if violations:
        return CheckResult("no_debug_code", 0, len(violations), violations)
    return CheckResult("no_debug_code", 1, 1, [])


def _check_no_todos(comments: list[tuple[int, str]]) -> CheckResult:
    """No TODO/FIXME/HACK/XXX/NOQA comments."""
    total = max(len(comments), 1)
    violations: list[dict] = []

    for lineno, text in comments:
        if _TODO_RE.search(text):
            violations.append(
                {"line": lineno, "msg": f"unresolved marker: {text.strip()}"}
            )

    compliant = total - len(violations)
    return CheckResult("no_todos", max(compliant, 0), total, violations)


def _check_exports_defined(
    tree: ast.Module,
    rel_path: str,
) -> CheckResult:
    """
    Modules with at least one public name should define `__all__`.
    Score: 1 if `__all__` present, 0 otherwise.
    """
    public_names = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and _is_public(n.name)
    ]
    if not public_names:
        return CheckResult("exports_defined", 1, 1, [])

    has_all = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
        for n in ast.walk(tree)
    )

    if has_all:
        return CheckResult("exports_defined", 1, 1, [])

    return CheckResult(
        "exports_defined",
        0,
        1,
        [
            {
                "line": 1,
                "msg": f"`{rel_path}` has public names but no `__all__`",
            }
        ],
    )


def _check_resource_safety(tree: ast.Module) -> CheckResult:
    """
    File/socket opens not inside a `with` block are flagged.

    Detection: find all `ast.Call` where the function is `open` or
    `socket.socket` (and common aliases), then check whether the
    immediate parent is a `ast.withitem` context expression.

    We track parent relationships via a single pre-pass.
    """
    unsafe_funcs = {"open", "socket", "connect", "urlopen", "urlretrieve"}

    # Build parent map
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    compliant = 0
    total = 0
    violations: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr

        if not name or name not in unsafe_funcs:
            continue

        total += 1

        # Walk up two levels: Call → value of withitem → With
        p1 = parent.get(id(node))
        p2 = parent.get(id(p1)) if p1 else None

        in_with = (
            isinstance(p1, ast.withitem)
            or isinstance(p2, ast.withitem)
            or isinstance(p1, ast.With)
            or isinstance(p2, ast.With)
        )

        if in_with:
            compliant += 1
        else:
            violations.append(
                {
                    "line": node.lineno,
                    "msg": f"`{name}()` not wrapped in `with` — resource leak risk",
                }
            )

    if total == 0:
        return CheckResult("resource_safety", 1, 1, [])
    return CheckResult("resource_safety", compliant, total, violations)


def _check_constant_naming(tree: ast.Module) -> CheckResult:
    """
    Module-level constants (non-dunder assignments to a single Name target,
    not inside a class/function) should be UPPER_SNAKE_CASE.
    """
    compliant = 0
    total = 0
    violations: list[dict] = []

    for node in tree.body:  # top-level only
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.target:
            targets = [node.target]

        for t in targets:
            if not isinstance(t, ast.Name):
                continue
            name = t.id
            if _DUNDER_RE.match(name):
                continue  # __version__ etc. are exempt
            if name.startswith("_"):
                continue  # private constants are exempt

            # Skip if the value is a function/class (it's not a constant)
            val = getattr(node, "value", None)
            if isinstance(
                val, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue

            total += 1
            if _UPPER_SNAKE.match(name):
                compliant += 1
            else:
                violations.append(
                    {
                        "line": node.lineno,
                        "msg": f"constant `{name}` should be UPPER_SNAKE_CASE",
                    }
                )

    if total == 0:
        return CheckResult("constant_naming", 1, 1, [])
    return CheckResult("constant_naming", compliant, total, violations)


def _check_error_messages(tree: ast.Module) -> CheckResult:
    """
    `raise SomeError()` must include a non-empty message.
    `raise SomeError` (no call) or `raise SomeError("")` are violations.
    Bare `raise` (re-raise) is fine.
    """
    compliant = 0
    total = 0
    violations: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        if node.exc is None:
            continue  # bare re-raise — exempt

        total += 1
        exc = node.exc

        # raise SomeError  (no call)
        if isinstance(exc, ast.Name):
            violations.append(
                {
                    "line": node.lineno,
                    "msg": f"`raise {exc.id}` without message — add context string",
                }
            )
            continue

        # raise SomeError(...)
        if isinstance(exc, ast.Call):
            args = exc.args
            if not args:
                name = ""
                if isinstance(exc.func, ast.Name):
                    name = exc.func.id
                elif isinstance(exc.func, ast.Attribute):
                    name = exc.func.attr
                violations.append(
                    {
                        "line": node.lineno,
                        "msg": f"`raise {name}()` with no message — add context string",
                    }
                )
                continue

            # First arg must be a non-empty string constant (or not a constant — OK)
            first = args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if not first.value.strip():
                    violations.append(
                        {
                            "line": node.lineno,
                            "msg": "raise with empty string message — add context",
                        }
                    )
                    continue

        compliant += 1

    if total == 0:
        return CheckResult("error_messages", 1, 1, [])
    return CheckResult("error_messages", compliant, total, violations)


# ---------------------------------------------------------------------------
# Per-file runner
# ---------------------------------------------------------------------------


def _analyse_file(
    repo_path: Path,
    fpath: Path,
    checks_enabled: set[str],
    strict_annotations: bool,
    require_module_doc: bool,
) -> FileResult | None:
    rel_path = str(fpath.relative_to(repo_path))
    try:
        source = fpath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        # Unparseable file: score all enabled checks as 0 (1 violation each)
        results = [
            CheckResult(
                c, 0, 1, [{"line": 1, "msg": "SyntaxError — file not parseable"}]
            )
            for c in checks_enabled
        ]
        return FileResult(rel_path, results)

    is_test = _is_test_file(rel_path)
    comments = _collect_comments(source)
    check_list: list[CheckResult] = []

    dispatch: dict[str, CheckResult] = {}

    if "type_annotations" in checks_enabled:
        dispatch["type_annotations"] = _check_type_annotations(tree, strict_annotations)
    if "docstrings" in checks_enabled:
        dispatch["docstrings"] = _check_docstrings(tree, rel_path, require_module_doc)
    if "exception_safety" in checks_enabled:
        dispatch["exception_safety"] = _check_exception_safety(tree)
    if "logging_hygiene" in checks_enabled:
        dispatch["logging_hygiene"] = _check_logging_hygiene(tree, is_test)
    if "no_debug_code" in checks_enabled:
        dispatch["no_debug_code"] = _check_no_debug_code(tree)
    if "no_todos" in checks_enabled:
        dispatch["no_todos"] = _check_no_todos(comments)
    if "exports_defined" in checks_enabled:
        dispatch["exports_defined"] = _check_exports_defined(tree, rel_path)
    if "resource_safety" in checks_enabled:
        dispatch["resource_safety"] = _check_resource_safety(tree)
    if "constant_naming" in checks_enabled:
        dispatch["constant_naming"] = _check_constant_naming(tree)
    if "error_messages" in checks_enabled:
        dispatch["error_messages"] = _check_error_messages(tree)

    return FileResult(rel_path, list(dispatch.values()))


# ---------------------------------------------------------------------------
# Aggregate across files
# ---------------------------------------------------------------------------


def _aggregate(
    file_results: list[FileResult],
    checks_enabled: set[str],
    weights: dict[str, float],
    min_items: int,
) -> tuple[float, dict[str, dict]]:
    """
    Returns (final_score, per_check_stats).

    per_check_stats[check] = {
        "score": float,
        "compliant": int,
        "total": int,
        "applicable": bool,
        "weight_used": float,
        "violations": list[dict],   # up to 20 across all files
    }
    """
    # Accumulate across files
    agg: dict[str, dict] = {
        c: {"compliant": 0, "total": 0, "violations": []} for c in checks_enabled
    }

    for fr in file_results:
        for cr in fr.checks:
            if cr.check not in agg:
                continue
            agg[cr.check]["compliant"] += cr.compliant
            agg[cr.check]["total"] += cr.total
            agg[cr.check]["violations"].extend(cr.violations)

    # Compute per-check score + weighted final
    weighted_sum = 0.0
    weight_sum = 0.0

    for check, data in agg.items():
        total = data["total"]
        comp = data["compliant"]
        score = comp / total if total > 0 else 1.0
        applicable = total >= min_items

        data["score"] = round(score, 4)
        data["applicable"] = applicable
        data["violations"] = data["violations"][:20]

        w = weights.get(check, 0.0)
        data["weight_used"] = w

        if applicable:
            weighted_sum += w * score
            weight_sum += w

    final = weighted_sum / max(weight_sum, 1e-9)
    return _clamp(final), agg


# ---------------------------------------------------------------------------
# ProductionGrader
# ---------------------------------------------------------------------------


class ProductionGrader(BaseGrader):
    """
    Scores production-readiness of Python source files using static analysis.

    Runs 10 deterministic AST-based checks, aggregates per-check scores
    with configurable weights, and computes a delta-proportional reward
    relative to the episode baseline.

    This is the primary grader for the `style-enforce` scenario (hard task),
    but is also enabled at lower weights for `lint-cleanup` and `api-rename`.
    """

    grader_id = "production"

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
        Run all enabled checks across all non-excluded .py files.

        Returns
        -------
        {
          "final_score"    : float
          "per_check"      : dict[str, dict]  — per-check aggregated stats
          "file_results"   : list[dict]        — per-file breakdown
          "files_checked"  : int
          "error"          : str | None
        }
        """
        repo_path = Path(repo_path)
        prod_cfg = config.get("graders", {}).get("production", {})

        checks_enabled_list: list[str] = prod_cfg.get("checks_enabled", _ALL_CHECKS)
        checks_enabled = set(checks_enabled_list) & set(_ALL_CHECKS)

        weights: dict[str, float] = {
            **_DEFAULT_WEIGHTS,
            **prod_cfg.get("check_weights", {}),
        }
        strict_annotations: bool = bool(prod_cfg.get("strict_annotations", False))
        require_module_doc: bool = bool(prod_cfg.get("require_module_doc", True))
        min_items: int = int(prod_cfg.get("min_public_items", 1))
        exclude_patterns: list[str] = prod_cfg.get("exclude_patterns", []) + config.get(
            "exclude_patterns", []
        )

        def _run() -> tuple[list[FileResult], int]:
            file_results: list[FileResult] = []
            files_checked = 0

            for fpath in sorted(repo_path.rglob("*.py")):
                rel_path = str(fpath.relative_to(repo_path))
                if any(
                    fpath.match(p) or rel_path.startswith(p.rstrip("/"))
                    for p in exclude_patterns
                ):
                    continue

                result = _analyse_file(
                    repo_path,
                    fpath,
                    checks_enabled,
                    strict_annotations,
                    require_module_doc,
                )
                if result is not None:
                    file_results.append(result)
                    files_checked += 1

            return file_results, files_checked

        try:
            file_results, files_checked = cache.get_or_compute(
                "production_analysis", None, _run
            )
        except Exception as exc:
            return {
                "final_score": 0.0,
                "per_check": {},
                "file_results": [],
                "files_checked": 0,
                "error": str(exc),
            }

        final_score, per_check = _aggregate(
            file_results, checks_enabled, weights, min_items
        )

        return {
            "final_score": final_score,
            "per_check": per_check,
            "file_results": [fr.to_dict() for fr in file_results],
            "files_checked": files_checked,
            "error": None,
        }

    # ------------------------------------------------------------------
    # grade
    # ------------------------------------------------------------------

    def grade(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
        config: dict[str, Any],
    ) -> GradeResult:
        if current.get("error"):
            return empty_grade(
                "Production", f"static analysis failed: {current['error']}"
            )
        if current.get("files_checked", 0) == 0:
            return empty_grade("Production", "no Python files found to analyse")

        b_score = float(baseline.get("final_score", 0.0))
        c_score = float(current.get("final_score", 0.0))

        if b_score >= 1.0:
            return already_gold("Production", current)

        is_regression = c_score < b_score - 1e-6
        solved = c_score >= 1.0

        # Final score: absolute (the score IS the reward component)
        # Regression → 0 to strongly discourage removal of annotations/docs
        final_score = 0.0 if is_regression else _clamp(c_score)

        # Sub-scores per check
        b_checks = baseline.get("per_check", {})
        c_checks = current.get("per_check", {})

        sub_scores: dict[str, float] = {}
        for check in _ALL_CHECKS:
            b_c = float(b_checks.get(check, {}).get("score", 1.0))
            c_c = float(c_checks.get(check, {}).get("score", 1.0))
            sub_scores[check] = c_c
            sub_scores[f"delta:{check}"] = round(c_c - b_c, 4)

        # Build human feedback
        feedback = _build_feedback(c_score, b_score, is_regression, c_checks)

        delta_score = c_score - b_score

        return GradeResult(
            score=final_score,
            gold_distance=1.0 - c_score,
            raw_baseline={
                "final_score": b_score,
                "per_check": {k: v.get("score", 0) for k, v in b_checks.items()},
            },
            raw_current={
                "final_score": c_score,
                "files_checked": current.get("files_checked", 0),
                "per_check": {k: v.get("score", 0) for k, v in c_checks.items()},
            },
            delta={
                "score": round(delta_score, 4),
                "per_check": {
                    k: round(
                        float(c_checks.get(k, {}).get("score", 1.0))
                        - float(b_checks.get(k, {}).get("score", 1.0)),
                        4,
                    )
                    for k in _ALL_CHECKS
                },
            },
            feedback=feedback,
            solved=solved,
            is_regression=is_regression,
            sub_scores=sub_scores,
        )

    # ------------------------------------------------------------------
    # gold_standard
    # ------------------------------------------------------------------

    def gold_standard(self, config: dict[str, Any]) -> dict[str, Any]:
        perfect_check = {
            "score": 1.0,
            "compliant": 1,
            "total": 1,
            "applicable": True,
            "weight_used": 0.1,
            "violations": [],
        }
        return {
            "final_score": 1.0,
            "per_check": {c: dict(perfect_check) for c in _ALL_CHECKS},
            "file_results": [],
            "files_checked": 1,
            "error": None,
        }


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


def _build_feedback(
    c_score: float,
    b_score: float,
    is_regression: bool,
    c_checks: dict[str, dict],
) -> str:
    if c_score >= 1.0:
        return "[Production] Gold standard — all production checks pass."

    if is_regression:
        return (
            f"[Production] Regression detected: score dropped from "
            f"{b_score:.0%} to {c_score:.0%}. "
            "Review recent edits — annotations or docstrings may have been removed."
        )

    # Find the two worst checks for actionable guidance
    ranked = sorted(
        [(check, data) for check, data in c_checks.items() if data.get("applicable")],
        key=lambda x: x[1].get("score", 1.0),
    )

    if not ranked:
        return f"[Production] Score: {c_score:.0%}."

    parts: list[str] = []
    for check, data in ranked[:2]:
        score = data.get("score", 1.0)
        if score >= 1.0:
            continue
        viols = data.get("violations", [])
        example = f" (e.g. L{viols[0]['line']}: {viols[0]['msg']})" if viols else ""
        parts.append(f"`{check}` {score:.0%}{example}")

    worst_str = "; ".join(parts) if parts else "all checks near passing"

    return (
        f"[Production] {c_score:.0%} production score "
        f"(Δ {c_score - b_score:+.0%} vs baseline). "
        f"Lowest: {worst_str}."
    )
