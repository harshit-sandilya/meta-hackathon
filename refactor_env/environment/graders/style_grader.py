"""
Style Grader — Google Python Style Guide compliance.

Reference specs:
  https://google.github.io/styleguide/pyguide.html
  https://google.github.io/eng-practices/review/reviewer/looking-for.html

Gold standard : zero style violations across all non-test .py files.

This grader is pure-AST + regex; it intentionally does NOT invoke ruff or
pylint at runtime so it stays fast and fully deterministic inside the sandbox.
It encodes the rules that matter most for the refactoring tasks — naming,
docstrings, imports, type annotations, complexity, and formatting signals —
and produces a weighted score across six sub-dimensions.

Sub-dimensions (weighted sum → final score)
────────────────────────────────────────────
  1. naming        (weight 0.30) — PEP 8 + Google naming conventions
  2. docstrings    (weight 0.25) — public symbol docstring coverage
  3. imports       (weight 0.20) — import ordering & style rules
  4. type_hints    (weight 0.10) — public function annotation coverage
  5. complexity    (weight 0.10) — cyclomatic complexity, line length
  6. formatting    (weight 0.05) — blank lines, trailing whitespace

Each sub-dimension produces a score in [0, 1] representing the fraction of
compliant items. The weighted sum is the final score.

Scenario config keys (under config["graders"]["style"]):
  weight          : float       — contribution to overall reward (default 0.10)
  sub_weights     : dict        — per-dimension weight overrides
  exclude_patterns: list[str]   — file globs to skip
  ignore_rules    : list[str]   — rule codes to skip, e.g. ["N002", "D001"]
  max_line_length : int         — line length limit (default 80, Google spec)
  min_complexity  : int         — cyclomatic complexity threshold (default 10)
  strict_docstrings: bool       — if True, also require private func docstrings
  strict_type_hints: bool       — if True, also require private func annotations

Rule code taxonomy
──────────────────
  N = Naming      N001–N010
  D = Docstrings  D001–D006
  I = Imports     I001–I006
  T = Type hints  T001–T003
  C = Complexity  C001–C003
  F = Formatting  F001–F004
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
from .utils import MetricCache, parse_repository, parse_file_safe


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StyleRule:
    code: str
    description: str
    dimension: (
        str  # naming | docstrings | imports | type_hints | complexity | formatting
    )
    severity: str  # error | warning


_RULES: dict[str, StyleRule] = {
    # ── Naming ─────────────────────────────────────────────────────────────
    "N001": StyleRule(
        "N001", "Module name must be lowercase_underscore", "naming", "error"
    ),
    "N002": StyleRule("N002", "Class name must be CapWords", "naming", "error"),
    "N003": StyleRule(
        "N003", "Function/method name must be lowercase_underscore", "naming", "error"
    ),
    "N004": StyleRule(
        "N004", "Constant must be UPPER_CASE (module-level)", "naming", "warning"
    ),
    "N005": StyleRule(
        "N005", "Variable must be lowercase_underscore", "naming", "warning"
    ),
    "N006": StyleRule(
        "N006", "Private name must use single leading underscore", "naming", "warning"
    ),
    "N007": StyleRule(
        "N007", "Avoid single-letter names except loop counters", "naming", "warning"
    ),
    "N008": StyleRule(
        "N008", "TypeVar name must be CapWords or single letter", "naming", "warning"
    ),
    "N009": StyleRule("N009", "Exception class must end in 'Error'", "naming", "error"),
    "N010": StyleRule(
        "N010", "Test function must start with 'test_'", "naming", "warning"
    ),
    # ── Docstrings ─────────────────────────────────────────────────────────
    "D001": StyleRule(
        "D001", "Public function/method missing docstring", "docstrings", "error"
    ),
    "D002": StyleRule("D002", "Public class missing docstring", "docstrings", "error"),
    "D003": StyleRule(
        "D003", "Module missing module-level docstring", "docstrings", "warning"
    ),
    "D004": StyleRule(
        "D004", "Docstring summary line exceeds 80 characters", "docstrings", "warning"
    ),
    "D005": StyleRule(
        "D005",
        "Multi-line docstring missing blank line after summary",
        "docstrings",
        "warning",
    ),
    "D006": StyleRule(
        "D006",
        "Generator function should use 'Yields:' not 'Returns:'",
        "docstrings",
        "warning",
    ),
    # ── Imports ────────────────────────────────────────────────────────────
    "I001": StyleRule(
        "I001", "Wildcard import (from x import *) forbidden", "imports", "error"
    ),
    "I002": StyleRule(
        "I002", "Relative import forbidden (use full package path)", "imports", "error"
    ),
    "I003": StyleRule(
        "I003", "Multiple imports on one line (import a, b)", "imports", "warning"
    ),
    "I004": StyleRule(
        "I004", "Import of individual class/function (use module)", "imports", "warning"
    ),
    "I005": StyleRule(
        "I005", "Imports not grouped: stdlib, third-party, local", "imports", "warning"
    ),
    "I006": StyleRule(
        "I006", "Mutable default argument in function signature", "imports", "error"
    ),  # grouped here as a "language rule"
    # ── Type hints ─────────────────────────────────────────────────────────
    "T001": StyleRule(
        "T001",
        "Public function missing return type annotation",
        "type_hints",
        "warning",
    ),
    "T002": StyleRule(
        "T002",
        "Public function missing parameter type annotation",
        "type_hints",
        "warning",
    ),
    "T003": StyleRule(
        "T003", "Bare 'except:' clause (catch-all) forbidden", "type_hints", "error"
    ),  # language rule, fits here
    # ── Complexity ─────────────────────────────────────────────────────────
    "C001": StyleRule(
        "C001",
        "Function exceeds cyclomatic complexity threshold",
        "complexity",
        "warning",
    ),
    "C002": StyleRule("C002", "Line exceeds maximum length", "complexity", "warning"),
    "C003": StyleRule(
        "C003",
        "Nested comprehension with multiple for-clauses",
        "complexity",
        "warning",
    ),
    # ── Formatting ─────────────────────────────────────────────────────────
    "F001": StyleRule("F001", "Trailing whitespace on line", "formatting", "warning"),
    "F002": StyleRule(
        "F002",
        "Missing two blank lines before top-level definition",
        "formatting",
        "warning",
    ),
    "F003": StyleRule(
        "F003", "Semicolon used to separate statements", "formatting", "error"
    ),
    "F004": StyleRule(
        "F004", "Tab character used for indentation", "formatting", "error"
    ),
}

_DIMENSION_ORDER = [
    "naming",
    "docstrings",
    "imports",
    "type_hints",
    "complexity",
    "formatting",
]

_DEFAULT_SUB_WEIGHTS: dict[str, float] = {
    "naming": 0.30,
    "docstrings": 0.25,
    "imports": 0.20,
    "type_hints": 0.10,
    "complexity": 0.10,
    "formatting": 0.05,
}

# Stdlib module names for import grouping check (top-level only)
_STDLIB_TOP = frozenset(
    {
        "abc",
        "ast",
        "asyncio",
        "builtins",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "datetime",
        "enum",
        "functools",
        "gc",
        "hashlib",
        "heapq",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "queue",
        "random",
        "re",
        "shutil",
        "signal",
        "socket",
        "sqlite3",
        "string",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "timeit",
        "traceback",
        "types",
        "typing",
        "unittest",
        "urllib",
        "uuid",
        "warnings",
        "weakref",
        "collections.abc",
        "typing_extensions",
    }
)


# ---------------------------------------------------------------------------
# Violation dataclass
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    rule: str
    file: str
    line: int
    context: str = ""  # snippet for feedback

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "dimension": _RULES[self.rule].dimension,
            "severity": _RULES[self.rule].severity,
            "message": _RULES[self.rule].description,
            "file": self.file,
            "line": self.line,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------


class _StyleVisitor(ast.NodeVisitor):
    """
    Single-pass AST visitor that emits Violation objects for a single file.
    Stateful: call visit(module), then read self.violations.
    """

    def __init__(
        self,
        rel_path: str,
        source_lines: list[str],
        max_line_length: int,
        min_complexity: int,
        strict_docstrings: bool,
        strict_type_hints: bool,
        ignore_rules: set[str],
    ) -> None:
        self.rel_path = rel_path
        self.source_lines = source_lines
        self.max_line_length = max_line_length
        self.min_complexity = min_complexity
        self.strict_docstrings = strict_docstrings
        self.strict_type_hints = strict_type_hints
        self.ignore_rules = ignore_rules
        self.violations: list[Violation] = []
        self._scope_stack: list[str] = []  # track class/function nesting

    # ── helpers ──────────────────────────────────────────────────────────

    def _v(self, rule: str, node: ast.AST, context: str = "") -> None:
        if rule in self.ignore_rules:
            return
        self.violations.append(
            Violation(rule, self.rel_path, getattr(node, "lineno", 0), context)
        )

    def _src(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].rstrip()
        return ""

    def _is_public(self, name: str) -> bool:
        return not name.startswith("_")

    def _is_snake(self, name: str) -> bool:
        return bool(re.match(r"^[a-z][a-z0-9_]*$", name)) or name == "_"

    def _is_upper(self, name: str) -> bool:
        return bool(re.match(r"^[A-Z][A-Z0-9_]*$", name))

    def _is_capwords(self, name: str) -> bool:
        return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", name))

    def _get_docstring(self, node: ast.AST) -> str | None:
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            return None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value
        return None

    def _cyclomatic(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Approximate cyclomatic complexity: 1 + branching nodes."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.While,
                    ast.For,
                    ast.AsyncFor,
                    ast.ExceptHandler,
                    ast.With,
                    ast.AsyncWith,
                    ast.Assert,
                    ast.comprehension,
                ),
            ):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _has_nested_comprehension(self, node: ast.AST) -> bool:
        """Check for multiple for-clauses in a single comprehension."""
        for child in ast.walk(node):
            if isinstance(
                child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                if hasattr(child, "generators") and len(child.generators) > 1:
                    return True
        return False

    # ── module ────────────────────────────────────────────────────────────

    def visit_Module(self, node: ast.Module) -> None:
        # N001: module name
        stem = Path(self.rel_path).stem
        if not self._is_snake(stem) and stem not in ("__init__", "__main__"):
            self._v("N001", node, stem)

        # D003: module docstring
        if self._get_docstring(node) is None:
            self._v("D003", node)

        # I005: import grouping (stdlib → third-party → local)
        self._check_import_grouping(node)

        self.generic_visit(node)

    # ── imports ────────────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        # I003: multiple imports on one line
        if len(node.names) > 1:
            self._v("I003", node, ", ".join(a.name for a in node.names))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # I001: wildcard import
        if any(a.name == "*" for a in node.names):
            self._v("I001", node, f"from {node.module} import *")

        # I002: relative import
        if node.level and node.level > 0:
            self._v("I002", node, f"relative level={node.level}")

        # I004: importing individual class/function (heuristic: all-caps or CapWords name)
        _typing_modules = {
            "typing",
            "collections.abc",
            "typing_extensions",
            "__future__",
        }
        if node.module and node.module not in _typing_modules:
            for alias in node.names:
                name = alias.name
                if self._is_capwords(name) and not name.isupper():
                    self._v("I004", node, f"from {node.module} import {name}")

    def _check_import_grouping(self, module: ast.Module) -> None:
        """
        Imports must be ordered: stdlib → third-party → local.
        A group transition is detected when a non-stdlib import appears before
        a stdlib import with no blank line separating them.
        """
        groups: list[tuple[str, int]] = []  # (group, lineno)

        for node in module.body:
            if isinstance(node, ast.Import):
                top = node.names[0].name.split(".")[0]
                groups.append(
                    ("stdlib" if top in _STDLIB_TOP else "third_party", node.lineno)
                )
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    groups.append(("local", node.lineno))
                else:
                    top = (node.module or "").split(".")[0]
                    groups.append(
                        ("stdlib" if top in _STDLIB_TOP else "third_party", node.lineno)
                    )

        # Detect order violations: stdlib after third_party/local
        order_map = {"stdlib": 0, "third_party": 1, "local": 2}
        seen_max = 0
        for group, lineno in groups:
            gval = order_map.get(group, 1)
            if gval < seen_max:
                fake = ast.AST()
                fake.lineno = lineno  # type: ignore[attr-defined]
                self._v("I005", fake)
                break
            seen_max = max(seen_max, gval)

    # ── classes ────────────────────────────────────────────────────────────

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # N002: class name CapWords
        if not self._is_capwords(node.name):
            self._v("N002", node, node.name)

        # N009: exception class must end in 'Error'
        base_names = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
        is_exception = any(
            "Exception" in b or "Error" in b or b == "BaseException" for b in base_names
        )
        if is_exception and not node.name.endswith("Error"):
            self._v("N009", node, node.name)

        # D002: public class docstring
        if self._is_public(node.name) and self._get_docstring(node) is None:
            self._v("D002", node, node.name)

        self._scope_stack.append(f"class:{node.name}")
        self.generic_visit(node)
        self._scope_stack.pop()

    # ── functions / methods ────────────────────────────────────────────────

    def _visit_funcdef(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        name = node.name
        is_public = self._is_public(name)
        in_class = any(s.startswith("class:") for s in self._scope_stack)
        is_async = isinstance(node, ast.AsyncFunctionDef)

        # N003: function/method name snake_case
        if not self._is_snake(name) and name not in (
            "__init__",
            "__new__",
            "__repr__",
            "__str__",
            "__len__",
            "__eq__",
            "__hash__",
            "__call__",
            "__enter__",
            "__exit__",
            "__iter__",
            "__next__",
            "__contains__",
            "__getitem__",
            "__setitem__",
            "__delitem__",
            "__aenter__",
            "__aexit__",
            "__await__",
            "__aiter__",
            "__anext__",
        ):
            self._v("N003", node, name)

        # N007: single-letter names (except common loop counters)
        _ok_single = {"i", "j", "k", "n", "x", "y", "z", "v", "e", "f", "k"}
        if len(name) == 1 and name not in _ok_single and name != "_":
            self._v("N007", node, name)

        # D001: public function/method docstring
        needs_doc = is_public or self.strict_docstrings
        if needs_doc and self._get_docstring(node) is None:
            self._v("D001", node, name)

        # D004: docstring summary line length
        docstr = self._get_docstring(node)
        if docstr:
            summary = docstr.strip().splitlines()[0]
            if len(summary) > self.max_line_length:
                self._v("D004", node, summary[:60] + "…")

        # D005: multi-line docstring blank line after summary
        if docstr and "\n" in docstr:
            lines = docstr.strip().splitlines()
            if len(lines) > 1 and lines[1].strip() != "":
                self._v("D005", node, name)

        # D006: generator with 'Returns:' in docstring
        if is_async is False and docstr:
            has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))
            if has_yield and "Returns:" in docstr and "Yields:" not in docstr:
                self._v("D006", node, name)

        # T001: public function return type annotation
        if (is_public or self.strict_type_hints) and node.returns is None:
            self._v("T001", node, name)

        # T002: public function parameter type annotations
        if is_public or self.strict_type_hints:
            unannotated = [
                arg.arg
                for arg in (
                    node.args.args + node.args.posonlyargs + node.args.kwonlyargs
                )
                if arg.annotation is None and arg.arg not in ("self", "cls")
            ]
            if node.args.vararg and node.args.vararg.annotation is None:
                unannotated.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg and node.args.kwarg.annotation is None:
                unannotated.append(f"**{node.args.kwarg.arg}")
            if unannotated:
                self._v("T002", node, f"{name}({', '.join(unannotated[:3])})")

        # T003: bare except clause
        for child in ast.walk(node):
            if isinstance(child, ast.ExceptHandler) and child.type is None:
                self._v("T003", child, name)

        # I006: mutable default argument
        _mutable_nodes = (ast.List, ast.Dict, ast.Set, ast.Call)
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None and isinstance(default, _mutable_nodes):
                if isinstance(default, ast.Call):
                    func_name = getattr(
                        default.func, "id", getattr(default.func, "attr", "")
                    )
                    if func_name.lower() in ("list", "dict", "set", "defaultdict"):
                        self._v("I006", node, name)
                else:
                    self._v("I006", node, name)

        # C001: cyclomatic complexity
        cc = self._cyclomatic(node)
        if cc > self.min_complexity:
            self._v("C001", node, f"{name} cc={cc}")

        # C003: nested comprehension
        if self._has_nested_comprehension(node):
            self._v("C003", node, name)

        self._scope_stack.append(f"func:{name}")
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_funcdef(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_funcdef(node)

    # ── module-level assignments (constants / variables) ──────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only check module-level assignments
        if self._scope_stack:
            return

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name.startswith("__") and name.endswith("__"):
                continue  # dunder globals are fine
            if isinstance(node.value, ast.Constant):
                # Module-level constant: should be UPPER_CASE
                if not self._is_upper(name) and not name.startswith("_"):
                    self._v("N004", node, name)
            # N005 intentionally lenient — only flag obviously wrong names
            # (e.g. CamelCase at module scope that's not a class)
            if self._is_capwords(name) and not name.isupper():
                self._v("N005", node, name)

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Source-level checks (line-by-line, no AST)
# ---------------------------------------------------------------------------


def _check_source_lines(
    rel_path: str,
    source_lines: list[str],
    max_line_length: int,
    ignore_rules: set[str],
) -> list[Violation]:
    violations: list[Violation] = []

    _blank_before_required = 0  # countdown for blank-line check
    prev_nonblank_was_toplevel = False

    for lineno, raw in enumerate(source_lines, start=1):
        line = raw.rstrip("\n")

        # F001: trailing whitespace
        if "F001" not in ignore_rules and line != line.rstrip():
            violations.append(Violation("F001", rel_path, lineno, repr(line[-10:])))

        # F004: tab indentation
        if "F004" not in ignore_rules and line.startswith("\t"):
            violations.append(Violation("F004", rel_path, lineno))

        # F003: semicolons separating statements (not in strings heuristic)
        if "F003" not in ignore_rules:
            stripped = line.lstrip()
            if ";" in stripped and not stripped.startswith("#"):
                # Quick heuristic: semicolon outside a string
                in_str = False
                quote = None
                for ch in stripped:
                    if ch in ('"', "'") and not in_str:
                        in_str = True
                        quote = ch
                    elif in_str and ch == quote:
                        in_str = False
                    elif ch == ";" and not in_str:
                        violations.append(
                            Violation("F003", rel_path, lineno, stripped[:40])
                        )
                        break

        # C002: line length
        if "C002" not in ignore_rules:
            if len(line) > max_line_length:
                # Exempt: long URLs, pylint directives
                stripped_c = line.strip()
                is_url_line = (
                    stripped_c.startswith("#")
                    and "http" in stripped_c
                    and " " not in stripped_c.lstrip("# ")
                )
                is_pylint_dir = "pylint:" in line
                is_long_str = not stripped_c.startswith("#") and " " not in stripped_c
                if not (is_url_line or is_pylint_dir or is_long_str):
                    violations.append(
                        Violation("C002", rel_path, lineno, f"len={len(line)}")
                    )

    return violations


def _check_blank_lines(
    rel_path: str,
    source_lines: list[str],
    ignore_rules: set[str],
) -> list[Violation]:
    """
    F002: top-level function/class definitions must be preceded by two blank lines.
    """
    if "F002" in ignore_rules:
        return []

    violations: list[Violation] = []
    _toplevel_re = re.compile(r"^(def |class |async def )")

    blank_count = 0
    for lineno, raw in enumerate(source_lines, start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped == "":
            blank_count += 1
            continue

        if _toplevel_re.match(line) and lineno > 1:
            if blank_count < 2:
                violations.append(Violation("F002", rel_path, lineno, stripped[:40]))

        blank_count = 0

    return violations


# ---------------------------------------------------------------------------
# Per-file entry point
# ---------------------------------------------------------------------------


def _check_file(
    rel_path: str,
    source: str,
    config_style: dict[str, Any],
    ignore_rules: set[str],
) -> list[Violation]:
    max_line_length = int(config_style.get("max_line_length", 80))
    min_complexity = int(config_style.get("min_complexity", 10))
    strict_docstrings = bool(config_style.get("strict_docstrings", False))
    strict_type_hints = bool(config_style.get("strict_type_hints", False))

    source_lines = source.splitlines()

    # Parse AST
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []  # syntax errors are caught by other graders

    visitor = _StyleVisitor(
        rel_path=rel_path,
        source_lines=source_lines,
        max_line_length=max_line_length,
        min_complexity=min_complexity,
        strict_docstrings=strict_docstrings,
        strict_type_hints=strict_type_hints,
        ignore_rules=ignore_rules,
    )
    visitor.visit(tree)

    violations = list(visitor.violations)
    violations += _check_source_lines(
        rel_path, source_lines, max_line_length, ignore_rules
    )
    violations += _check_blank_lines(rel_path, source_lines, ignore_rules)

    return violations


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _merge_sub_weights(overrides: dict[str, float] | None) -> dict[str, float]:
    w = dict(_DEFAULT_SUB_WEIGHTS)
    if overrides:
        w.update(overrides)
    total = sum(w.values())
    return (
        {k: v / total for k, v in w.items()}
        if total > 0
        else dict(_DEFAULT_SUB_WEIGHTS)
    )


def _count_checkable_items(
    modules: dict[str, ast.Module],
    rel_path: str,
    source: str,
    config_style: dict[str, Any],
) -> dict[str, int]:
    """
    Estimate the total number of checkable items per dimension so we can
    compute a meaningful fraction score rather than a raw violation count.
    """
    strict_docstrings = config_style.get("strict_docstrings", False)
    strict_type_hints = config_style.get("strict_type_hints", False)
    source_lines = source.splitlines()
    line_count = len(source_lines)

    # Parse tree for symbol counts
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    public_funcs = 0
    public_classes = 0
    all_names = 0
    imports = 0
    top_level_assigns = 0

    if tree:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") or strict_docstrings:
                    public_funcs += 1
                all_names += 1
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    public_classes += 1
                all_names += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1

        for node in tree.body:
            if isinstance(node, ast.Assign):
                top_level_assigns += 1

    # Naming: all named symbols + module name
    naming_total = max(1, all_names + 1 + top_level_assigns)

    # Docstrings: public functions + public classes + module
    docstring_total = max(1, public_funcs + public_classes + 1)

    # Imports: every import is a checkable item
    import_total = max(1, imports + 1)  # +1 for grouping check itself

    # Type hints: public functions × 2 (params + return)
    type_hint_total = max(
        1, (public_funcs if (not strict_type_hints or public_funcs > 0) else 1) * 2
    )

    # Complexity: all functions + line count
    complexity_total = max(1, public_funcs + line_count)

    # Formatting: line count
    formatting_total = max(1, line_count)

    return {
        "naming": naming_total,
        "docstrings": docstring_total,
        "imports": import_total,
        "type_hints": type_hint_total,
        "complexity": complexity_total,
        "formatting": formatting_total,
    }


def _compute_dim_scores(
    violations: list[dict],
    totals_by_dim: dict[str, int],
) -> dict[str, float]:
    """
    For each dimension: score = 1 - (violations / total_checkable_items).
    Clamped to [0, 1].
    """
    counts: dict[str, int] = {d: 0 for d in _DIMENSION_ORDER}
    for v in violations:
        dim = v.get("dimension", "")
        if dim in counts:
            counts[dim] += 1

    scores: dict[str, float] = {}
    for dim in _DIMENSION_ORDER:
        total = totals_by_dim.get(dim, 1)
        viols = counts[dim]
        scores[dim] = _clamp(1.0 - viols / max(total, 1))

    return scores


def _build_feedback(
    score: float,
    dim_scores: dict[str, float],
    violations: list[dict],
    is_regression: bool,
    baseline_score: float,
) -> str:
    if score >= 1.0:
        return "[Style] Gold standard — zero violations (Google Python Style Guide)."

    if is_regression:
        new_viols = [v for v in violations if v.get("severity") == "error"][:3]
        details = "; ".join(f"{v['rule']}:{v['file']}:{v['line']}" for v in new_viols)
        return (
            f"[Style] Regression vs baseline ({score:.0%} < {baseline_score:.0%}). "
            f"New errors: {details or 'see violations list'}."
        )

    # Worst two dimensions
    worst = sorted(dim_scores.items(), key=lambda kv: kv[1])[:2]
    worst_str = ", ".join(f"{d}={s:.0%}" for d, s in worst if s < 1.0)

    # Top 3 most common rule violations
    from collections import Counter

    rule_counts = Counter(v["rule"] for v in violations)
    top_rules = ", ".join(
        f"{rule}×{cnt} ({_RULES[rule].description[:35]}…)"
        for rule, cnt in rule_counts.most_common(3)
        if rule in _RULES
    )

    return (
        f"[Style] {score:.0%} compliant ({len(violations)} violation(s)). "
        f"Weakest: {worst_str or 'none'}. "
        f"Top issues: {top_rules or 'see violations list'}."
    )


# ---------------------------------------------------------------------------
# StyleGrader
# ---------------------------------------------------------------------------


class StyleGrader(BaseGrader):
    """
    Scores compliance with the Google Python Style Guide and eng-practices.

    Runs entirely via AST + line-level analysis — no external linter process.
    Six sub-dimensions each produce a [0,1] score (fraction of items passing),
    combined via weighted sum.

    Scoring model:
      - Items, not just violations: each dimension tracks the total number
        of checkable items (functions, lines, imports, etc.) so the score is
        proportional to the file's actual content, not just an absence of errors.
      - Regression: if the current weighted violation count exceeds baseline
        by any amount, is_regression=True and the score is capped at baseline.

    Gold standard: zero violations across all six dimensions.
    """

    grader_id = "style"

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
        Run all style checks across every non-test .py file.

        Returns
        -------
        {
          "violations"      : list[dict]          — per-violation records
          "violation_count" : int
          "dim_scores"      : dict[str, float]    — per-dimension scores
          "dim_violations"  : dict[str, int]      — per-dimension counts
          "weighted_score"  : float               — final combined score
          "files_checked"   : int
          "error"           : str|None
        }
        """
        repo_path = Path(repo_path)
        style_cfg = config.get("graders", {}).get("style", {})
        ignore_rules: set[str] = set(
            style_cfg.get("ignore_rules", []) + config.get("ignore_rules", [])
        )
        exclude_patterns: list[str] = style_cfg.get(
            "exclude_patterns", []
        ) + config.get("exclude_patterns", [])

        def _run() -> tuple[list[Violation], dict[str, int], int]:
            all_violations: list[Violation] = []
            totals_by_dim: dict[str, int] = {d: 0 for d in _DIMENSION_ORDER}
            files_checked = 0

            py_files = sorted(repo_path.rglob("*.py"))
            for fpath in py_files:
                rel_path = str(fpath.relative_to(repo_path))

                # Skip test files and excluded patterns
                if _is_test_file(rel_path):
                    continue
                if any(
                    fpath.match(pat) or rel_path.startswith(pat.rstrip("/"))
                    for pat in exclude_patterns
                ):
                    continue

                try:
                    source = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                viols = _check_file(rel_path, source, style_cfg, ignore_rules)
                all_violations.extend(viols)

                # Count checkable items
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    tree = None

                item_counts = _count_checkable_items(
                    {rel_path: tree} if tree else {},
                    rel_path,
                    source,
                    style_cfg,
                )
                for dim, cnt in item_counts.items():
                    totals_by_dim[dim] = totals_by_dim.get(dim, 0) + cnt

                files_checked += 1

            return all_violations, totals_by_dim, files_checked

        try:
            violations, totals_by_dim, files_checked = cache.get_or_compute(
                "style_checks", None, _run
            )
        except Exception as exc:
            return {
                "violations": [],
                "violation_count": 0,
                "dim_scores": {d: 0.0 for d in _DIMENSION_ORDER},
                "dim_violations": {d: 0 for d in _DIMENSION_ORDER},
                "weighted_score": 0.0,
                "files_checked": 0,
                "error": str(exc),
            }

        serialised = [v.to_dict() for v in violations]
        dim_scores = _compute_dim_scores(serialised, totals_by_dim)

        sub_weights = _merge_sub_weights(style_cfg.get("sub_weights", None))
        weighted_score = _clamp(
            sum(sub_weights[d] * dim_scores[d] for d in _DIMENSION_ORDER)
        )

        dim_violations: dict[str, int] = {d: 0 for d in _DIMENSION_ORDER}
        for v in serialised:
            dim = v.get("dimension", "")
            if dim in dim_violations:
                dim_violations[dim] += 1

        return {
            "violations": serialised,
            "violation_count": len(violations),
            "dim_scores": dim_scores,
            "dim_violations": dim_violations,
            "weighted_score": weighted_score,
            "files_checked": files_checked,
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
        style_cfg = config.get("graders", {}).get("style", {})
        sub_weights = _merge_sub_weights(style_cfg.get("sub_weights", None))

        if current.get("error"):
            return empty_grade("Style", f"analysis failed: {current['error']}")

        if current.get("files_checked", 0) == 0:
            return empty_grade("Style", "no Python files found to check")

        b_score = baseline.get("weighted_score", 0.0)
        c_score = current.get("weighted_score", 0.0)
        b_count = baseline.get("violation_count", 0)
        c_count = current.get("violation_count", 0)

        # Already at gold at baseline
        if b_score >= 1.0 and b_count == 0:
            return already_gold("Style", current)

        b_dim = baseline.get("dim_scores", {})
        c_dim = current.get("dim_scores", {})

        # ── Regression: more weighted violations than baseline ──────────
        is_regression = c_count > b_count or c_score < b_score - 1e-6

        # Score is absolute — fraction of items passing — but we cap at
        # baseline on regression to prevent the agent gaming it.
        final_score = c_score
        if is_regression:
            final_score = min(c_score, b_score)

        solved = c_score >= 1.0 - 1e-9

        # Per-dimension sub-scores relative to baseline
        sub_scores: dict[str, float] = {}
        for dim in _DIMENSION_ORDER:
            b_d = b_dim.get(dim, 0.0)
            c_d = c_dim.get(dim, 0.0)
            # Progress from baseline toward 1.0
            if b_d >= 1.0:
                sub_scores[f"dim:{dim}"] = c_d  # already perfect at baseline
            else:
                gap = 1.0 - b_d
                improvement = c_d - b_d
                sub_scores[f"dim:{dim}"] = _clamp(improvement / max(gap, 1e-9))

        feedback = _build_feedback(
            score=final_score,
            dim_scores=c_dim,
            violations=current.get("violations", [])[:20],
            is_regression=is_regression,
            baseline_score=b_score,
        )

        # Delta per-dimension violation counts
        b_dv = baseline.get("dim_violations", {})
        c_dv = current.get("dim_violations", {})
        delta_dim_viols = {d: b_dv.get(d, 0) - c_dv.get(d, 0) for d in _DIMENSION_ORDER}

        return GradeResult(
            score=final_score,
            gold_distance=1.0 - c_score,
            raw_baseline={
                "weighted_score": b_score,
                "violation_count": b_count,
                "dim_scores": b_dim,
                "dim_violations": b_dv,
            },
            raw_current={
                "weighted_score": c_score,
                "violation_count": c_count,
                "dim_scores": c_dim,
                "dim_violations": c_dv,
                "files_checked": current.get("files_checked", 0),
            },
            delta={
                "weighted_score": c_score - b_score,
                "violation_count": b_count - c_count,
                "dim_violations": delta_dim_viols,
            },
            feedback=feedback,
            solved=solved,
            is_regression=is_regression,
            sub_scores={
                "baseline_score": b_score,
                "current_score": c_score,
                **sub_scores,
            },
        )

    # ------------------------------------------------------------------
    # gold_standard
    # ------------------------------------------------------------------

    def gold_standard(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "violations": [],
            "violation_count": 0,
            "dim_scores": {d: 1.0 for d in _DIMENSION_ORDER},
            "dim_violations": {d: 0 for d in _DIMENSION_ORDER},
            "weighted_score": 1.0,
            "files_checked": 1,
            "error": None,
        }


# ---------------------------------------------------------------------------
# Module-level helper (shared with symbol_grader)
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
