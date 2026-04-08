"""
Style Grader — Google Python Style Guide compliance.

Reference specs:
  https://google.github.io/styleguide/pyguide.html
  https://google.github.io/eng-practices/review/reviewer/looking-for.html

Gold standard : zero style violations across all non-test .py files.

Pure-AST + regex; does NOT invoke ruff or pylint at runtime — stays fast
and fully deterministic inside the sandbox.

Scoring model (delta-based)
───────────────────────────
  raw_score   = weighted_sum(dim_scores)                    ∈ [0, 1]
  delta_score = (raw_now − raw_baseline) /
                max(1.0 − raw_baseline, 0.01)               clamped [0, 1]

  1.0 = agent fully closed the gap from starting state to zero-violation.
  Partial improvement → proportional credit.
  Regression (raw_now < raw_baseline) → 0.0, is_regression=True.

Sub-dimensions (weighted sum → raw_score)
──────────────────────────────────────────
  naming     0.30  PEP 8 + Google naming conventions
  docstrings 0.25  public symbol docstring coverage
  imports    0.20  import ordering & style rules
  type_hints 0.10  public function annotation coverage
  complexity 0.10  cyclomatic complexity, line length
  formatting 0.05  blank lines, trailing whitespace

Each sub-dimension score = 1 − (violations / checkable_items), clamped [0, 1].

Scenario config keys (under spec.config["style"]):
  sub_weights      : dict   per-dimension weight overrides
  exclude_patterns : list   file globs to skip (default: test files)
  ignore_rules     : list   rule codes to suppress, e.g. ["N002", "D001"]
  max_line_length  : int    line length limit (default 80, Google spec)
  min_complexity   : int    cyclomatic complexity threshold (default 10)
  strict_docstrings: bool   also require private func docstrings
  strict_type_hints: bool   also require private func annotations

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
from dataclasses import dataclass
from typing import ClassVar

from .base import BaseGrader, GradeResult


# ── Constants ─────────────────────────────────────────────────────────────────

_DIMENSIONS: list[str] = [
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

# stdlib top-level names for import-group ordering (I005)
_STDLIB_TOP: frozenset[str] = frozenset(
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

# Typing-only modules — skip I004 for imports from these
_TYPING_MODULES: frozenset[str] = frozenset(
    {
        "typing",
        "collections.abc",
        "typing_extensions",
        "__future__",
    }
)

# Dunder methods exempt from all naming checks
_DUNDER_METHODS: frozenset[str] = frozenset(
    {
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
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__radd__",
        "__iadd__",
        "__mul__",
        "__rmul__",
        "__imul__",
        "__sub__",
        "__rsub__",
        "__isub__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__pow__",
        "__and__",
        "__or__",
        "__xor__",
        "__invert__",
        "__lshift__",
        "__rshift__",
        "__neg__",
        "__pos__",
        "__abs__",
        "__bool__",
        "__int__",
        "__float__",
        "__complex__",
        "__bytes__",
        "__format__",
        "__sizeof__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
        "__copy__",
        "__deepcopy__",
        "__class_getitem__",
        "__init_subclass__",
        "__set_name__",
        "__get__",
        "__set__",
        "__delete__",
        "__missing__",
        "__reversed__",
        "__del__",
        "__slots__",
    }
)


# ── Rule registry ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Rule:
    code: str
    description: str
    dimension: (
        str  # naming | docstrings | imports | type_hints | complexity | formatting
    )
    severity: str  # error | warning


_RULES: dict[str, _Rule] = {
    # Naming
    "N001": _Rule(
        "N001", "Module name must be lowercase_underscore", "naming", "error"
    ),
    "N002": _Rule("N002", "Class name must be CapWords", "naming", "error"),
    "N003": _Rule(
        "N003", "Function/method name must be lowercase_underscore", "naming", "error"
    ),
    "N004": _Rule(
        "N004",
        "Constant must be UPPER_CASE (module-level literal)",
        "naming",
        "warning",
    ),
    "N005": _Rule(
        "N005", "CamelCase variable at module scope (not a class)", "naming", "warning"
    ),
    "N006": _Rule(
        "N006", "Private name must use single leading underscore", "naming", "warning"
    ),
    "N007": _Rule(
        "N007", "Avoid single-letter names except loop counters", "naming", "warning"
    ),
    "N008": _Rule(
        "N008", "TypeVar name must be CapWords or single letter", "naming", "warning"
    ),
    "N009": _Rule("N009", "Exception class must end in 'Error'", "naming", "error"),
    "N010": _Rule("N010", "Test function must start with 'test_'", "naming", "warning"),
    # Docstrings
    "D001": _Rule(
        "D001", "Public function/method missing docstring", "docstrings", "error"
    ),
    "D002": _Rule("D002", "Public class missing docstring", "docstrings", "error"),
    "D003": _Rule(
        "D003", "Module missing module-level docstring", "docstrings", "warning"
    ),
    "D004": _Rule(
        "D004",
        "Docstring summary line exceeds max_line_length chars",
        "docstrings",
        "warning",
    ),
    "D005": _Rule(
        "D005",
        "Multi-line docstring missing blank line after summary",
        "docstrings",
        "warning",
    ),
    "D006": _Rule(
        "D006",
        "Generator function should use 'Yields:' not 'Returns:'",
        "docstrings",
        "warning",
    ),
    # Imports
    "I001": _Rule(
        "I001", "Wildcard import (from x import *) forbidden", "imports", "error"
    ),
    "I002": _Rule(
        "I002", "Relative import forbidden (use full package path)", "imports", "error"
    ),
    "I003": _Rule(
        "I003", "Multiple imports on one line (import a, b)", "imports", "warning"
    ),
    "I004": _Rule(
        "I004",
        "Importing individual class/function (use module import)",
        "imports",
        "warning",
    ),
    "I005": _Rule(
        "I005",
        "Imports not grouped: stdlib → third-party → local",
        "imports",
        "warning",
    ),
    "I006": _Rule(
        "I006", "Mutable default argument in function signature", "imports", "error"
    ),
    # Type hints
    "T001": _Rule(
        "T001",
        "Public function missing return type annotation",
        "type_hints",
        "warning",
    ),
    "T002": _Rule(
        "T002",
        "Public function missing parameter type annotation(s)",
        "type_hints",
        "warning",
    ),
    "T003": _Rule(
        "T003", "Bare 'except:' clause (catch-all) forbidden", "type_hints", "error"
    ),
    # Complexity
    "C001": _Rule(
        "C001",
        "Function exceeds cyclomatic complexity threshold",
        "complexity",
        "warning",
    ),
    "C002": _Rule("C002", "Line exceeds maximum length", "complexity", "warning"),
    "C003": _Rule(
        "C003",
        "Nested comprehension with multiple for-clauses",
        "complexity",
        "warning",
    ),
    # Formatting
    "F001": _Rule("F001", "Trailing whitespace on line", "formatting", "warning"),
    "F002": _Rule(
        "F002",
        "Missing two blank lines before top-level definition",
        "formatting",
        "warning",
    ),
    "F003": _Rule(
        "F003", "Semicolon used to separate statements", "formatting", "error"
    ),
    "F004": _Rule("F004", "Tab character used for indentation", "formatting", "error"),
}


# ── Violation ─────────────────────────────────────────────────────────────────


@dataclass
class _Violation:
    rule: str
    file: str
    line: int
    context: str = ""

    def to_dict(self) -> dict:
        r = _RULES[self.rule]
        return {
            "rule": self.rule,
            "dimension": r.dimension,
            "severity": r.severity,
            "message": r.description,
            "file": self.file,
            "line": self.line,
            "context": self.context,
        }


# ── AST visitor (single pass per file) ───────────────────────────────────────


class _FileVisitor(ast.NodeVisitor):
    """
    Single-pass AST visitor that collects _Violation objects for one file.
    Usage: v = _FileVisitor(...); v.visit(tree); violations = v.violations
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
        self.violations: list[_Violation] = []
        self._scope_stack: list[str] = []

    # ── emit ──────────────────────────────────────────────────────────────

    def _emit(self, rule: str, node: ast.AST, context: str = "") -> None:
        if rule in self.ignore_rules:
            return
        self.violations.append(
            _Violation(rule, self.rel_path, getattr(node, "lineno", 0), context)
        )

    # ── predicates ────────────────────────────────────────────────────────

    @staticmethod
    def _is_public(name: str) -> bool:
        return not name.startswith("_")

    @staticmethod
    def _is_snake(name: str) -> bool:
        return bool(re.match(r"^[a-z][a-z0-9_]*$", name)) or name == "_"

    @staticmethod
    def _is_upper(name: str) -> bool:
        return bool(re.match(r"^[A-Z][A-Z0-9_]*$", name))

    @staticmethod
    def _is_capwords(name: str) -> bool:
        return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", name))

    # ── AST helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _docstring(node: ast.AST) -> str | None:
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

    @staticmethod
    def _cyclomatic(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        cc = 1
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
                cc += 1
            elif isinstance(child, ast.BoolOp):
                cc += len(child.values) - 1
        return cc

    @staticmethod
    def _has_nested_comprehension(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(
                child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                if len(getattr(child, "generators", [])) > 1:
                    return True
        return False

    # ── Module ────────────────────────────────────────────────────────────

    def visit_Module(self, node: ast.Module) -> None:
        from pathlib import Path

        stem = Path(self.rel_path).stem
        if stem not in ("__init__", "__main__") and not self._is_snake(stem):
            self._emit("N001", node, stem)
        if self._docstring(node) is None:
            self._emit("D003", node)
        self._check_import_groups(node)
        self.generic_visit(node)

    # ── Imports ───────────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        if len(node.names) > 1:
            self._emit("I003", node, ", ".join(a.name for a in node.names))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(a.name == "*" for a in node.names):
            self._emit("I001", node, f"from {node.module} import *")
        if node.level and node.level > 0:
            self._emit("I002", node, f"relative level={node.level}")
        if node.module and node.module not in _TYPING_MODULES:
            for alias in node.names:
                if self._is_capwords(alias.name) and not alias.name.isupper():
                    self._emit("I004", node, f"from {node.module} import {alias.name}")

    def _check_import_groups(self, module: ast.Module) -> None:
        order = {"stdlib": 0, "third_party": 1, "local": 2}
        seen_max = 0
        for node in module.body:
            if isinstance(node, ast.Import):
                top = node.names[0].name.split(".")[0]
                group = "stdlib" if top in _STDLIB_TOP else "third_party"
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    group = "local"
                else:
                    top = (node.module or "").split(".")[0]
                    group = "stdlib" if top in _STDLIB_TOP else "third_party"
            else:
                continue
            gval = order[group]
            if gval < seen_max:
                fake = type("_N", (), {"lineno": node.lineno})()
                self._emit("I005", fake)
                break
            seen_max = max(seen_max, gval)

    # ── Classes ───────────────────────────────────────────────────────────

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not self._is_capwords(node.name):
            self._emit("N002", node, node.name)
        base_names = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
        if any(
            "Exception" in b or "Error" in b or b == "BaseException" for b in base_names
        ) and not node.name.endswith("Error"):
            self._emit("N009", node, node.name)
        if self._is_public(node.name) and self._docstring(node) is None:
            self._emit("D002", node, node.name)
        self._scope_stack.append(f"class:{node.name}")
        self.generic_visit(node)
        self._scope_stack.pop()

    # ── Functions ─────────────────────────────────────────────────────────

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = node.name
        is_public = self._is_public(name)
        is_generator = any(isinstance(n, ast.Yield) for n in ast.walk(node))

        # Naming
        if name not in _DUNDER_METHODS and not self._is_snake(name):
            self._emit("N003", node, name)
        _ok_single = {"i", "j", "k", "n", "x", "y", "z", "v", "e", "f", "_"}
        if len(name) == 1 and name not in _ok_single:
            self._emit("N007", node, name)
        if (
            "test" in self.rel_path.lower()
            and is_public
            and not name.startswith("test_")
        ):
            if any(isinstance(n, ast.Assert) for n in ast.walk(node)):
                self._emit("N010", node, name)

        # Docstrings
        docstr = self._docstring(node)
        if (is_public or self.strict_docstrings) and docstr is None:
            self._emit("D001", node, name)
        if docstr is not None:
            summary = docstr.strip().splitlines()[0]
            if len(summary) > self.max_line_length:
                self._emit("D004", node, summary[: self.max_line_length - 10] + "…")
            lines = docstr.strip().splitlines()
            if len(lines) > 1 and lines[1].strip() != "":
                self._emit("D005", node, name)
            if is_generator and "Returns:" in docstr and "Yields:" not in docstr:
                self._emit("D006", node, name)

        # Type hints
        if is_public or self.strict_type_hints:
            if node.returns is None:
                self._emit("T001", node, name)
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
                self._emit("T002", node, f"{name}({', '.join(unannotated[:3])})")

        # Bare except
        for child in ast.walk(node):
            if isinstance(child, ast.ExceptHandler) and child.type is None:
                self._emit("T003", child, name)

        # Mutable defaults
        _mutable_nodes = (ast.List, ast.Dict, ast.Set)
        _mutable_calls = {"list", "dict", "set", "defaultdict"}
        for default in node.args.defaults + node.args.kw_defaults:
            if default is None:
                continue
            if isinstance(default, _mutable_nodes):
                self._emit("I006", node, name)
                break
            if isinstance(default, ast.Call):
                fn = getattr(default.func, "id", getattr(default.func, "attr", ""))
                if fn.lower() in _mutable_calls:
                    self._emit("I006", node, name)
                    break

        # Complexity
        cc = self._cyclomatic(node)
        if cc > self.min_complexity:
            self._emit("C001", node, f"{name} cc={cc}")
        if self._has_nested_comprehension(node):
            self._emit("C003", node, name)

        self._scope_stack.append(f"func:{name}")
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    # ── Module-level assignments ──────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._scope_stack:
            return
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(node.value, ast.Constant):
                if not self._is_upper(name) and not name.startswith("_"):
                    self._emit("N004", node, name)
            elif self._is_capwords(name) and not name.isupper():
                self._emit("N005", node, name)
        self.generic_visit(node)


# ── StyleGrader ───────────────────────────────────────────────────────────────


class StyleGrader(BaseGrader):
    """
    Grader for Google Python Style Guide compliance.

    Inherits from BaseGrader:
      _compute_metrics()  →  runs AST + regex analysis, returns raw metrics dict.
      grade()             →  delta-scores improvement from self._baseline.
    """

    grader_id: ClassVar[str] = "style"

    # ── BaseGrader interface ──────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        """
        Walk all non-test .py files, run AST + line checks, aggregate results.

        Returns:
            violations     : list[dict]       one entry per violation
            by_dim         : dict[str, int]   violation count per dimension
            by_rule        : dict[str, int]   violation count per rule code
            totals_by_dim  : dict[str, int]   checkable items (denominator)
            dim_scores     : dict[str, float] per-dimension compliance score
            raw_score      : float            weighted compliance ∈ [0, 1]
            total_viols    : int
            error_viols    : int              severity == "error" count
            files_checked  : int
        """
        cfg = self._style_cfg()
        ignore = set(cfg.get("ignore_rules", []))
        exclude = cfg.get("exclude_patterns", ["test_*.py", "*_test.py", "tests/**"])
        sub_weights = self._merge_weights(cfg.get("sub_weights"))

        py_files = self.file_handler.list_python_files(exclude_patterns=exclude)

        all_violations: list[dict] = []
        totals_by_dim: dict[str, int] = {d: 0 for d in _DIMENSIONS}
        files_checked = 0

        for rel_path in py_files:
            source = self.file_handler.read(rel_path)
            file_viols = self._check_file(rel_path, source, cfg, ignore)
            all_violations.extend(v.to_dict() for v in file_viols)
            for dim, count in self._checkable_items(source, cfg).items():
                totals_by_dim[dim] += count
            files_checked += 1

        if files_checked == 0:
            return self._empty_metrics()

        by_dim, by_rule = self._aggregate(all_violations)
        dim_scores = {
            d: self._clamp(1.0 - by_dim.get(d, 0) / max(totals_by_dim.get(d, 1), 1))
            for d in _DIMENSIONS
        }
        raw = sum(sub_weights.get(d, 0.0) * dim_scores[d] for d in _DIMENSIONS)

        return {
            "violations": all_violations,
            "by_dim": by_dim,
            "by_rule": by_rule,
            "totals_by_dim": totals_by_dim,
            "dim_scores": {k: round(v, 4) for k, v in dim_scores.items()},
            "raw_score": round(raw, 4),
            "total_viols": len(all_violations),
            "error_viols": sum(1 for v in all_violations if v["severity"] == "error"),
            "files_checked": files_checked,
        }

    def grade(self) -> GradeResult:
        """
        Compute current metrics and delta-score improvement from baseline.

        score = clamp((raw_now − raw_baseline) / max(1 − raw_baseline, 0.01))

        1.0  →  agent fully closed gap to zero-violation codebase.
        0.0  →  no improvement or regression.
        """
        current = self._compute_metrics()

        baseline_raw: float = self._baseline.get("raw_score", 0.0)
        current_raw: float = current.get("raw_score", 0.0)
        is_regression = current_raw < baseline_raw

        if baseline_raw >= 1.0:
            # Baseline was already gold — full credit only if still perfect.
            score = 1.0 if current.get("total_viols", 0) == 0 else 0.0
        elif is_regression:
            score = 0.0
        else:
            gap = max(1.0 - baseline_raw, 0.01)
            score = self._clamp((current_raw - baseline_raw) / gap)

        return GradeResult(
            score=round(score, 4),
            feedbacks=[self._feedback(
                score, current, baseline_raw, current_raw, is_regression
            )],
            errors=[],
            tool_errors=[],
            added_violations=round(max(0, current.get("error_viols", 0) - self._baseline.get("error_viols", 0)), 0),
        )

    # ── Config helpers ────────────────────────────────────────────────────

    def _style_cfg(self) -> dict:
        # Return hardcoded default configuration
        # Remove dependency on spec.config to maintain interface stability
        return {
            "sub_weights": None,  # Use default weights
            "exclude_patterns": ["test_*.py", "*_test.py", "tests/**"],
            "ignore_rules": [],
            "max_line_length": 80,
            "min_complexity": 10,
            "strict_docstrings": False,
            "strict_type_hints": False,
        }

    @staticmethod
    def _merge_weights(overrides: dict[str, float] | None) -> dict[str, float]:
        w = dict(_DEFAULT_SUB_WEIGHTS)
        if overrides:
            w.update(overrides)
        total = sum(w.values())
        return (
            {k: v / total for k, v in w.items()}
            if total > 0
            else dict(_DEFAULT_SUB_WEIGHTS)
        )

    # ── File analysis ─────────────────────────────────────────────────────

    def _check_file(
        self,
        rel_path: str,
        source: str,
        cfg: dict,
        ignore: set[str],
    ) -> list[_Violation]:
        """Parse and lint a single file. Returns all violations."""
        max_ll = int(cfg.get("max_line_length", 80))
        min_cc = int(cfg.get("min_complexity", 10))
        strict_d = bool(cfg.get("strict_docstrings", False))
        strict_t = bool(cfg.get("strict_type_hints", False))
        lines = source.splitlines()

        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError:
            return []  # syntax errors handled by other graders

        visitor = _FileVisitor(
            rel_path, lines, max_ll, min_cc, strict_d, strict_t, ignore
        )
        visitor.visit(tree)

        return visitor.violations + self._line_checks(rel_path, lines, max_ll, ignore)

    # ── Line-level checks (no AST needed) ─────────────────────────────────

    @staticmethod
    def _line_checks(
        rel_path: str,
        lines: list[str],
        max_line_length: int,
        ignore: set[str],
    ) -> list[_Violation]:
        viols: list[_Violation] = []
        blank_count = 0
        _toplevel_re = re.compile(r"^(def |class |async def )")

        for lineno, raw in enumerate(lines, start=1):
            line = raw.rstrip("\n")
            stripped = line.strip()

            # F001: trailing whitespace
            if "F001" not in ignore and line != line.rstrip():
                viols.append(_Violation("F001", rel_path, lineno, repr(line[-10:])))

            # F004: tab indentation
            if "F004" not in ignore and line.startswith("\t"):
                viols.append(_Violation("F004", rel_path, lineno))

            # F003: semicolons (heuristic, skips string contents)
            if (
                "F003" not in ignore
                and ";" in stripped
                and not stripped.startswith("#")
            ):
                in_str, quote = False, None
                for ch in stripped:
                    if ch in ('"', "'") and not in_str:
                        in_str, quote = True, ch
                    elif in_str and ch == quote:
                        in_str = False
                    elif ch == ";" and not in_str:
                        viols.append(
                            _Violation("F003", rel_path, lineno, stripped[:40])
                        )
                        break

            # C002: line too long (exempt URL-only comments, noqa, single tokens)
            if "C002" not in ignore and len(line) > max_line_length:
                s = line.strip()
                if not (
                    (s.startswith("#") and "http" in s and " " not in s.lstrip("# "))
                    or "pylint:" in line
                    or "noqa" in line
                    or " " not in s
                ):
                    viols.append(
                        _Violation("C002", rel_path, lineno, f"len={len(line)}")
                    )

            # F002: two blank lines before top-level defs
            if "F002" not in ignore:
                if stripped == "":
                    blank_count += 1
                else:
                    if _toplevel_re.match(line) and lineno > 1 and blank_count < 2:
                        viols.append(
                            _Violation("F002", rel_path, lineno, stripped[:40])
                        )
                    blank_count = 0

        return viols

    # ── Scoring helpers ───────────────────────────────────────────────────

    @staticmethod
    def _aggregate(violations: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
        by_dim: dict[str, int] = {d: 0 for d in _DIMENSIONS}
        by_rule: dict[str, int] = {}
        for v in violations:
            dim = v.get("dimension", "")
            rule = v.get("rule", "")
            if dim in by_dim:
                by_dim[dim] += 1
            by_rule[rule] = by_rule.get(rule, 0) + 1
        return by_dim, by_rule

    @staticmethod
    def _checkable_items(source: str, cfg: dict) -> dict[str, int]:
        """Count denominator items per dimension for one file."""
        strict_d = cfg.get("strict_docstrings", False)
        strict_t = cfg.get("strict_type_hints", False)
        lines = source.splitlines()
        n_lines = max(len(lines), 1)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {d: n_lines for d in _DIMENSIONS}

        public_funcs = total_symbols = import_nodes = top_assigns = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") or strict_d:
                    public_funcs += 1
                total_symbols += 1
            elif isinstance(node, ast.ClassDef):
                total_symbols += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes += 1

        for node in tree.body:
            if isinstance(node, ast.Assign):
                top_assigns += 1

        return {
            "naming": max(1, total_symbols + 1 + top_assigns),
            "docstrings": max(1, public_funcs + 1),
            "imports": max(1, import_nodes + 1),
            "type_hints": max(1, (public_funcs if not strict_t else total_symbols) * 2),
            "complexity": max(1, total_symbols + n_lines),
            "formatting": max(1, n_lines),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "violations": [],
            "by_dim": {d: 0 for d in _DIMENSIONS},
            "by_rule": {},
            "totals_by_dim": {d: 1 for d in _DIMENSIONS},
            "dim_scores": {d: 1.0 for d in _DIMENSIONS},
            "raw_score": 1.0,
            "total_viols": 0,
            "error_viols": 0,
            "files_checked": 0,
        }

    # ── Feedback ──────────────────────────────────────────────────────────

    @staticmethod
    def _feedback(
        score: float,
        current: dict,
        baseline_raw: float,
        current_raw: float,
        is_regression: bool,
    ) -> str:
        if score >= 1.0:
            return (
                "[Style] Gold standard — zero violations (Google Python Style Guide)."
            )

        by_rule: dict[str, int] = current.get("by_rule", {})
        top_rules = sorted(by_rule.items(), key=lambda kv: -kv[1])[:3]
        rules_str = ", ".join(
            f"{r}×{c} ({_RULES[r].description[:30]}…)"
            for r, c in top_rules
            if r in _RULES
        )

        if is_regression:
            return (
                f"[Style] Regression vs baseline "
                f"(raw {current_raw:.2f} < baseline {baseline_raw:.2f}). "
                f"Top violations: {rules_str or 'see violations list'}."
            )

        dim_scores: dict[str, float] = current.get("dim_scores", {})
        worst = sorted(dim_scores.items(), key=lambda kv: kv[1])[:2]
        worst_str = ", ".join(f"{d}={s:.0%}" for d, s in worst if s < 1.0)

        return (
            f"[Style] delta={score:.0%} "
            f"(raw {baseline_raw:.2f}→{current_raw:.2f}). "
            f"Weakest dims: {worst_str}. "
            f"Top rules: {rules_str}."
        )


__all__ = ["StyleGrader"]
