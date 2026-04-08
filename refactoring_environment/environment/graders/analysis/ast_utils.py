"""
AST parsing and analysis utilities.

All functions are pure/side-effect-free except parse_file_safe and
parse_repository which do filesystem reads. Every function is
deterministic: same input → same output.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolDef:
    """A definition site: function, class, import, or variable."""

    name: str
    qualified_name: str  # e.g. "utils.process_data"
    kind: str  # "function" | "async_function" | "class" | "import" | "variable"
    file: str  # relative path from repo root
    line: int
    is_public: bool  # no leading underscore
    is_in_all: bool  # listed in __all__
    col: int = 0


@dataclass(frozen=True)
class SymbolRef:
    """A usage site of a name."""

    name: str  # raw name as written
    qualified_name: str  # resolved qualified name if possible, else same as name
    kind: str  # "call" | "load" | "attribute" | "import"
    file: str
    line: int
    col: int = 0


@dataclass
class FunctionInfo:
    qualified_name: str
    file: str
    line: int
    is_async: bool
    arg_count: int
    has_return_annotation: bool
    has_docstring: bool
    body_line_count: int
    decorator_names: list[str] = field(default_factory=list)
    is_method: bool = False
    node: ast.FunctionDef | None = field(default=None, compare=False, repr=False)


@dataclass
class ClassInfo:
    qualified_name: str
    file: str
    line: int
    base_names: list[str]
    has_docstring: bool
    method_count: int
    node: ast.ClassDef | None = field(default=None, compare=False, repr=False)


@dataclass
class AnnotationStats:
    annotated_params: int
    total_params: int
    annotated_returns: int
    total_functions: int

    @property
    def param_density(self) -> float:
        return self.annotated_params / max(self.total_params, 1)

    @property
    def return_density(self) -> float:
        return self.annotated_returns / max(self.total_functions, 1)

    @property
    def overall_density(self) -> float:
        total = self.total_params + self.total_functions
        annotated = self.annotated_params + self.annotated_returns
        return annotated / max(total, 1)


@dataclass(frozen=True)
class UnreachableBlock:
    file: str
    line: int  # line of the unreachable statement
    after_kind: str  # "return" | "raise" | "continue" | "break"
    function_name: str


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_file_safe(path: Path) -> ast.Module | None:
    """
    Parse a Python file into an AST.

    Error-tolerant: returns None on UnicodeDecodeError, returns a best-effort
    partial module on SyntaxError by truncating to the last valid statement.
    Never raises.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    try:
        return ast.parse(source, filename=str(path), type_comments=False)
    except SyntaxError as exc:
        # Attempt partial parse: truncate source to line before the error
        if exc.lineno and exc.lineno > 1:
            lines = source.splitlines()
            truncated = "\n".join(lines[: exc.lineno - 1])
            try:
                return ast.parse(truncated, filename=str(path), type_comments=False)
            except SyntaxError:
                pass
        return None


_DEFAULT_EXCLUDES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "*.egg-info",
    "migrations",
}

_GENERATED_SUFFIXES = ("_pb2.py", "_pb2_grpc.py", "_thrift.py")


def parse_repository(
    repo_path: Path,
    exclude_patterns: list[str] | None = None,
) -> dict[str, ast.Module]:
    """
    Walk all .py files under repo_path and return {relative_path: ast.Module}.

    Files that fail to parse return None values and are excluded from the dict.
    exclude_patterns is a list of glob-style directory/file names to skip
    (merged with _DEFAULT_EXCLUDES).
    """
    excludes = set(_DEFAULT_EXCLUDES)
    if exclude_patterns:
        excludes.update(exclude_patterns)

    result: dict[str, ast.Module] = {}

    for py_file in sorted(repo_path.rglob("*.py")):
        # Skip excluded directories anywhere in the path
        parts = py_file.relative_to(repo_path).parts
        if any(part in excludes for part in parts):
            continue
        # Skip generated files by suffix
        if py_file.name.endswith(_GENERATED_SUFFIXES):
            continue

        module = parse_file_safe(py_file)
        if module is not None:
            rel = str(py_file.relative_to(repo_path))
            result[rel] = module

    return result


# ---------------------------------------------------------------------------
# Definition collection
# ---------------------------------------------------------------------------


def _module_name_from_path(rel_path: str) -> str:
    """Convert 'pkg/utils.py' → 'pkg.utils'."""
    return rel_path.replace("/", ".").replace("\\", ".").removesuffix(".py")


def _all_names(module: ast.Module) -> set[str]:
    """Return names listed in module-level __all__ = [...], or empty set."""
    for node in ast.iter_child_nodes(module):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            return {
                elt.s
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.s, str)
            }
    return set()


def collect_definitions(module: ast.Module, file_path: str) -> list[SymbolDef]:
    """
    Walk an AST and return all definition sites.

    Handles: FunctionDef, AsyncFunctionDef, ClassDef, Import, ImportFrom,
    module-level Assign / AnnAssign.
    """
    mod_name = _module_name_from_path(file_path)
    all_names = _all_names(module)
    defs: list[SymbolDef] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._scope_stack: list[str] = [mod_name]

        def _qname(self, name: str) -> str:
            return f"{self._scope_stack[-1]}.{name}"

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            qname = self._qname(node.name)
            defs.append(
                SymbolDef(
                    name=node.name,
                    qualified_name=qname,
                    kind=(
                        "async_function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function"
                    ),
                    file=file_path,
                    line=node.lineno,
                    col=node.col_offset,
                    is_public=not node.name.startswith("_"),
                    is_in_all=node.name in all_names,
                )
            )
            self._scope_stack.append(qname)
            self.generic_visit(node)
            self._scope_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            qname = self._qname(node.name)
            defs.append(
                SymbolDef(
                    name=node.name,
                    qualified_name=qname,
                    kind="class",
                    file=file_path,
                    line=node.lineno,
                    col=node.col_offset,
                    is_public=not node.name.startswith("_"),
                    is_in_all=node.name in all_names,
                )
            )
            self._scope_stack.append(qname)
            self.generic_visit(node)
            self._scope_stack.pop()

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                defs.append(
                    SymbolDef(
                        name=local,
                        qualified_name=self._qname(local),
                        kind="import",
                        file=file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        is_public=not local.startswith("_"),
                        is_in_all=local in all_names,
                    )
                )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                defs.append(
                    SymbolDef(
                        name=local,
                        qualified_name=self._qname(local),
                        kind="import",
                        file=file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        is_public=not local.startswith("_"),
                        is_in_all=local in all_names,
                    )
                )

        def visit_Assign(self, node: ast.Assign) -> None:
            # Only module-level or class-level simple name assignments
            if len(self._scope_stack) <= 2:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defs.append(
                            SymbolDef(
                                name=target.id,
                                qualified_name=self._qname(target.id),
                                kind="variable",
                                file=file_path,
                                line=node.lineno,
                                col=target.col_offset,
                                is_public=not target.id.startswith("_"),
                                is_in_all=target.id in all_names,
                            )
                        )
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if len(self._scope_stack) <= 2 and isinstance(node.target, ast.Name):
                defs.append(
                    SymbolDef(
                        name=node.target.id,
                        qualified_name=self._qname(node.target.id),
                        kind="variable",
                        file=file_path,
                        line=node.lineno,
                        col=node.target.col_offset,
                        is_public=not node.target.id.startswith("_"),
                        is_in_all=node.target.id in all_names,
                    )
                )
            self.generic_visit(node)

    _Visitor().visit(module)
    return defs


# ---------------------------------------------------------------------------
# Usage collection
# ---------------------------------------------------------------------------


def collect_usages(module: ast.Module, file_path: str) -> list[SymbolRef]:
    """
    Walk an AST and return all name usage sites (Load, Call, Attribute).
    """
    refs: list[SymbolRef] = []

    for node in ast.walk(module):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            refs.append(
                SymbolRef(
                    name=node.id,
                    qualified_name=node.id,
                    kind="load",
                    file=file_path,
                    line=node.lineno,
                    col=node.col_offset,
                )
            )
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Name):
                full = f"{node.value.id}.{node.attr}"
                refs.append(
                    SymbolRef(
                        name=node.attr,
                        qualified_name=full,
                        kind="attribute",
                        file=file_path,
                        line=node.lineno,
                        col=node.col_offset,
                    )
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                refs.append(
                    SymbolRef(
                        name=func.id,
                        qualified_name=func.id,
                        kind="call",
                        file=file_path,
                        line=func.lineno,
                        col=func.col_offset,
                    )
                )
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                full = f"{func.value.id}.{func.attr}"
                refs.append(
                    SymbolRef(
                        name=func.attr,
                        qualified_name=full,
                        kind="call",
                        file=file_path,
                        line=func.lineno,
                        col=func.col_offset,
                    )
                )

    return refs


# ---------------------------------------------------------------------------
# AST normalization (for duplication detection)
# ---------------------------------------------------------------------------


class _NormalizeNames(ast.NodeTransformer):
    """Rename all local identifiers to positional placeholders."""

    def __init__(self) -> None:
        self._counter = 0
        self._mapping: dict[str, str] = {}

    def _get(self, name: str) -> str:
        if name not in self._mapping:
            self._mapping[name] = f"var_{self._counter}"
            self._counter += 1
        return self._mapping[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._get(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._get(node.arg)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        # Normalize all literals to type-tagged placeholders
        if isinstance(node.value, str):
            node.value = "<STR>"
        elif isinstance(node.value, (int, float)):
            node.value = 0
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        # Strip docstrings (first statement Constant in function/class/module)
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def normalize_ast_subtree(node: ast.AST) -> str:
    """
    Return a canonical, hashable string representation of an AST subtree.

    Renames all local variables to var_0, var_1, ..., strips docstrings,
    normalizes all literals. Used for structural clone detection.
    """
    import copy

    node_copy = copy.deepcopy(node)
    normalized = _NormalizeNames().visit(node_copy)
    try:
        return ast.dump(normalized, indent=None)
    except Exception:
        return ""


def hash_subtree(node: ast.AST) -> str:
    return hashlib.md5(normalize_ast_subtree(node).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Repository-level iterators
# ---------------------------------------------------------------------------


def iter_functions(
    modules: dict[str, ast.Module],
) -> Generator[FunctionInfo, None, None]:
    """Yield FunctionInfo for every function/method across all modules."""
    for rel_path, module in modules.items():
        mod_name = _module_name_from_path(rel_path)

        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            is_async = isinstance(node, ast.AsyncFunctionDef)
            has_docstring = (
                bool(node.body)
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )
            decorator_names = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorator_names.append(dec.id)
                elif isinstance(dec, ast.Attribute):
                    decorator_names.append(f"{ast.dump(dec.value)}.{dec.attr}")

            # Count lines: end_lineno is available in Python 3.8+
            body_lines = (
                (node.end_lineno - node.lineno + 1)
                if hasattr(node, "end_lineno") and node.end_lineno
                else len(node.body)
            )

            # Arg count excludes self/cls for methods
            args = node.args
            all_args = args.args + args.posonlyargs + args.kwonlyargs
            if args.vararg:
                all_args = all_args + [args.vararg]
            if args.kwarg:
                all_args = all_args + [args.kwarg]
            arg_count = len(all_args)

            yield FunctionInfo(
                qualified_name=f"{mod_name}.{node.name}",
                file=rel_path,
                line=node.lineno,
                is_async=is_async,
                arg_count=arg_count,
                has_return_annotation=node.returns is not None,
                has_docstring=has_docstring,
                body_line_count=body_lines,
                decorator_names=decorator_names,
                node=node,
            )


def iter_classes(
    modules: dict[str, ast.Module],
) -> Generator[ClassInfo, None, None]:
    """Yield ClassInfo for every class across all modules."""
    for rel_path, module in modules.items():
        mod_name = _module_name_from_path(rel_path)
        for node in ast.walk(module):
            if not isinstance(node, ast.ClassDef):
                continue

            base_names = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(f"{ast.dump(base.value)}.{base.attr}")

            has_docstring = (
                bool(node.body)
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )
            method_count = sum(
                1
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            yield ClassInfo(
                qualified_name=f"{mod_name}.{node.name}",
                file=rel_path,
                line=node.lineno,
                base_names=base_names,
                has_docstring=has_docstring,
                method_count=method_count,
                node=node,
            )


def get_annotation_density(modules: dict[str, ast.Module]) -> AnnotationStats:
    """Compute repo-wide type annotation coverage."""
    total_params = 0
    annotated_params = 0
    total_functions = 0
    annotated_returns = 0

    for func in iter_functions(modules):
        node = func.node
        if node is None:
            continue
        total_functions += 1
        if node.returns is not None:
            annotated_returns += 1

        args = node.args
        all_arg_nodes = args.posonlyargs + args.args + args.kwonlyargs
        if args.vararg:
            all_arg_nodes.append(args.vararg)
        if args.kwarg:
            all_arg_nodes.append(args.kwarg)

        for arg in all_arg_nodes:
            # Skip self / cls — conventionally unannotated
            if arg.arg in ("self", "cls"):
                continue
            total_params += 1
            if arg.annotation is not None:
                annotated_params += 1

    return AnnotationStats(
        annotated_params=annotated_params,
        total_params=total_params,
        annotated_returns=annotated_returns,
        total_functions=total_functions,
    )


# ---------------------------------------------------------------------------
# Unreachable block detection
# ---------------------------------------------------------------------------

_TERMINATOR_KINDS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def detect_unreachable_blocks(
    module: ast.Module, file_path: str
) -> list[UnreachableBlock]:
    """
    Find statements that immediately follow a terminator (return/raise/
    continue/break) in the same block — i.e., code that can never execute.
    """
    results: list[UnreachableBlock] = []

    def _check_body(stmts: list[ast.stmt], fn_name: str) -> None:
        for i, stmt in enumerate(stmts[:-1]):
            if isinstance(stmt, _TERMINATOR_KINDS):
                next_stmt = stmts[i + 1]
                # Ignore trailing pass/... used as placeholder
                if isinstance(next_stmt, (ast.Pass,)):
                    continue
                if (
                    isinstance(next_stmt, ast.Expr)
                    and isinstance(next_stmt.value, ast.Constant)
                    and next_stmt.value.value is ...
                ):
                    continue
                results.append(
                    UnreachableBlock(
                        file=file_path,
                        line=next_stmt.lineno,
                        after_kind=type(stmt).__name__.lower(),
                        function_name=fn_name,
                    )
                )

    class _Visitor(ast.NodeVisitor):
        def _visit_func(self, node: ast.FunctionDef) -> None:
            fn_name = node.name
            _check_body(node.body, fn_name)
            for child in ast.walk(node):
                if child is node:
                    continue
                for attr in ("body", "orelse", "handlers", "finalbody"):
                    block = getattr(child, attr, [])
                    if isinstance(block, list) and block:
                        _check_body(block, fn_name)
            self.generic_visit(node)

        visit_FunctionDef = _visit_func
        visit_AsyncFunctionDef = _visit_func  # type: ignore[assignment]

    _Visitor().visit(module)
    return results
