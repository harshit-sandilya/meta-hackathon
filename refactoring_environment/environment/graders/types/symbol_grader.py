"""
symbol_grader.py — Dead-code elimination grader.

Detects 4 classes of dead code via AST analysis:
  - Unused imports        (weight 1.0)
  - Unused variables      (weight 0.8)
  - Dead functions/classes (weight 1.5) — in-degree 0 in dep graph
  - Unreachable blocks    (weight 1.2) — statements after return/raise/break/continue

Score = clamp((weighted_baseline - weighted_now) / max(weighted_baseline, 1), 0, 1)
"""

from __future__ import annotations

import ast
import logging
from collections import defaultdict
from pathlib import Path

from ..analysis import DependencyGraph, collect_usages, parse_file_safe
from .base import BaseGrader, GradeResult

logger = logging.getLogger(__name__)

WEIGHT_UNUSED_IMPORT = 1.0
WEIGHT_UNUSED_VARIABLE = 0.8
WEIGHT_DEAD_FUNCTION = 1.5
WEIGHT_UNREACHABLE = 1.2


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _is_test_file(path: Path) -> bool:
    parts = path.parts
    name = path.name
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "tests" in parts
        or "test" in parts
    )


def _python_files(root: Path, exclude_tests: bool = True) -> list[Path]:
    return [
        p
        for p in sorted(root.rglob("*.py"))
        if not (exclude_tests and _is_test_file(p))
    ]


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _find_def_line(file_path: Path, symbol_name: str) -> int:
    try:
        tree = parse_file_safe(file_path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol_name:
                    return node.lineno
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Detection — each returns list[dict] with keys: kind, name, file, line, weight
# ---------------------------------------------------------------------------


def _unused_imports(tree: ast.Module, fp: Path, root: Path) -> list[dict]:
    used = {u.name for u in collect_usages(tree, str(fp))}
    dead = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name.split(".")[0]
                if local not in used:
                    dead.append(
                        {
                            "kind": "unused_import",
                            "name": local,
                            "file": _rel(root, fp),
                            "line": node.lineno,
                            "weight": WEIGHT_UNUSED_IMPORT,
                        }
                    )
    return dead


def _unused_variables(tree: ast.Module, fp: Path, root: Path) -> list[dict]:
    rel = _rel(root, fp)
    dead = []

    def _check(body: list[ast.stmt], prefix: str) -> None:
        assigned: dict[str, int] = {}
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if (
                        isinstance(t, ast.Name)
                        and not t.id.startswith("_")
                        and not t.id.isupper()
                    ):
                        assigned[t.id] = stmt.lineno
            elif isinstance(stmt, ast.AnnAssign):
                if (
                    isinstance(stmt.target, ast.Name)
                    and not stmt.target.id.startswith("_")
                    and not stmt.target.id.isupper()
                ):
                    assigned[stmt.target.id] = stmt.lineno

        loaded = {
            n.id
            for stmt in body
            for n in ast.walk(stmt)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        for name, lineno in assigned.items():
            if name not in loaded:
                dead.append(
                    {
                        "kind": "unused_variable",
                        "name": f"{prefix}.{name}",
                        "file": rel,
                        "line": lineno,
                        "weight": WEIGHT_UNUSED_VARIABLE,
                    }
                )

    mod = rel.replace("/", ".").removesuffix(".py")
    _check(tree.body, mod)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check(node.body, f"{mod}:{node.name}")

    return dead


def _dead_callables(
    dep_graph: DependencyGraph,
    root: Path,
    all_files: list[Path],
) -> list[dict]:
    dead = []
    # No public_api from spec — skip nothing, flag all in-degree-0 symbols
    for dead_symbol in dep_graph.get_dead_symbols([]):
        # dead_symbol is a DeadSymbol object, not a string
        parts = dead_symbol.qualified_name.rsplit(".", 1)
        module_stem = parts[0] if len(parts) == 2 else ""
        symbol = parts[-1]

        file_path = next(
            (
                fp
                for fp in all_files
                if (
                    fp.relative_to(root).with_suffix("").as_posix().replace("/", ".")
                    == module_stem
                    or fp.stem == module_stem
                )
            ),
            None,
        )
        if file_path and _is_test_file(file_path):
            continue

        dead.append(
            {
                "kind": "dead_class" if symbol[:1].isupper() else "dead_function",
                "name": dead_symbol.qualified_name,
                "file": dead_symbol.file if dead_symbol.file else (module_stem or "unknown"),
                "line": _find_def_line(file_path, symbol) if file_path else dead_symbol.line,
                "weight": WEIGHT_DEAD_FUNCTION,
            }
        )
    return dead


def _unreachable_blocks(tree: ast.Module, fp: Path, root: Path) -> list[dict]:
    rel = _rel(root, fp)
    dead = []
    _TERM = (ast.Return, ast.Raise, ast.Continue, ast.Break)

    def _check(stmts: list[ast.stmt], ctx: str) -> None:
        terminated = False
        for stmt in stmts:
            if terminated:
                dead.append(
                    {
                        "kind": "unreachable_block",
                        "name": f"{ctx}:unreachable@L{stmt.lineno}",
                        "file": rel,
                        "line": stmt.lineno,
                        "weight": WEIGHT_UNREACHABLE,
                    }
                )
            if isinstance(stmt, _TERM):
                terminated = True
            # recurse into nested blocks
            for attr in ("body", "orelse", "finalbody"):
                block = getattr(stmt, attr, None)
                if block:
                    label = f"{ctx}:{attr}@L{stmt.lineno}"
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) or isinstance(stmt, ast.ClassDef):
                        label = f"{ctx}.{stmt.name}"
                    _check(block, label)
            if isinstance(stmt, ast.Try):
                for h in stmt.handlers:
                    _check(h.body, f"{ctx}:except@L{h.lineno}")

    _check(tree.body, rel.replace("/", ".").removesuffix(".py"))
    return dead


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(symbols: list[dict]) -> dict:
    seen: set[str] = set()
    unique: list[dict] = []
    by_kind: dict[str, int] = defaultdict(int)

    for s in symbols:
        key = f"{s['file']}:{s['name']}"
        if key not in seen:
            seen.add(key)
            unique.append(s)
            by_kind[s["kind"]] += 1

    return {
        "weighted_total": round(sum(s["weight"] for s in unique), 3),
        "raw_count": len(unique),
        "by_kind": dict(by_kind),
        "symbols": unique,
    }


# ---------------------------------------------------------------------------
# SymbolGrader
# ---------------------------------------------------------------------------


class SymbolGrader(BaseGrader):
    grader_id = "symbol"

    def _compute_metrics(self) -> dict:
        root: Path = self.file_handler.root

        all_files = _python_files(root, exclude_tests=False)
        source_files = _python_files(root, exclude_tests=True)

        symbols: list[dict] = []

        # Dep graph — best effort, don't crash if it fails
        dep_graph = DependencyGraph()
        has_graph = False
        try:
            # Build modules dict for DependencyGraph
            modules = {}
            for fp in source_files:
                try:
                    tree = parse_file_safe(fp)
                    if tree:
                        rel_path = _rel(root, fp)
                        modules[rel_path] = tree
                except Exception as parse_exc:
                    logger.debug("Failed to parse %s for dep graph: %s", fp, parse_exc)
                    continue

            dep_graph.build(modules)
            has_graph = True
        except Exception as exc:
            logger.warning(
                "DepGraph build failed (%s); skipping dead-callable pass", exc
            )

        for fp in source_files:
            try:
                tree = parse_file_safe(fp)
            except SyntaxError as exc:
                logger.warning("Skipping %s (SyntaxError: %s)", fp, exc)
                continue
            except Exception as exc:
                logger.warning("Skipping %s (parse error: %s)", fp, exc)
                continue

            # Skip if parsing failed
            if tree is None:
                logger.debug("Skipping %s (parse returned None)", fp)
                continue

            symbols.extend(_unused_imports(tree, fp, root))
            symbols.extend(_unused_variables(tree, fp, root))
            symbols.extend(_unreachable_blocks(tree, fp, root))

        if has_graph:
            symbols.extend(_dead_callables(dep_graph, root, all_files))

        return _aggregate(symbols)

    def grade(self) -> GradeResult:
        current = self._compute_metrics()

        w_base = self._baseline.get("weighted_total", 0.0)
        w_now = current.get("weighted_total", 0.0)
        n_base = self._baseline.get("raw_count", 0)
        n_now = current.get("raw_count", 0)

        score = self._delta_score(w_base, w_now)

        delta = n_base - n_now
        direction = (
            f"-{delta}"
            if delta > 0
            else f"+{abs(delta)} (regression)" if delta < 0 else "no change"
        )

        remaining = [
            f"{k.replace('_', ' ')} ×{v}"
            for k, v in current.get("by_kind", {}).items()
            if v > 0
        ]
        remaining_str = f" Remaining: {', '.join(remaining[:3])}." if remaining else ""

        feedback = (
            f"Dead code: {n_base} → {n_now} ({direction}). "
            f"Weighted: {w_base:.1f} → {w_now:.1f} ({score * 100:.0f}% eliminated)."
            f"{remaining_str}"
        )

        return GradeResult(
            score=score,
            feedbacks=[feedback],
            errors=[],
            tool_errors=[],
            added_violations=round(max(0, w_now - w_base), 0),
        )
