"""
Heuristic Big-O time and space complexity estimation via AST analysis.

All estimates are best-effort approximations — they signal to the RL agent
which functions are worth simplifying. Not a formal complexity prover.
All functions are pure and deterministic.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Complexity class enumeration (ordered by cost)
# ---------------------------------------------------------------------------


class ComplexityClass(IntEnum):
    O1 = 0  # constant
    OLOGN = 1  # logarithmic
    ON = 2  # linear
    ONLOGN = 3  # linearithmic
    ON2 = 4  # quadratic
    ON3 = 5  # cubic
    ONEXP = 6  # exponential
    UNKNOWN = 7  # could not determine

    def __str__(self) -> str:
        return {
            0: "O(1)",
            1: "O(log n)",
            2: "O(n)",
            3: "O(n log n)",
            4: "O(n²)",
            5: "O(n³)",
            6: "O(2ⁿ)",
            7: "O(?)",
        }[self.value]


# Score cost for reward computation — higher = worse, more room for improvement
COMPLEXITY_COST: dict[ComplexityClass, float] = {
    ComplexityClass.O1: 0.0,
    ComplexityClass.OLOGN: 0.0,
    ComplexityClass.ON: 0.0,
    ComplexityClass.ONLOGN: 1.0,
    ComplexityClass.ON2: 3.0,
    ComplexityClass.ON3: 6.0,
    ComplexityClass.ONEXP: 10.0,
    ComplexityClass.UNKNOWN: 2.0,
}


@dataclass
class ComplexityEstimate:
    time: ComplexityClass
    space: ComplexityClass
    time_reason: str  # human-readable explanation of the worst pattern found
    space_reason: str
    patterns_found: list[str]  # list of anti-pattern tags for feedback


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _max_loop_nesting(node: ast.AST) -> int:
    """Return maximum loop nesting depth inside a node (0 = no loops)."""
    max_depth = [0]

    def _walk(n: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.For, ast.While)):
                new_depth = depth + 1
                if new_depth > max_depth[0]:
                    max_depth[0] = new_depth
                _walk(child, new_depth)
            else:
                _walk(child, depth)

    _walk(node, 0)
    return max_depth[0]


def _count_recursive_calls(func_node: ast.FunctionDef) -> int:
    """Count how many times a function calls itself directly."""
    name = func_node.name
    count = 0
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == name:
                count += 1
    return count


def _has_memoization(func_node: ast.FunctionDef) -> bool:
    """Check for @lru_cache, @cache, or @functools.cache decorators."""
    memo_names = {"lru_cache", "cache"}
    for dec in func_node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in memo_names:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr in memo_names:
            return True
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Name)
            and dec.func.id in memo_names
        ):
            return True
    return False


def _has_halving_pattern(func_node: ast.FunctionDef) -> bool:
    """
    Detect divide-and-conquer or binary search patterns:
    - `i //= 2` or `i >>= 1` in a loop
    - `bisect` module call
    - slice with //2 (e.g. arr[mid:] where mid = len(arr) // 2)
    """
    for node in ast.walk(func_node):
        if isinstance(node, ast.AugAssign):
            if isinstance(node.op, (ast.FloorDiv, ast.RShift)):
                if isinstance(node.value, ast.Constant) and node.value.value in (1, 2):
                    return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith(
                "bisect"
            ):
                return True
            if isinstance(node.func, ast.Name) and node.func.id.startswith("bisect"):
                return True
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            for sub in ast.walk(node.slice):
                if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.FloorDiv):
                    return True
    return False


# ---------------------------------------------------------------------------
# Anti-pattern detection
# Each returns Optional[tuple[tag_str, ComplexityClass]] — None = not found
# ---------------------------------------------------------------------------


def _detect_str_concat_in_loop(
    func_node: ast.FunctionDef,
) -> tuple[str, ComplexityClass] | None:
    """str += ... inside a loop → O(n²) due to string copies."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.AugAssign)
                    and isinstance(child.op, ast.Add)
                    and isinstance(child.target, ast.Name)
                ):
                    return (
                        "str_concat_in_loop → use ''.join(parts)",
                        ComplexityClass.ON2,
                    )
    return None


def _detect_list_search_in_loop(
    func_node: ast.FunctionDef,
) -> tuple[str, ComplexityClass] | None:
    """x in list or list.index(x) inside a loop → O(n²)."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Compare):
                    for op in child.ops:
                        if isinstance(op, ast.In):
                            return (
                                "linear_search_in_loop → convert to set()",
                                ComplexityClass.ON2,
                            )
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "index"
                ):
                    return (
                        "list.index_in_loop → use dict/set for O(1) lookup",
                        ComplexityClass.ON2,
                    )
    return None


def _detect_sort_in_loop(
    func_node: ast.FunctionDef,
) -> tuple[str, ComplexityClass] | None:
    """sorted() or .sort() called inside a loop → O(n² log n)."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Name) and func.id == "sorted":
                        return (
                            "sort_in_loop → O(n² log n), sort once outside loop",
                            ComplexityClass.ON2,
                        )
                    if isinstance(func, ast.Attribute) and func.attr == "sort":
                        return (
                            "sort_in_loop → O(n² log n), sort once outside loop",
                            ComplexityClass.ON2,
                        )
    return None


def _detect_list_insert_head_in_loop(
    func_node: ast.FunctionDef,
) -> tuple[str, ComplexityClass] | None:
    """.insert(0, x) in a loop → O(n²) due to list shifts."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "insert"
                    and child.args
                    and isinstance(child.args[0], ast.Constant)
                    and child.args[0].value == 0
                ):
                    return (
                        "list.insert(0,x)_in_loop → use collections.deque.appendleft()",
                        ComplexityClass.ON2,
                    )
    return None


def _detect_nested_comprehension(
    func_node: ast.FunctionDef,
) -> tuple[str, ComplexityClass] | None:
    """Nested list comprehension with multiple generators → O(n²)."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            if len(node.generators) >= 2:
                return ("nested_comprehension → O(n²) or worse", ComplexityClass.ON2)
    return None


def _detect_full_load(
    func_node: ast.FunctionDef,
) -> tuple[str, ComplexityClass] | None:
    """Accumulating list-of-lists inside a loop → O(n²) space."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "append"
                    and child.args
                    and isinstance(child.args[0], (ast.ListComp, ast.List))
                ):
                    return (
                        "list_of_lists_accumulation → consider generator/streaming",
                        ComplexityClass.ON2,
                    )
    return None


_ALL_TIME_DETECTORS = [
    _detect_str_concat_in_loop,
    _detect_list_search_in_loop,
    _detect_sort_in_loop,
    _detect_list_insert_head_in_loop,
    _detect_nested_comprehension,
]

_ALL_SPACE_DETECTORS = [
    _detect_full_load,
]


# ---------------------------------------------------------------------------
# Main estimation functions
# ---------------------------------------------------------------------------


def estimate_time_complexity(
    func_node: ast.FunctionDef,
    func_name: str | None = None,
) -> ComplexityEstimate:
    """
    Estimate time complexity of a single function node.

    Algorithm (in priority order):
    1. Anti-pattern detectors (most actionable signal)
    2. Recursion without memoization → exponential
    3. Recursion with memoization / halving → O(n log n) or better
    4. Loop nesting depth → O(n), O(n²), O(n³)
    5. Default → O(1)
    """
    patterns: list[str] = []
    worst_time = ComplexityClass.O1
    time_reason = "no loops or recursion detected"

    # 1. Anti-pattern scan
    for detector in _ALL_TIME_DETECTORS:
        result = detector(func_node)
        if result:
            tag, cls = result
            patterns.append(tag)
            if cls > worst_time:
                worst_time = cls
                time_reason = tag

    # 2 & 3. Recursion detection
    recursive_calls = _count_recursive_calls(func_node)
    if recursive_calls > 0:
        if _has_memoization(func_node):
            cls = ComplexityClass.ONLOGN
            tag = "memoized_recursion → O(n log n) or better"
            patterns.append(tag)
            if cls > worst_time:
                worst_time = cls
                time_reason = tag
        elif recursive_calls >= 2:
            cls = ComplexityClass.ONEXP
            tag = f"recursive_{recursive_calls}_branches_no_memo → add @lru_cache"
            patterns.append(tag)
            if cls > worst_time:
                worst_time = cls
                time_reason = tag
        else:
            if _has_halving_pattern(func_node):
                cls = ComplexityClass.OLOGN
                tag = "tail_recursion_with_halving → O(log n)"
            else:
                cls = ComplexityClass.ON
                tag = "single_recursion → O(n)"
            patterns.append(tag)
            if cls > worst_time:
                worst_time = cls
                time_reason = tag

    # 4. Loop nesting (only if no worse pattern already found)
    if worst_time < ComplexityClass.ON2:
        nesting = _max_loop_nesting(func_node)
        if nesting == 1:
            # Check for O(n log n): sort or halving at the direct function body level
            has_top_level_sort = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, (ast.Name, ast.Attribute))
                and (
                    (isinstance(n.func, ast.Name) and n.func.id == "sorted")
                    or (
                        isinstance(n.func, ast.Attribute)
                        and n.func.attr in ("sort", "sorted")
                    )
                )
                for n in ast.iter_child_nodes(func_node)
            )
            if has_top_level_sort or _has_halving_pattern(func_node):
                cls = ComplexityClass.ONLOGN
                tag = "single_loop_with_sort_or_halving → O(n log n)"
            else:
                cls = ComplexityClass.ON
                tag = "single_loop → O(n)"
            patterns.append(tag)
            if cls > worst_time:
                worst_time = cls
                time_reason = tag
        elif nesting == 2:
            worst_time = ComplexityClass.ON2
            time_reason = "2_nested_loops → O(n²)"
            patterns.append(time_reason)
        elif nesting >= 3:
            worst_time = ComplexityClass.ON3
            time_reason = f"{nesting}_nested_loops → O(n³) or worse"
            patterns.append(time_reason)

    # Space complexity
    worst_space = ComplexityClass.O1
    space_reason = "in-place or constant extra space"

    for detector in _ALL_SPACE_DETECTORS:
        result = detector(func_node)
        if result:
            tag, cls = result
            if cls > worst_space:
                worst_space = cls
                space_reason = tag

    # Any comprehension that builds a collection → at least O(n) space
    for node in ast.walk(func_node):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            if worst_space < ComplexityClass.ON:
                worst_space = ComplexityClass.ON
                space_reason = "list/set/dict comprehension → O(n) space"
            break

    # Recursive without memoization → O(depth) call stack space
    if recursive_calls > 0 and not _has_memoization(func_node):
        if worst_space < ComplexityClass.ON:
            worst_space = ComplexityClass.ON
            space_reason = "recursion_stack → O(depth) space"

    return ComplexityEstimate(
        time=worst_time,
        space=worst_space,
        time_reason=time_reason,
        space_reason=space_reason,
        patterns_found=patterns,
    )


def estimate_space_complexity(func_node: ast.FunctionDef) -> ComplexityClass:
    """Convenience wrapper — returns just the space ComplexityClass."""
    return estimate_time_complexity(func_node).space


def estimate_repo_complexity(
    modules: dict[str, ast.Module],
    exclude_test_files: bool = True,
) -> dict[str, ComplexityEstimate]:
    """
    Return {qualified_function_name: ComplexityEstimate} for all functions
    in the repository. Used by ComplexityGrader alongside radon cc output.
    """
    from .ast_utils import _module_name_from_path

    results: dict[str, ComplexityEstimate] = {}
    for rel_path, module in modules.items():
        if exclude_test_files:
            base = Path(rel_path).name
            if base.startswith("test_") or base.endswith("_test.py"):
                continue
        mod_name = _module_name_from_path(rel_path)
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = f"{mod_name}.{node.name}"
                results[qname] = estimate_time_complexity(node, func_name=qname)
    return results
