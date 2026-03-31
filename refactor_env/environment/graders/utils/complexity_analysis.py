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
from typing import Optional


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

    def _depth(n: ast.AST, current: int) -> int:
        if isinstance(n, (ast.For, ast.While)):
            return max(_depth(child, current + 1) for child in ast.walk(n))
        depths = [_depth(child, current) for child in ast.iter_child_nodes(n)]
        return max(depths, default=current)

    result = _depth(node, 0)
    # Subtract 1 because the root contributes 0 loops
    return max(0, result)


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
        # @lru_cache(maxsize=...) call form
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
    - recursive call with argument of the form `arr[mid:]` or `len(x) // 2`
    """
    for node in ast.walk(func_node):
        # i //= 2 or i >>= 1
        if isinstance(node, ast.AugAssign):
            if isinstance(node.op, (ast.FloorDiv, ast.RShift)):
                if isinstance(node.value, ast.Constant) and node.value.value in (1, 2):
                    return True
        # bisect.bisect_left(...) etc
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith(
                "bisect"
            ):
                return True
            if isinstance(node.func, ast.Name) and node.func.id.startswith("bisect"):
                return True
        # Slice with //2
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            for sub in ast.walk(node.slice):
                if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.FloorDiv):
                    return True
    return False


# ---------------------------------------------------------------------------
# Anti-pattern detection — each returns (found: bool, tag: str, worst_class)
# ---------------------------------------------------------------------------


def _detect_str_concat_in_loop(
    func_node: ast.FunctionDef,
) -> Optional[tuple[str, ComplexityClass]]:
    """str += ... inside a loop → O(n²) due to string copies."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.AugAssign)
                    and isinstance(child.op, ast.Add)
                    and isinstance(child.target, ast.Name)
                ):
                    # Heuristic: if the right side is a Str or Name, likely str concat
                    return (
                        "str_concat_in_loop → use ''.join(parts)",
                        ComplexityClass.ON2,
                    )
    return None


def _detect_list_search_in_loop(
    func_node: ast.FunctionDef,
) -> Optional[tuple[str, ComplexityClass]]:
    """x in list or list.index(x) inside a loop → O(n²)."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                # `x in some_var` where some_var is not a known set/dict
                if isinstance(child, ast.Compare):
                    for op in child.ops:
                        if isinstance(op, ast.In):
                            return (
                                "linear_search_in_loop → convert to set()",
                                ComplexityClass.ON2,
                            )
                # `.index(x)` call
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
) -> Optional[tuple[str, ComplexityClass]]:
    """sorted() or .sort() called inside a loop → O(n² log n)."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Name) and func.id == "sorted":
                        return (
                            "sort_in_loop → sort once outside loop",
                            ComplexityClass.ON2,
                        )
                    if isinstance(func, ast.Attribute) and func.attr == "sort":
                        return (
                            "sort_in_loop → sort once outside loop",
                            ComplexityClass.ON2,
                        )
    return None


def _detect_list_insert_head_in_loop(
    func_node: ast.FunctionDef,
) -> Optional[tuple[str, ComplexityClass]]:
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
                        "list.insert(0, x)_in_loop → use collections.deque.appendleft()",
                        ComplexityClass.ON2,
                    )
    return None


def _detect_nested_comprehension(
    func_node: ast.FunctionDef,
) -> Optional[tuple[str, ComplexityClass]]:
    """Nested list comprehension with condition → O(n²)."""
    for node in ast.walk(func_node):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            if len(node.generators) >= 2:
                return ("nested_comprehension → O(n²) or worse", ComplexityClass.ON2)
    return None


# Space complexity patterns
def _detect_full_load(
    func_node: ast.FunctionDef,
) -> Optional[tuple[str, ComplexityClass]]:
    """Accumulating all results into a list inside a loop → O(n) space (fine) or O(n²)."""
    # Detect building list-of-lists (nested accumulation)
    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "append"
                    and child.args
                ):
                    # If arg is a list comprehension or another list → O(n²) space
                    if isinstance(child.args[0], (ast.ListComp, ast.List)):
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
    func_name: Optional[str] = None,
) -> ComplexityEstimate:
    """
    Estimate time complexity of a single function node.

    Algorithm (in priority order):
    1. Check anti-pattern detectors (most actionable signal)
    2. Check recursion without memoization → exponential
    3. Check recursion with memoization / halving → O(n log n)
    4. Check loop nesting depth → O(n), O(n²), O(n³)
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

    # 2. Recursion detection
    recursive_calls = _count_recursive_calls(func_node)
    if recursive_calls > 0:
        if _has_memoization(func_node):
            # Memoized recursion: typically O(n) or O(n log n)
            cls = ComplexityClass.ONLOGN
            tag = "memoized_recursion → O(n log n) or better"
            patterns.append(tag)
            if cls > worst_time:
                worst_time = cls
                time_reason = tag
        elif recursive_calls >= 2:
            # Multiple recursive branches without memo → likely exponential
            cls = ComplexityClass.ONEXP
            tag = f"recursive_{recursive_calls}_branches_no_memo → add @lru_cache"
            patterns.append(tag)
            if cls > worst_time:
                worst_time = cls
                time_reason = tag
        else:
            # Single recursive call → check for halving (log n) vs linear
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

    # 3. Loop nesting (only if no worse pattern already found from anti-patterns)
    if worst_time < ComplexityClass.ON2:
        nesting = _max_loop_nesting(func_node)
        if nesting == 1:
            # Check for O(n log n) — sort call at top level (not inside loop)
            has_sort = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, (ast.Name, ast.Attribute))
                and (
                    (isinstance(n.func, ast.Name) and n.func.id == "sorted")
                    or (isinstance(n.func, ast.Attribute) and n.func.attr == "sort")
                )
                for n in ast.walk(func_node)
                if not any(
                    isinstance(parent, (ast.For, ast.While))
                    for parent in ast.walk(func_node)
                )
            )
            if has_sort or _has_halving_pattern(func_node):
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
            if worst_time < ComplexityClass.ON2:
                worst_time = ComplexityClass.ON2
                time_reason = "2_nested_loops → O(n²)"
                patterns.append(time_reason)
        elif nesting >= 3:
            if worst_time < ComplexityClass.ON3:
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

    # Basic space: any list comprehension or accumulator → at least O(n)
    for node in ast.walk(func_node):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            if worst_space < ComplexityClass.ON:
                worst_space = ComplexityClass.ON
                space_reason = "list/set/dict comprehension → O(n) space"
            break

    # Recursive without TCO → O(depth) stack space
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
    in the repository. Used by ComplexityGrader.
    """
    from .ast_utils import _module_name_from_path

    results: dict[str, ComplexityEstimate] = {}
    for rel_path, module in modules.items():
        if exclude_test_files:
            import os

            base = os.path.basename(rel_path)
            if base.startswith("test_") or base.endswith("_test.py"):
                continue
        mod_name = _module_name_from_path(rel_path)
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = f"{mod_name}.{node.name}"
                results[qname] = estimate_time_complexity(node, func_name=qname)
    return results
