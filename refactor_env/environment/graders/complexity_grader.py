"""
Complexity Grader — multi-axis code complexity measurement.

Gold standard : every measured axis is at or below its target threshold.

This grader is the quantitative complement to StyleGrader. Where StyleGrader
checks *conventions*, ComplexityGrader checks *structural complexity* —
the things that make code hard to understand, test, and maintain regardless
of naming or formatting.

Axes measured
─────────────
  1. cyclomatic     — McCabe cyclomatic complexity per function
  2. cognitive      — Cognitive complexity (Sonar model) per function
  3. halstead       — Halstead volume, difficulty, effort per module
  4. maintainability— Maintainability Index (MI) per module
  5. loc            — Lines of Code: total, logical, comment ratio
  6. depth          — Max nesting depth per function
  7. fanin_fanout   — Module coupling: fan-in (dependents) + fan-out (imports)

Score formula
─────────────
For each axis a:
  axis_score[a] = fraction of functions/modules that are at or below threshold

final_score = Σ axis_weight[a] * axis_score[a]

Improvement is measured delta-from-baseline so the agent is rewarded
proportionally for reducing complexity even if gold isn't yet reached.

Scenario config keys (under config["graders"]["complexity"]):
  weight                  : float  — contribution to overall reward (default 0.10)
  axis_weights            : dict   — per-axis weight overrides
  thresholds              : dict   — per-axis threshold overrides
  exclude_patterns        : list   — file globs to skip
  ignore_axes             : list   — axes to skip, e.g. ["halstead", "fanin_fanout"]

Default thresholds (Google / industry standard baselines)
──────────────────────────────────────────────────────────
  cyclomatic      : 10    — McCabe; >10 is "complex", >15 is "untestable"
  cognitive       : 15    — Sonar; stricter than cyclomatic for nested logic
  halstead_volume : 1000  — bits of information in a module
  halstead_effort : 30000 — mental effort to understand a module
  maintainability : 20    — MI < 20 = unmaintainable (0–100 scale, higher better)
  loc_per_function: 50    — logical lines per function
  max_depth       : 4     — nesting depth; >4 is a code smell
  fanout          : 10    — distinct modules imported (coupling)
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base_grader import BaseGrader, GradeResult, _clamp, already_gold, empty_grade
from .utils import MetricCache, parse_repository


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AXIS_ORDER = [
    "cyclomatic",
    "cognitive",
    "halstead",
    "maintainability",
    "loc",
    "depth",
    "fanin_fanout",
]

_DEFAULT_AXIS_WEIGHTS: dict[str, float] = {
    "cyclomatic": 0.25,
    "cognitive": 0.20,
    "halstead": 0.15,
    "maintainability": 0.15,
    "loc": 0.10,
    "depth": 0.10,
    "fanin_fanout": 0.05,
}

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "cyclomatic": 10.0,
    "cognitive": 15.0,
    "halstead_volume": 1000.0,
    "halstead_effort": 30000.0,
    "maintainability": 20.0,  # lower bound — MI should be ABOVE this
    "loc_per_function": 50.0,
    "max_depth": 4.0,
    "fanout": 10.0,
}


# ---------------------------------------------------------------------------
# Dataclasses for intermediate results
# ---------------------------------------------------------------------------


@dataclass
class FunctionMetrics:
    name: str
    qualified: str
    file: str
    line: int
    cyclomatic: int = 0
    cognitive: int = 0
    loc: int = 0  # logical lines (non-blank, non-comment)
    max_depth: int = 0


@dataclass
class HalsteadMetrics:
    n1: int = 0  # distinct operators
    n2: int = 0  # distinct operands
    N1: int = 0  # total operators
    N2: int = 0  # total operands

    @property
    def vocabulary(self) -> int:
        return self.n1 + self.n2

    @property
    def length(self) -> int:
        return self.N1 + self.N2

    @property
    def volume(self) -> float:
        v = self.vocabulary
        if v <= 0:
            return 0.0
        return self.length * math.log2(v)

    @property
    def difficulty(self) -> float:
        if self.n2 == 0:
            return 0.0
        return (self.n1 / 2.0) * (self.N2 / self.n2)

    @property
    def effort(self) -> float:
        return self.difficulty * self.volume

    @property
    def time_sec(self) -> float:
        return self.effort / 18.0

    @property
    def bugs(self) -> float:
        return self.volume / 3000.0


@dataclass
class ModuleMetrics:
    file: str
    halstead: HalsteadMetrics = field(default_factory=HalsteadMetrics)
    mi: float = 0.0  # Maintainability Index
    total_loc: int = 0  # raw lines
    logical_loc: int = 0  # non-blank, non-comment lines
    comment_lines: int = 0
    fanout: int = 0  # distinct imported modules
    fanin: int = 0  # set externally after graph build


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

# Python operator token types for Halstead counting
_OPERATOR_NODE_TYPES = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.MatMult,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Invert,
    ast.UAdd,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.If,  # ternary counts as operator
    ast.IfExp,
    ast.comprehension,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Return,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.Delete,
    ast.Assert,
    ast.Raise,
)

_OPERAND_NODE_TYPES = (
    ast.Name,
    ast.Constant,
    ast.Attribute,
)

_BRANCH_NODES = (
    ast.If,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
)


def _cyclomatic(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """McCabe cyclomatic complexity: 1 + decision points."""
    cc = 1
    for node in ast.walk(func_node):
        if isinstance(node, _BRANCH_NODES):
            cc += 1
        elif isinstance(node, ast.BoolOp):
            cc += len(node.values) - 1
        elif isinstance(node, ast.comprehension) and node.ifs:
            cc += len(node.ifs)
    return cc


def _cognitive(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """
    Sonar cognitive complexity approximation.

    Rules (simplified):
      +1  per branching structure (if, for, while, try, with)
      +1  per nesting level for nested structures
      +1  per logical operator sequence break (&&, ||)
      +1  per recursion call
      No increment for else/elif/finally continuations
    """
    score = 0
    func_name = func_node.name

    def _walk(node: ast.AST, depth: int) -> None:
        nonlocal score
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.While,
                    ast.For,
                    ast.AsyncFor,
                    ast.Try,
                    ast.With,
                    ast.AsyncWith,
                    ast.ExceptHandler,
                ),
            ):
                score += 1 + depth
                _walk(child, depth + 1)
            elif isinstance(
                child,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Lambda,
                ),
            ):
                score += 1
                _walk(child, depth + 1)
            elif isinstance(child, ast.BoolOp):
                score += 1
                _walk(child, depth)
            elif isinstance(child, ast.comprehension):
                score += len(child.ifs)
                _walk(child, depth)
            elif isinstance(child, ast.Call):
                func = child.func
                called = getattr(func, "id", getattr(func, "attr", ""))
                if called == func_name:
                    score += 1  # recursion
                _walk(child, depth)
            else:
                _walk(child, depth)

    _walk(func_node, 0)
    return score


def _max_depth(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Maximum nesting depth inside a function."""
    _depth_nodes = (
        ast.If,
        ast.While,
        ast.For,
        ast.AsyncFor,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    def _walk(node: ast.AST, depth: int) -> int:
        max_d = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _depth_nodes):
                max_d = max(max_d, _walk(child, depth + 1))
            else:
                max_d = max(max_d, _walk(child, depth))
        return max_d

    return _walk(func_node, 0)


def _logical_loc(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count logical lines: AST statements inside the function body."""
    count = 0
    for node in ast.walk(func_node):
        if isinstance(node, ast.stmt) and node is not func_node:
            count += 1
    return count


def _halstead_for_module(tree: ast.Module) -> HalsteadMetrics:
    """Count Halstead operators and operands for a whole module."""
    hm = HalsteadMetrics()

    operators_seen: set[type] = set()
    operands_seen: set[str] = set()

    for node in ast.walk(tree):
        for op_type in _OPERATOR_NODE_TYPES:
            if isinstance(node, op_type):
                hm.N1 += 1
                operators_seen.add(type(node))
                break

        if isinstance(node, ast.Name):
            hm.N2 += 1
            operands_seen.add(node.id)
        elif isinstance(node, ast.Constant):
            hm.N2 += 1
            operands_seen.add(repr(node.value))
        elif isinstance(node, ast.Attribute):
            hm.N2 += 1
            operands_seen.add(node.attr)

    hm.n1 = len(operators_seen)
    hm.n2 = len(operands_seen)
    return hm


def _maintainability_index(
    halstead_volume: float,
    cyclomatic_avg: float,
    loc: int,
    comment_ratio: float,
) -> float:
    """
    Classic Maintainability Index formula (SEI variant, 0–100 scale).

    MI = max(0, (171
                 - 5.2 * ln(HV)
                 - 0.23 * CC
                 - 16.2 * ln(LOC)
                 + 50 * sin(√(2.4 * comment_ratio))
               ) * 100 / 171)

    Higher is better. Below 20 = unmaintainable.
    """
    hv = max(halstead_volume, 1.0)
    l = max(loc, 1)
    raw = (
        171.0
        - 5.2 * math.log(hv)
        - 0.23 * cyclomatic_avg
        - 16.2 * math.log(l)
        + 50.0 * math.sin(math.sqrt(2.4 * comment_ratio))
    )
    return max(0.0, raw * 100.0 / 171.0)


def _fanout_for_module(tree: ast.Module) -> int:
    """Count distinct top-level module names imported."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return len(modules)


def _loc_stats(source: str) -> tuple[int, int, int]:
    """Returns (total_lines, logical_lines, comment_lines)."""
    lines = source.splitlines()
    total = len(lines)
    logical = 0
    comment_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment_count += 1
        else:
            logical += 1
    return total, logical, comment_count


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


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------


def _analyse_file(
    rel_path: str,
    source: str,
) -> tuple[list[FunctionMetrics], ModuleMetrics]:
    """
    Full complexity analysis of a single .py file.
    Returns (function_metrics_list, module_metrics).
    """
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return [], ModuleMetrics(file=rel_path)

    func_metrics: list[FunctionMetrics] = []
    module_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    # Collect all functions/methods (one level deep for qualified name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_funcs.append(node)

    for func in module_funcs:
        # Build qualified name: walk parent chain via lineno heuristic
        # (full parent tracking requires a parent map)
        qualified = f"{rel_path}::{func.name}"

        fm = FunctionMetrics(
            name=func.name,
            qualified=qualified,
            file=rel_path,
            line=func.lineno,
        )
        fm.cyclomatic = _cyclomatic(func)
        fm.cognitive = _cognitive(func)
        fm.max_depth = _max_depth(func)
        fm.loc = _logical_loc(func)
        func_metrics.append(fm)

    # Module-level metrics
    total_loc, logical_loc, comment_lines = _loc_stats(source)
    halstead = _halstead_for_module(tree)
    fanout = _fanout_for_module(tree)

    # Average cyclomatic for MI calculation
    cc_avg = (
        sum(fm.cyclomatic for fm in func_metrics) / len(func_metrics)
        if func_metrics
        else 1.0
    )
    comment_ratio = comment_lines / max(total_loc, 1)
    mi = _maintainability_index(halstead.volume, cc_avg, logical_loc, comment_ratio)

    mm = ModuleMetrics(
        file=rel_path,
        halstead=halstead,
        mi=mi,
        total_loc=total_loc,
        logical_loc=logical_loc,
        comment_lines=comment_lines,
        fanout=fanout,
    )

    return func_metrics, mm


# ---------------------------------------------------------------------------
# Score computation helpers
# ---------------------------------------------------------------------------


def _axis_score_functions(
    metrics: list[FunctionMetrics],
    threshold: float,
    attr: str,
) -> tuple[float, list[dict]]:
    """
    Score an axis that operates on FunctionMetrics.
    Returns (score, list of over-threshold records).
    """
    if not metrics:
        return (1.0, [])

    over: list[dict] = []
    for fm in metrics:
        val = getattr(fm, attr, 0)
        if val > threshold:
            over.append(
                {
                    "file": fm.file,
                    "line": fm.line,
                    "name": fm.name,
                    "value": val,
                    "threshold": threshold,
                }
            )

    score = _clamp(1.0 - len(over) / len(metrics))
    return (score, over)


def _axis_score_modules(
    modules: list[ModuleMetrics],
    threshold: float,
    attr: str,
    higher_is_better: bool = False,
) -> tuple[float, list[dict]]:
    """
    Score an axis that operates on ModuleMetrics.
    If higher_is_better, we penalise modules *below* threshold.
    """
    if not modules:
        return (1.0, [])

    over: list[dict] = []
    for mm in modules:
        if attr.startswith("halstead."):
            prop = attr.split(".")[1]
            val = getattr(mm.halstead, prop, 0.0)
        else:
            val = getattr(mm, attr, 0.0)

        if higher_is_better:
            violated = val < threshold
        else:
            violated = val > threshold

        if violated:
            over.append(
                {
                    "file": mm.file,
                    "value": val,
                    "threshold": threshold,
                    "higher_is_better": higher_is_better,
                }
            )

    score = _clamp(1.0 - len(over) / len(modules))
    return (score, over)


def _merge_axis_weights(overrides: dict[str, float] | None) -> dict[str, float]:
    w = dict(_DEFAULT_AXIS_WEIGHTS)
    if overrides:
        w.update(overrides)
    total = sum(w.values())
    return (
        {k: v / total for k, v in w.items()}
        if total > 0
        else dict(_DEFAULT_AXIS_WEIGHTS)
    )


def _merge_thresholds(overrides: dict[str, float] | None) -> dict[str, float]:
    t = dict(_DEFAULT_THRESHOLDS)
    if overrides:
        t.update(overrides)
    return t


def _build_feedback(
    score: float,
    baseline_score: float,
    is_regression: bool,
    axis_scores: dict[str, float],
    hotspots: list[dict],
) -> str:
    if score >= 1.0:
        return "[Complexity] Gold standard — all axes within thresholds."

    if is_regression:
        return (
            f"[Complexity] Regression: score dropped "
            f"{baseline_score:.0%} → {score:.0%}. "
            f"A recent edit increased complexity — check cyclomatic / depth."
        )

    worst = sorted(axis_scores.items(), key=lambda kv: kv[1])[:2]
    worst_str = ", ".join(f"{a}={s:.0%}" for a, s in worst if s < 1.0)

    hot = hotspots[:3]
    hot_str = "; ".join(
        f"{h.get('name', h.get('file', '?'))}@{h.get('file','?')}:{h.get('line',0)} "
        f"({h.get('axis','?')}={h.get('value',0):.0f}>{h.get('threshold',0):.0f})"
        for h in hot
    )

    return (
        f"[Complexity] {score:.0%} of functions/modules within thresholds. "
        f"Weakest axes: {worst_str or 'none'}. "
        f"Top hotspots: {hot_str or 'none'}."
    )


def _top_hotspots(
    violations_by_axis: dict[str, list[dict]],
    axis_weights: dict[str, float],
    n: int = 10,
) -> list[dict]:
    """
    Flatten all violations into a ranked hotspot list.
    Rank by (axis_weight × normalised_excess).
    """
    flat: list[dict] = []
    for axis, viols in violations_by_axis.items():
        w = axis_weights.get(axis, 0.1)
        for v in viols:
            threshold = v.get("threshold", 1.0)
            value = v.get("value", 0.0)
            if threshold > 0:
                excess = (
                    (value - threshold) / threshold
                    if not v.get("higher_is_better")
                    else (threshold - value) / threshold
                )
            else:
                excess = 0.0
            flat.append({**v, "axis": axis, "weight_score": w * max(excess, 0.0)})

    return sorted(flat, key=lambda x: x["weight_score"], reverse=True)[:n]


# ---------------------------------------------------------------------------
# ComplexityGrader
# ---------------------------------------------------------------------------


class ComplexityGrader(BaseGrader):
    """
    Scores code complexity across seven axes using pure-AST analysis.

    Each axis produces a [0,1] score = fraction of symbols/modules within
    their threshold. Combined via configurable weighted sum.

    Delta-based: the reward is proportional to improvement from baseline,
    so the agent gets gradient signal even for partial progress.

    Axes:
      cyclomatic      — McCabe per function
      cognitive       — Sonar cognitive complexity per function
      halstead        — Halstead volume + effort per module
      maintainability — Maintainability Index per module (higher = better)
      loc             — Logical LOC per function
      depth           — Max nesting depth per function
      fanin_fanout    — Import fan-out per module
    """

    grader_id = "complexity"

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
        Analyse all non-test .py files and compute complexity metrics.

        Returns
        -------
        {
          "function_metrics" : list[dict]       — per-function records
          "module_metrics"   : list[dict]        — per-module records
          "axis_scores"      : dict[str, float]  — per-axis [0,1] scores
          "axis_violations"  : dict[str, list]   — per-axis over-threshold records
          "hotspots"         : list[dict]         — top ranked violations
          "weighted_score"   : float
          "files_checked"    : int
          "thresholds"       : dict
          "error"            : str|None
        }
        """
        repo_path = Path(repo_path)
        cplx_cfg = config.get("graders", {}).get("complexity", {})
        thresholds = _merge_thresholds(cplx_cfg.get("thresholds", None))
        ignore_axes: set[str] = set(cplx_cfg.get("ignore_axes", []))
        axis_weights = _merge_axis_weights(cplx_cfg.get("axis_weights", None))
        exclude_patterns: list[str] = cplx_cfg.get("exclude_patterns", []) + config.get(
            "exclude_patterns", []
        )

        def _run() -> tuple[list[FunctionMetrics], list[ModuleMetrics]]:
            all_funcs: list[FunctionMetrics] = []
            all_modules: list[ModuleMetrics] = []

            for fpath in sorted(repo_path.rglob("*.py")):
                rel_path = str(fpath.relative_to(repo_path))
                if _is_test_file(rel_path):
                    continue
                if any(
                    fpath.match(p) or rel_path.startswith(p.rstrip("/"))
                    for p in exclude_patterns
                ):
                    continue
                try:
                    source = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                funcs, mod = _analyse_file(rel_path, source)
                all_funcs.extend(funcs)
                all_modules.append(mod)

            # Fan-in: count how many other modules import each module
            # (simple heuristic: count occurrences of each file stem in import lists)
            module_stems = {Path(mm.file).stem: mm for mm in all_modules}
            for mm in all_modules:
                for other in all_modules:
                    if other.file == mm.file:
                        continue
                    # Check if this module imports mm via its stem
                    # (approximation — full resolution needs the dep graph)
                    if Path(mm.file).stem in [
                        Path(mm2.file).stem for mm2 in all_modules if mm2.fanout > 0
                    ]:
                        pass  # fan-in populated via dep graph in structure_grader
                # Fan-in is informational only; not scored directly

            return all_funcs, all_modules

        try:
            all_funcs, all_modules = cache.get_or_compute(
                "complexity_analysis", None, _run
            )
        except Exception as exc:
            return {
                "function_metrics": [],
                "module_metrics": [],
                "axis_scores": {a: 0.0 for a in _AXIS_ORDER},
                "axis_violations": {a: [] for a in _AXIS_ORDER},
                "hotspots": [],
                "weighted_score": 0.0,
                "files_checked": 0,
                "thresholds": thresholds,
                "error": str(exc),
            }

        # ── Per-axis scoring ───────────────────────────────────────────

        axis_scores: dict[str, float] = {}
        axis_violations: dict[str, list] = {}

        if "cyclomatic" not in ignore_axes:
            s, v = _axis_score_functions(
                all_funcs, thresholds["cyclomatic"], "cyclomatic"
            )
            axis_scores["cyclomatic"] = s
            axis_violations["cyclomatic"] = v
        else:
            axis_scores["cyclomatic"] = 1.0
            axis_violations["cyclomatic"] = []

        if "cognitive" not in ignore_axes:
            s, v = _axis_score_functions(
                all_funcs, thresholds["cognitive"], "cognitive"
            )
            axis_scores["cognitive"] = s
            axis_violations["cognitive"] = v
        else:
            axis_scores["cognitive"] = 1.0
            axis_violations["cognitive"] = []

        if "halstead" not in ignore_axes:
            sv, vv = _axis_score_modules(
                all_modules, thresholds["halstead_volume"], "halstead.volume"
            )
            se, ve = _axis_score_modules(
                all_modules, thresholds["halstead_effort"], "halstead.effort"
            )
            # Combine: score is min of volume + effort scores
            axis_scores["halstead"] = (sv + se) / 2.0
            axis_violations["halstead"] = vv + ve
        else:
            axis_scores["halstead"] = 1.0
            axis_violations["halstead"] = []

        if "maintainability" not in ignore_axes:
            s, v = _axis_score_modules(
                all_modules, thresholds["maintainability"], "mi", higher_is_better=True
            )
            axis_scores["maintainability"] = s
            axis_violations["maintainability"] = v
        else:
            axis_scores["maintainability"] = 1.0
            axis_violations["maintainability"] = []

        if "loc" not in ignore_axes:
            s, v = _axis_score_functions(
                all_funcs, thresholds["loc_per_function"], "loc"
            )
            axis_scores["loc"] = s
            axis_violations["loc"] = v
        else:
            axis_scores["loc"] = 1.0
            axis_violations["loc"] = []

        if "depth" not in ignore_axes:
            s, v = _axis_score_functions(
                all_funcs, thresholds["max_depth"], "max_depth"
            )
            axis_scores["depth"] = s
            axis_violations["depth"] = v
        else:
            axis_scores["depth"] = 1.0
            axis_violations["depth"] = []

        if "fanin_fanout" not in ignore_axes:
            s, v = _axis_score_modules(all_modules, thresholds["fanout"], "fanout")
            axis_scores["fanin_fanout"] = s
            axis_violations["fanin_fanout"] = v
        else:
            axis_scores["fanin_fanout"] = 1.0
            axis_violations["fanin_fanout"] = []

        weighted_score = _clamp(
            sum(axis_weights.get(a, 0.0) * axis_scores.get(a, 1.0) for a in _AXIS_ORDER)
        )

        hotspots = _top_hotspots(axis_violations, axis_weights, n=10)

        # Serialise dataclasses
        func_dicts = [
            {
                "name": fm.name,
                "qualified": fm.qualified,
                "file": fm.file,
                "line": fm.line,
                "cyclomatic": fm.cyclomatic,
                "cognitive": fm.cognitive,
                "loc": fm.loc,
                "max_depth": fm.max_depth,
            }
            for fm in all_funcs
        ]
        mod_dicts = [
            {
                "file": mm.file,
                "mi": round(mm.mi, 2),
                "total_loc": mm.total_loc,
                "logical_loc": mm.logical_loc,
                "comment_lines": mm.comment_lines,
                "fanout": mm.fanout,
                "halstead_volume": round(mm.halstead.volume, 2),
                "halstead_effort": round(mm.halstead.effort, 2),
                "halstead_bugs": round(mm.halstead.bugs, 4),
            }
            for mm in all_modules
        ]

        return {
            "function_metrics": func_dicts,
            "module_metrics": mod_dicts,
            "axis_scores": axis_scores,
            "axis_violations": axis_violations,
            "hotspots": hotspots,
            "weighted_score": weighted_score,
            "files_checked": len(all_modules),
            "thresholds": thresholds,
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
        cplx_cfg = config.get("graders", {}).get("complexity", {})
        axis_weights = _merge_axis_weights(cplx_cfg.get("axis_weights", None))
        thresholds = _merge_thresholds(cplx_cfg.get("thresholds", None))

        if current.get("error"):
            return empty_grade("Complexity", f"analysis failed: {current['error']}")

        if current.get("files_checked", 0) == 0:
            return empty_grade("Complexity", "no Python files found to check")

        b_score = baseline.get("weighted_score", 0.0)
        c_score = current.get("weighted_score", 0.0)

        if b_score >= 1.0:
            return already_gold("Complexity", current)

        is_regression = c_score < b_score - 1e-6

        # Final score: delta progress from baseline toward 1.0
        if b_score >= 1.0:
            final_score = c_score
        else:
            gap = 1.0 - b_score
            improvement = c_score - b_score
            final_score = _clamp(improvement / max(gap, 1e-9))

        # Regression cap
        if is_regression:
            final_score = 0.0

        solved = c_score >= 1.0 - 1e-9

        b_axis = baseline.get("axis_scores", {})
        c_axis = current.get("axis_scores", {})

        # Per-axis sub-scores (delta)
        sub_scores: dict[str, float] = {}
        for axis in _AXIS_ORDER:
            b_a = b_axis.get(axis, 0.0)
            c_a = c_axis.get(axis, 0.0)
            if b_a >= 1.0:
                sub_scores[f"axis:{axis}"] = c_a
            else:
                gap_a = 1.0 - b_a
                sub_scores[f"axis:{axis}"] = _clamp((c_a - b_a) / max(gap_a, 1e-9))

        hotspots = current.get("hotspots", [])

        feedback = _build_feedback(
            score=final_score,
            baseline_score=b_score,
            is_regression=is_regression,
            axis_scores=c_axis,
            hotspots=hotspots,
        )

        # Violation count deltas per axis
        b_viols = {a: len(v) for a, v in baseline.get("axis_violations", {}).items()}
        c_viols = {a: len(v) for a, v in current.get("axis_violations", {}).items()}
        delta_viols = {a: b_viols.get(a, 0) - c_viols.get(a, 0) for a in _AXIS_ORDER}

        return GradeResult(
            score=final_score,
            gold_distance=1.0 - c_score,
            raw_baseline={
                "weighted_score": b_score,
                "axis_scores": b_axis,
                "violation_counts": b_viols,
            },
            raw_current={
                "weighted_score": c_score,
                "axis_scores": c_axis,
                "violation_counts": c_viols,
                "files_checked": current.get("files_checked", 0),
            },
            delta={
                "weighted_score": c_score - b_score,
                "axis_violations": delta_viols,
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
        thresholds = _merge_thresholds(
            config.get("graders", {}).get("complexity", {}).get("thresholds", None)
        )
        return {
            "function_metrics": [],
            "module_metrics": [],
            "axis_scores": {a: 1.0 for a in _AXIS_ORDER},
            "axis_violations": {a: [] for a in _AXIS_ORDER},
            "hotspots": [],
            "weighted_score": 1.0,
            "files_checked": 1,
            "thresholds": thresholds,
            "error": None,
        }
