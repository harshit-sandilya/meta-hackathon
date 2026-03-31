"""
environment/registry/graders/__init__.py

Public surface of the graders sub-package.

Exports
───────
  BaseGrader          — abstract base every grader inherits from
  GradeResult         — typed return value of BaseGrader.grade()

  # Concrete graders
  LintGrader          — ruff / flake8 error counts          (lint-cleanup)
  SymbolGrader        — function / class rename invariants   (api-rename)
  CoverageGrader      — pytest-cov branch + line coverage    (test-coverage)
  StructureGrader     — target module layout enforcement     (module-decompose)
  StyleGrader         — cyclomatic complexity + radon        (style-enforce)
  DuplicationGrader   — Type I/II/III clone detection (DRY)  (module-decompose, style-enforce)
  ProductionGrader    — 10-check production-hygiene suite    (style-enforce)

  # Registry helpers
  GRADER_REGISTRY     — dict[str, type[BaseGrader]]  grader_id → class
  get_grader          — factory: grader_id → BaseGrader instance
  get_graders_for_scenario
                      — scenario config → list[BaseGrader]  (ordered by weight)

Usage
─────
  # Direct instantiation
  from environment.registry.graders import LintGrader
  grader = LintGrader()
  metrics  = grader.compute_metrics(repo_path, config, cache)
  result   = grader.grade(baseline_metrics, metrics, config)

  # Via factory (preferred in environment.py)
  from environment.registry.graders import get_graders_for_scenario
  graders = get_graders_for_scenario(scenario_config)
  for g in graders:
      m = g.compute_metrics(sandbox_path, scenario_config, cache)
      r = g.grade(baseline[g.grader_id], m, scenario_config)

  # ID-based lookup (useful in tests)
  from environment.registry.graders import get_grader, GRADER_REGISTRY
  assert "lint" in GRADER_REGISTRY
  g = get_grader("duplication")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# ── Base types ─────────────────────────────────────────────────────────────
from .base_grader import BaseGrader, GradeResult

# ── Concrete graders ───────────────────────────────────────────────────────
from .lint_grader import LintGrader
from .symbol_grader import SymbolGrader
from .coverage_grader import CoverageGrader
from .structure_grader import StructureGrader
from .style_grader import StyleGrader
from .duplication_grader import DuplicationGrader
from .production_grader import ProductionGrader

if TYPE_CHECKING:
    # Avoid circular import at runtime; used only for type annotations
    from typing import Any


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADER_REGISTRY: dict[str, type[BaseGrader]] = {
    LintGrader.grader_id: LintGrader,
    SymbolGrader.grader_id: SymbolGrader,
    CoverageGrader.grader_id: CoverageGrader,
    StructureGrader.grader_id: StructureGrader,
    StyleGrader.grader_id: StyleGrader,
    DuplicationGrader.grader_id: DuplicationGrader,
    ProductionGrader.grader_id: ProductionGrader,
}
"""
Mapping of grader_id → grader class.

grader_id values (canonical, match keys in scenario.yaml ``graders`` block):
  "lint"        LintGrader
  "symbol"      SymbolGrader
  "coverage"    CoverageGrader
  "structure"   StructureGrader
  "style"       StyleGrader
  "duplication" DuplicationGrader
  "production"  ProductionGrader
"""


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def get_grader(grader_id: str) -> BaseGrader:
    """
    Return a fresh instance of the grader identified by *grader_id*.

    Raises
    ------
    KeyError
        If *grader_id* is not registered.  The error message lists all
        valid ids so the caller (and agent feedback) is immediately
        actionable.

    Example
    -------
    >>> g = get_grader("lint")
    >>> isinstance(g, LintGrader)
    True
    """
    try:
        cls = GRADER_REGISTRY[grader_id]
    except KeyError:
        valid = ", ".join(sorted(GRADER_REGISTRY))
        raise KeyError(
            f"Unknown grader id {grader_id!r}. " f"Registered ids: {valid}"
        ) from None
    return cls()


def get_graders_for_scenario(config: "Any") -> list[BaseGrader]:
    """
    Return an ordered list of grader instances for the given scenario config.

    Resolution order
    ────────────────
    1.  Read ``config["graders"]`` — a dict whose keys are grader ids and
        whose values are per-grader config dicts (may include ``weight``).
    2.  Instantiate each grader that is listed AND present in
        GRADER_REGISTRY.  Unknown ids emit a warning but do not raise, so
        a misspelled id in scenario.yaml doesn't crash an episode.
    3.  Sort the resulting list by descending ``weight`` (from the per-grader
        config, defaulting to 0.0) so the highest-impact grader runs first.
        This matters for the ``MetricCache`` — heavy graders (coverage, style)
        populate shared cache entries that lighter graders may reuse.

    Parameters
    ----------
    config : dict
        Parsed scenario.yaml dict.  Must contain a top-level ``"graders"``
        key mapping grader_id → grader-specific config dict.

    Returns
    -------
    list[BaseGrader]
        Ordered by descending weight.  Empty list if ``config["graders"]``
        is absent or empty.

    Example
    -------
    Given scenario.yaml:

        graders:
          lint:
            weight: 0.30
          coverage:
            weight: 0.25
          production:
            weight: 0.20
          duplication:
            weight: 0.15
          style:
            weight: 0.10

    >>> graders = get_graders_for_scenario(config)
    >>> [g.grader_id for g in graders]
    ['lint', 'coverage', 'production', 'duplication', 'style']
    """
    import warnings

    graders_cfg: dict = config.get("graders", {})
    if not graders_cfg:
        return []

    instances: list[tuple[float, BaseGrader]] = []

    for grader_id, grader_cfg in graders_cfg.items():
        if grader_id not in GRADER_REGISTRY:
            warnings.warn(
                f"scenario.yaml references unknown grader {grader_id!r} — skipping. "
                f"Valid ids: {', '.join(sorted(GRADER_REGISTRY))}",
                stacklevel=2,
            )
            continue

        weight = float((grader_cfg or {}).get("weight", 0.0))
        instances.append((weight, GRADER_REGISTRY[grader_id]()))

    # Stable sort: descending weight, then alphabetical grader_id as tiebreak
    instances.sort(key=lambda t: (-t[0], t[1].grader_id))
    return [g for _, g in instances]


# ---------------------------------------------------------------------------
# Public API surface — explicit __all__ guards accidental re-exports
# ---------------------------------------------------------------------------

__all__ = [
    # Base types
    "BaseGrader",
    "GradeResult",
    # Concrete graders
    "LintGrader",
    "SymbolGrader",
    "CoverageGrader",
    "StructureGrader",
    "StyleGrader",
    "DuplicationGrader",
    "ProductionGrader",
    # Registry + factory
    "GRADER_REGISTRY",
    "get_grader",
    "get_graders_for_scenario",
]
