"""
refactoring_environment/environment/graders/registry.py

Grader registry — maps grader_id → weight for the active scenario.
No grader instances yet; this is the single source of truth for IDs.
"""

from __future__ import annotations

import warnings

from ...models_internal.grader_spec import GraderSpec
from ..sandbox.files import FileHandler
from ..sandbox.runner import ShellExecutor
from .types import (
    BaseGrader,
    make_mock_grader,
    LintGrader,
    SymbolGrader,
    StyleGrader,
    CoverageGrader,
    ComplexityGrader,
)

# All known grader IDs in the system.
# Extend this tuple as graders are implemented.
GRADER_REGISTRY: dict[str, type[BaseGrader]] = {
    "lint": LintGrader,
    "symbol": SymbolGrader,
    "style": StyleGrader,
    "coverage": CoverageGrader,
    "complexity": ComplexityGrader,
}


def build_grader(
    grader_id: str,
    spec: GraderSpec,
    executor: ShellExecutor,
    file_handler: FileHandler,
) -> BaseGrader:
    """
    Instantiate the grader for grader_id.
    Falls back to MockGrader with a warning when the real class is not yet wired.
    Raises KeyError for completely unknown IDs.
    """
    if grader_id not in GRADER_REGISTRY:
        raise KeyError(
            f"Unknown grader_id {grader_id!r}. " f"Valid IDs: {list(GRADER_REGISTRY)}"
        )

    cls = GRADER_REGISTRY[grader_id]

    if cls is None:
        warnings.warn(
            f"Grader {grader_id!r} is not yet implemented — using MockGrader.",
            stacklevel=2,
        )
        return make_mock_grader(grader_id, spec, executor, file_handler)

    return cls(spec=spec, executor=executor, file_handler=file_handler)


def parse_graders(
    scenario_graders: dict,
    executor: ShellExecutor,
    file_handler: FileHandler,
) -> list[BaseGrader]:
    """
    Parse the `graders:` block from scenario.yaml into an ordered list of
    BaseGrader instances, sorted descending by weight.
    Unknown grader IDs emit a warning and are skipped.
    """
    graders: list[BaseGrader] = []

    for grader_id, spec in scenario_graders.items():
        try:
            grader = build_grader(grader_id, spec, executor, file_handler)
            graders.append(grader)
        except KeyError as exc:
            warnings.warn(str(exc), stacklevel=2)
            continue

    return sorted(graders, key=lambda g: g.spec.weight, reverse=True)
