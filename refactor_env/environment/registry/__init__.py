"""
environment/registry  —  Task discovery, loading, and scenario contracts.

Public exports:
  TaskRegistry  — discovers and loads task scenarios from disk
  ScenarioSpec  — immutable description of one refactoring task
  RewardWeights — per-task component weighting
  PenaltyConfig — per-task penalty magnitudes
"""

from .loader import TaskRegistry
from .scenario import (
    ScenarioSpec,
    RewardWeights,
    PenaltyConfig,
    GraderConfig,
    LintConfig,
    EffConfig,
)
from .invariant import (
    AnyInvariant,
    NoEditInvariant,
    FileExistsInvariant,
    NoDeleteInvariant,
    SymbolPresentInvariant,
    build_invariant,
)

__all__ = [
    "TaskRegistry",
    "ScenarioSpec",
    "RewardWeights",
    "PenaltyConfig",
    "AnyInvariant",
    "NoEditInvariant",
    "FileExistsInvariant",
    "NoDeleteInvariant",
    "SymbolPresentInvariant",
    "build_invariant",
    "GraderConfig",
    "LintConfig",
    "EffConfig",
]
