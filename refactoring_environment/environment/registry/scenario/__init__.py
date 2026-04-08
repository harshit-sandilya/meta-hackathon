"""
environment/registry/scenario.py

Typed, immutable scenario configuration loaded from scenario.yaml.

This module only parses task metadata/config:
- identity
- grader config
- penalties
- efficiency config
- invariants
- test paths

Those concerns belong to the registry/environment layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ....models.grader_spec import GraderSpec
from .invariant import AnyInvariant, build_invariant


@dataclass(frozen=True)
class EffConfig:
    """
    Step-efficiency decay parameters.

        eff = max(0.0, 1.0 − (steps_used / max_steps) · decay_rate)

    decay_rate = 0.5 means reaching max_steps gives eff = 0.5, not 0.0,
    so any completion is rewarded over silent timeout.
    """

    decay_rate: float = 0.5

    @classmethod
    def from_raw(cls, raw: dict | None) -> EffConfig:
        raw = raw or {}
        return cls(decay_rate=float(raw.get("decay_rate", 0.5)))


@dataclass(frozen=True)
class PenaltyConfig:
    """
    Penalty magnitudes used later by reward calculation.
    """

    syntax_error: float = 0.30
    repeated_noop: float = 0.10
    broken_import: float = 0.20
    test_regression: float = 0.25

    @classmethod
    def from_raw(cls, raw: dict | None) -> PenaltyConfig:
        raw = raw or {}
        return cls(
            syntax_error=float(raw.get("syntax_error", 0.30)),
            repeated_noop=float(raw.get("repeated_noop", 0.10)),
            broken_import=float(raw.get("broken_import", 0.20)),
            test_regression=float(raw.get("test_regression", 0.25)),
        )


@dataclass(frozen=True)
class ScenarioSpec:
    """
    Immutable task specification parsed from scenario.yaml.
    """

    slug: str
    name: str
    description: str

    language: str
    max_steps: int

    graders: dict[str, GraderSpec]
    penalties: PenaltyConfig
    eff_config: EffConfig
    invariants: list[AnyInvariant]
    test_paths: list[str]

    @property
    def task_id(self) -> str:
        return self.slug

    def active_graders(self) -> dict[str, GraderSpec]:
        return {
            grader_id: spec
            for grader_id, spec in self.graders.items()
            if spec.weight > 0.0
        }

    def check_invariants(self, sandbox_root: str, git_diff_output: str) -> list[str]:
        violations: list[str] = []
        for invariant in self.invariants:
            violations.extend(invariant.check(sandbox_root, git_diff_output))
        return violations

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> ScenarioSpec:
        path = Path(yaml_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"scenario.yaml not found: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"scenario.yaml did not parse to a dict: {path}")

        graders_raw = raw.get("graders", {})
        if not isinstance(graders_raw, dict):
            raise ValueError("`graders` must be a mapping in scenario.yaml")

        invariants_raw = raw.get("invariants", [])
        if not isinstance(invariants_raw, list):
            raise ValueError("`invariants` must be a list in scenario.yaml")

        test_paths_raw = raw.get("test_paths", ["tests/"])
        if not isinstance(test_paths_raw, list):
            raise ValueError("`test_paths` must be a list in scenario.yaml")

        return cls(
            slug=str(raw["slug"]),
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            language=str(raw.get("language", "python")),
            max_steps=int(raw.get("max_steps", 20)),
            graders={
                grader_id: GraderSpec.from_raw(entry)
                for grader_id, entry in graders_raw.items()
            },
            penalties=PenaltyConfig.from_raw(raw.get("penalties")),
            eff_config=EffConfig.from_raw(raw.get("eff_config")),
            invariants=[build_invariant(item) for item in invariants_raw],
            test_paths=[str(p) for p in test_paths_raw],
        )
