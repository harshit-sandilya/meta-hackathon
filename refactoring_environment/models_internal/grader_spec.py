from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=False)
class GraderSpec:
    """
    Configuration for one grader entry under `graders:`.

    Supports both YAML forms:

        lint:
          weight: 0.50
          target_coverage: 1.00

    and shorthand:

        lint: 0.50
    """

    weight: float
    target_coverage: float = 0.80
    config: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict | float | int) -> GraderSpec:
        if isinstance(raw, (int, float)):
            return cls(weight=float(raw), target_coverage=0.80)

        return cls(
            weight=float(raw.get("weight", 0.0)),
            target_coverage=float(raw.get("target_coverage", 0.80)),
            config=raw.get("config", {}),
        )
