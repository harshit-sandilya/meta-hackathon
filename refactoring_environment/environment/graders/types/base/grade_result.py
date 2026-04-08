from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GradeResult:
    score: float  # 0.0–1.0 normalised improvement over baseline
    feedbacks: list[str]  # stdout — progress messages, improvement summaries
    errors: list[str]  # stderr — agent-caused (syntax errors, test failures)
    tool_errors: list[str]  # grader-caused (tool missing, subprocess timeout)
    added_violations: int = 0  # regression delta — fed to penalty layer in reward.py
