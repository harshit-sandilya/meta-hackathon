from __future__ import annotations

from ....models.grader_spec import GraderSpec
from ...sandbox.files import FileHandler
from ...sandbox.runner import ShellExecutor
from .base import BaseGrader, GradeResult


class MockGrader(BaseGrader):
    """
    Placeholder grader used until a real implementation is wired.
    Always returns score=0.5 with a clear warning in feedbacks.
    _compute_metrics is a no-op — no subprocess calls.
    """

    grader_id = "_mock"  # not a real scenario.yaml key — never appears in registry

    def _compute_metrics(self) -> dict:
        return {}

    def grade(self) -> GradeResult:
        return GradeResult(
            score=0.5,
            feedbacks=[
                f"[MOCK] {self._name!r} grader not yet implemented — returning 0.5"
            ],
            errors=[],
            tool_errors=[],
            added_violations=0.0,
        )

    # set by registry when building a named mock so feedbacks identify which grader it stands in for
    _name: str = "_mock"


def make_mock_grader(
    name: str,
    spec: GraderSpec,
    executor: ShellExecutor,
    file_handler: FileHandler,
) -> MockGrader:
    g = MockGrader(spec=spec, executor=executor, file_handler=file_handler)
    object.__setattr__(g, "_name", name)  # frozen-safe write — MockGrader is not frozen
    return g
