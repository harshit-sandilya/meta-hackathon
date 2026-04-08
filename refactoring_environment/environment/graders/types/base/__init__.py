from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .....models.actions import RunShellParams
from .....models.grader_spec import GraderSpec
from ....sandbox.files import FileHandler
from ....sandbox.runner import ShellExecutor
from .grade_result import GradeResult


class BaseGrader(ABC):
    grader_id: ClassVar[str]

    def __init__(
        self,
        spec: GraderSpec,
        executor: ShellExecutor,
        file_handler: FileHandler,
    ) -> None:
        self.spec = spec
        self.executor = executor
        self.file_handler = file_handler
        self._baseline = self._compute_metrics()

    # ── Internal — subclasses implement both ──────────────────────────────────

    @abstractmethod
    def _compute_metrics(self) -> dict:
        """
        Run analysis tools against the current sandbox state.
        Called once in __init__ to establish baseline, then again inside grade().
        Has side effects — spawns subprocesses, reads files.
        """

    @abstractmethod
    def grade(self) -> GradeResult:
        """
        Compute current metrics, score delta from self._baseline, return GradeResult.
        This is the only public method — no arguments, no external state needed.
        """

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _delta_score(baseline: float, current: float) -> float:
        """Standard proportional improvement score, clamped to [0, 1]."""
        if baseline <= 0:
            return 1.0
        return max(0.0, min(1.0, (baseline - current) / baseline))


__all__ = ["BaseGrader", "GradeResult"]
