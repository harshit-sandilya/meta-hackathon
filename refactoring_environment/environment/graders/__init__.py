"""
refactoring_environment/environment/graders/__init__.py
"""

from __future__ import annotations

import logging

from ...models.observations import GraderContext, RewardContext
from ..registry.scenario import ScenarioSpec
from ..sandbox.files import FileHandler
from ..sandbox.runner import ShellExecutor
from .registry import parse_graders
from .types.base import BaseGrader, GradeResult

logger = logging.getLogger(__name__)


class GraderDispatcher:
    """
    Holds the active grader list for one episode and runs them on demand.

    sandbox wires it in at SandboxEnv.__init__:
        self.graders = GraderDispatcher(scenario)

    Then calls on every edit:
        grader_ctx, reward_ctx = self.graders.grade(step, cumulative_penalty)
    """

    def __init__(
        self,
        scenario: ScenarioSpec,
        executor: ShellExecutor,
        file_handler: FileHandler,
    ) -> None:
        self._graders: list[BaseGrader] = parse_graders(
            scenario_graders=getattr(scenario, "graders", {}),
            executor=executor,
            file_handler=file_handler,
        )
        logger.info(
            "GraderDispatcher ready — task=%r  graders=%s",
            scenario.task_id,
            [(g.__class__.__name__, g.spec.weight) for g in self._graders],
        )

    # ── Public ────────────────────────────────────────────────────────

    def grade(
        self, step: int = 0, cumulative_penalty: float = 0.0
    ) -> tuple[GraderContext, RewardContext]:
        """Run all graders and return results in weight-descending order."""
        results: list[tuple[BaseGrader, GradeResult]] = []
        for grader in self._graders:
            result = grader.grade()
            logger.debug(
                "grader=%r  score=%.4f  tool_errors=%s",
                grader.__class__.__name__,
                result.score,
                result.tool_errors,
            )
            results.append((grader, result))

        scores: dict[str, float] = {}
        feedbacks: list[str] = []
        errors: list[str] = []
        tool_errors: list[str] = []
        penalties: list[float] = []

        for grader, result in results:
            gid = grader.grader_id
            scores[gid] = result.score
            feedbacks.extend(result.feedbacks)
            errors.extend(result.errors)
            tool_errors.extend(result.tool_errors)

            if result.added_violations > 0:
                penalties.append(str(result.added_violations))

        step_score = sum(g.spec.weight * scores[g.grader_id] for g, _ in results)
        is_regression = len(penalties) > 0

        grader_ctx = GraderContext(
            scores=scores,
            is_regression=is_regression,
            feedbacks=feedbacks,
            errors=errors,
            tool_errors=tool_errors,
            penalties=penalties,
        )
        reward_ctx = RewardContext(
            step_score=round(step_score, 4),
            cumulative_penalty=cumulative_penalty,
        )

        return grader_ctx, reward_ctx


__all__ = ["GraderDispatcher"]
