"""
refactoring_environment/environment/sandbox/__init__.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ...environment.registry import ScenarioSpec
from ...models_internal import RefactorAction, RefactorObservation
from ...models_internal.actions import ActionType
from ...models_internal.observations import (
    CodebaseContext,
    ExecutionContext,
    GitStatus,
    GraderContext,
    RewardContext,
)
from ..graders import GraderDispatcher
from .files import FileHandler
from .filters import make_ignore_fn
from .git import GitHandler
from .runner import ShellExecutor


class SandboxEnv:
    """
    One instance per episode.
    """

    def __init__(
        self,
        repo_root: Path,
        scenario: ScenarioSpec,
        wall_timeout: int = 90,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.scenario = scenario
        self._wall_timeout = wall_timeout

        if not self.repo_root.exists():
            raise FileNotFoundError(f"repo_root does not exist: {self.repo_root}")

        # Copy repo → tmpdir
        self._tmpdir = tempfile.TemporaryDirectory(prefix="refenv_")
        self._root = (Path(self._tmpdir.name) / "repo").resolve()
        shutil.copytree(
            str(self.repo_root),
            str(self._root),
            ignore=make_ignore_fn(self.repo_root),
        )
        self.runner = ShellExecutor(self._root, wall_timeout=wall_timeout)
        self.git_manager = GitHandler(self._root, self.runner)
        self.file_handler = FileHandler(self._root, self.runner)
        self.graders = GraderDispatcher(
            scenario=self.scenario, executor=self.runner, file_handler=self.file_handler
        )

        self.baseline_metrics: dict[str, Any] = {}
        self.last_metrics: dict[str, Any] = {}
        self.best_metrics: dict[str, Any] = {}
        self.cumulative_penalty: float = 0.0

    def get_initial_observation(self) -> tuple[CodebaseContext, GitStatus]:
        return self.file_handler.context, self.git_manager.status()

    def act(
        self,
        action: RefactorAction,
        step: int,
        episode_id: str,
        task_id: str,
    ) -> RefactorObservation:
        """
        Dispatch action → handler → return fully assembled observation.
        edit_file / edit_files also commit to git and regrade.
        """
        params = action.typed_params
        action_type = action.action_type

        codebase = self.file_handler.context
        execution = ExecutionContext()
        grader = GraderContext()
        reward = RewardContext(cumulative_penalty=self.cumulative_penalty)

        if action_type == ActionType.view_file:
            codebase = self.file_handler.view_file(params)
        elif action_type == ActionType.list_directory:
            codebase = self.file_handler.list_dir(params)
        elif action_type == ActionType.search_codebase:
            codebase = self.file_handler.search(params)
        elif action_type == ActionType.git_diff:
            diff_output = self.git_manager.diff(params)
            execution = ExecutionContext(
                command="git diff",
                stdout=diff_output or "(no diff)",
            )
        elif action_type == ActionType.edit_file:
            codebase = self.file_handler.apply_patch(params)
            self.git_manager.commit(step)
            grader, reward = self.graders.grade(
                step=step, cumulative_penalty=self.cumulative_penalty
            )
        elif action_type == ActionType.edit_files:
            codebase = self.file_handler.apply_patches(params)
            self.git_manager.commit(step)
            grader, reward = self.graders.grade(
                step=step, cumulative_penalty=self.cumulative_penalty
            )
        elif action_type == ActionType.run_shell:
            execution = self.runner.run(params)

        return RefactorObservation(
            done=False,
            reward=reward.step_score,
            episode_id=episode_id,
            task_id=task_id,
            current_step=step,
            max_steps=self.scenario.max_steps,
            remaining_steps=self.scenario.max_steps - step,
            codebase=codebase,
            execution=execution,
            grader=grader,
            git=self.git_manager.status(),
            reward_context=reward,
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        return self._root

    @property
    def baseline_commit(self) -> str:
        return self.git_manager.baseline_commit

    # ── Lifecycle ────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Clean up the tmpdir. Call this at the start of each reset()."""
        self._tmpdir.cleanup()

    def __enter__(self) -> SandboxEnv:
        return self

    def __exit__(self, *_) -> None:
        self.destroy()
