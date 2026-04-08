import uuid

from openenv.core.env_server import Environment

from ..models import RefactorAction, RefactorObservation, RefactorState
from ..models.actions import GitDiffParams
from ..models.observations import ExecutionContext, GraderContext, RewardContext
from .registry import Registry
from .sandbox import SandboxEnv


class RefactorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._registry = Registry()
        self.sandbox: SandboxEnv | None = None

        # State variables
        self._episode_id: str | None = None
        self._task_id: str | None = None
        self._step_count: int = 0
        self._done: bool = False

    def reset(
        self, task_name: str = "lint-cleanup", episode_id: str | None = None
    ) -> RefactorObservation:
        if self.sandbox is not None:
            self.sandbox.destroy()

        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_id = task_name
        self._step_count = 0
        self._accumulated_penalty = 0.0
        self._done = False

        scenario = self._registry.load_scenario(task_name)
        repo_root = self._registry.repo_path(task_name)
        self.sandbox = SandboxEnv(repo_root=repo_root, scenario=scenario)

        codebase, git = self.sandbox.get_initial_observation()

        return RefactorObservation(
            done=False,
            reward=None,
            episode_id=self._episode_id,
            task_id=task_name,
            current_step=0,
            max_steps=scenario.max_steps,
            remaining_steps=scenario.max_steps,
            codebase=codebase,
            execution=ExecutionContext(),
            grader=GraderContext(),
            git=git,
            reward_context=RewardContext(),
        )

    def step(self, action: RefactorAction) -> RefactorObservation:
        self._require_reset()
        self._step_count += 1
        return self.sandbox.act(
            action,
            step=self._step_count,
            episode_id=self._episode_id,
            task_id=self._task_id,
        )

    @property
    def state(self) -> RefactorState:
        self._require_reset()
        action_history_diff = self.sandbox.git_manager.diff(GitDiffParams())
        return RefactorState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            done=self._done,
            sandbox_path=str(self.sandbox.root),
            baseline_commit=self.sandbox.baseline_commit,
            action_history=(
                [{"diff": action_history_diff}] if action_history_diff else []
            ),
            accumulated_penalty=self.sandbox.cumulative_penalty,
            violations=[],
        )

    def _require_reset(self) -> None:
        if self.sandbox is None or self._episode_id is None:
            raise RuntimeError("Call reset() before step() or state()")


__all__ = ["RefactorEnvironment"]
