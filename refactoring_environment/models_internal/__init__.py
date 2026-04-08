from typing import Any

from openenv.core.env_server import Action, Observation, State
from pydantic import Field, model_validator

from .actions import _PARAMS_MAP, ActionParams, ActionType
from .observations import (
    CodebaseContext,
    ExecutionContext,
    GitStatus,
    GraderContext,
    RewardContext,
)


class RefactorAction(Action):
    action_type: ActionType
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_params(self) -> "RefactorAction":
        expected_cls = _PARAMS_MAP[self.action_type]
        if self.action_type == ActionType.submit and not self.params:
            self.params = {}
        try:
            validated = expected_cls(**self.params).model_dump()
            object.__setattr__(self, "params", validated)
        except Exception as exc:
            raise ValueError(
                f"Invalid params for action_type='{self.action_type}': {exc}"
            ) from exc
        return self

    @property
    def typed_params(self) -> ActionParams:
        cls = _PARAMS_MAP[self.action_type]
        return cls(**self.params)


class RefactorObservation(Observation):
    # Task identity
    episode_id: str
    task_id: str
    description: str | None = None

    # Step budget
    current_step: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    remaining_steps: int = Field(..., ge=0)

    # Contexts
    codebase: CodebaseContext = Field(default_factory=CodebaseContext)
    execution: ExecutionContext = Field(default_factory=ExecutionContext)
    grader: GraderContext = Field(default_factory=GraderContext)
    git: GitStatus = Field(default_factory=GitStatus)
    reward_context: RewardContext = Field(default_factory=RewardContext)


class RefactorState(State):
    task_id: str
    description: str | None = None
    done: bool = False

    sandbox_path: str
    baseline_commit: str

    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    last_metrics: dict[str, Any] = Field(default_factory=dict)
    best_metrics: dict[str, Any] = Field(default_factory=dict)

    action_history: list[dict[str, Any]] = Field(default_factory=list)
    accumulated_penalty: float = 0.0
    violations: list[str] = Field(default_factory=list)


__all__ = [
    "RefactorAction",
    "RefactorObservation",
    "RefactorState",
]
