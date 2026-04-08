from pydantic import BaseModel, Field

from .primitives import FileTreeEntry


# ---------------------------------------------------------------------------
# Observation contexts
# ---------------------------------------------------------------------------
class CodebaseContext(BaseModel):
    """Current view into the sandbox filesystem."""

    file_tree: list[FileTreeEntry] = Field(default_factory=list)
    active_file: str | None = None
    file_content: str | None = None
    file_line_start: int | None = None
    file_line_end: int | None = None
    total_file_lines: int | None = None


class ExecutionContext(BaseModel):
    """Raw output from the most recently executed action."""

    command: str | None = None
    stdout: str | None = None  # stdout, truncated to 8 KB
    stderr: str | None = None  # stderr, truncated to 8 KB
    return_code: int | None = None
    timed_out: bool = False
    run_error: str | None = None  # error message if applicable


class GraderContext(BaseModel):
    """Aggregated results from all graders after the last step.
    Populated from GradeResult fields — raw metrics and deltas stay
    inside the graders and never reach the agent.
    """

    scores: dict[str, float] = Field(default_factory=dict)  # grader_name -> score
    is_regression: bool = False
    feedbacks: list[str] = Field(default_factory=list)  # feedback, stdout
    errors: list[str] = Field(default_factory=list)  # errors, stderr
    tool_errors: list[str] = Field(
        default_factory=list
    )  # any tool error, grader didn't run
    penalties: list[str] = Field(
        default_factory=list
    )  # penalities due to added violations


class GitStatus(BaseModel):
    staged_files: list[str] = Field(default_factory=list)
    unstaged_files: list[str] = Field(default_factory=list)
    untracked_files: list[str] = Field(default_factory=list)
    diff_stat: str | None = Field(None)

    @property
    def has_changes(self) -> bool:
        """True if any tracked or untracked changes exist in the sandbox."""
        return bool(
            self.staged_files
            or self.unstaged_files
            or self.untracked_files
            or self.diff_stat
        )


class RewardContext(BaseModel):
    """Step-level reward signal surfaced to the agent."""

    step_score: float | None = Field(None, ge=0.0, le=1.0)
    cumulative_penalty: float = Field(0.0, ge=0.0)
