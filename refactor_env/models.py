"""
models.py  —  Public Pydantic contract for refactor-env
All types are importable directly from the package root via __init__.py.

Design philosophy (drawn from Claude Code / oh-my-claudecode):
  - Actions mirror Claude Code's tool-call surface:
      view_file, list_directory, search_codebase, git_diff,
      edit_file, edit_files, run_shell, submit
  - Observations carry the full context window a model needs at every step
    (file tree, active file window, test/lint summaries, git diff, step budget).
  - State is the environment's private ledger; never sent to the agent directly.
  - Reward is shaped and dense; partial credit at every step, not just on submit.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# 0.  Shared primitives
# ---------------------------------------------------------------------------


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class FileTreeEntry(BaseModel):
    path: str = Field(..., description="Relative path from sandbox root")
    is_dir: bool = Field(False)
    size_bytes: int = Field(0)
    last_modified: Optional[str] = Field(None)


class TestSummary(BaseModel):
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0

    @property
    def pass_ratio(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


class LintSummary(BaseModel):
    total_errors: int = 0
    error_by_code: Dict[str, int] = Field(default_factory=dict)
    type_errors: int = 0
    complexity_max: Optional[int] = Field(None)


class GitStatus(BaseModel):
    staged_files: List[str] = Field(default_factory=list)
    unstaged_files: List[str] = Field(default_factory=list)
    untracked_files: List[str] = Field(default_factory=list)
    diff_stat: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# 1.  ACTION
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    view_file = "view_file"
    list_directory = "list_directory"
    search_codebase = "search_codebase"
    git_diff = "git_diff"
    edit_file = "edit_file"
    edit_files = "edit_files"
    run_shell = "run_shell"
    submit = "submit"


class FilePatch(BaseModel):
    path: str
    unified_diff: Optional[str] = None
    new_content: Optional[str] = None

    @model_validator(mode="after")
    def exactly_one_change(self) -> "FilePatch":
        if self.unified_diff is None and self.new_content is None:
            raise ValueError("FilePatch must supply either unified_diff or new_content")
        if self.unified_diff is not None and self.new_content is not None:
            raise ValueError(
                "FilePatch must supply unified_diff OR new_content, not both"
            )
        return self


class ViewFileParams(BaseModel):
    path: str
    line_start: Optional[int] = Field(None, ge=1)
    line_end: Optional[int] = Field(None, ge=1)


class ListDirectoryParams(BaseModel):
    path: str = "."
    recursive: bool = False
    max_depth: int = Field(3, ge=1, le=8)


class SearchCodebaseParams(BaseModel):
    query: str = Field(..., min_length=1)
    file_glob: str = "*.py"
    case_insensitive: bool = False
    context_lines: int = Field(2, ge=0, le=10)
    max_results: int = Field(50, ge=1, le=200)


class GitDiffParams(BaseModel):
    paths: List[str] = Field(default_factory=list)
    stat_only: bool = False


class EditFileParams(BaseModel):
    patch: FilePatch


class EditFilesParams(BaseModel):
    patches: List[FilePatch] = Field(..., min_length=1, max_length=20)


ALLOWED_SHELL_COMMANDS: frozenset[str] = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "ruff",
        "mypy",
        "radon",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "ls",
        "wc",
        "git",
        "black",
        "isort",
    }
)


class RunShellParams(BaseModel):
    command: str
    timeout_sec: int = Field(30, ge=1, le=120)
    workdir: str = "."

    @model_validator(mode="after")
    def command_is_allowed(self) -> "RunShellParams":
        binary = self.command.strip().split()[0] if self.command.strip() else ""
        if binary not in ALLOWED_SHELL_COMMANDS:
            raise ValueError(
                f"Binary '{binary}' is not in the allowed command list. "
                f"Allowed: {sorted(ALLOWED_SHELL_COMMANDS)}"
            )
        return self


class SubmitParams(BaseModel):
    note: Optional[str] = Field(None, max_length=500)


ActionParams = Union[
    ViewFileParams,
    ListDirectoryParams,
    SearchCodebaseParams,
    GitDiffParams,
    EditFileParams,
    EditFilesParams,
    RunShellParams,
    SubmitParams,
]

_ACTION_TYPE_TO_PARAMS: Dict[ActionType, type] = {
    ActionType.view_file: ViewFileParams,
    ActionType.list_directory: ListDirectoryParams,
    ActionType.search_codebase: SearchCodebaseParams,
    ActionType.git_diff: GitDiffParams,
    ActionType.edit_file: EditFileParams,
    ActionType.edit_files: EditFilesParams,
    ActionType.run_shell: RunShellParams,
    ActionType.submit: SubmitParams,
}


class RefactorAction(BaseModel):
    action_type: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_params(self) -> "RefactorAction":
        expected_cls = _ACTION_TYPE_TO_PARAMS[self.action_type]
        if self.action_type == ActionType.submit and not self.params:
            self.params = {}
        try:
            validated = expected_cls(**self.params)
            self.params = validated.model_dump()
        except Exception as exc:
            raise ValueError(
                f"Invalid params for action_type='{self.action_type}': {exc}"
            ) from exc
        return self

    @property
    def typed_params(self) -> ActionParams:
        cls = _ACTION_TYPE_TO_PARAMS[self.action_type]
        return cls(**self.params)


# ---------------------------------------------------------------------------
# 2.  OBSERVATION
# ---------------------------------------------------------------------------


class RefactorObservation(BaseModel):
    # Task identity
    task_id: str
    difficulty: Difficulty
    objective: str
    constraints: List[str] = Field(default_factory=list)

    # Step budget
    current_step: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    remaining_steps: int = Field(..., ge=0)

    # Codebase context
    file_tree: List[FileTreeEntry] = Field(default_factory=list)
    active_file: Optional[str] = None
    file_content: Optional[str] = None
    file_line_start: Optional[int] = None
    file_line_end: Optional[int] = None
    total_file_lines: Optional[int] = None

    # Execution feedback
    test_summary: TestSummary = Field(default_factory=TestSummary)
    lint_summary: LintSummary = Field(default_factory=LintSummary)
    last_action_output: Optional[str] = None  # stdout+stderr, truncated to 8 KB
    last_action_error: Optional[str] = None

    # Change awareness
    git_status: GitStatus = Field(default_factory=GitStatus)
    baseline_test_summary: Optional[TestSummary] = None
    baseline_lint_summary: Optional[LintSummary] = None

    # Dense reward signal (so inference.py can include it in prompts)
    step_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    cumulative_penalty: float = Field(0.0, ge=0.0)
    violations: List[str] = Field(default_factory=list)

    # Meta
    episode_id: str


# ---------------------------------------------------------------------------
# 3.  STATE  (private ledger — /state endpoint only)
# ---------------------------------------------------------------------------


class RefactorState(BaseModel):
    episode_id: str
    task_id: str
    difficulty: Difficulty
    step_count: int = 0
    done: bool = False

    sandbox_path: str
    baseline_commit: str

    baseline_metrics: Dict[str, Any] = Field(default_factory=dict)
    last_metrics: Dict[str, Any] = Field(default_factory=dict)
    best_metrics: Dict[str, Any] = Field(default_factory=dict)

    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    consecutive_noop_count: int = 0
    accumulated_penalty: float = 0.0
    violations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4.  REWARD
# ---------------------------------------------------------------------------


class RewardBreakdown(BaseModel):
    # Components [0.0, 1.0]
    accuracy: float = Field(0.0, ge=0.0, le=1.0)
    quality: float = Field(0.0, ge=0.0, le=1.0)
    efficiency: float = Field(0.0, ge=0.0, le=1.0)
    format_ok: float = Field(0.0, ge=0.0, le=1.0)
    weighted_raw: float = Field(0.0, ge=0.0)

    # Penalty sub-fields (negative contributions)
    penalty_syntax_error: float = 0.0  # -0.30
    penalty_test_regression: float = 0.0  # -0.20
    penalty_noop_loop: float = 0.0  # -0.10 each, cap 0.30
    penalty_invariant_breach: float = 0.0  # -0.20 each, cap 0.40
    total_penalty: float = 0.0  # sum, cap 0.70


class RefactorReward(BaseModel):
    # OpenEnv required fields
    score: float = Field(..., ge=0.0, le=1.0)
    partial_credit: float = Field(..., ge=0.0, le=1.0)
    penalty: float = Field(0.0, ge=0.0, le=1.0)

    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    feedback: str = ""
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# 5.  Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "ActionType",
    "Difficulty",
    "FileTreeEntry",
    "TestSummary",
    "LintSummary",
    "GitStatus",
    "FilePatch",
    "ViewFileParams",
    "ListDirectoryParams",
    "SearchCodebaseParams",
    "GitDiffParams",
    "EditFileParams",
    "EditFilesParams",
    "RunShellParams",
    "SubmitParams",
    "ALLOWED_SHELL_COMMANDS",
    "RefactorAction",
    "RefactorObservation",
    "RefactorState",
    "RefactorReward",
    "RewardBreakdown",
]
