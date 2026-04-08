from enum import Enum

from pydantic import BaseModel, Field

from .primitives import FilePatch


# ---------------------------------------------------------------------------
# Action Types
# ---------------------------------------------------------------------------
class ActionType(Enum):
    view_file = "view_file"
    list_directory = "list_directory"
    search_codebase = "search_codebase"
    git_diff = "git_diff"
    edit_file = "edit_file"
    edit_files = "edit_files"
    run_shell = "run_shell"
    submit = "submit"


class ViewFileParams(BaseModel):
    path: str
    line_start: int | None = Field(None, ge=1)
    line_end: int | None = Field(None, ge=1)


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
    paths: list[str] = Field(default_factory=list)
    stat_only: bool = False


class EditFileParams(BaseModel):
    patch: FilePatch


class EditFilesParams(BaseModel):
    patches: list[FilePatch] = Field(..., min_length=1, max_length=20)


class RunShellParams(BaseModel):
    command: str
    timeout_sec: int = Field(30, ge=1, le=120)
    workdir: str = "."


class SubmitParams(BaseModel):
    note: str | None = Field(None, max_length=500)


ActionParams = (
    ViewFileParams
    | ListDirectoryParams
    | SearchCodebaseParams
    | GitDiffParams
    | EditFileParams
    | EditFilesParams
    | RunShellParams
    | SubmitParams
)

_PARAMS_MAP: dict[ActionType, type] = {
    ActionType.view_file: ViewFileParams,
    ActionType.list_directory: ListDirectoryParams,
    ActionType.search_codebase: SearchCodebaseParams,
    ActionType.git_diff: GitDiffParams,
    ActionType.edit_file: EditFileParams,
    ActionType.edit_files: EditFilesParams,
    ActionType.run_shell: RunShellParams,
    ActionType.submit: SubmitParams,
}
