"""
protocols.py  —  Abstract interfaces (Protocols) for the sandbox layer.

Using typing.Protocol instead of ABC gives structural subtyping:
any class with the right methods satisfies the protocol without
explicit inheritance.  This makes mocking in tests trivial.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from refactor_env.models import FilePatch, FileTreeEntry


@runtime_checkable
class ExecutorProtocol(Protocol):
    """Safe subprocess runner."""

    def run(
        self,
        command: str,
        *,
        workdir: Optional[str] = None,
        timeout_sec: int = 30,
        check: bool = False,
    ) -> tuple[int, str, str]:
        """
        Run *command* in a shell.

        Returns (returncode, stdout, stderr).
        Raises CommandNotAllowedError before spawning if binary is blocked.
        Raises ExecutorTimeoutError on timeout.
        Raises ExecutorError if check=True and returncode != 0.
        """
        ...


@runtime_checkable
class SandboxProtocol(Protocol):
    """Git-backed, isolated workspace for one RL episode."""

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def init(self, task_files: dict[str, str]) -> str:
        """
        Materialise task_files into a fresh temp directory, git-init it,
        and record the baseline commit SHA.

        task_files: {relative_path: file_content}
        Returns: baseline commit SHA.
        Raises: SandboxAlreadyInitializedError, GitInitError.
        """
        ...

    def teardown(self) -> None:
        """
        Delete the sandbox directory and release all resources.
        Safe to call multiple times.
        Raises: SandboxTeardownError on hard failures.
        """
        ...

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def root(self) -> str:
        """Absolute path to the sandbox root directory."""
        ...

    @property
    def baseline_commit(self) -> str:
        """SHA of the initial commit (set during init)."""
        ...

    # ── Read operations ─────────────────────────────────────────────────────

    def read_file(
        self,
        path: str,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
    ) -> str:
        """
        Read *path* (relative to sandbox root), optionally sliced.
        Raises: PathTraversalError, FileNotFoundInSandboxError, FileTooLargeError.
        """
        ...

    def file_tree(
        self,
        path: str = ".",
        *,
        recursive: bool = True,
        max_depth: int = 4,
    ) -> List["FileTreeEntry"]:
        """
        Return a sorted list of FileTreeEntry for every node under *path*.
        Skips .git/, __pycache__/, and *.pyc automatically.
        """
        ...

    def git_diff(
        self,
        paths: Optional[List[str]] = None,
        *,
        stat_only: bool = False,
    ) -> str:
        """
        Return unified diff (or --stat) against the baseline commit.
        Empty string if no changes.
        """
        ...

    # ── Write operations ────────────────────────────────────────────────────

    def apply_atomic_patches(self, patches: List["FilePatch"]) -> List[str]:
        """
        Apply a list of FilePatch objects atomically:
          - unified_diff  → patch(1) applied via in-process difflib
          - new_content   → direct file write (creates dirs as needed)

        All patches are validated before any write.  If any validation
        fails, no files are touched.

        Returns: list of relative paths that were modified.
        Raises: PatchApplyError on the first patch that fails.
        """
        ...

    def reset_to_baseline(self) -> None:
        """
        Hard-reset the working tree to the baseline commit.
        Used to restart an episode without re-init.
        """
        ...

    def checkpoint(self, message: str = "checkpoint") -> str:
        """
        Commit the current working tree state and return the new SHA.
        Useful for rollback points during an episode.
        """
        ...
