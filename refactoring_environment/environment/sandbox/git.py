"""
refactoring_environment/environment/sandbox/_git.py

GitHandler
----------
Owns all git operations inside the sandbox.
Every step commits → no untracked/unstaged files ever exist mid-episode.
status() is used internally to guard against empty commits.
"""

from __future__ import annotations

from pathlib import Path

from ...models.actions import GitDiffParams, RunShellParams
from ...models.observations import GitStatus
from .runner import ShellExecutor

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "refenv",
    "GIT_AUTHOR_EMAIL": "refenv@local",
    "GIT_COMMITTER_NAME": "refenv",
    "GIT_COMMITTER_EMAIL": "refenv@local",
}


class GitHandler:
    def __init__(self, root: Path, executor: ShellExecutor) -> None:
        """
        Initialise a fresh git repo in the sandbox root.
        Stages everything, commits as "reset", tags the commit "reset".
        """
        self._root = root
        self._executor = executor
        steps = [
            "git init -q -b main",
            "git add -A",
            "git commit -q -m 'reset'",
            "git tag reset",
        ]
        for cmd in steps:
            self._run(cmd)

        sha = self._head_sha()
        self._baseline_commit = sha
        self._current_commit = sha

    # ── Public API ────────────────────────────────────────────────────

    def commit(self, step: int) -> str | None:
        """
        Stage all changes and commit as 'step {N}'.
        Returns the new SHA, or None if there was nothing to commit.
        """
        if not self.status().has_changes:
            return None

        self._run("git add -A")
        self._run(f"git commit -q -m 'step {step}'")
        sha = self._head_sha()
        self._current_commit = sha
        return sha

    def diff(self, params: GitDiffParams) -> str:
        """
        Diff from the 'reset' tag to HEAD.
        Optionally scoped to specific paths.
        Optionally --stat only.
        """
        stat_flag = "--stat" if params.stat_only else ""
        paths_str = (
            "-- " + " ".join(f"'{p}'" for p in params.paths) if params.paths else ""
        )
        cmd = f"git diff reset HEAD {stat_flag} {paths_str}".strip()
        result = self._run(cmd)
        return result.stdout or ""

    def status(self) -> GitStatus:
        """
        Parse `git status --porcelain` into a GitStatus.
        Since every step commits, this mainly guards the commit() call.
        """
        result = self._run("git status --porcelain")
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []

        for line in (result.stdout or "").splitlines():
            if len(line) < 4:
                continue
            index_flag = line[0]
            worktree_flag = line[1]
            path = line[3:]

            if index_flag == "?" and worktree_flag == "?":
                untracked.append(path)
            else:
                if index_flag not in (" ", "?"):
                    staged.append(path)
                if worktree_flag not in (" ", "?"):
                    unstaged.append(path)

        # diff --stat for the summary line
        stat_result = self._run("git diff --stat HEAD")
        diff_stat = (stat_result.stdout or "").strip()

        return GitStatus(
            staged_files=staged,
            unstaged_files=unstaged,
            untracked_files=untracked,
            diff_stat=diff_stat,
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def baseline_commit(self) -> str:
        if self._baseline_commit is None:
            raise RuntimeError("GitHandler.init() has not been called yet.")
        return self._baseline_commit

    @property
    def current_commit(self) -> str:
        if self._current_commit is None:
            raise RuntimeError("GitHandler.init() has not been called yet.")
        return self._current_commit

    # ── Private ───────────────────────────────────────────────────────

    def _run(self, command: str) -> any:
        """Internal git run — never surfaces to agent ExecutionContext."""
        return self._executor.run(
            RunShellParams(command=command, timeout_sec=30, workdir="."),
            env_extra=_GIT_ENV,
        )

    def _head_sha(self) -> str:
        result = self._executor.run(
            RunShellParams(command="git rev-parse HEAD", timeout_sec=10, workdir="."),
            env_extra=_GIT_ENV,
        )
        return result.stdout.strip() if result.stdout else ""

    def __repr__(self) -> str:
        return (
            f"GitHandler("
            f"baseline={self._baseline_commit!r}, "
            f"current={self._current_commit!r})"
        )
