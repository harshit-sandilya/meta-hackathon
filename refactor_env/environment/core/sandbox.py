"""
sandbox.py  —  Git-backed isolated workspace for one RL episode.

Lifecycle
---------
  1. init(task_files)     — create tempdir, write files, git init + baseline commit
  2. [actions run]        — read_file, file_tree, git_diff, apply_atomic_patches
  3. checkpoint(msg)      — optional mid-episode commit for rollback
  4. reset_to_baseline()  — hard reset to step-0 state (episode restart)
  5. teardown()           — delete tempdir, mark as dead

Design notes (drawn from OpenSandbox and Claude Code patterns)
--------------------------------------------------------------
  - All paths are normalised through _safe_path() before any I/O.
    This is the single chokepoint for path traversal defence.
  - Patches are validated in a dry-run pass before any file is touched,
    so apply_atomic_patches is truly atomic on the filesystem level.
  - Git operations are run via Executor (not raw subprocess) so they go
    through the same timeout / logging path as agent-issued commands.
  - file_tree() skips .git/, __pycache__/, *.pyc, .DS_Store automatically.
  - The sandbox holds a reference to the Executor it was given; callers
    inject it (dependency injection) so tests can pass a MockExecutor.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import textwrap
import time
import uuid
from io import StringIO
from pathlib import Path
from typing import List, Optional

import unidiff

from refactor_env.models import FilePatch, FileTreeEntry
from .exceptions import (
    FileNotFoundInSandboxError,
    FileTooLargeError,
    GitDiffError,
    GitInitError,
    PatchApplyError,
    PathTraversalError,
    SandboxAlreadyInitializedError,
    SandboxNotInitializedError,
    SandboxTeardownError,
    GitError,
)
from .executor import Executor

# ── Constants ────────────────────────────────────────────────────────────────

FILE_SIZE_LIMIT: int = 512 * 1024  # 512 KB per read
MAX_TREE_ENTRIES: int = 500
_SKIP_DIRS: frozenset = frozenset(
    {".git", "__pycache__", ".mypy_cache", ".ruff_cache", "node_modules"}
)
_SKIP_EXTS: frozenset = frozenset({".pyc", ".pyo", ".DS_Store"})

# Git identity used for baseline commit (consistent, deterministic)
_GIT_AUTHOR_NAME = "refactor-env"
_GIT_AUTHOR_EMAIL = "env@refactor-env.local"


class Sandbox:
    """
    Manages one isolated, git-tracked workspace for a single RL episode.

    Parameters
    ----------
    executor:
        Injected Executor instance.  The sandbox sets its own root as
        the executor's sandbox_root after init().
    episode_id:
        Stable identifier carried through from RefactorState.
        If None, a fresh UUID is generated.
    """

    def __init__(
        self,
        executor: Optional[Executor] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        self._executor: Optional[Executor] = executor
        self._episode_id: str = episode_id or str(uuid.uuid4())
        self._root: Optional[str] = None
        self._baseline_commit: Optional[str] = None
        self._alive: bool = False

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def root(self) -> str:
        self._assert_alive()
        return self._root  # type: ignore[return-value]

    @property
    def baseline_commit(self) -> str:
        self._assert_alive()
        return self._baseline_commit  # type: ignore[return-value]

    @property
    def episode_id(self) -> str:
        return self._episode_id

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def init(self, task_files: dict[str, str]) -> str:
        """
        Materialise task_files, git-init, and record the baseline commit.

        Parameters
        ----------
        task_files:
            Mapping of relative path → file content (str).
            Directories are created as needed.

        Returns
        -------
        str
            SHA of the baseline (initial) commit.
        """
        if self._alive:
            raise SandboxAlreadyInitializedError(
                f"Sandbox for episode '{self._episode_id}' is already initialised."
            )

        # 1. Create temp directory
        prefix = f"refactor_env_{self._episode_id[:8]}_"
        self._root = tempfile.mkdtemp(prefix=prefix)

        # 2. Write task files
        for rel_path, content in task_files.items():
            abs_path = self._safe_path(rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            Path(abs_path).write_text(content, encoding="utf-8")

        # 3. Create Executor scoped to this root
        self._executor = Executor(sandbox_root=self._root)

        # 4. Git init + baseline commit
        self._baseline_commit = self._git_init_and_commit()
        self._alive = True
        return self._baseline_commit

    def teardown(self) -> None:
        """Delete sandbox directory and mark as dead. Safe to call multiple times."""
        if not self._root or not os.path.exists(self._root):
            self._alive = False
            return
        try:
            shutil.rmtree(self._root, onerror=_force_remove_readonly)
        except Exception as exc:
            raise SandboxTeardownError(
                f"Failed to remove sandbox at '{self._root}': {exc}"
            ) from exc
        finally:
            self._alive = False
            self._root = None
            self._baseline_commit = None

    # ── Read operations ───────────────────────────────────────────────────────

    def read_file(
        self,
        path: str,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
    ) -> str:
        """
        Read *path* relative to sandbox root, optionally sliced to
        [line_start, line_end] (1-indexed, inclusive).
        """
        self._assert_alive()
        abs_path = self._safe_path(path)

        if not os.path.isfile(abs_path):
            raise FileNotFoundInSandboxError(path)

        size = os.path.getsize(abs_path)
        if size > FILE_SIZE_LIMIT:
            raise FileTooLargeError(path, size, FILE_SIZE_LIMIT)

        content = Path(abs_path).read_text(encoding="utf-8", errors="replace")

        if line_start is not None or line_end is not None:
            lines = content.splitlines(keepends=True)
            lo = (line_start - 1) if line_start else 0
            hi = line_end if line_end else len(lines)
            content = "".join(lines[lo:hi])

        return content

    def file_tree(
        self,
        path: str = ".",
        *,
        recursive: bool = True,
        max_depth: int = 4,
    ) -> List[FileTreeEntry]:
        """
        Return sorted FileTreeEntry list for every node under *path*.
        """
        self._assert_alive()
        base = self._safe_path(path)
        entries: List[FileTreeEntry] = []
        self._walk(
            base, base, depth=0, max_depth=max_depth, recursive=recursive, out=entries
        )
        return sorted(entries, key=lambda e: (e.is_dir, e.path))[:MAX_TREE_ENTRIES]

    def git_diff(
        self,
        paths: Optional[List[str]] = None,
        *,
        stat_only: bool = False,
    ) -> str:
        """
        Return unified diff (or --stat) against the baseline commit.
        """
        self._assert_alive()
        flag = "--stat" if stat_only else "--unified=3"
        path_args = ""
        if paths:
            safe_paths = [self._safe_path(p) for p in paths]
            path_args = " -- " + " ".join(f'"{p}"' for p in safe_paths)
        cmd = f"git diff {flag} {self._baseline_commit}{path_args}"
        rc, stdout, stderr = self._executor.run(  # type: ignore[union-attr]
            cmd, workdir=self._root, timeout_sec=15
        )
        if rc != 0:
            raise GitDiffError(f"git diff failed: {stderr}")
        return stdout

    # ── Write operations ──────────────────────────────────────────────────────

    def apply_atomic_patches(self, patches: List[FilePatch]) -> List[str]:
        """
        Apply all patches atomically.

        Pass 1 — validate every patch.
        Pass 2 — apply all patches.
        No files are touched if Pass 1 fails for any patch.
        """
        self._assert_alive()

        # ── Pass 1: Validate ────────────────────────────────────────────────
        validated: list[tuple[str, Optional[str]]] = (
            []
        )  # (abs_path, new_text | None→unified)
        for patch in patches:
            abs_path = self._safe_path(patch.path)
            if patch.new_content is not None:
                validated.append((abs_path, patch.new_content))
            else:
                # unified_diff — parse and dry-run
                assert patch.unified_diff is not None
                new_text = self._apply_unified_diff(
                    abs_path, patch.path, patch.unified_diff, dry_run=True
                )
                validated.append((abs_path, new_text))

        # ── Pass 2: Write ───────────────────────────────────────────────────
        modified: List[str] = []
        for (abs_path, new_text), patch in zip(validated, patches):
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            Path(abs_path).write_text(new_text or "", encoding="utf-8")
            modified.append(patch.path)

        return modified

    def reset_to_baseline(self) -> None:
        """Hard reset to baseline commit — wipes all agent edits."""
        self._assert_alive()
        rc, _, stderr = self._executor.run(  # type: ignore[union-attr]
            f"git reset --hard {self._baseline_commit}",
            workdir=self._root,
            timeout_sec=15,
        )
        if rc != 0:
            raise GitError(f"reset failed: {stderr}")
        # Also clean untracked files
        self._executor.run("git clean -fdq", workdir=self._root, timeout_sec=10)

    def checkpoint(self, message: str = "checkpoint") -> str:
        """Commit current state and return new SHA."""
        self._assert_alive()
        self._executor.run(  # type: ignore[union-attr]
            "git add -A", workdir=self._root, timeout_sec=10
        )
        ts = int(time.time())
        safe_msg = message.replace('"', "'")
        self._executor.run(
            f'git commit --allow-empty -m "{safe_msg} [{ts}]"',
            workdir=self._root,
            timeout_sec=10,
        )
        rc, sha, _ = self._executor.run(
            "git rev-parse HEAD", workdir=self._root, timeout_sec=5
        )
        return sha.strip()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _safe_path(self, rel_path: str) -> str:
        """
        Resolve *rel_path* against sandbox root and verify it does not
        escape.  This is the single chokepoint for path traversal defence.
        """
        root = self._root or ""
        candidate = os.path.realpath(os.path.join(root, rel_path))
        if not candidate.startswith(os.path.realpath(root)):
            raise PathTraversalError(rel_path)
        return candidate

    def _git_init_and_commit(self) -> str:
        """Initialise git repo and create the baseline commit. Returns SHA."""
        exec_ = self._executor
        assert exec_ is not None

        git_env = {
            "GIT_AUTHOR_NAME": _GIT_AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": _GIT_AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": _GIT_AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": _GIT_AUTHOR_EMAIL,
        }

        steps = [
            "git init -q",
            f'git config user.email "{_GIT_AUTHOR_EMAIL}"',
            f'git config user.name "{_GIT_AUTHOR_NAME}"',
            "git add -A",
            'git commit -q -m "baseline: task files loaded"',
        ]
        for cmd in steps:
            rc, _, stderr = exec_.run(
                cmd, workdir=self._root, env_extra=git_env, timeout_sec=15
            )
            if rc != 0:
                raise GitInitError(f"'{cmd}' failed: {stderr}")

        rc, sha, stderr = exec_.run(
            "git rev-parse HEAD", workdir=self._root, timeout_sec=5
        )
        if rc != 0:
            raise GitInitError(f"Could not read HEAD SHA: {stderr}")
        return sha.strip()

    def _apply_unified_diff(
        self,
        abs_path: str,
        rel_path: str,
        diff_text: str,
        *,
        dry_run: bool = False,
    ) -> str:
        """
        Apply a unified diff string to an existing file.
        Uses the `unidiff` library to parse hunks, then applies them
        line-by-line.  Raises PatchApplyError on mismatch.

        Returns the new file content as a string.
        """
        # Read original (may not exist for new-file patches)
        if os.path.isfile(abs_path):
            original = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        else:
            original = ""

        try:
            patch_set = unidiff.PatchSet(StringIO(diff_text))
        except Exception as exc:
            raise PatchApplyError(rel_path, f"could not parse diff: {exc}") from exc

        if not patch_set:
            raise PatchApplyError(rel_path, "diff is empty or has no hunks")

        lines = original.splitlines(keepends=True)
        patched: list[str] = list(lines)

        # Apply hunks in reverse order (so line numbers stay valid)
        for patched_file in patch_set:
            hunks = list(patched_file)
            for hunk in reversed(hunks):
                source_start = hunk.source_start - 1  # 0-indexed
                source_len = hunk.source_length
                # Validate context matches
                context_lines = [
                    line.value for line in hunk if line.is_context or line.is_removed
                ]
                original_slice = [
                    l for l in patched[source_start : source_start + source_len]
                ]
                if context_lines and "".join(context_lines) != "".join(original_slice):
                    raise PatchApplyError(
                        rel_path, f"hunk context mismatch at line {hunk.source_start}"
                    )
                # Build replacement
                new_lines = [
                    line.value for line in hunk if line.is_context or line.is_added
                ]
                patched[source_start : source_start + source_len] = new_lines

        result = "".join(patched)
        return result

    def _walk(
        self,
        base: str,
        current: str,
        depth: int,
        max_depth: int,
        recursive: bool,
        out: list,
    ) -> None:
        """Recursive directory walker producing FileTreeEntry objects."""
        if depth > max_depth or len(out) >= MAX_TREE_ENTRIES:
            return
        try:
            entries = sorted(
                os.scandir(current), key=lambda e: (not e.is_dir(), e.name)
            )
        except PermissionError:
            return

        for entry in entries:
            if entry.name in _SKIP_DIRS:
                continue
            _, ext = os.path.splitext(entry.name)
            if ext in _SKIP_EXTS:
                continue

            rel = os.path.relpath(entry.path, base)
            if entry.is_dir(follow_symlinks=False):
                out.append(FileTreeEntry(path=rel, is_dir=True))
                if recursive:
                    self._walk(base, entry.path, depth + 1, max_depth, recursive, out)
            else:
                size = entry.stat(follow_symlinks=False).st_size
                mtime = str(int(entry.stat(follow_symlinks=False).st_mtime))
                out.append(
                    FileTreeEntry(
                        path=rel, is_dir=False, size_bytes=size, last_modified=mtime
                    )
                )

    def _assert_alive(self) -> None:
        if not self._alive or self._root is None:
            raise SandboxNotInitializedError(
                f"Sandbox for episode '{self._episode_id}' is not initialised. Call init() first."
            )


# ── Module-level helper ───────────────────────────────────────────────────────


def _force_remove_readonly(func, path, _excinfo):
    """Error handler for shutil.rmtree on Windows read-only files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)
