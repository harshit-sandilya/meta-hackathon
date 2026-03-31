"""
Git utilities for sandbox repos.

All functions operate on a git-initialized directory (the sandbox).
Each episode sandbox is a git repo — sandbox.py calls `git init && git add -A
&& git commit -m 'baseline'` at reset() so every edit can be diffed against
the original state.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(args: list[str], cwd: Path) -> tuple[str, str, int]:
    """Run a git command and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def get_changed_files(sandbox_path: Path) -> list[str]:
    """
    Return relative paths of all .py files modified since the baseline commit
    (the commit made at reset()).

    Combines staged + unstaged changes.
    """
    stdout, _, rc = _git(["diff", "--name-only", "HEAD"], sandbox_path)
    if rc != 0:
        return []
    return [f for f in stdout.splitlines() if f.endswith(".py") and f.strip()]


def get_diff(sandbox_path: Path, file_path: str) -> str:
    """Return unified diff for a single file relative to baseline commit."""
    stdout, _, _ = _git(["diff", "HEAD", "--", file_path], sandbox_path)
    return stdout


def get_full_diff(sandbox_path: Path) -> str:
    """Return unified diff of all changes since baseline commit."""
    stdout, _, _ = _git(["diff", "HEAD"], sandbox_path)
    return stdout


def get_added_files(sandbox_path: Path) -> list[str]:
    """
    Return relative paths of .py files that are new (untracked or staged-new)
    since the baseline commit.
    """
    # Untracked files
    stdout, _, _ = _git(["ls-files", "--others", "--exclude-standard"], sandbox_path)
    untracked = [f for f in stdout.splitlines() if f.endswith(".py")]

    # Staged new files
    stdout2, _, _ = _git(
        ["diff", "--name-only", "--diff-filter=A", "HEAD"], sandbox_path
    )
    staged_new = [f for f in stdout2.splitlines() if f.endswith(".py")]

    return list(set(untracked + staged_new))


def get_deleted_files(sandbox_path: Path) -> list[str]:
    """
    Return relative paths of .py files deleted since the baseline commit.
    """
    stdout, _, _ = _git(
        ["diff", "--name-only", "--diff-filter=D", "HEAD"], sandbox_path
    )
    return [f for f in stdout.splitlines() if f.endswith(".py")]


def files_modified_in_dirs(sandbox_path: Path, dirs: list[str]) -> list[str]:
    """
    Return files changed since baseline that fall under any of the given
    directory prefixes. Used by StructureGrader to enforce no_edit_files
    invariants (e.g. agent must not touch tests/).

    Parameters
    ----------
    dirs : list of directory prefixes, e.g. ["tests/", "migrations/"]
    """
    changed = get_changed_files(sandbox_path)
    added = get_added_files(sandbox_path)
    all_changed = changed + added

    violations: list[str] = []
    for f in all_changed:
        for d in dirs:
            # Normalize: both with and without trailing slash
            prefix = d.rstrip("/") + "/"
            if f.startswith(prefix) or f.startswith(d):
                violations.append(f)
                break
    return violations


def get_file_at_baseline(sandbox_path: Path, file_path: str) -> Optional[str]:
    """
    Return the content of file_path as it was at the baseline commit.
    Returns None if the file didn't exist at baseline.
    """
    stdout, _, rc = _git(["show", f"HEAD:{file_path}"], sandbox_path)
    if rc != 0:
        return None
    return stdout


def init_baseline_commit(sandbox_path: Path) -> bool:
    """
    Initialize a git repo and create the baseline commit.
    Called by sandbox.py at the end of reset().
    Returns True on success.
    """
    cmds = [
        ["init"],
        ["config", "user.email", "env@refactor.local"],
        ["config", "user.name", "RefactorEnv"],
        ["add", "-A"],
        ["commit", "-m", "baseline", "--allow-empty"],
    ]
    for cmd in cmds:
        _, _, rc = _git(cmd, sandbox_path)
        if rc != 0 and cmd[0] not in ("init",):
            return False
    return True


# Optional import guard — used in get_file_at_baseline
from typing import Optional
