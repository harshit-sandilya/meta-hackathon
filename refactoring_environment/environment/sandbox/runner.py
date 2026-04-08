"""
refactoring_environment/environment/sandbox/runner.py

ShellExecutor
-------------
Knows nothing about git, files, or the environment.
Takes a command string, runs it in the sandbox root, returns ExecutionContext.

Internal calls (is_internal=True) return a default ExecutionContext so the
agent's observation is never polluted with framework noise.
"""

from __future__ import annotations

import os
import platform
import resource
import subprocess
from pathlib import Path

from ...models_internal.actions import RunShellParams
from ...models_internal.observations import ExecutionContext

_1_GB = 1 * 1024**3
_64_MB = 64 * 1024**2
_8_KB = 8 * 1024
_IS_LINUX = platform.system() == "Linux"


def _apply_limits() -> None:
    """
    preexec_fn — only apply limits that are safe on the current platform.
    RLIMIT_AS and RLIMIT_NPROC are Linux-only; skip on macOS.
    """
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    except (OSError, ValueError):
        pass

    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (_64_MB, _64_MB))
    except (OSError, ValueError):
        pass

    if _IS_LINUX:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (_1_GB, _1_GB))
        except (OSError, ValueError):
            pass

        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (OSError, ValueError):
            pass

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    except (OSError, ValueError):
        pass


class ShellExecutor:
    """
    Singleton passed into GitHandler and FileHandler.
    All subprocess calls in the sandbox go through here.
    """

    def __init__(self, root: Path, wall_timeout: int = 90) -> None:
        self._root = root
        self._wall_timeout = wall_timeout

    # ── Public ────────────────────────────────────────────────────────

    def run(
        self,
        params: RunShellParams,
        env_extra: dict[str, str] | None = None,
    ) -> ExecutionContext:
        """
        Run params.command inside the sandbox.

        Parameters
        ----------
        params:
            RunShellParams — carries command, timeout_sec, workdir.
        is_internal:
            When True (git ops, file ops) returns a blank ExecutionContext
            so the agent observation is never polluted.
        env_extra:
            Additional env vars merged on top of os.environ.
            Used by GitHandler to inject GIT_AUTHOR_* vars.
        """
        effective_timeout = min(params.timeout_sec, self._wall_timeout)
        cwd = self._resolve_workdir(params.workdir)
        env = {**os.environ, **(env_extra or {})}
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        timed_out = False
        stdout = stderr = ""
        returncode = -1

        try:
            proc = subprocess.run(
                params.command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(cwd),
                timeout=effective_timeout,
                preexec_fn=_apply_limits,
                env=env,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode

        except subprocess.TimeoutExpired as exc:
            timed_out = True
            raw_out = exc.stdout or b""
            raw_err = exc.stderr or b""
            stdout = (
                raw_out.decode("utf-8", errors="replace")
                if isinstance(raw_out, bytes)
                else raw_out
            )
            stderr = (
                raw_err.decode("utf-8", errors="replace")
                if isinstance(raw_err, bytes)
                else raw_err
            )

        # Determine run_error based on return code or exceptions
        run_error = None
        if timed_out:
            run_error = f"Command timed out after {effective_timeout}s"
        elif returncode != 0:
            run_error = stderr or "Command failed"

        return ExecutionContext(
            command=params.command,
            stdout=self._truncate(stdout),
            stderr=self._truncate(stderr),
            return_code=returncode,
            timed_out=timed_out,
            run_error=run_error,
        )

    # ── Private ───────────────────────────────────────────────────────

    def _resolve_workdir(self, workdir: str) -> Path:
        """
        Resolve workdir relative to sandbox root.
        Blocks path traversal outside the root.
        """
        target = (self._root / workdir).resolve()
        if not str(target).startswith(str(self._root.resolve())):
            raise PermissionError(f"workdir '{workdir}' escapes sandbox root.")
        if not target.is_dir():
            raise NotADirectoryError(f"workdir '{workdir}' does not exist in sandbox.")
        return target

    @staticmethod
    def _truncate(text: str) -> str | None:
        if not text:
            return None
        if len(text.encode("utf-8")) <= _8_KB:
            return text
        truncated = text.encode("utf-8")[:_8_KB].decode("utf-8", errors="replace")
        return truncated + "\n… [truncated to 8 KB]"

    def __repr__(self) -> str:
        return f"ShellExecutor(root={self._root}, wall_timeout={self._wall_timeout}s)"
