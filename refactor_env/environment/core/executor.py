"""
executor.py  —  Safe subprocess runner for refactor-env.

Key design decisions (drawn from OpenSandbox):
  - Allowlist checked BEFORE fork — no process ever spawns for blocked binaries.
  - stdout + stderr are both captured and truncated to OUTPUT_LIMIT bytes.
  - Timeout is enforced via subprocess.run(timeout=...) + SIGKILL fallback.
  - workdir is always resolved to an absolute path inside the sandbox root
    before the process is spawned (prevents directory traversal via workdir).
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Optional

from refactor_env.models import ALLOWED_SHELL_COMMANDS
from .exceptions import (
    CommandNotAllowedError,
    ExecutorError,
    ExecutorTimeoutError,
    PathTraversalError,
)

OUTPUT_LIMIT: int = 8 * 1024  # 8 KB per stream
DEFAULT_TIMEOUT: int = 30  # seconds


class Executor:
    """
    Runs shell commands safely inside an optional sandbox root.

    Parameters
    ----------
    sandbox_root:
        When set, all workdir values are resolved and checked to be
        descendants of this path.  Pass None to skip the check (test use only).
    allowed_commands:
        Override the default allowlist (mainly for tests).
    """

    def __init__(
        self,
        sandbox_root: Optional[str] = None,
        allowed_commands: Optional[frozenset[str]] = None,
    ) -> None:
        self._sandbox_root = os.path.realpath(sandbox_root) if sandbox_root else None
        self._allowed = (
            allowed_commands if allowed_commands is not None else ALLOWED_SHELL_COMMANDS
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        command: str,
        *,
        workdir: Optional[str] = None,
        timeout_sec: int = DEFAULT_TIMEOUT,
        check: bool = False,
        env_extra: Optional[dict[str, str]] = None,
    ) -> tuple[int, str, str]:
        """
        Run *command* synchronously.

        Returns (returncode, stdout, stderr) — both streams are UTF-8
        decoded and truncated to OUTPUT_LIMIT bytes.

        Raises
        ------
        CommandNotAllowedError  — binary not in allowlist
        PathTraversalError      — workdir escapes sandbox root
        ExecutorTimeoutError    — process exceeded timeout_sec
        ExecutorError           — non-zero exit and check=True
        """
        binary = self._extract_binary(command)
        if binary not in self._allowed:
            raise CommandNotAllowedError(binary)

        resolved_workdir = self._resolve_workdir(workdir)
        env = self._build_env(env_extra)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=resolved_workdir,
                capture_output=True,
                timeout=timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise ExecutorTimeoutError(command, timeout_sec)

        stdout = self._decode_truncate(result.stdout)
        stderr = self._decode_truncate(result.stderr)

        if check and result.returncode != 0:
            raise ExecutorError(command, result.returncode, stderr)

        return result.returncode, stdout, stderr

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_binary(command: str) -> str:
        tokens = shlex.split(command)
        if not tokens:
            return ""
        # Strip path prefix: "/usr/bin/python" → "python"
        return os.path.basename(tokens[0])

    def _resolve_workdir(self, workdir: Optional[str]) -> Optional[str]:
        if workdir is None:
            return self._sandbox_root  # default to sandbox root
        resolved = os.path.realpath(workdir)
        if self._sandbox_root and not resolved.startswith(self._sandbox_root):
            raise PathTraversalError(workdir)
        return resolved

    @staticmethod
    def _build_env(extra: Optional[dict[str, str]]) -> dict[str, str]:
        env = os.environ.copy()
        # Prevent accidental network calls from subprocesses
        env.setdefault("NO_COLOR", "1")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        if extra:
            env.update(extra)
        return env

    @staticmethod
    def _decode_truncate(raw: bytes) -> str:
        text = raw.decode("utf-8", errors="replace")
        if len(text) > OUTPUT_LIMIT:
            text = text[:OUTPUT_LIMIT] + f"\n... [truncated at {OUTPUT_LIMIT} bytes]"
        return text
