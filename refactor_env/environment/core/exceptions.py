"""
exceptions.py  —  Typed exception hierarchy for refactor-env sandbox.
All public exceptions inherit from RefactorEnvError so callers can
catch the whole family or specific sub-types.
"""


class RefactorEnvError(Exception):
    """Base for all refactor-env errors."""


# ── Sandbox lifecycle ──────────────────────────────────────────────────────


class SandboxNotInitializedError(RefactorEnvError):
    """Raised when a sandbox method is called before init() completes."""


class SandboxAlreadyInitializedError(RefactorEnvError):
    """Raised when init() is called on an already-active sandbox."""


class SandboxTeardownError(RefactorEnvError):
    """Raised when teardown fails (e.g. locked files on Windows)."""


# ── File operations ────────────────────────────────────────────────────────


class PathTraversalError(RefactorEnvError):
    """Raised when a path escapes the sandbox root (../ attack)."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Path '{path}' escapes the sandbox root.")


class FileNotFoundInSandboxError(RefactorEnvError):
    """Raised when a requested file does not exist inside the sandbox."""

    def __init__(self, path: str) -> None:
        super().__init__(f"File not found in sandbox: '{path}'")


class FileTooLargeError(RefactorEnvError):
    """Raised when a file exceeds the read size limit."""

    def __init__(self, path: str, size: int, limit: int) -> None:
        super().__init__(
            f"File '{path}' is {size} bytes, exceeds limit of {limit} bytes."
        )


class PatchApplyError(RefactorEnvError):
    """Raised when a unified diff patch cannot be applied cleanly."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Patch failed on '{path}': {reason}")


# ── Executor ───────────────────────────────────────────────────────────────


class CommandNotAllowedError(RefactorEnvError):
    """Raised when a shell command binary is not in the allowlist."""

    def __init__(self, binary: str) -> None:
        super().__init__(f"Command '{binary}' is not in the allowed list.")


class ExecutorTimeoutError(RefactorEnvError):
    """Raised when a subprocess exceeds its timeout."""

    def __init__(self, cmd: str, timeout: int) -> None:
        super().__init__(f"Command timed out after {timeout}s: {cmd!r}")


class ExecutorError(RefactorEnvError):
    """Raised when a subprocess exits with a non-zero code and caller treats it as fatal."""

    def __init__(self, cmd: str, returncode: int, stderr: str) -> None:
        super().__init__(
            f"Command {cmd!r} exited {returncode}.\nSTDERR: {stderr[:500]}"
        )


# ── Git ────────────────────────────────────────────────────────────────────


class GitError(RefactorEnvError):
    """Raised when a git operation fails."""


class GitInitError(GitError):
    """Raised when git init or initial commit fails."""


class GitDiffError(GitError):
    """Raised when git diff cannot be computed."""
