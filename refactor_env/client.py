# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
client.py  —  RefactorEnv HTTP/WebSocket client.

Wraps the OpenEnv EnvClient base class with:
  - Correct _step_payload / _parse_result / _parse_state implementations
    that match RefactorAction / RefactorObservation / State models.
  - A reset() override that accepts task_id so callers don't have to
    construct the kwargs dict manually.
  - Ergonomic one-liner helpers (view_file, edit_file, run_shell, submit …)
    that build the correct RefactorAction and call step() internally.
    These make inference.py and test code significantly cleaner.

Wire protocol (what the OpenEnv server sends / receives)
--------------------------------------------------------
  step   POST/WS  payload : {"action_type": str, "params": {…}}
  result payload : {"observation": {…}, "reward": float|dict|null, "done": bool}
  state  GET/WS   payload : {"episode_id": str, "step_count": int}

All nested objects (TestSummary, LintSummary, FileTreeEntry, …) are
serialised as plain dicts by the server and reconstructed by Pydantic's
model_validate() on the client side.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    ActionType,
    FilePatch,
    RefactorAction,
    RefactorObservation,
)


# ---------------------------------------------------------------------------
# Internal helper — builds a validated RefactorAction from keyword params
# ---------------------------------------------------------------------------


def _action(action_type: ActionType, **params: Any) -> RefactorAction:
    """Construct a validated RefactorAction from keyword params."""
    return RefactorAction(action_type=action_type, params=params)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RefactorEnv(EnvClient[RefactorAction, RefactorObservation, State]):
    """
    Client for the RefactorEnv environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each instance owns one isolated episode on the server side (one sandbox,
    one git-backed workspace).  Multiple instances can run concurrently
    because the server sets SUPPORTS_CONCURRENT_SESSIONS = True.

    Typical usage — high-level helpers
    ------------------------------------
    >>> with RefactorEnv(base_url="http://localhost:8000") as env:
    ...     obs = env.reset(task_id="lint-cleanup").observation
    ...     result = env.view_file("utils.py")
    ...     result = env.edit_file("utils.py", unified_diff="--- a\\n+++ b\\n…")
    ...     result = env.run_shell("pytest tests/")
    ...     result = env.submit()
    ...     print(result.reward)                    # float step score
    ...     print(result.observation.step_score)    # same, embedded in obs

    Typical usage — raw actions (for inference.py)
    -----------------------------------------------
    >>> result = env.step(RefactorAction(
    ...     action_type=ActionType.edit_file,
    ...     params={"patch": {"path": "utils.py", "new_content": "…"}},
    ... ))

    Using Docker
    -------------
    >>> env = RefactorEnv.from_docker_image("refactor-env:latest")
    >>> try:
    ...     env.reset(task_id="lint-cleanup")
    ...     env.submit()
    ... finally:
    ...     env.close()
    """

    # ── OpenEnv wire-protocol implementation ─────────────────────────────────

    def _step_payload(self, action: RefactorAction) -> Dict:
        """
        Serialise RefactorAction to the JSON dict sent over the wire.

        Uses Pydantic's model_dump(mode="json") so that:
          - ActionType enum  → string value  (e.g. "view_file")
          - nested Pydantic models inside params → plain dicts
          - Optional fields with None values → passed through correctly

        The original stub used {"message": action.message} which
        referenced a non-existent field.  The correct wire payload is the
        full model dump:  {"action_type": "view_file", "params": {"path": "…"}}
        """
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict) -> StepResult[RefactorObservation]:
        """
        Parse the server's step / reset response into a typed StepResult.

        Expected payload shape::

            {
                "observation": { <RefactorObservation fields> },
                "reward":      float | {"score": float, …} | null,
                "done":        bool
            }

        The ``reward`` field may arrive as:
          - a plain float  (some server versions return step_score directly)
          - a dict matching RefactorReward  (richer breakdown)
          - null / missing  (on reset, before any step score exists)

        In all cases we normalise to a float (or None) for StepResult.
        The full breakdown is always accessible via
        ``result.observation.step_score`` and ``result.observation.cumulative_penalty``.

        The original stub constructed RefactorObservation with entirely wrong
        fields (echoed_message, message_length, etc.) copied from the Echo
        env template.  This version uses model_validate() which lets Pydantic
        handle all nested model reconstruction automatically.
        """
        obs_data = payload.get("observation")
        if not obs_data:
            raise ValueError(
                "Server response is missing the 'observation' key. "
                f"Received keys: {list(payload.keys())}"
            )

        # Pydantic handles all nested model reconstruction:
        # FileTreeEntry list, TestSummary, LintSummary, GitStatus, etc.
        observation = RefactorObservation.model_validate(obs_data)

        # Normalise reward → float | None
        raw_reward = payload.get("reward")
        if isinstance(raw_reward, dict):
            # Full RefactorReward dict — extract primary score scalar
            reward: Optional[float] = raw_reward.get("score")
        elif isinstance(raw_reward, (int, float)):
            reward = float(raw_reward)
        else:
            # None or absent — fall back to the step_score embedded in obs
            reward = observation.step_score

        done: bool = bool(payload.get("done", False))

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse the server's /state response into a State object.

        Expected payload shape::

            {"episode_id": str, "step_count": int}
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    # ── reset override — adds task_id support ────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StepResult[RefactorObservation]:
        """
        Start a new episode, optionally choosing a specific task.

        The base class reset() accepts arbitrary **kwargs and passes them as
        the ``data`` dict in the reset wire message, so passing task_id here
        routes it straight through to RefactorEnvironment.reset(task_id=…)
        on the server.

        Parameters
        ----------
        task_id : str, optional
            Slug of the task to load.  Available tasks:
              - ``"lint-cleanup"``     (easy)        — remove dead code & lint errors
              - ``"api-rename"``       (easy/medium) — rename a function & propagate
              - ``"test-coverage"``    (medium)      — raise coverage to target threshold
              - ``"module-decompose"`` (medium/hard) — split god module
              - ``"style-enforce"``    (hard)        — full production-hygiene pass
            When omitted, the server falls back to REFACTOR_ENV_DEFAULT_TASK
            env var, then ``"lint-cleanup"``.
        **kwargs :
            Forwarded to the base EnvClient.reset() for extensibility.

        Returns
        -------
        StepResult[RefactorObservation]
            ``result.observation`` holds the full initial state: file tree,
            baseline test/lint summaries, and the task objective string.
        """
        if task_id is not None:
            kwargs["task_id"] = task_id
        return super().reset(**kwargs)

    # ── Ergonomic action helpers ──────────────────────────────────────────────
    # These are thin wrappers around step() with no extra logic.
    # They exist to make inference.py, tests, and interactive use readable.

    def view_file(
        self,
        path: str,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
    ) -> StepResult[RefactorObservation]:
        """
        Open a file and load its content (or a line-range window) into
        ``observation.file_content``.

        Parameters
        ----------
        path : str
            Relative path from the sandbox root (e.g. ``"utils.py"``).
        line_start : int, optional
            First line to include (1-indexed).
        line_end : int, optional
            Last line to include (inclusive, 1-indexed).
        """
        params: Dict[str, Any] = {"path": path}
        if line_start is not None:
            params["line_start"] = line_start
        if line_end is not None:
            params["line_end"] = line_end
        return self.step(_action(ActionType.view_file, **params))

    def list_directory(
        self,
        path: str = ".",
        recursive: bool = False,
        max_depth: int = 3,
    ) -> StepResult[RefactorObservation]:
        """
        List files in a directory.  Updates ``observation.file_tree``.

        Parameters
        ----------
        path : str
            Directory to list, relative to sandbox root.
        recursive : bool
            Whether to descend into subdirectories.
        max_depth : int
            Maximum recursion depth (1–8).
        """
        return self.step(
            _action(
                ActionType.list_directory,
                path=path,
                recursive=recursive,
                max_depth=max_depth,
            )
        )

    def search_codebase(
        self,
        query: str,
        file_glob: str = "*.py",
        case_insensitive: bool = False,
        context_lines: int = 2,
        max_results: int = 50,
    ) -> StepResult[RefactorObservation]:
        """
        Grep-style search across the sandbox.  Matches appear in
        ``observation.last_action_output``.

        Parameters
        ----------
        query : str
            Search string or regex pattern.
        file_glob : str
            Glob pattern to restrict the search (default ``"*.py"``).
        case_insensitive : bool
            Enable case-insensitive matching.
        context_lines : int
            Lines of context around each match (0–10).
        max_results : int
            Cap on returned matches (1–200).
        """
        return self.step(
            _action(
                ActionType.search_codebase,
                query=query,
                file_glob=file_glob,
                case_insensitive=case_insensitive,
                context_lines=context_lines,
                max_results=max_results,
            )
        )

    def git_diff(
        self,
        paths: Optional[List[str]] = None,
        stat_only: bool = False,
    ) -> StepResult[RefactorObservation]:
        """
        Show the current git diff vs the baseline commit.

        Parameters
        ----------
        paths : list[str], optional
            Restrict the diff to specific files.  Empty list = all files.
        stat_only : bool
            Return only ``--stat`` summary instead of full diff.
        """
        return self.step(
            _action(
                ActionType.git_diff,
                paths=paths or [],
                stat_only=stat_only,
            )
        )

    def edit_file(
        self,
        path: str,
        *,
        unified_diff: Optional[str] = None,
        new_content: Optional[str] = None,
    ) -> StepResult[RefactorObservation]:
        """
        Apply a single-file change.  Supply **exactly one** of the two
        mutually-exclusive parameters (enforced by FilePatch.model_validator).

        Parameters
        ----------
        path : str
            Relative path of the file to modify.
        unified_diff : str, optional
            Standard unified-diff patch (``--- a/file  +++ b/file  @@ …``).
            Preferred — keeps the context window small in prompts.
        new_content : str, optional
            Full replacement content for the file.  Use for small files or
            when the diff would be longer than the whole file.

        Raises
        ------
        ValueError
            If neither or both of ``unified_diff`` / ``new_content`` are given.

        Example
        -------
        >>> result = env.edit_file(
        ...     "utils.py",
        ...     unified_diff=(
        ...         "--- a/utils.py\\n"
        ...         "+++ b/utils.py\\n"
        ...         "@@ -1,4 +1,3 @@\\n"
        ...         "-import os\\n"
        ...         " import re\\n"
        ...     ),
        ... )
        """
        patch = FilePatch(
            path=path,
            unified_diff=unified_diff,
            new_content=new_content,
        )
        return self.step(
            _action(ActionType.edit_file, patch=patch.model_dump(mode="json"))
        )

    def edit_files(
        self,
        patches: List[FilePatch],
    ) -> StepResult[RefactorObservation]:
        """
        Apply changes to multiple files in a single atomic step.

        Parameters
        ----------
        patches : list[FilePatch]
            Between 1 and 20 FilePatch objects.  Each must supply exactly one
            of ``unified_diff`` or ``new_content``.

        Example
        -------
        >>> result = env.edit_files([
        ...     FilePatch(path="utils.py",   unified_diff="--- …"),
        ...     FilePatch(path="helpers.py", new_content="def helper(): …"),
        ... ])
        """
        serialised = [p.model_dump(mode="json") for p in patches]
        return self.step(_action(ActionType.edit_files, patches=serialised))

    def run_shell(
        self,
        command: str,
        timeout_sec: int = 30,
        workdir: str = ".",
    ) -> StepResult[RefactorObservation]:
        """
        Run a whitelisted shell command inside the sandbox.

        stdout + stderr (truncated to 8 KB) appear in
        ``observation.last_action_output``.  Non-zero exit sets
        ``observation.last_action_error``.

        Allowed binaries: python, python3, pytest, ruff, mypy, radon, cat,
        head, tail, grep, find, ls, wc, git, black, isort.

        Parameters
        ----------
        command : str
            Full command string, e.g. ``"pytest tests/ -q --tb=short"`` or
            ``"ruff check utils.py --output-format=json"``.
        timeout_sec : int
            Max wall-clock seconds before the process is killed (1–120).
        workdir : str
            Working directory relative to sandbox root.

        Example
        -------
        >>> result = env.run_shell("pytest tests/ --tb=short -q")
        >>> print(result.observation.last_action_output)
        >>> print(result.observation.test_summary.pass_ratio)
        """
        return self.step(
            _action(
                ActionType.run_shell,
                command=command,
                timeout_sec=timeout_sec,
                workdir=workdir,
            )
        )

    def submit(self, note: Optional[str] = None) -> StepResult[RefactorObservation]:
        """
        Signal that the agent is done with this episode.

        Triggers the final grader evaluation and returns the terminal
        observation with the episode's final score.  After calling submit
        the episode is closed; call ``reset()`` to start a new one.

        Parameters
        ----------
        note : str, optional
            Optional free-text note (≤ 500 chars) logged alongside the
            episode record.  Useful for debugging multi-run experiments.

        Returns
        -------
        StepResult[RefactorObservation]
            ``result.done`` is always ``True``.
            ``result.reward`` holds the final episode score in [0.0, 1.0].

        Example
        -------
        >>> final = env.submit(note="Removed unused imports, preserved tests")
        >>> print(f"Final score: {final.reward:.4f}")
        """
        params: Dict[str, Any] = {}
        if note is not None:
            params["note"] = note
        return self.step(_action(ActionType.submit, **params))
