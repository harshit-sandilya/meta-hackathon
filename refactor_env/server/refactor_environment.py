# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
refactor_environment.py  —  OpenEnv adapter for refactor-env.

Bridges OpenEnv's Environment interface (reset / step / state) to
RefactorEnvCore, which handles all sandbox, grader, and reward logic.

Session lifecycle
-----------------
  1. reset(task_id=...) → spins up a Sandbox, loads the ScenarioSpec,
                          records baseline metrics, returns initial obs.
  2. step(action)       → dispatches action to core, returns obs + reward.
  3. state              → exposes current RefactorState as a dict.
  4. On episode end (submit or max_steps reached), teardown is called
     automatically by the core.  On premature disconnect, __del__ handles it.

Concurrency
-----------
  SUPPORTS_CONCURRENT_SESSIONS = True because each instance gets its own
  Sandbox in a separate tempdir.  The OpenEnv server creates one instance
  per WebSocket client in factory mode.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..environment.core.environment import RefactorEnvCore
    from ..models import (
        ActionType,
        RefactorAction,
        RefactorObservation,
        RefactorReward,
        Difficulty,
        TestSummary,
        LintSummary,
        GitStatus,
    )
except ImportError:
    from environment.core.environment import RefactorEnvCore
    from models import (
        ActionType,
        RefactorAction,
        RefactorObservation,
        RefactorReward,
        Difficulty,
        TestSummary,
        LintSummary,
        GitStatus,
    )


_DEFAULT_TASK_ID = os.environ.get("REFACTOR_ENV_DEFAULT_TASK", "lint-cleanup")
_TASKS_ROOT = os.environ.get(
    "REFACTOR_ENV_TASKS_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "tasks"),
)
_PROJECT_ROOT = os.environ.get(
    "REFACTOR_ENV_PROJECT_ROOT",
    os.path.join(os.path.dirname(__file__), ".."),
)


class RefactorEnvironment(Environment):
    """
    OpenEnv-compatible adapter for the refactor-env RL environment.

    Each instance represents one isolated episode with its own git-backed
    sandbox.  Multiple instances run concurrently without interference
    because each Sandbox lives in a separate OS tempdir.

    Parameters
    ----------
    tasks_root : str, optional
        Path to the tasks directory.  Defaults to REFACTOR_ENV_TASKS_ROOT
        env var or ``../tasks`` relative to this file.
    project_root : str, optional
        Project root for resolving repo_path in scenario.yaml files.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        tasks_root: Optional[str] = None,
        project_root: Optional[str] = None,
    ) -> None:
        self._tasks_root = os.path.realpath(tasks_root or _TASKS_ROOT)
        self._project_root = os.path.realpath(project_root or _PROJECT_ROOT)

        # Core orchestrator — created fresh on each reset()
        self._core: Optional[RefactorEnvCore] = None

        # OpenEnv state object — updated after every interaction
        self._openenv_state = State(episode_id=str(uuid4()), step_count=0)

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(
        self, task_id: Optional[str] = None, **kwargs: Any
    ) -> RefactorObservation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str, optional
            Slug of the task to load (e.g. "lint-cleanup").
            Falls back to REFACTOR_ENV_DEFAULT_TASK env var, then "lint-cleanup".
        **kwargs :
            Passed through to RefactorEnvCore.reset() for future extensibility.

        Returns
        -------
        RefactorObservation
            Initial observation with full file tree, baseline metrics, and
            the task objective embedded in the observation.
        """
        # Teardown any live episode before starting a new one
        if self._core is not None:
            try:
                self._core.teardown()
            except Exception:
                pass

        resolved_task_id = task_id or _DEFAULT_TASK_ID

        # Fresh OpenEnv State for this episode
        episode_id = str(uuid4())
        self._openenv_state = State(episode_id=episode_id, step_count=0)

        # Build the core orchestrator
        self._core = RefactorEnvCore(
            tasks_root=self._tasks_root,
            project_root=self._project_root,
            episode_id=episode_id,
        )

        # Delegate to core — this inits sandbox, records baseline metrics
        obs = self._core.reset(task_id=resolved_task_id)
        return obs

    def step(self, action: RefactorAction) -> RefactorObservation:
        """
        Execute one agent action.

        Parameters
        ----------
        action : RefactorAction
            Typed action from models.py.  The adapter validates that the
            core is alive before delegating.

        Returns
        -------
        RefactorObservation
            Updated observation with:
              - last_action_output / last_action_error
              - step_score (dense reward for this step)
              - cumulative_penalty so far
              - updated file_tree, git_status, test_summary, lint_summary
              - done=True if this was a submit action or max_steps reached
        """
        self._assert_episode_live()
        assert self._core is not None

        obs = self._core.step(action)

        # Keep OpenEnv State in sync
        self._openenv_state.step_count += 1
        if obs.done:
            # Episode over — teardown is handled by core internally on submit
            self._core = None

        return obs

    @property
    def state(self) -> State:
        """
        Return the current OpenEnv State.

        OpenEnv uses this to expose episode_id and step_count via /state.
        Richer state (RefactorState) is available via obs.episode_id + core.
        """
        return self._openenv_state

    # ── Lifecycle helpers ─────────────────────────────────────────────────────

    def teardown(self) -> None:
        """
        Explicitly tear down the live episode.
        Called by the OpenEnv server on WebSocket disconnect.
        """
        if self._core is not None:
            try:
                self._core.teardown()
            except Exception:
                pass
            finally:
                self._core = None

    def __del__(self) -> None:
        """Best-effort cleanup on GC — catches stray instances on hard crashes."""
        self.teardown()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_episode_live(self) -> None:
        if self._core is None:
            raise RuntimeError("No active episode. Call reset() before step().")
