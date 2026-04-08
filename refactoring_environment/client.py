"""client.py — RefactoringEnvironment WebSocket client

Import this in training or evaluation code to interact with the server.

Async usage:
    async with RefactoringEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="t1", difficulty="medium", max_steps=50)
        while not result.done:
            action = RefactorAction(
                action_type="view_file",
                params={"path": "app.py"},
            )
            result = await env.step(action)
            print(result.observation.grader.feedbacks)

Sync usage (scripts / notebooks):
    with RefactoringEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        result = env.step(action)
        state  = env.state()
"""

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models_internal import RefactorAction, RefactorObservation, RefactorState


class RefactoringEnv(EnvClient[RefactorAction, RefactorObservation, RefactorState]):
    """WebSocket client for the openenv-refactoring-environment server.

    Subclasses EnvClient — all connection management, reconnect logic,
    the sync() wrapper, and the async context manager are handled by the
    base class.  This class only owns the three parsing/serialization hooks.
    """

    # ------------------------------------------------------------------
    # 1. Serialize action → wire dict
    # ------------------------------------------------------------------

    def _step_payload(self, action: RefactorAction) -> dict:
        """Serialize a RefactorAction to the JSON-safe dict sent over WebSocket.
        Uses mode="json" so ActionType enums become plain strings and all
        nested FilePatch / params dicts are JSON-serializable.
        """
        return action.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 2. Deserialize reset/step response → typed StepResult
    # ------------------------------------------------------------------

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse the server's step or reset response into a StepResult.

        OpenEnv wire convention:
            {
              "done":        <bool>,
              "reward":      <float | null>,
              "observation": { ...all observation fields... }
            }

        done and reward live at the payload root; everything else is
        nested under "observation".  We merge them before calling
        model_validate so RefactorObservation (which inherits done/reward
        from the Observation base) can be built in one shot.
        """
        obs_raw = payload.get("observation", {})

        # Merge top-level done/reward into the observation dict.
        # Base Observation fields take precedence if the server also
        # echoes them inside "observation" (harmless either way).
        obs_full = {
            "done": payload.get("done", False),
            "reward": payload.get("reward"),
            **obs_raw,
        }

        observation = RefactorObservation.model_validate(obs_full)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    # ------------------------------------------------------------------
    # 3. Deserialize /state response → typed RefactorState
    # ------------------------------------------------------------------

    def _parse_state(self, payload: dict) -> RefactorState:
        """Parse the /state response into a RefactorState.

        The server returns the full State dict; model_validate handles
        all nested types including the inherited episode_id / step_count.
        """
        return RefactorState.model_validate(payload)

    # ------------------------------------------------------------------
    # 4. Optional: customize the reset payload
    # ------------------------------------------------------------------

    def _reset_payload(self, **kwargs) -> dict:
        """Build the payload sent to the server's /reset endpoint.

        Supported kwargs (all optional):
            task_id    (str)  — pin a specific task; server picks randomly if omitted
            difficulty (str)  — "easy" | "medium" | "hard"
            max_steps  (int)  — override the episode step budget
            seed       (int)  — for deterministic task selection
            episode_id (str)  — reuse a specific episode ID (replay / eval)

        Unknown kwargs are forwarded as-is so the server can extend the
        contract without requiring a client update.
        """
        return {k: v for k, v in kwargs.items() if v is not None}
