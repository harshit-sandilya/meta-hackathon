# inference.py
# Baseline inference script for the Refactoring Environment
# Meta PyTorch OpenEnv Hackathon — harshit-sandilya/refactoring-environment
#
# MANDATORY environment variables:
#   API_BASE_URL  — Base URL of the OpenAI-compatible LLM endpoint (e.g. ngrok tunnel)
#   MODEL_NAME    — Model identifier (e.g. "meta-llama/Llama-3.1-8B-Instruct")
#   API_KEY       — API key for the LLM endpoint (use "no-key" for local/ngrok)
#   HF_TOKEN      — Hugging Face token (used to pull the env from HF Spaces)

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Optional

from openai import OpenAI
from refactoring_environment import RefactorAction, RefactoringEnv

# ── Credentials & config ──────────────────────────────────────────────────────
API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME:   str = os.environ["MODEL_NAME"]
HF_TOKEN:     str = os.environ["HF_TOKEN"]
API_KEY:      str = os.environ.get("API_KEY") or "no-key"

HF_REPO_ID = "harshit-sandilya/refactoring-environment"

MAX_STEPS   = 10   # steps per episode
TEMPERATURE = 0.0  # deterministic for reproducibility
MAX_TOKENS  = 512

# Episode counts map to tasks: 0→lint-cleanup, 1→style-enforcement, 2→module-decompose
TASK_EPISODES = {
    "lint-cleanup":       0,
    "style-enforcement":  1,
    "module-decompose":   2,
}

# ── LLM client (OpenAI-compatible, works with ngrok local LLM) ───────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Python refactoring agent. Your job is to improve the quality
    of Python code in a repository by taking precise, targeted actions.

    You have access to these action types:
      - view_file        : Read a file (args: path, optional start_line/end_line)
      - list_directory   : List files in a directory (args: path)
      - search_codebase  : Search for a pattern (args: pattern, optional path)
      - edit_file        : Edit a file (args: path, new_content OR unified_diff)
      - run_shell        : Run a shell command (args: command)
      - git_diff         : View current diff vs baseline (no args needed)

    Respond with ONLY a valid JSON object with keys "action_type" and "args".
    No explanation. No markdown fences. Just the raw JSON.

    Examples:
      {"action_type": "view_file", "args": {"path": "utils.py"}}
      {"action_type": "edit_file", "args": {"path": "utils.py", "new_content": "..."}}
      {"action_type": "run_shell", "args": {"command": "ruff check ."}}
""").strip()


# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    """Call the LLM and return the raw text response."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"    [LLM ERROR] {exc}")
        return ""


# ── Parse model output into a RefactorAction ─────────────────────────────────
def parse_action(raw_response: str) -> RefactorAction:
    """
    Extract a JSON action dict from the model response.
    Falls back to a safe git_diff if parsing fails.
    """
    # Strip markdown fences if the model adds them
    cleaned = re.sub(r"```(?:json)?|```", "", raw_response).strip()
    try:
        data = json.loads(cleaned)
        action_type = data.get("action_type", "git_diff")
        args        = data.get("args", {})
        return RefactorAction(action_type=action_type, args=args)
    except (json.JSONDecodeError, Exception):
        # Safe fallback — just inspect the diff
        return RefactorAction(action_type="git_diff", args={})


# ── Build the agent prompt from an observation ────────────────────────────────
def build_prompt(obs, step: int, history: list[str]) -> str:
    """
    Construct the per-step prompt from the current observation.
    obs is a RefactorObservation from the environment.
    """
    # Pull relevant fields; env guarantees these exist
    task_id     = getattr(obs, "task_id", "unknown")
    task_desc   = getattr(obs, "task_description", "")
    file_tree   = getattr(obs, "file_tree", "")
    active_file = getattr(obs, "active_file", "")
    file_content= getattr(obs, "file_content", "")
    grader_feedback = getattr(obs, "grader_feedback", "")
    current_score   = getattr(obs, "current_score", 0.0)

    history_block = "\n".join(history[-5:]) if history else "None"

    return textwrap.dedent(f"""
        TASK:    {task_id}
        GOAL:    {task_desc}
        STEP:    {step}/{MAX_STEPS}

        CURRENT SCORE: {current_score:.3f}

        FILE TREE:
        {file_tree}

        ACTIVE FILE: {active_file}
        --- FILE CONTENT ---
        {file_content[:3000]}
        --- END ---

        GRADER FEEDBACK:
        {grader_feedback}

        RECENT HISTORY:
        {history_block}

        Choose the single best action to improve code quality.
        Respond with ONLY a valid JSON action object.
    """).strip()


# ── Single episode runner ─────────────────────────────────────────────────────
async def run_episode(env: RefactoringEnv, episode_count: int, task_name: str) -> float:
    """
    Run one episode for the given episode_count (maps to a task).
    Returns the final scalar reward from the last step.
    """
    print(f"\n{'─'*55}")
    print(f"  TASK: {task_name}  (episode_count={episode_count})")
    print(f"{'─'*55}")

    # Reset with episode_count to select the task
    obs = await env.reset(episode_count=episode_count)
    print(f"  Initial score: {getattr(obs, 'current_score', 0.0):.3f}")

    history: list[str] = []
    done   = False
    final_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        if done:
            print(f"  Environment signalled done at step {step - 1}. Stopping.")
            break

        # Build prompt & get action from LLM
        prompt     = build_prompt(obs, step, history)
        raw_resp   = call_llm(prompt)
        action     = parse_action(raw_resp)

        print(f"  Step {step:2d} | action={action.action_type:18s}", end="")

        # Step the environment
        step_result = await env.step(action)

        # Unpack (obs, reward, done, info) — env returns a StepResult object
        obs    = step_result.observation
        reward = step_result.reward
        done   = step_result.done
        info   = step_result.info or {}

        score   = getattr(reward, "score",          0.0)
        partial = getattr(reward, "partial_credit",  score)
        final_reward = score

        print(f" | reward={score:.3f}  partial={partial:.3f}  done={done}")

        # Log step into history for context
        history.append(
            f"Step {step}: {action.action_type} → score={score:.3f}"
        )

        if done:
            print(f"  ✓ Episode complete.")
            break
    else:
        print(f"  Reached max steps ({MAX_STEPS}).")

    print(f"  FINAL SCORE [{task_name}]: {final_reward:.4f}")
    return final_reward


# ── Main ──────────────────────────────────────────────────────────────────────
async def async_main() -> None:
    print("=" * 55)
    print("  Refactoring Environment — Baseline Inference")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Endpoint  : {API_BASE_URL}")
    print(f"  HF Repo   : {HF_REPO_ID}")
    print("=" * 55)

    results: dict[str, float] = {}

    async with RefactoringEnv.from_env(HF_REPO_ID, hf_token=HF_TOKEN) as env:
        for task_name, episode_count in TASK_EPISODES.items():
            score = await run_episode(env, episode_count, task_name)
            results[task_name] = score

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  BASELINE SCORES SUMMARY")
    print("=" * 55)
    for task, score in results.items():
        print(f"  {task:<28} {score:.4f}")
    overall = sum(results.values()) / len(results)
    print(f"  {'─'*38}")
    print(f"  {'Overall average':<28} {overall:.4f}")
    print("=" * 55)

    # Persist scores for README / CI artifacts
    with open("baseline_scores.json", "w") as f:
        json.dump(
            {
                "scores": results,
                "overall": overall,
                "model":   MODEL_NAME,
                "env":     HF_REPO_ID,
                "max_steps_per_episode": MAX_STEPS,
            },
            f,
            indent=2,
        )
    print("\n  Scores saved → baseline_scores.json")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()