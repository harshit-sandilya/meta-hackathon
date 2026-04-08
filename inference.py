# inference.py
# Baseline inference script for the Refactoring Environment
# Meta PyTorch OpenEnv Hackathon — harshit-sandilya/refactoring-environment
#
# Required environment variables:
#   API_BASE_URL  — Base URL of the OpenAI-compatible LLM endpoint (e.g. ngrok tunnel)
#   MODEL_NAME    — Model identifier
#   API_KEY       — API key for the LLM endpoint ("no-key" for local/ngrok)
#   HF_TOKEN      — Hugging Face token (for pulling the HF Space env)

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap

from openai import OpenAI
from refactoring_environment import RefactoringEnv, RefactorAction

# ── Credentials & config ──────────────────────────────────────────────────────
API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME: str = os.environ["MODEL_NAME"]
HF_TOKEN: str = os.environ["HF_TOKEN"]
API_KEY: str = os.environ.get("API_KEY") or "no-key"

HF_REPO_ID = "harshit-sandilya/refactoring-environment"

MAX_STEPS = 10  # steps per episode
TEMPERATURE = 0.0  # deterministic — required for reproducible grader scores
MAX_TOKENS = 512

# episode_count → task (cycles through 3 tasks: 0, 1, 2)
TASK_EPISODES = {
    "lint-cleanup": 0,
    "style-enforcement": 1,
    "module-decompose": 2,
}

# ── LLM client ────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ── System prompt (MUST MATCH EXACT FORMAT) ────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Python refactoring agent. Improve the quality of Python
    code in a repository by taking precise, targeted actions. You will get a
    reward between 0.0-1.0 for your every action, when you reach 1.0 that's the final stage.

    Available action_type values:
      view_file        — read a file  (args: path, optional line_start, line_end)
      list_directory   — list files   (args: path, optional recursive, max_depth)
      search_codebase  — regex search (args: query, optional file_glob, case_insensitive, context_lines, max_results)
      edit_file        — edit a file  (args: patch (path, unified_diff/new_content))
      edit_files       — edit multiple files (args: patches)
      run_shell        — run a shell command (args: command, timeout_sec, workdir)
      git_diff         — view diff vs baseline (args: paths, optional stat_only)
      submit           - submit the code when done, i.e, reward is maximised (reward = 1.0)

    STEP-BY-STEP WORKFLOW:
    1. First, use list_directory to understand the repository structure
    2. Identify the target file(s) to refactor from the file tree
    3. Use view_file to examine the specific file(s) that need changes
    4. Make targeted edits using edit_file or edit_files
    5. Use run_shell to run scripts and verify your changes

    BEST PRACTICES:
    - After listing directory, ALWAYS view a file next (don't list again)
    - When you see an error about a missing file, list_directory ONCE, then view the correct file
    - Focus on ONE file at a time for targeted refactoring
    - Use search_codebase to find specific patterns across files
    - Check git_diff periodically to review your changes

    ERROR HANDLING:
    - If file not found: list_directory once → view_file → edit_file
    - If edit fails: view_file to see current state → retry edit
    - Never call the same action type more than 2 times in a row

    Reply with ONLY a raw JSON object — no markdown, no explanation.
    Examples:
      {"action_type": "view_file", "args": {"path": "utils.py"}}
      {"action_type": "edit_file", "args": {"path": "utils.py", "new_content": "..."}}
      {"action_type": "run_shell", "args": {"command": "python utils.py"}}
      {"action_type": "list_directory", "args": {"path": "."}}
"""
).strip()


# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"    [LLM ERROR] {exc}")
        return ""


# ── Parse model output → RefactorAction ──────────────────────────────────────
def parse_action(raw: str) -> RefactorAction:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        return RefactorAction(
            action_type=data.get("action_type", "list_directory"),
            params=data.get("args", {"path": "."}),
        )
    except Exception:
        return RefactorAction(action_type="list_directory", params={"path": "."})


# ── Build per-step prompt from observation ────────────────────────────────────
def build_prompt(obs, step: int, history: list[str]) -> str:

    task_id = getattr(obs, "task_id", "unknown")
    task_desc = getattr(obs, "task_description", "")
    codebase = getattr(obs, "codebase", None)
    grader = getattr(obs, "grader", None)
    reward_ctx = getattr(obs, "reward_context", None)
    max_steps = getattr(obs, "max_steps", MAX_STEPS)

    file_tree = ""
    active_file = ""
    file_content = ""
    if codebase:
        entries = getattr(codebase, "file_tree", [])
        active_file = getattr(codebase, "active_file", "") or ""
        file_content = getattr(codebase, "file_content", "") or ""
        file_tree = "\n".join(
            f"{'  ' * e.path.count('/') if hasattr(e, 'path') else ''}"
            f"{'📁' if getattr(e, 'is_dir', False) else '📄'} "
            f"{getattr(e, 'path', str(e))}"
            for e in (entries or [])
        )

    feedbacks = getattr(grader, "feedbacks", []) if grader else []
    step_score = getattr(reward_ctx, "step_score", None) if reward_ctx else None
    score_str = f"{step_score:.3f}" if step_score is not None else "n/a"
    fb_str = "\n".join(feedbacks) if feedbacks else "No feedback yet."
    hist_str = "\n".join(history[-5:]) if history else "None"

    return textwrap.dedent(
        f"""
        TASK:         {task_id}
        GOAL:         {task_desc}
        STEP:         {step}/{max_steps}
        CURRENT SCORE:{score_str}

        FILE TREE:
        {file_tree}

        ACTIVE FILE: {active_file}
        --- CONTENT (first 3000 chars) ---
        {file_content[:3000]}
        --- END ---

        GRADER FEEDBACK:
        {fb_str}

        RECENT HISTORY (last 5 steps):
        {hist_str}

        Choose the single best action to improve code quality.
        Remember: list_directory → view_file → edit_file (don't get stuck looping!)
        Reply with ONLY a raw JSON action object.
    """
    ).strip()


# ── Single episode ────────────────────────────────────────────────────────────
async def run_episode(env: RefactoringEnv, episode_count: int, task_name: str) -> float:
    print(f"[EPISODE_START] task={task_name} episode_count={episode_count}")

    # reset() returns a StepResult; .observation is a RefactorObservation
    reset_result = await env.reset(episode_count=episode_count)
    obs = reset_result.observation
    done = reset_result.done

    initial_score = getattr(getattr(obs, "reward_context", None), "step_score", None)
    print(
        f"[EPISODE_START] initial_score={initial_score if initial_score is not None else 'n/a'}"
    )

    history: list[str] = []
    final_reward = 0.0

    for step in range(1, MAX_STEPS + 1):

        prompt = build_prompt(obs, step, history)
        raw_resp = call_llm(prompt)
        action = parse_action(raw_resp)

        print(
            f"[STEP] episode={task_name} step={step}/{MAX_STEPS} action={action.action_type}",
            flush=True,
        )

        # Debug: Log action details
        action_details = f"Action details: {action.action_type}"
        if action.params:
            action_details += f" with params: {action.params}"

        # step() also returns a StepResult
        try:
            step_result = await env.step(action)

            obs = step_result.observation
            done = step_result.done
            # reward is a float on the wire (weight-summed score)
            reward = step_result.reward if step_result.reward is not None else 0.0

            # richer grader score lives in obs.reward_context.step_score
            ctx_score = getattr(
                getattr(obs, "reward_context", None), "step_score", reward
            )
            final_reward = ctx_score if ctx_score is not None else reward

            print(
                f"[STEP] reward={float(reward):.3f} score={final_reward:.3f} done={done}"
            )

            history.append(
                f"Step {step}: {action.action_type} → score={final_reward:.3f}"
            )

            if done:
                print(f"[EPISODE_END] task={task_name} reason=done step={step}")
                break

        except RuntimeError as e:
            error_msg = str(e)
            print(f"[STEP] error={error_msg}")

            if "File not found" in error_msg:
                if "File not found in sandbox: " in error_msg:
                    file_path = error_msg.split("File not found in sandbox: ")[
                        -1
                    ].split(" (code:")[0]
                    error_feedback = (
                        f"File not found: {file_path}. "
                        "Use list_directory to verify available files."
                    )
                else:
                    error_feedback = (
                        "File not found. Use list_directory to verify available files."
                    )
            elif "EXECUTION_ERROR" in error_msg:
                error_feedback = (
                    f"Execution error: {error_msg}. "
                    "Check action parameters and try again."
                )
            else:
                error_feedback = f"Environment error: {error_msg}"

            history.append(
                f"Step {step}: {action.action_type} → ERROR: {error_feedback}"
            )
            print(f"[STEP] error_feedback={error_feedback}")
    else:
        print(f"[EPISODE_END] task={task_name} reason=max_steps step={MAX_STEPS}")

    print(f"[EPISODE_END] task={task_name} final_score={final_reward:.4f}")
    return final_reward


# ── Main ──────────────────────────────────────────────────────────────────────
async def async_main() -> None:
    print(f"[START] env={HF_REPO_ID} model={MODEL_NAME} endpoint={API_BASE_URL}")

    results: dict[str, float] = {}

    # ── Correct async pattern for openenv EnvClient ───────────────────────
    # from_env() is a coroutine — await it first, then use the returned
    # env object as an async context manager.
    env = await RefactoringEnv.from_env(HF_REPO_ID, hf_token=HF_TOKEN)
    async with env:
        for task_name, episode_count in TASK_EPISODES.items():
            score = await run_episode(env, episode_count, task_name)
            results[task_name] = score

    # ── Summary ───────────────────────────────────────────────────────────
    overall = sum(results.values()) / len(results)
    for task, score in results.items():
        print(f"[END] task={task} score={score:.4f}")
    print(f"[END] overall={overall:.4f}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
