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
TEMPERATURE = 1.0
MAX_TOKENS = 4096

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
    except json.JSONDecodeError as e:
        print(
            f"[PARSE_ERROR] JSONDecodeError: {e} | truncated={len(raw)>=MAX_TOKENS*3} | raw_tail={raw[-80:]!r}"
        )
        return RefactorAction(action_type="list_directory", params={"path": "."})
    except Exception:
        return RefactorAction(action_type="list_directory", params={"path": "."})


# ── Build per-step prompt from observation ────────────────────────────────────
def build_prompt(obs, step: int, history: list[str], file_cache) -> str:
    task_id = getattr(obs, "task_id", "unknown")
    task_desc = getattr(obs, "description", "")
    codebase = getattr(obs, "codebase", None)
    execution = getattr(obs, "execution", None)
    grader = getattr(obs, "grader", None)
    reward_ctx = getattr(obs, "reward_context", None)
    git_status = getattr(obs, "git", None)
    max_steps = getattr(obs, "max_steps", MAX_STEPS)
    remaining = getattr(obs, "remaining_steps", MAX_STEPS - step)

    # ── CodebaseContext ───────────────────────────────────────────────────
    file_tree = ""
    active_file = ""
    file_content = ""
    line_range = ""
    if codebase:
        entries = getattr(codebase, "file_tree", []) or []
        active_file = getattr(codebase, "active_file", "") or ""
        file_content = getattr(codebase, "file_content", "") or ""
        line_start = getattr(codebase, "file_line_start", None)
        line_end = getattr(codebase, "file_line_end", None)
        total_lines = getattr(codebase, "total_file_lines", None)
        if line_start and line_end:
            line_range = f"lines {line_start}-{line_end} of {total_lines}"

        tree_lines = []
        for e in entries:
            path = getattr(e, "path", str(e))
            is_dir = getattr(e, "is_dir", False)
            size = getattr(e, "size_bytes", 0)
            indent = "  " * path.count("/")
            kind = "[DIR] " if is_dir else "[FILE]"
            size_str = "" if is_dir else f" ({size} B)"
            tree_lines.append(f"{indent}{kind}{path}{size_str}")
        file_tree = (
            "\n".join(tree_lines) or "(empty — run list_directory path='.' to populate)"
        )
        print(f"[STEP_DEBUG] file_tree={file_tree}")
        print(f"[STEP_DEBUG] active_file={active_file} content_len={len(file_content)}")
        if active_file and file_content:
            if active_file not in file_cache:
                file_cache[active_file] = file_content
            elif line_start and line_start > 1:
                # Append new chunk if it's a later segment
                if file_content not in file_cache[active_file]:
                    file_cache[active_file] += "\n" + file_content
                    print(
                        f"[STEP_DEBUG] file_cache updated: {active_file} total_len={len(file_cache[active_file])}"
                    )

    # ── ExecutionContext ──────────────────────────────────────────────────
    exec_block = "None"
    if execution:
        cmd = getattr(execution, "command", "") or ""
        stdout = getattr(execution, "stdout", "") or ""
        stderr = getattr(execution, "stderr", "") or ""
        retcode = getattr(execution, "return_code", None)
        timed_out = getattr(execution, "timed_out", False)
        run_error = getattr(execution, "run_error", "") or ""
        parts = []
        if cmd:
            parts.append(f"$ {cmd}")
        if stdout.strip():
            parts.append(stdout.strip()[:2000])
        if stderr.strip():
            parts.append(f"[stderr] {stderr.strip()[:500]}")
        if run_error:
            parts.append(f"[error] {run_error}")
        if timed_out:
            parts.append("[TIMED OUT]")
        if retcode is not None:
            parts.append(f"[exit code: {retcode}]")
        if parts:
            exec_block = "\n".join(parts)
        print(f"[STEP_DEBUG] exec_stdout={stdout[:300].replace(chr(10),' ')}")
        if stderr:
            print(f"[STEP_DEBUG] exec_stderr={stderr[:200].replace(chr(10),' ')}")
        if run_error:
            print(f"[STEP_DEBUG] exec_run_error={run_error}")

    # ── GraderContext ─────────────────────────────────────────────────────
    scores = getattr(grader, "scores", {}) if grader else {}
    feedbacks = getattr(grader, "feedbacks", []) if grader else []
    errors = getattr(grader, "errors", []) if grader else []
    penalties = getattr(grader, "penalties", []) if grader else []
    is_regress = getattr(grader, "is_regression", False) if grader else False

    scores_str = ", ".join(f"{k}={v:.3f}" for k, v in scores.items()) or "none yet"
    fb_str = "\n".join(feedbacks) if feedbacks else "No feedback yet."
    errors_str = "\n".join(errors) if errors else "None"
    penalties_str = "\n".join(penalties) if penalties else "None"
    regress_tag = "  ⚠ REGRESSION" if is_regress else ""

    # ── RewardContext ─────────────────────────────────────────────────────
    step_score = getattr(reward_ctx, "step_score", None) if reward_ctx else None
    cum_penalty = getattr(reward_ctx, "cumulative_penalty", 0.0) if reward_ctx else 0.0
    score_str = f"{step_score:.3f}" if step_score is not None else "n/a"

    # ── GitStatus ─────────────────────────────────────────────────────────
    git_block = "No changes yet."
    if git_status:
        staged = getattr(git_status, "staged_files", [])
        unstaged = getattr(git_status, "unstaged_files", [])
        diff_stat = getattr(git_status, "diff_stat", None)
        git_parts = []
        if staged:
            git_parts.append(f"Staged:   {', '.join(staged)}")
        if unstaged:
            git_parts.append(f"Unstaged: {', '.join(unstaged)}")
        if diff_stat:
            git_parts.append(f"Diff:     {diff_stat}")
        if git_parts:
            git_block = "\n".join(git_parts)

    # ── History ───────────────────────────────────────────────────────────
    hist_str = "\n".join(history[-6:]) if history else "None"
    total_lines = getattr(codebase, "total_file_lines", None) if codebase else None
    cached_len = len((file_cache or {}).get(active_file, "")) if active_file else 0
    active_header = f"ACTIVE FILE: {active_file or 'none'}"
    if line_range:
        active_header += f" (viewing {line_range})"
    if total_lines:
        active_header += f" | TOTAL LINES: {total_lines}"
    if cached_len > 0:
        active_header += f" | CACHED: {cached_len} B"
    cached = (file_cache or {}).get(active_file, "") if active_file else ""
    full_content = cached if len(cached) > len(file_content) else file_content
    file_body = (
        file_content[:65536]
        if file_content
        else "(no file loaded — use view_file to open one)"
    )
    if full_content:
        file_body += f"\n\n[CACHE: {len(full_content)} B total — {'COMPLETE' if len(full_content) >= (getattr(codebase, 'total_file_lines', 0) or 0) * 30 else 'MAY BE PARTIAL — use view_file for more lines'}]"

    return textwrap.dedent(
        f"""
        TASK:           {task_id}
        GOAL:           {task_desc}
        STEP:           {step}/{max_steps}  (remaining: {remaining})
        CURRENT SCORE:  {score_str}  |  graders: [{scores_str}]
        CUMUL PENALTY:  {cum_penalty:.3f}{regress_tag}
        PENALTIES:      {penalties_str}
        GRADER ERRORS:  {errors_str}

        ── REPOSITORY FILE TREE ─────────────────────────────────
        {file_tree}

        ── {active_header} ──
        {file_body}

        ── LAST COMMAND OUTPUT ──────────────────────────────────
        {exec_block}

        ── GRADER FEEDBACK ──────────────────────────────────────
        {fb_str}

        ── GIT STATUS ───────────────────────────────────────────
        {git_block}

        ── RECENT HISTORY (last 6 steps) ────────────────────────
        {hist_str}

        Choose the single best action to improve code quality.
        Remember: list_directory → view_file → edit_file (don't loop!)
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
    file_cache: dict[str, str] = {}
    final_reward = 0.0

    for step in range(1, MAX_STEPS + 1):

        prompt = build_prompt(obs, step, history, file_cache)
        loop = asyncio.get_event_loop()
        raw_resp = await loop.run_in_executor(None, call_llm, prompt)
        action = parse_action(raw_resp)
        print(f"[STEP_DEBUG] raw_response={raw_resp[:300].replace(chr(10), ' ')}")
        print(
            f"[STEP_DEBUG] parsed_action={action.action_type} params={json.dumps(action.params)}"
        )

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
            from websockets.exceptions import ConnectionClosedError as WsClosedError

            try:
                step_result = await env.step(action)
            except (WsClosedError, ConnectionResetError, BrokenPipeError) as conn_err:
                print(f"[STEP_DEBUG] WebSocket dropped: {conn_err} — skipping step")
                history.append(f"Step {step}: {action.action_type} → WS_DISCONNECT")
                continue

            obs = step_result.observation
            task_desc = getattr(obs, "description", "")
            print(f"[EPISODE_START] task_description={task_desc}")
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

            exec_preview = ""
            if obs.execution:
                out = (getattr(obs.execution, "stdout", "") or "").strip()
                err = (getattr(obs.execution, "run_error", "") or "").strip()
                if out:
                    exec_preview = f" | stdout={out[:150].replace(chr(10), ' ')}"
                elif err:
                    exec_preview = f" | error={err[:100]}"
            history.append(
                f"Step {step}: {action.action_type} → score={final_reward:.3f}{exec_preview}"
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
