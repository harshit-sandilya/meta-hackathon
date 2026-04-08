# inference.py
# Baseline inference script for the Refactoring Environment
# Meta PyTorch OpenEnv Hackathon — harshit-sandilya/refactoring-environment
#
# Required environment variables:
#   API_BASE_URL  — Base URL of the OpenAI-compatible LLM endpoint
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
from websockets.exceptions import ConnectionClosedError as WsClosedError
from refactoring_environment import RefactoringEnv, RefactorAction


# ── Credentials & config ──────────────────────────────────────────────────────
API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME: str = os.environ["MODEL_NAME"]
HF_TOKEN: str = os.environ["HF_TOKEN"]
API_KEY: str = os.environ.get("API_KEY") or "no-key"

HF_REPO_ID = "harshit-sandilya/refactoring-environment"

MAX_STEPS = 10
TEMPERATURE = 0.0
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


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Python refactoring agent. Improve the quality of Python
    code in a repository by taking precise, targeted actions. You will get a
    reward between 0.0-1.0 for every action. Reach 1.0 to complete the task. 
    The grader's context will provide you what went wrong and what needs to be fixed

    ── ACTION REFERENCE ──────────────────────────────────────────────────────

    view_file
      {"action_type": "view_file", "args": {"path": "utils.py"}}
      {"action_type": "view_file", "args": {"path": "utils.py", "line_start": 50, "line_end": 100}}

    list_directory
      {"action_type": "list_directory", "args": {"path": "."}}
      {"action_type": "list_directory", "args": {"path": ".", "recursive": true, "max_depth": 3}}

    search_codebase
      {"action_type": "search_codebase", "args": {"query": "def process", "file_glob": "*.py", "context_lines": 2}}

    edit_file  ← patch is REQUIRED wrapper; use new_content OR unified_diff, not both
      {"action_type": "edit_file", "args": {"patch": {"path": "utils.py", "new_content": "...full file content..."}}}
      {"action_type": "edit_file", "args": {"patch": {"path": "utils.py", "unified_diff": "--- a/utils.py\n+++ b/utils.py\n@@ -1,4 +1,4 @@\n context\n-old line\n+new line\n context"}}}

    edit_files  ← patches is a list of patch objects
      {"action_type": "edit_files", "args": {"patches": [{"path": "a.py", "new_content": "..."}, {"path": "b.py", "unified_diff": "..."}]}}

    run_shell
      {"action_type": "run_shell", "args": {"command": "python -m ruff check . --output-format=concise", "timeout_sec": 30}}

    git_diff
      {"action_type": "git_diff", "args": {"paths": [], "stat_only": false}}

    submit  ← only when score is maximised
      {"action_type": "submit", "args": {}}

    ── CRITICAL RULES ────────────────────────────────────────────────────────

    edit_file ALWAYS uses this exact structure:
      "args": {"patch": {"path": "<file>", "new_content": "<full content>"}}
    NEVER:
      "args": {"path": "...", "new_content": "..."}   ← WRONG, missing patch wrapper

    ── UNIFIED DIFF RULES (only when using unified_diff) ────────────────────

    A valid unified diff MUST follow this exact format — every character matters:

      --- a/utils.py
      +++ b/utils.py
      @@ -<src_start>,<src_count> +<dst_start>,<dst_count> @@
       <unchanged context line>   ← leading SPACE
      -<removed line>             ← leading MINUS
      +<added line>               ← leading PLUS
       <unchanged context line>   ← leading SPACE

    Rules:
    1. Line numbers in @@ must EXACTLY match the CURRENT file state.
       If the file was already edited this episode, its line numbers changed.
       ALWAYS use new_content (full rewrite) instead of unified_diff when:
         • You already edited this file earlier in the episode, OR
         • You are unsure of exact current line numbers.
    2. Every context line (space-prefixed) must match the file verbatim.
    3. src_count = number of lines from source (context + removed).
       dst_count = number of lines in result (context + added).
    4. Include 2-3 context lines before and after each change.
    5. Do NOT include inline comments in the diff (e.g. "# F401 – ...").
       Only show the final clean lines, no annotation comments.
    6. Separate hunks with a blank line between @@ blocks.
    7. If multiple hunks are needed, recalculate each @@ line number
       accounting for the line count delta of all previous hunks.

    PREFER new_content over unified_diff for any file under ~200 lines.
    unified_diff is only beneficial for very large files (500+ lines)
    where you are making 1-2 small, targeted changes.

    ── WORKFLOW ──────────────────────────────────────────────────────────────

    1. list_directory → understand repo structure
    2. view_file → read the target file fully before editing
    3. edit_file with new_content → apply ALL fixes in one shot
    4. run_shell → python a/utils.py
    5. submit → when score is 1.0

    BEST PRACTICES:
    - Prefer new_content for a complete file rewrite over unified_diff.
    - After list_directory, ALWAYS view a file next (never list again).
    - After view_file, proceed to edit_file immediately.
    - Never call the same action_type more than 2 times in a row.

    Reply with ONLY a raw JSON object — no markdown, no explanation, no extra keys.
"""
).strip()


# ── LLM call (sync — must be run via run_in_executor to avoid blocking loop) ──
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
        print(f"[LLM_ERROR] {exc}")
        return ""


# ── Parse model output → RefactorAction ──────────────────────────────────────
def parse_action(raw: str) -> RefactorAction:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        action_type = data.get("action_type", "list_directory")
        args = data.get("args", {"path": "."})

        # Auto-fix: model sends edit_file without patch wrapper
        if (
            action_type == "edit_file"
            and "patch" not in args
            and ("path" in args and ("new_content" in args or "unified_diff" in args))
        ):
            print("[PARSE_FIX] Wrapping bare edit_file args into patch structure")
            args = {"patch": args}

        # Auto-fix: model sends edit_files without patches list
        if action_type == "edit_files" and "patches" not in args and "patch" in args:
            print("[PARSE_FIX] Wrapping single patch into patches list")
            args = {"patches": [args["patch"]]}

        return RefactorAction(action_type=action_type, params=args)

    except json.JSONDecodeError as e:
        is_truncated = len(raw) >= int(MAX_TOKENS * 3.5)
        print(
            f"[PARSE_ERROR] JSONDecodeError: {e} | len={len(raw)} "
            f"| likely_truncated={is_truncated} | tail={raw[-80:]!r}"
        )
        return RefactorAction(action_type="list_directory", params={"path": "."})
    except Exception as e:
        print(f"[PARSE_ERROR] Unexpected: {e} | raw={raw[:200]!r}")
        return RefactorAction(action_type="list_directory", params={"path": "."})


# ── Build per-step prompt from observation ────────────────────────────────────
def build_prompt(
    obs,
    step: int,
    history: list[str],
    file_cache: dict[str, str],
    diff_failures: dict[str, int] | None = None,
) -> str:
    task_id = getattr(obs, "task_id", "unknown")
    task_desc = getattr(obs, "description", "") or getattr(obs, "task_description", "")
    codebase = getattr(obs, "codebase", None)
    execution = getattr(obs, "execution", None)
    grader = getattr(obs, "grader", None)
    reward_ctx = getattr(obs, "reward_context", None)
    git_status = getattr(obs, "git", None)
    max_steps = getattr(obs, "max_steps", MAX_STEPS)

    # ── CodebaseContext ───────────────────────────────────────────────────
    file_tree = ""
    active_file = ""
    file_content = ""
    line_range = ""
    total_lines = None
    remaining = getattr(obs, "remaining_steps", MAX_STEPS - step)
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

        # Accumulate file content across view_file calls
        if active_file and file_content:
            line_start_val = getattr(codebase, "file_line_start", None)
            if active_file not in file_cache:
                file_cache[active_file] = file_content
            elif line_start_val and line_start_val > 1:
                if file_content not in file_cache[active_file]:
                    file_cache[active_file] += "\n" + file_content
                    print(
                        f"[STEP_DEBUG] file_cache updated: {active_file} "
                        f"total_len={len(file_cache[active_file])}"
                    )

    print(f"[STEP_DEBUG] file_tree={file_tree}")
    print(f"[STEP_DEBUG] active_file={active_file} content_len={len(file_content)}")

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

        print(f"[STEP_DEBUG] exec_stdout={stdout[:300].replace(chr(10), ' ')}")
        if stderr:
            print(f"[STEP_DEBUG] exec_stderr={stderr[:200].replace(chr(10), ' ')}")
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

    print(f"[STEP_DEBUG] grader_scores={scores}")
    if feedbacks:
        print(f"[STEP_DEBUG] grader_feedback={feedbacks[0][:200]}")
    if errors:
        print(f"[STEP_DEBUG] grader_errors={errors}")

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

    # ── File body — use cache if larger than current chunk ────────────────
    hist_str = "\n".join(history[-6:]) if history else "None"
    cached = file_cache.get(active_file, "") if active_file else ""
    full_content = cached if len(cached) > len(file_content) else file_content

    active_header = f"ACTIVE FILE: {active_file or 'none'}"
    if line_range:
        active_header += f" (viewing {line_range})"
    if total_lines:
        active_header += f" | TOTAL LINES: {total_lines}"
    if cached:
        active_header += f" | CACHED: {len(cached)} B"

    if full_content:
        completeness = (
            "COMPLETE"
            if len(full_content) >= (total_lines or 0) * 30
            else "MAY BE PARTIAL — use view_file for remaining lines"
        )
        file_body = f"{full_content[:10000]}\n\n[CACHE: {len(full_content)} B total — {completeness}]"
    else:
        file_body = "(no file loaded — use view_file to open one)"

    # ── Diff failure warning ──────────────────────────────────────────────
    diff_warn = ""
    if diff_failures:
        bad = [f"{p} ({n}x)" for p, n in diff_failures.items() if n >= 1]
        if bad:
            diff_warn = (
                f"\n⚠ DIFF FAILURES on: {', '.join(bad)}. "
                "The file was already edited this episode — its line numbers changed. "
                "You MUST use new_content (full file rewrite) for these files. "
                "Do NOT attempt unified_diff on them again."
            )

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
        {diff_warn}
        Choose the single best action to improve code quality.
        Remember: list_directory → view_file → edit_file (don't loop!)
        PREFER new_content over unified_diff unless the file is 500+ lines.
        Reply with ONLY a raw JSON action object.
    """
    ).strip()


# ── Single episode ────────────────────────────────────────────────────────────
async def run_episode(env: RefactoringEnv, episode_count: int, task_name: str) -> float:
    print(f"[EPISODE_START] task={task_name} episode_count={episode_count}")

    reset_result = await env.reset(episode_count=episode_count)
    obs = reset_result.observation
    done = reset_result.done

    task_desc = getattr(obs, "description", "") or getattr(obs, "task_description", "")
    print(f"[EPISODE_START] task_description={task_desc}")

    initial_score = getattr(getattr(obs, "reward_context", None), "step_score", None)
    print(
        f"[EPISODE_START] initial_score={initial_score if initial_score is not None else 'n/a'}"
    )

    history: list[str] = []
    file_cache: dict[str, str] = {}
    diff_failures: dict[str, int] = {}
    final_reward = 0.0

    for step in range(1, MAX_STEPS + 1):

        prompt = build_prompt(obs, step, history, file_cache, diff_failures)

        # ── Run LLM in executor so WebSocket keepalives are not blocked ───
        loop = asyncio.get_event_loop()
        raw_resp = await loop.run_in_executor(None, call_llm, prompt)

        # ── Loop detection: inject ruff if stuck viewing same file ────────
        if len(history) >= 3:
            last_3 = [h.split(":")[1].split("→")[0].strip() for h in history[-3:]]
            if len(set(last_3)) == 1 and "view_file" in last_3[0]:
                print(
                    "[LOOP_DETECTED] Stuck on view_file for 3 steps — injecting ruff check"
                )
                action = RefactorAction(
                    action_type="run_shell",
                    params={
                        "command": "python -m ruff check . --output-format=concise 2>&1 | head -60",
                        "timeout_sec": 30,
                    },
                )
            else:
                action = parse_action(raw_resp)
        else:
            action = parse_action(raw_resp)

        print(f"[STEP_DEBUG] raw_response={raw_resp[:300].replace(chr(10), ' ')}")
        print(
            f"[STEP_DEBUG] parsed_action={action.action_type} "
            f"params={json.dumps(action.params)}"
        )
        print(
            f"[STEP] episode={task_name} step={step}/{MAX_STEPS} "
            f"action={action.action_type}",
            flush=True,
        )

        try:
            step_result = await env.step(action)

            obs = step_result.observation

            # ── Invalidate file cache on successful edit ──────────────────
            if action.action_type in ("edit_file", "edit_files"):
                edited_paths = []
                if action.action_type == "edit_file":
                    patch = action.params.get("patch", {})
                    if isinstance(patch, dict):
                        edited_paths.append(patch.get("path", ""))
                else:
                    edited_paths.extend(
                        p.get("path", "")
                        for p in action.params.get("patches", [])
                        if isinstance(p, dict)
                    )
                for p in edited_paths:
                    if p in file_cache:
                        del file_cache[p]
                        print(
                            f"[STEP_DEBUG] file_cache invalidated: {p} (edit succeeded)"
                        )
                    # Clear diff failure counter on successful edit
                    if p in diff_failures:
                        del diff_failures[p]
                        print(
                            f"[STEP_DEBUG] diff_failures cleared: {p} (edit succeeded)"
                        )

            done = step_result.done
            reward = step_result.reward if step_result.reward is not None else 0.0
            ctx_score = getattr(
                getattr(obs, "reward_context", None), "step_score", reward
            )
            final_reward = ctx_score if ctx_score is not None else reward

            print(
                f"[STEP] reward={float(reward):.3f} "
                f"score={final_reward:.3f} done={done}"
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

        except (WsClosedError, ConnectionResetError, BrokenPipeError) as conn_err:
            print(f"[WS_DISCONNECT] step={step} error={conn_err} — skipping step")
            history.append(f"Step {step}: {action.action_type} → WS_DISCONNECT")
            continue

        except RuntimeError as e:
            error_msg = str(e)
            print(f"[STEP] error={error_msg}")

            if "File not found in sandbox: " in error_msg:
                file_path = error_msg.split("File not found in sandbox: ")[-1].split(
                    " (code:"
                )[0]
                error_feedback = (
                    f"File not found: {file_path}. "
                    "Use list_directory to verify available files."
                )
            elif "File not found" in error_msg:
                error_feedback = (
                    "File not found. Use list_directory to verify available files."
                )
            elif "EXECUTION_ERROR" in error_msg:
                error_feedback = (
                    f"Execution error: {error_msg}. "
                    "Check action parameters and try again."
                )
                # ── Track unified_diff failures per file ──────────────────
                if "Hunk" in error_msg or "hunk" in error_msg:
                    patch = action.params.get("patch", {})
                    failed_path = (
                        patch.get("path", "") if isinstance(patch, dict) else ""
                    )
                    if failed_path:
                        diff_failures[failed_path] = (
                            diff_failures.get(failed_path, 0) + 1
                        )
                        print(
                            f"[STEP_DEBUG] diff_failures[{failed_path}]="
                            f"{diff_failures[failed_path]} — will force new_content next"
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

    env = await RefactoringEnv.from_env(HF_REPO_ID, hf_token=HF_TOKEN)
    async with env:
        for task_name, episode_count in TASK_EPISODES.items():
            score = await run_episode(env, episode_count, task_name)
            results[task_name] = score

    overall = sum(results.values()) / len(results)
    for task, score in results.items():
        print(f"[END] task={task} score={score:.4f}")
    print(f"[END] overall={overall:.4f}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
