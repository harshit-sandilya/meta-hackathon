# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
inference.py  —  Baseline inference script for RefactorEnv
===========================================================

MANDATORY environment variables (set before running):
    API_BASE_URL   Base URL of the OpenAI-compatible LLM API endpoint.
    MODEL_NAME     Model identifier (e.g. "meta-llama/Llama-3.1-8B-Instruct").
    HF_TOKEN       Hugging Face token used as the API key.

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 \\
    MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \\
    HF_TOKEN=hf_... \\
    python inference.py

The script:
  1. Runs all three tasks in sequence: lint-cleanup → api-rename → style-enforce
  2. For each task, resets the environment, runs a coding-agent loop, then submits.
  3. Prints a final score table and writes baseline_scores.json to the project root.
  4. Must complete in under 20 minutes on 2 vCPU / 8 GB RAM.

Agent strategy (kept intentionally simple for a reproducible baseline):
  - view_file  → read each source file once
  - edit_file  → apply the model's suggested unified diff
  - run_shell  → verify with pytest / ruff after every edit
  - submit     → signal completion

All LLM calls use temperature=0.0 for full reproducibility.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from refactoring_environment.client import RefactorEnv
from refactoring_environment.models_internal import (
    ActionType,
    RefactorAction,
    RefactorObservation,
)

# ---------------------------------------------------------------------------
# Configuration — all credentials MUST come from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME: str = os.environ["MODEL_NAME"]
HF_TOKEN: str = os.environ["HF_TOKEN"]

SERVER_URL: str = os.environ.get("REFACTOR_ENV_URL", "http://localhost:8000")

# Per-task step budgets (kept tight to stay well inside the 20-min wall clock)
MAX_STEPS_EASY: int = 12
MAX_STEPS_MEDIUM: int = 18
MAX_STEPS_HARD: int = 25

TEMPERATURE: float = 0.0  # deterministic for reproducibility
MAX_TOKENS: int = 1024
MAX_FILE_CHARS: int = 6000  # truncate large files in prompts
MAX_OUTPUT_CHARS: int = 2000  # truncate shell output in prompts

TASKS: List[Tuple[str, int]] = [
    ("lint_cleanup", MAX_STEPS_EASY),
    ("api_rename", MAX_STEPS_MEDIUM),
    ("style_enforce", MAX_STEPS_HARD),
]

# ---------------------------------------------------------------------------
# System prompt — describes the coding-agent contract to the model
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a senior Python engineer performing code refactoring tasks.
    You work inside a git-backed sandbox through a strict JSON action interface.

    AVAILABLE ACTIONS (choose exactly one per turn):

    1. view_file        — read a file (or a line window)
       {"action_type": "view_file", "params": {"path": "utils.py"}}
       {"action_type": "view_file", "params": {"path": "utils.py", "line_start": 1, "line_end": 40}}

    2. list_directory   — list files in a directory
       {"action_type": "list_directory", "params": {"path": ".", "recursive": true}}

    3. search_codebase  — grep across .py files
       {"action_type": "search_codebase", "params": {"query": "process_data", "context_lines": 2}}

    4. git_diff         — show current changes vs baseline
       {"action_type": "git_diff", "params": {"paths": [], "stat_only": false}}

    5. edit_file        — apply a unified diff to one file
       {"action_type": "edit_file", "params": {"patch": {"path": "utils.py", "unified_diff": "--- a/utils.py\\n+++ b/utils.py\\n@@ … @@\\n-old line\\n+new line"}}}

    6. edit_files       — apply diffs to multiple files atomically (max 20)
       {"action_type": "edit_files", "params": {"patches": [{"path": "a.py", "unified_diff": "…"}, {"path": "b.py", "new_content": "…"}]}}

    7. run_shell        — run a whitelisted command (pytest, ruff, mypy, …)
       {"action_type": "run_shell", "params": {"command": "pytest tests/ -q --tb=short"}}
       {"action_type": "run_shell", "params": {"command": "ruff check . --output-format=json"}}

    8. submit           — declare you are done (ends the episode)
       {"action_type": "submit", "params": {}}

    RULES:
    - Reply with ONE valid JSON object and nothing else — no prose, no markdown fences.
    - Use unified_diff when changing existing files; use new_content only when creating a new file or replacing a tiny file entirely.
    - Unified diff format: "--- a/path\\n+++ b/path\\n@@ -start,count +start,count @@\\n context\\n-removed\\n+added"
    - Never edit test files unless the task explicitly asks you to improve test coverage.
    - After every edit_file, call run_shell with pytest to verify tests still pass.
    - Call submit only when all tests pass and the objective is satisfied.
    - If you are unsure what to do next, call view_file or search_codebase — never submit prematurely.
"""
).strip()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int, label: str = "") -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    note = f"\n… [{label} truncated — {len(text) - max_chars} chars omitted] …\n"
    return text[:half] + note + text[-half:]


def build_initial_prompt(obs: RefactorObservation) -> str:
    """Build the opening user message from the reset observation."""
    file_list = (
        "\n".join(
            f"  {'[dir] ' if e.is_dir else '      '}{e.path}" for e in obs.file_tree
        )
        or "  (empty)"
    )

    baseline_tests = obs.baseline_test_summary
    baseline_lint = obs.baseline_lint_summary

    return textwrap.dedent(
        f"""
        TASK: {obs.task_id}  (difficulty: {obs.difficulty.value})
        OBJECTIVE: {obs.objective}

        CONSTRAINTS:
        {chr(10).join(f'  - {c}' for c in obs.constraints) or '  (none)'}

        STEP BUDGET: {obs.max_steps} steps remaining

        FILE TREE:
        {file_list}

        BASELINE METRICS:
          Tests : {baseline_tests.passed}/{baseline_tests.total} passing
          Lint  : {baseline_lint.total_errors} violations
                  {json.dumps(baseline_lint.error_by_code, indent=4) if baseline_lint.error_by_code else '{}'}

        Start by viewing the source files to understand what needs to change,
        then make targeted edits, verify with pytest and ruff, then submit.
    """
    ).strip()


def build_step_prompt(
    obs: RefactorObservation,
    history: List[str],
) -> str:
    """Build the user message for step N from the latest observation."""
    # Last action feedback
    last_output = ""
    if obs.last_action_output:
        last_output = _truncate(obs.last_action_output, MAX_OUTPUT_CHARS, "output")
    if obs.last_action_error:
        last_output += f"\n⚠ ERROR: {obs.last_action_error}"

    # Active file content (if a view_file was the previous action)
    file_section = ""
    if obs.active_file and obs.file_content:
        content = _truncate(obs.file_content, MAX_FILE_CHARS, obs.active_file)
        lines_info = (
            f"lines {obs.file_line_start}–{obs.file_line_end} of {obs.total_file_lines}"
            if obs.file_line_start
            else f"all {obs.total_file_lines} lines"
        )
        file_section = (
            f"\nFILE [{obs.active_file}  {lines_info}]:\n```python\n{content}\n```"
        )

    # Recent git changes
    git_section = ""
    if obs.git_status.diff_stat:
        git_section = f"\nGIT DIFF STAT:\n{obs.git_status.diff_stat}"

    # Step score feedback (dense reward signal embedded in obs)
    score_line = ""
    if obs.step_score is not None:
        score_line = f"  Step score : {obs.step_score:.3f}"
    if obs.cumulative_penalty > 0:
        score_line += f"  |  Cumulative penalty: {obs.cumulative_penalty:.3f}"
    if obs.violations:
        score_line += f"\n  Violations : {'; '.join(obs.violations)}"

    # History (last 6 steps to stay within context)
    history_text = "\n".join(history[-6:]) if history else "  (none yet)"

    return textwrap.dedent(
        f"""
        STEP {obs.current_step}/{obs.max_steps}  (remaining: {obs.remaining_steps})
        {score_line}

        CURRENT METRICS:
          Tests : {obs.test_summary.passed}/{obs.test_summary.total} passing
                  failed={obs.test_summary.failed}  errors={obs.test_summary.errors}
          Lint  : {obs.lint_summary.total_errors} violations
        {file_section}
        {git_section}

        LAST ACTION OUTPUT:
        {last_output or '(none)'}

        RECENT HISTORY:
        {history_text}

        What is your next action? Reply with a single JSON object.
    """
    ).strip()


# ---------------------------------------------------------------------------
# Action parsing — model output → RefactorAction
# ---------------------------------------------------------------------------

# Strip optional markdown code fences the model might accidentally add
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def parse_action(response_text: str) -> Optional[RefactorAction]:
    """
    Parse the LLM's response into a validated RefactorAction.

    Returns None if parsing or validation fails so the caller can
    fall back to a safe no-op (submit with a note).
    """
    if not response_text:
        return None

    cleaned = _strip_fences(response_text)

    # The model sometimes wraps the JSON in outer prose; try to extract
    # the first {...} block if direct parsing fails.
    try:
        data: Dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    try:
        return RefactorAction.model_validate(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------


def run_task(
    env: RefactorEnv,
    client: OpenAI,
    task_id: str,
    max_steps: int,
) -> float:
    """
    Run one complete episode on *task_id* and return the final score.

    Episode lifecycle:
      reset() → [view / edit / run_shell …] × N → submit()

    Returns the final episode score in [0.0, 1.0].
    """
    print(f"\n{'=' * 55}")
    print(f"  TASK: {task_id}  (budget: {max_steps} steps)")
    print(f"{'=' * 55}")

    # ── reset ────────────────────────────────────────────────────────────────
    reset_result = env.reset(task_id=task_id)
    obs: RefactorObservation = reset_result.observation

    print(f"  Objective : {obs.objective}")
    print(
        f"  Baseline  : {obs.baseline_test_summary.passed}/"
        f"{obs.baseline_test_summary.total} tests passing, "
        f"{obs.baseline_lint_summary.total_errors} lint violations"
    )

    # Build conversation: system + initial user message
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_prompt(obs)},
    ]

    history: List[str] = []
    final_score: float = 0.0

    # ── episode loop ─────────────────────────────────────────────────────────
    for step in range(1, max_steps + 1):
        if reset_result.done:
            print(f"  Environment signalled done at step {step - 1}.")
            break

        # Add the step observation as the next user turn
        if step > 1:
            messages.append(
                {"role": "user", "content": build_step_prompt(obs, history)}
            )

        # ── LLM call ─────────────────────────────────────────────────────────
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text: str = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  ⚠ LLM request failed at step {step}: {exc}")
            response_text = ""

        # Append assistant turn to the running conversation
        messages.append({"role": "assistant", "content": response_text})

        # ── Parse action ──────────────────────────────────────────────────────
        action = parse_action(response_text)

        if action is None:
            print(f"  Step {step:>2}: ⚠ could not parse action — submitting early")
            print(f"           raw response: {response_text[:200]!r}")
            # Safe fallback: submit so we get whatever score we earned so far
            action = RefactorAction(
                action_type=ActionType.submit,
                params={"note": f"parse failure at step {step}"},
            )

        action_label = action.action_type.value
        if action.action_type == ActionType.edit_file:
            patch_path = action.params.get("patch", {}).get("path", "?")
            action_label = f"edit_file({patch_path})"
        elif action.action_type == ActionType.run_shell:
            action_label = f"run_shell({action.params.get('command', '')[:60]})"
        elif action.action_type == ActionType.view_file:
            action_label = f"view_file({action.params.get('path', '?')})"

        # ── Step the environment ──────────────────────────────────────────────
        try:
            step_result = env.step(action)
        except Exception as exc:
            print(f"  Step {step:>2}: ⚠ env.step() raised: {exc}")
            traceback.print_exc()
            break

        obs = step_result.observation
        reward = step_result.reward or 0.0
        done = step_result.done

        score_str = (
            f"{reward:+.3f}" if obs.step_score is None else f"{obs.step_score:.3f}"
        )
        error_flag = "  ⚠ ERROR" if obs.last_action_error else ""
        print(f"  Step {step:>2}: {action_label:<45} score={score_str}{error_flag}")

        history_line = f"Step {step}: {action_label} → score {score_str}" + (
            f" | error: {obs.last_action_error[:80]}" if obs.last_action_error else ""
        )
        history.append(history_line)

        if reward > 0:
            final_score = reward

        if done:
            final_score = reward
            print(f"  Episode complete at step {step}.")
            break

        # Auto-submit if we hit the last step
        if step == max_steps:
            print(f"  Reached max steps ({max_steps}) — auto-submitting.")
            try:
                submit_result = env.submit(note=f"auto-submit at step {step}")
                final_score = submit_result.reward or final_score
            except Exception as exc:
                print(f"  ⚠ Auto-submit failed: {exc}")

    print(f"  Final score: {final_score:.4f}")
    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Validate required env vars early so the error is clear
    missing = [
        v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(v)
    ]
    if missing:
        print(
            f"ERROR: missing required environment variables: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print(f"\nRefactorEnv Baseline Inference")
    print(f"  Server     : {SERVER_URL}")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Tasks      : {[t for t, _ in TASKS]}")

    results: Dict[str, float] = {}

    with RefactorEnv(base_url=SERVER_URL) as env:
        for task_id, max_steps in TASKS:
            try:
                score = run_task(env, client, task_id, max_steps)
            except Exception as exc:
                print(f"\n⚠ Task '{task_id}' crashed: {exc}")
                traceback.print_exc()
                score = 0.0
            results[task_id] = score

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("  BASELINE SCORES SUMMARY")
    print(f"{'=' * 55}")
    for task_id, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<22}  {score:.4f}  {bar}")
    overall = sum(results.values()) / len(results) if results else 0.0
    print(f"  {'overall average':<22}  {overall:.4f}")
    print(f"{'=' * 55}\n")

    # ── Persist for README / reproducibility ─────────────────────────────────
    output = {
        "model": MODEL_NAME,
        "server": SERVER_URL,
        "tasks": results,
        "overall_average": overall,
    }
    out_path = os.path.join(os.path.dirname(__file__), "baseline_scores.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Scores written to {out_path}")


if __name__ == "__main__":
    main()
