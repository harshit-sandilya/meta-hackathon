"""
environment.py  —  RefactorEnvCore orchestrator.

This class owns the episode lifecycle:
  reset()  → init sandbox, collect baseline metrics, build initial obs
  step()   → dispatch action, update state, collect metrics, return obs
  teardown() → clean up sandbox

It is NOT an OpenEnv Environment subclass — that's the adapter's job.
This separation keeps core logic testable without any HTTP/WebSocket layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from refactor_env.environment.core.executor import Executor
from refactor_env.environment.core.sandbox import Sandbox
from refactor_env.environment.core.metrics import collect_metrics, GraderResults
from refactor_env.environment.core.reward import compute_reward, compute_step_reward
from refactor_env.environment.core.exceptions import SandboxNotInitializedError
from refactor_env.environment.registry import TaskRegistry, ScenarioSpec
from refactor_env.models import (
    ActionType,
    RefactorAction,
    RefactorObservation,
    RefactorReward,
    RefactorState,
    Difficulty,
    FileTreeEntry,
    TestSummary,
    LintSummary,
    GitStatus,
    ViewFileParams,
    ListDirectoryParams,
    SearchCodebaseParams,
    GitDiffParams,
    EditFileParams,
    EditFilesParams,
    RunShellParams,
    SubmitParams,
)


class RefactorEnvCore:
    """
    Core episode orchestrator for refactor-env.

    One instance = one episode.  Not thread-safe within a single instance
    (OpenEnv ensures sequential step() calls per session).

    Parameters
    ----------
    tasks_root : str
        Path to the tasks/ directory.
    project_root : str
        Project root for resolving repo_path in scenario.yaml.
    episode_id : str, optional
        Stable ID carried through State and observations.
    """

    def __init__(
        self,
        tasks_root: str,
        project_root: str,
        episode_id: Optional[str] = None,
    ) -> None:
        from uuid import uuid4

        self._episode_id = episode_id or str(uuid4())
        self._tasks_root = tasks_root
        self._project_root = project_root
        self._registry = TaskRegistry.from_tasks_dir(tasks_root, project_root)
        self._sandbox: Optional[Sandbox] = None
        self._spec: Optional[ScenarioSpec] = None
        self._state: Optional[RefactorState] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, task_id: str) -> RefactorObservation:
        """Init sandbox with task files, record baseline, return initial obs."""
        spec = self._registry.get(task_id)
        self._spec = spec

        executor = Executor()
        sandbox = Sandbox(executor=executor, episode_id=self._episode_id)
        baseline_sha = sandbox.init(spec.task_files)
        self._sandbox = sandbox

        # Collect baseline metrics (used for regression detection in reward.py)
        baseline_results, _ = collect_metrics(
            sandbox=sandbox,
            spec=spec,
            step_count=0,
        )

        baseline_metrics = {
            "acc": baseline_results.score_for("acc"),
            "qual": baseline_results.score_for("qual"),
            "lint_errors": baseline_results.lint_summary.total_errors,
        }

        self._state = RefactorState(
            episode_id=self._episode_id,
            task_id=task_id,
            difficulty=Difficulty(spec.difficulty),
            step_count=0,
            done=False,
            sandbox_path=sandbox.root,
            baseline_commit=baseline_sha,
            baseline_metrics=baseline_metrics,
            last_metrics=baseline_metrics,
            best_metrics=baseline_metrics,
        )

        return self._build_observation(
            last_output=None,
            last_error=None,
            step_score=None,
            reward_result=None,
        )

    def step(self, action: RefactorAction) -> RefactorObservation:
        """Dispatch action → update state → build observation."""
        self._assert_live()
        assert self._state is not None and self._sandbox is not None
        assert self._spec is not None

        last_output: Optional[str] = None
        last_error: Optional[str] = None
        reward_result: Optional[RefactorReward] = None

        # ── Dispatch action ───────────────────────────────────────────────────
        try:
            last_output = self._dispatch(action)
        except Exception as exc:
            last_error = str(exc)

        # ── Update step count + noop detection ────────────────────────────────
        self._state = RefactorState(
            **{
                **self._state.__dict__,
                "step_count": self._state.step_count + 1,
                "action_history": self._state.action_history
                + [
                    {
                        "step": self._state.step_count + 1,
                        "action_type": action.action_type,
                        "params_hash": _hash_params(action.params),
                        "had_error": last_error is not None,
                    }
                ],
                "consecutive_noop_count": self._update_noop_count(action),
            }
        )

        is_submit = action.action_type == ActionType.submit
        max_reached = self._state.step_count >= self._spec.max_steps

        # ── Collect metrics (always — dense reward signal) ────────────────────
        grader_results, grader_errors = collect_metrics(
            sandbox=self._sandbox,
            spec=self._spec,
            step_count=self._state.step_count,
        )

        # ── Compute reward ────────────────────────────────────────────────────
        if is_submit or max_reached:
            git_diff = self._sandbox.git_diff()
            violations = self._spec.check_invariants(self._sandbox.root, git_diff)
            reward_result = compute_reward(
                grader_results=grader_results,
                state=self._state,
                spec=self._spec,
                invariant_violations=violations,
                is_submit=True,
            )
            self._state = RefactorState(
                **{
                    **self._state.__dict__,
                    "done": True,
                    "accumulated_penalty": (
                        self._state.accumulated_penalty + reward_result.penalty
                    ),
                    "violations": violations,
                }
            )
        else:
            reward_result = compute_step_reward(
                grader_results=grader_results,
                state=self._state,
                spec=self._spec,
            )

        # Update best_metrics
        self._state = RefactorState(
            **{
                **self._state.__dict__,
                "last_metrics": {
                    "acc": grader_results.score_for("acc"),
                    "qual": grader_results.score_for("qual"),
                    "lint_errors": grader_results.lint_summary.total_errors,
                },
                "best_metrics": _update_best(self._state.best_metrics, grader_results),
            }
        )

        # Teardown on episode end
        if self._state.done:
            self._sandbox.teardown()

        return self._build_observation(
            last_output=last_output,
            last_error=last_error,
            step_score=reward_result.score,
            reward_result=reward_result,
        )

    def teardown(self) -> None:
        """Force-cleanup sandbox regardless of episode state."""
        if self._sandbox is not None:
            try:
                self._sandbox.teardown()
            except Exception:
                pass
            finally:
                self._sandbox = None

    # ── Action dispatch ───────────────────────────────────────────────────────

    def _dispatch(self, action: RefactorAction) -> str:
        """Route action_type to the correct sandbox/executor method."""
        t = action.action_type
        p = action.typed_params  # typed sub-model via property on RefactorAction
        sb = self._sandbox
        assert sb is not None

        if t == ActionType.view_file:
            p: ViewFileParams
            return sb.read_file(p.path, p.line_start, p.line_end)

        elif t == ActionType.list_directory:
            p: ListDirectoryParams
            entries = sb.file_tree(p.path, recursive=p.recursive, max_depth=p.max_depth)
            return _format_tree(entries)

        elif t == ActionType.search_codebase:
            p: SearchCodebaseParams
            executor = Executor(sandbox_root=sb.root)
            flag_i = "-i" if p.case_insensitive else ""
            cmd = (
                f"grep -rn {flag_i} "
                f"-A {p.context_lines} -B {p.context_lines} "
                f"--include='{p.file_glob}' "
                f"'{p.query}' ."
            )
            rc, stdout, stderr = executor.run(cmd, workdir=sb.root, timeout_sec=15)
            output = stdout or "(no matches)"
            # Trim to max_results lines
            lines = output.splitlines()
            if len(lines) > p.max_results:
                output = (
                    "\n".join(lines[: p.max_results])
                    + f"\n... [{len(lines)-p.max_results} lines truncated]"
                )
            return output

        elif t == ActionType.git_diff:
            p: GitDiffParams
            return sb.git_diff(p.paths or None, stat_only=p.stat_only)

        elif t == ActionType.edit_file:
            p: EditFileParams
            modified = sb.apply_atomic_patches([p.patch])
            return f"Modified: {', '.join(modified)}"

        elif t == ActionType.edit_files:
            p: EditFilesParams
            modified = sb.apply_atomic_patches(p.patches)
            return f"Modified {len(modified)} file(s): {', '.join(modified)}"

        elif t == ActionType.run_shell:
            p: RunShellParams
            executor = Executor(sandbox_root=sb.root)
            rc, stdout, stderr = executor.run(
                p.command,
                workdir=os.path.join(sb.root, p.workdir),
                timeout_sec=p.timeout_sec,
            )
            output = stdout
            if stderr:
                output += f"\n[stderr]\n{stderr}"
            if rc != 0:
                output += f"\n[exit code: {rc}]"
            return output

        elif t == ActionType.submit:
            # submit has no side-effects in dispatch — reward computed in step()
            p: SubmitParams
            note = p.note or ""
            return f"Submit received. {note}".strip()

        else:
            raise ValueError(f"Unknown action_type: {t}")

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_observation(
        self,
        last_output: Optional[str],
        last_error: Optional[str],
        step_score: Optional[float],
        reward_result: Optional[RefactorReward],
    ) -> RefactorObservation:
        assert self._state is not None
        assert self._spec is not None
        sb = self._sandbox

        # File tree (None after teardown on terminal step)
        file_tree: List[FileTreeEntry] = []
        git_status = GitStatus()
        if sb is not None:
            try:
                file_tree = sb.file_tree()
                diff_stat = sb.git_diff(stat_only=True)
                git_status = GitStatus(diff_stat=diff_stat)
            except Exception:
                pass

        # Test / lint summaries from last metric collection
        # On reset, pull from baseline_metrics
        ts = TestSummary()
        ls = LintSummary()
        if reward_result is not None:
            # reward_result is only set after step(); it carries grader detail
            # via its info dict — we rebuild here
            ts = TestSummary(
                total=reward_result.info.get("tests_total", 0),
                passed=reward_result.info.get("tests_passed", 0),
            )
            ls = LintSummary(total_errors=reward_result.info.get("lint_errors", 0))

        remaining = max(0, self._spec.max_steps - self._state.step_count)

        return RefactorObservation(
            # Task identity
            task_id=self._spec.slug,
            difficulty=Difficulty(self._spec.difficulty),
            objective=self._spec.description,
            constraints=[inv.message for inv in self._spec.invariants],
            # Step budget
            current_step=self._state.step_count,
            max_steps=self._spec.max_steps,
            remaining_steps=remaining,
            # Codebase context
            file_tree=file_tree,
            git_status=git_status,
            # Execution feedback
            test_summary=ts,
            lint_summary=ls,
            last_action_output=last_output,
            last_action_error=last_error,
            # Baseline for delta display
            baseline_test_summary=TestSummary(
                passed=int(self._state.baseline_metrics.get("acc", 1.0) * 1),
            ),
            baseline_lint_summary=LintSummary(
                total_errors=int(self._state.baseline_metrics.get("lint_errors", 0))
            ),
            # Dense reward signal (embedded so LLM can see progress in context)
            step_score=step_score,
            cumulative_penalty=self._state.accumulated_penalty,
            violations=self._state.violations,
            # Meta
            episode_id=self._episode_id,
        )

    # ── Noop detection ────────────────────────────────────────────────────────

    def _update_noop_count(self, action: RefactorAction) -> int:
        """
        Increment noop counter if the last N actions are identical by
        (action_type, params_hash).  Resets on any novel action.
        """
        assert self._state is not None
        history = self._state.action_history
        if len(history) < 2:
            return 0
        last = history[-1]
        prev = history[-2]
        if (
            last["action_type"] == prev["action_type"]
            and last["params_hash"] == prev["params_hash"]
        ):
            return self._state.consecutive_noop_count + 1
        return 0

    def _assert_live(self) -> None:
        if self._sandbox is None or self._state is None:
            raise SandboxNotInitializedError(
                "Episode not initialised. Call reset() first."
            )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hash_params(params: dict) -> str:
    """Deterministic 8-char hash of action params for noop detection."""
    serialised = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(serialised.encode()).hexdigest()[:8]


def _format_tree(entries: List[FileTreeEntry]) -> str:
    """Render file tree entries as an indented string."""
    lines = []
    for e in entries:
        prefix = "  " * e.path.count(os.sep)
        icon = "📁 " if e.is_dir else "📄 "
        size = f"  ({e.size_bytes} B)" if not e.is_dir and e.size_bytes else ""
        lines.append(f"{prefix}{icon}{os.path.basename(e.path)}{size}")
    return "\n".join(lines) if lines else "(empty)"


def _update_best(current_best: Dict[str, Any], gr: GraderResults) -> Dict[str, Any]:
    """Keep a high-water mark of best metrics seen so far in the episode."""
    return {
        "acc": max(current_best.get("acc", 0.0), gr.score_for("acc")),
        "qual": max(current_best.get("qual", 0.0), gr.score_for("qual")),
        "lint_errors": min(
            current_best.get("lint_errors", 9999),
            gr.lint_summary.total_errors,
        ),
    }
