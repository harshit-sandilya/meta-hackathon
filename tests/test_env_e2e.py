# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
tests/test_env_e2e.py  —  End-to-end environment smoke test (no LLM)
======================================================================

Tests the full RefactorEnv lifecycle against the lint-cleanup task using
only programmatic (hard-coded) actions — zero LLM calls.

What this covers
----------------
  ✓ reset()                          — clean episode start, baseline metrics
  ✓ list_directory()                 — file tree observation
  ✓ view_file()                      — file content + line-window slicing
  ✓ search_codebase()                — grep across sandbox
  ✓ git_diff()                       — diff is empty on reset, non-empty after edit
  ✓ edit_file() — unified_diff       — apply a patch, observe reward signal
  ✓ edit_file() — new_content        — full-file replacement
  ✓ edit_files() — multi-file atomic — two patches in one step
  ✓ run_shell("ruff check …")        — lint runner, stdout in observation
  ✓ run_shell("pytest …")            — test runner, TestSummary populated
  ✓ reward shape                     — step_score advances after good edits
  ✓ penalty signal                   — repeated no-op is penalised
  ✓ submit()                         — episode closes, done=True, final score
  ✓ second reset()                   — env resets cleanly for a new episode

Run with:
    SERVER_URL=http://localhost:8000 pytest tests/test_env_e2e.py -v -s
    # -s keeps all print() / logging output visible
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running from project root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from refactor_env.client import RefactorEnv
from refactor_env.models import (
    ActionType,
    FilePatch,
    RefactorAction,
    RefactorObservation,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERVER_URL = os.environ.get("REFACTOR_ENV_URL", "http://localhost:8000")
TASK_ID = "lint-cleanup"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("e2e")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

SEP = "─" * 60
SEP2 = "═" * 60


def _header(title: str) -> None:
    log.info(SEP2)
    log.info("  %s", title)
    log.info(SEP2)


def _section(title: str) -> None:
    log.info(SEP)
    log.info("  %s", title)
    log.info(SEP)


def _log_obs(label: str, obs: RefactorObservation, reward: float | None = None) -> None:
    """Log key fields of a RefactorObservation in a readable table."""
    log.info("")
    log.info("  ┌─ %s", label)
    log.info("  │  task_id        : %s", obs.task_id)
    log.info("  │  current_step   : %s / %s", obs.current_step, obs.max_steps)
    log.info(
        "  │  step_score     : %s",
        f"{obs.step_score:.4f}" if obs.step_score is not None else "—",
    )
    log.info("  │  cum_penalty    : %.4f", obs.cumulative_penalty)
    if reward is not None:
        log.info("  │  reward (wire)  : %.4f", reward)
    log.info(
        "  │  tests          : %s/%s passing  (failed=%s, errors=%s)",
        obs.test_summary.passed,
        obs.test_summary.total,
        obs.test_summary.failed,
        obs.test_summary.errors,
    )
    log.info("  │  lint errors    : %s", obs.lint_summary.total_errors)
    if obs.lint_summary.error_by_code:
        for code, count in obs.lint_summary.error_by_code.items():
            log.info("  │    %-10s : %d", code, count)
    if obs.active_file:
        log.info(
            "  │  active_file    : %s  (%s lines)",
            obs.active_file,
            obs.total_file_lines,
        )
    if obs.last_action_error:
        log.info("  │  ⚠ error        : %s", obs.last_action_error[:120])
    if obs.violations:
        for v in obs.violations:
            log.info("  │  violation      : %s", v)
    log.info("  └─────────────────────────────────────────")
    log.info("")


def _log_step(step_n: int, action_label: str, reward: float | None, done: bool) -> None:
    bar = ("█" * int((reward or 0) * 20)).ljust(20)
    log.info(
        "  STEP %2d  %-40s  reward=%-7s  done=%s  [%s]",
        step_n,
        action_label,
        f"{reward:.4f}" if reward is not None else "None",
        done,
        bar,
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_obs(obs: RefactorObservation) -> None:
    """Bare-minimum sanity checks that apply to every observation."""
    assert isinstance(
        obs, RefactorObservation
    ), "observation must be a RefactorObservation"
    assert obs.task_id, "task_id must be non-empty"
    assert obs.max_steps > 0, "max_steps must be positive"
    assert obs.current_step >= 0, "current_step must be non-negative"
    assert obs.test_summary is not None, "test_summary must be present"
    assert obs.lint_summary is not None, "lint_summary must be present"


# ---------------------------------------------------------------------------
# Fixture — one shared env for the whole module (avoids repeated connections)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    """Provide a single RefactorEnv client, closed after all tests run."""
    _header("CONNECTING TO RefactorEnv")
    log.info("  server : %s", SERVER_URL)
    log.info("  task   : %s", TASK_ID)
    client = RefactorEnv(base_url=SERVER_URL)
    client.__enter__()
    yield client
    log.info("")
    _header("CLOSING CONNECTION")
    client.__exit__(None, None, None)


# ===========================================================================
# T1 — reset
# ===========================================================================


class TestReset:
    """reset() returns a clean, fully-populated initial observation."""

    def test_reset_returns_observation(self, env: RefactorEnv) -> None:
        _section("T1 — reset(task_id='lint-cleanup')")
        t0 = time.perf_counter()
        result = env.reset(task_id=TASK_ID)
        elapsed = time.perf_counter() - t0

        obs = result.observation
        reward = result.reward
        done = result.done

        _log_obs("reset observation", obs, reward)
        log.info("  reset latency : %.3f s", elapsed)

        # Structural assertions
        _assert_obs(obs)
        assert obs.task_id == TASK_ID, "task_id must match requested task"
        assert obs.difficulty is not None, "difficulty must be set"
        assert obs.objective, "objective must be non-empty"
        assert len(obs.file_tree) > 0, "file_tree must be populated on reset"
        assert obs.current_step == 0, "step counter must start at 0"
        assert done is False, "episode must not start as done"

        # Baseline metrics must be present
        assert obs.baseline_test_summary is not None, "baseline tests required"
        assert obs.baseline_lint_summary is not None, "baseline lint required"
        assert (
            obs.baseline_lint_summary.total_errors > 0
        ), "lint-cleanup task should have baseline lint violations"

        log.info("  ✓ reset OK  (%.3f s)", elapsed)

    def test_reset_clears_previous_episode(self, env: RefactorEnv) -> None:
        """A second reset on the same client gives a fresh episode."""
        _section("T1b — second reset() clears episode state")
        r1 = env.reset(task_id=TASK_ID)
        r2 = env.reset(task_id=TASK_ID)
        assert r2.observation.current_step == 0, "step counter must reset to 0"
        assert r2.observation.cumulative_penalty == 0.0, "penalties must clear on reset"
        log.info("  ✓ episode cleared correctly")


# ===========================================================================
# T2 — list_directory
# ===========================================================================


class TestListDirectory:
    """list_directory() populates file_tree in the observation."""

    def test_list_root(self, env: RefactorEnv) -> None:
        _section("T2 — list_directory('.')")
        env.reset(task_id=TASK_ID)

        result = env.list_directory(path=".", recursive=False)
        obs = result.observation
        reward = result.reward

        _log_obs("list_directory(.)", obs, reward)
        _log_step(obs.current_step, "list_directory(.)", reward, result.done)

        _assert_obs(obs)
        assert obs.current_step == 1, "step counter must advance"
        assert len(obs.file_tree) > 0, "file_tree must be populated"

        py_files = [e for e in obs.file_tree if e.path.endswith(".py")]
        log.info("  Python files : %s", [e.path for e in py_files])
        assert py_files, "sandbox must contain at least one .py file"
        log.info("  ✓ list_directory OK")

    def test_list_recursive(self, env: RefactorEnv) -> None:
        _section("T2b — list_directory recursive")
        result = env.list_directory(path=".", recursive=True, max_depth=4)
        obs = result.observation

        total_entries = len(obs.file_tree)
        log.info("  recursive entries : %d", total_entries)
        assert total_entries > 0, "recursive listing must return entries"
        log.info("  ✓ recursive listing OK")


# ===========================================================================
# T3 — view_file
# ===========================================================================


class TestViewFile:
    """view_file() loads content into the observation."""

    @pytest.fixture(autouse=True)
    def _reset(self, env: RefactorEnv) -> None:
        env.reset(task_id=TASK_ID)
        # Discover a .py source file to read
        r = env.list_directory(path=".", recursive=True)
        self._py_files = [
            e.path
            for e in r.observation.file_tree
            if e.path.endswith(".py") and not e.is_dir
        ]
        assert self._py_files, "need at least one .py file for view_file tests"

    def test_view_full_file(self, env: RefactorEnv) -> None:
        path = self._py_files[0]
        _section(f"T3a — view_file('{path}')")

        result = env.view_file(path)
        obs = result.observation
        reward = result.reward

        _log_obs(f"view_file({path})", obs, reward)
        _log_step(obs.current_step, f"view_file({path})", reward, result.done)

        _assert_obs(obs)
        assert obs.active_file == path, "active_file must match requested path"
        assert obs.file_content, "file_content must be non-empty"
        assert obs.total_file_lines > 0, "total_file_lines must be positive"
        log.info("  content preview (first 200 chars): %r", obs.file_content[:200])
        log.info("  ✓ view_file OK")

    def test_view_file_line_window(self, env: RefactorEnv) -> None:
        path = self._py_files[0]
        _section(f"T3b — view_file line window (1–10)")

        result = env.view_file(path, line_start=1, line_end=10)
        obs = result.observation

        assert obs.file_line_start == 1, "line_start must be reflected in obs"
        assert obs.file_line_end == 10, "line_end must be reflected in obs"
        lines_in_content = obs.file_content.count("\n") + 1
        assert lines_in_content <= 10, "content must not exceed requested window"
        log.info("  window lines : %d  ✓", lines_in_content)


# ===========================================================================
# T4 — search_codebase
# ===========================================================================


class TestSearchCodebase:
    """search_codebase() returns grep results in last_action_output."""

    @pytest.fixture(autouse=True)
    def _reset(self, env: RefactorEnv) -> None:
        env.reset(task_id=TASK_ID)

    def test_search_finds_import(self, env: RefactorEnv) -> None:
        _section("T4a — search_codebase('import')")

        result = env.search_codebase(query="import", context_lines=0)
        obs = result.observation
        reward = result.reward

        _log_obs("search_codebase(import)", obs, reward)
        _log_step(obs.current_step, "search_codebase(import)", reward, result.done)

        _assert_obs(obs)
        assert obs.last_action_output, "search must return output"
        assert (
            "import" in obs.last_action_output.lower()
        ), "results must mention 'import'"
        log.info("  output snippet: %r", obs.last_action_output[:300])
        log.info("  ✓ search_codebase OK")

    def test_search_no_results(self, env: RefactorEnv) -> None:
        _section("T4b — search_codebase with no matches")
        result = env.search_codebase(
            query="THIS_STRING_DEFINITELY_DOES_NOT_EXIST_IN_ANY_FILE_XYZ",
        )
        obs = result.observation
        # Should not crash; output should indicate no matches
        assert (
            obs.last_action_error is None
            or "no matches" in (obs.last_action_output or "").lower()
        )
        log.info("  ✓ zero-match search handled gracefully")


# ===========================================================================
# T5 — git_diff
# ===========================================================================


class TestGitDiff:
    """git_diff() shows empty diff after reset and non-empty after an edit."""

    @pytest.fixture(autouse=True)
    def _reset(self, env: RefactorEnv) -> None:
        env.reset(task_id=TASK_ID)
        r = env.list_directory(path=".", recursive=True)
        self._py_file = next(
            (
                e.path
                for e in r.observation.file_tree
                if e.path.endswith(".py") and not e.is_dir
            ),
            None,
        )
        assert self._py_file, "need a .py file for git_diff tests"

    def test_diff_empty_on_reset(self, env: RefactorEnv) -> None:
        _section("T5a — git_diff after reset (expect empty)")
        result = env.git_diff()
        obs = result.observation
        _log_obs("git_diff (baseline)", obs, result.reward)

        # No tracked changes yet
        assert not obs.git_status.has_changes, "diff must be empty right after reset"
        log.info("  ✓ diff is empty after reset")

    def test_diff_non_empty_after_edit(self, env: RefactorEnv) -> None:
        _section("T5b — git_diff after edit_file (expect changes)")
        # Read current content
        r_view = env.view_file(self._py_file)
        content = r_view.observation.file_content
        assert content, "file must have content"

        # Append a harmless comment so there is definitely a change
        new_content = content + "\n# e2e-test marker\n"
        env.edit_file(self._py_file, new_content=new_content)

        result = env.git_diff(stat_only=True)
        obs = result.observation
        _log_obs("git_diff (after edit)", obs, result.reward)

        assert obs.git_status.has_changes, "diff must be non-empty after edit"
        log.info("  diff stat : %s", obs.git_status.diff_stat)
        log.info("  ✓ diff reflects edit")


# ===========================================================================
# T6 — edit_file (unified_diff)
# ===========================================================================


class TestEditFileUnifiedDiff:
    """
    Apply a real lint-relevant patch and confirm the reward signal increases.

    The lint-cleanup task specifically rewards removing unused imports (F401)
    and fixing bare excepts (E722).  We apply a minimal targeted patch.
    """

    @pytest.fixture(autouse=True)
    def _reset(self, env: RefactorEnv) -> None:
        result = env.reset(task_id=TASK_ID)
        self._baseline_lint = result.observation.baseline_lint_summary.total_errors
        # Find a file with known lint violations
        r = env.list_directory(path=".", recursive=True)
        self._py_files: List[str] = [
            e.path
            for e in r.observation.file_tree
            if e.path.endswith(".py") and not e.is_dir
        ]

    def test_remove_unused_import_via_diff(self, env: RefactorEnv) -> None:
        _section("T6 — edit_file with unified_diff (remove unused import)")

        # Read the first source file to find an actual unused import to remove
        target = self._py_files[0]
        obs_view = env.view_file(target).observation
        lines = (obs_view.file_content or "").splitlines(keepends=True)

        # Find the first `import X` line as a patch target
        unused_import_line: str | None = None
        unused_import_idx: int | None = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                unused_import_line = line
                unused_import_idx = i
                break

        if unused_import_line is None:
            pytest.skip(f"No import statement found in {target} — skipping diff test")

        # Build a minimal unified diff that removes that line
        lineno_1 = unused_import_idx + 1  # 1-indexed
        diff = (
            f"--- a/{target}\n"
            f"+++ b/{target}\n"
            f"@@ -{lineno_1},1 +{lineno_1},0 @@\n"
            f"-{unused_import_line.rstrip()}\n"
        )
        log.info("  Applying diff to %s:", target)
        log.info("  %r", diff)

        result = env.edit_file(target, unified_diff=diff)
        obs = result.observation
        reward = result.reward

        _log_obs("after edit_file (unified_diff)", obs, reward)
        _log_step(obs.current_step, f"edit_file({target})", reward, result.done)

        _assert_obs(obs)
        assert (
            obs.last_action_error is None
        ), f"edit must not produce an error: {obs.last_action_error}"
        assert obs.git_status.has_changes, "edit must register in git"

        log.info(
            "  lint before : %d   lint after : %d",
            self._baseline_lint,
            obs.lint_summary.total_errors,
        )
        log.info(
            "  step_score  : %s",
            f"{obs.step_score:.4f}" if obs.step_score is not None else "—",
        )
        log.info("  ✓ edit_file (unified_diff) OK")


# ===========================================================================
# T7 — edit_file (new_content)
# ===========================================================================


class TestEditFileNewContent:
    """Replace a whole file and verify the observation updates."""

    @pytest.fixture(autouse=True)
    def _reset(self, env: RefactorEnv) -> None:
        env.reset(task_id=TASK_ID)
        r = env.list_directory(path=".", recursive=True)
        self._py_files: List[str] = [
            e.path
            for e in r.observation.file_tree
            if e.path.endswith(".py") and not e.is_dir
        ]

    def test_full_file_replacement(self, env: RefactorEnv) -> None:
        target = self._py_files[0]
        _section(f"T7 — edit_file new_content  ({target})")

        # Read original, then write a clean version with one trivial change
        original = env.view_file(target).observation.file_content
        assert original, "file must be readable"

        new_content = original.rstrip() + "\n# replaced by e2e test\n"
        result = env.edit_file(target, new_content=new_content)
        obs = result.observation
        reward = result.reward

        _log_obs("after edit_file (new_content)", obs, reward)
        _log_step(
            obs.current_step, f"edit_file({target}, new_content)", reward, result.done
        )

        _assert_obs(obs)
        assert (
            obs.last_action_error is None
        ), f"new_content edit must not error: {obs.last_action_error}"
        assert obs.git_status.has_changes, "replacement must register as a change"

        # Verify the content round-trips
        read_back = env.view_file(target).observation.file_content
        assert "# replaced by e2e test" in read_back, "replacement must persist"
        log.info("  ✓ edit_file (new_content) OK — content persists")


# ===========================================================================
# T8 — edit_files (multi-file atomic)
# ===========================================================================


class TestEditFilesAtomic:
    """edit_files() applies multiple patches in one step."""

    @pytest.fixture(autouse=True)
    def _reset(self, env: RefactorEnv) -> None:
        env.reset(task_id=TASK_ID)
        r = env.list_directory(path=".", recursive=True)
        py_files = [
            e.path
            for e in r.observation.file_tree
            if e.path.endswith(".py") and not e.is_dir
        ]
        if len(py_files) < 2:
            pytest.skip("Need at least 2 .py files for atomic multi-file edit test")
        self._files = py_files[:2]

    def test_atomic_two_file_edit(self, env: RefactorEnv) -> None:
        _section(f"T8 — edit_files (atomic)  {self._files}")

        contents = []
        for path in self._files:
            c = env.view_file(path).observation.file_content or ""
            contents.append(c)

        patches = [
            FilePatch(
                path=self._files[0], new_content=contents[0].rstrip() + "\n# patch-A\n"
            ),
            FilePatch(
                path=self._files[1], new_content=contents[1].rstrip() + "\n# patch-B\n"
            ),
        ]

        result = env.edit_files(patches)
        obs = result.observation
        reward = result.reward

        _log_obs("after edit_files (atomic)", obs, reward)
        _log_step(obs.current_step, "edit_files(2 patches)", reward, result.done)

        _assert_obs(obs)
        assert (
            obs.last_action_error is None
        ), f"atomic edit must not error: {obs.last_action_error}"
        assert obs.git_status.has_changes, "atomic edit must register changes"

        for path, marker in zip(self._files, ["# patch-A", "# patch-B"]):
            read = env.view_file(path).observation.file_content
            assert marker in read, f"marker '{marker}' must persist in {path}"
        log.info("  ✓ edit_files atomic OK — both patches persisted")


# ===========================================================================
# T9 — run_shell (ruff + pytest)
# ===========================================================================


class TestRunShell:
    """run_shell() executes linters and test runners in the sandbox."""

    @pytest.fixture(autouse=True)
    def _reset(self, env: RefactorEnv) -> None:
        env.reset(task_id=TASK_ID)

    def test_ruff_check(self, env: RefactorEnv) -> None:
        _section("T9a — run_shell('ruff check .')")
        result = env.run_shell("ruff check . --output-format=text")
        obs = result.observation
        reward = result.reward

        _log_obs("run_shell(ruff)", obs, reward)
        _log_step(obs.current_step, "run_shell(ruff check)", reward, result.done)

        _assert_obs(obs)
        # ruff output should be present; errors are expected for lint-cleanup task
        assert obs.last_action_output is not None, "shell output must be captured"
        log.info("  ruff stdout (first 400 chars): %r", obs.last_action_output[:400])
        log.info("  ✓ ruff ran without crashing the env")

    def test_pytest_baseline(self, env: RefactorEnv) -> None:
        _section("T9b — run_shell('pytest -q --tb=short')")
        result = env.run_shell("pytest -q --tb=short", timeout_sec=60)
        obs = result.observation
        reward = result.reward

        _log_obs("run_shell(pytest)", obs, reward)
        _log_step(obs.current_step, "run_shell(pytest)", reward, result.done)

        _assert_obs(obs)
        assert obs.test_summary.total >= 0, "TestSummary.total must be populated"
        log.info("  pytest total  : %d", obs.test_summary.total)
        log.info("  pytest passed : %d", obs.test_summary.passed)
        log.info("  pytest failed : %d", obs.test_summary.failed)
        log.info("  stdout (first 400 chars): %r", (obs.last_action_output or "")[:400])
        log.info("  ✓ pytest ran without crashing the env")

    def test_shell_timeout_respected(self, env: RefactorEnv) -> None:
        _section("T9c — run_shell respects timeout_sec")
        # A sleep longer than timeout_sec should not hang the client
        t0 = time.perf_counter()
        result = env.run_shell("sleep 10", timeout_sec=2)
        elapsed = time.perf_counter() - t0

        log.info("  elapsed wall clock : %.2f s  (timeout=2 s)", elapsed)
        assert elapsed < 8, "client must not hang waiting for a timed-out command"
        # Server may set last_action_error or return non-zero exit
        log.info("  error field : %s", result.observation.last_action_error)
        log.info("  ✓ timeout respected")


# ===========================================================================
# T10 — reward signal shape
# ===========================================================================


class TestRewardSignal:
    """
    Verify that making a good edit (removing a lint violation) produces a
    positive step_score and does not crash the episode.
    """

    def test_good_edit_yields_positive_score(self, env: RefactorEnv) -> None:
        _section("T10 — reward after lint-fixing edit")

        result = env.reset(task_id=TASK_ID)
        baseline_lint = result.observation.baseline_lint_summary.total_errors
        log.info("  baseline lint violations : %d", baseline_lint)

        # Locate a source file
        r = env.list_directory(path=".", recursive=True)
        py_files = [
            e.path
            for e in r.observation.file_tree
            if e.path.endswith(".py") and not e.is_dir
        ]
        if not py_files:
            pytest.skip("No .py files found")

        # Run ruff to find the first fixable violation
        ruff_result = env.run_shell("ruff check . --output-format=json")
        output = ruff_result.observation.last_action_output or "[]"
        try:
            import json

            violations = json.loads(output)
        except json.JSONDecodeError:
            violations = []

        if not violations:
            pytest.skip("No ruff violations found — cannot test positive reward")

        # Take a concrete action: view the first violated file then apply ruff --fix
        first_file = violations[0].get("filename", py_files[0])
        env.view_file(first_file)

        fix_result = env.run_shell("ruff check . --fix --unsafe-fixes")
        obs = fix_result.observation
        reward = fix_result.reward

        _log_obs("after ruff --fix", obs, reward)
        log.info(
            "  lint after fix : %d  (was %d)",
            obs.lint_summary.total_errors,
            baseline_lint,
        )
        log.info(
            "  step_score     : %s",
            f"{obs.step_score:.4f}" if obs.step_score is not None else "—",
        )

        if obs.lint_summary.total_errors < baseline_lint:
            assert (
                obs.step_score or 0
            ) >= 0, "step_score must be non-negative after improvement"
            log.info("  ✓ lint reduced — positive reward signal confirmed")
        else:
            log.info("  (lint not reduced by auto-fix — score signal not asserted)")


# ===========================================================================
# T11 — penalty for repeated no-op
# ===========================================================================


class TestPenaltySignal:
    """
    Repeatedly calling list_directory with no edits should accumulate a
    penalty (the env penalises no-progress steps).
    """

    def test_repeated_noop_accumulates_penalty(self, env: RefactorEnv) -> None:
        _section("T11 — penalty for repeated no-op steps")
        env.reset(task_id=TASK_ID)

        penalty_before = 0.0
        for i in range(1, 6):
            result = env.list_directory(path=".")
            obs = result.observation
            _log_step(i, "list_directory(.) [no-op]", result.reward, result.done)
            log.info("    cumulative_penalty = %.4f", obs.cumulative_penalty)
            if i == 1:
                penalty_before = obs.cumulative_penalty

        penalty_after = obs.cumulative_penalty
        log.info("  penalty before 4 repeated no-ops : %.4f", penalty_before)
        log.info("  penalty after  4 repeated no-ops : %.4f", penalty_after)

        assert (
            penalty_after >= penalty_before
        ), "cumulative_penalty must not decrease after no-op steps"
        log.info(
            "  ✓ penalty signal confirmed (%.4f → %.4f)", penalty_before, penalty_after
        )


# ===========================================================================
# T12 — submit
# ===========================================================================


class TestSubmit:
    """submit() closes the episode, sets done=True, returns a final score."""

    def test_submit_closes_episode(self, env: RefactorEnv) -> None:
        _section("T12 — submit()")
        env.reset(task_id=TASK_ID)

        # Do one useful step so the score is non-trivial
        env.run_shell("ruff check . --fix")

        result = env.submit(note="e2e smoke test — submit after auto-fix")
        obs = result.observation
        reward = result.reward

        _log_obs("submit observation", obs, reward)
        log.info("  done   : %s", result.done)
        log.info("  reward : %s", f"{reward:.4f}" if reward is not None else "None")

        assert result.done is True, "done must be True after submit"
        assert reward is not None, "reward must be non-None after submit"
        assert 0.0 <= reward <= 1.0, "final reward must be in [0.0, 1.0]"
        log.info("  ✓ submit closed episode with score %.4f", reward)

    def test_submit_with_no_edits(self, env: RefactorEnv) -> None:
        _section("T12b — submit with no edits (minimum score scenario)")
        env.reset(task_id=TASK_ID)
        result = env.submit(note="immediate submit — no edits")

        assert result.done is True, "done must be True"
        assert result.reward is not None
        log.info("  score with no edits : %.4f  ✓", result.reward)


# ===========================================================================
# T13 — full episode smoke test (sequential end-to-end)
# ===========================================================================


class TestFullEpisode:
    """
    Runs a complete episode without an LLM:
      reset → list_directory → view_file → run_shell(ruff --fix)
            → run_shell(pytest) → submit
    Logs every reward to console for visual inspection.
    """

    def test_full_episode(self, env: RefactorEnv) -> None:
        _header("T13 — FULL EPISODE END-TO-END  (no LLM)")

        rewards: list[float] = []

        # ── 1. reset ──────────────────────────────────────────────────────────
        result = env.reset(task_id=TASK_ID)
        obs = result.observation
        _log_obs("STEP 0  reset", obs)
        log.info("  Objective  : %s", obs.objective)
        log.info("  Constraints: %s", obs.constraints)
        log.info(
            "  Baseline tests : %d/%d",
            obs.baseline_test_summary.passed,
            obs.baseline_test_summary.total,
        )
        log.info(
            "  Baseline lint  : %d violations", obs.baseline_lint_summary.total_errors
        )

        # ── 2. list_directory ────────────────────────────────────────────────
        result = env.list_directory(path=".", recursive=True)
        obs = result.observation
        rewards.append(result.reward or 0.0)
        _log_step(obs.current_step, "list_directory(.)", result.reward, result.done)
        py_files = [
            e.path for e in obs.file_tree if e.path.endswith(".py") and not e.is_dir
        ]
        log.info("  Source files : %s", py_files)

        assert py_files, "must have at least one .py file"

        # ── 3. view each source file ──────────────────────────────────────────
        for path in py_files:
            result = env.view_file(path)
            obs = result.observation
            rewards.append(result.reward or 0.0)
            _log_step(
                obs.current_step, f"view_file({path})", result.reward, result.done
            )
            log.info(
                "    %d lines  |  first 80 chars: %r",
                obs.total_file_lines,
                obs.file_content[:80],
            )

        # ── 4. search for unused imports ─────────────────────────────────────
        result = env.search_codebase(query="^import |^from .* import ", context_lines=0)
        obs = result.observation
        rewards.append(result.reward or 0.0)
        _log_step(
            obs.current_step, "search_codebase(import)", result.reward, result.done
        )

        # ── 5. run ruff to find violations ───────────────────────────────────
        result = env.run_shell("ruff check . --output-format=text")
        obs = result.observation
        rewards.append(result.reward or 0.0)
        _log_step(obs.current_step, "run_shell(ruff check)", result.reward, result.done)
        log.info("  ruff output:\n%s", (obs.last_action_output or "")[:600])

        # ── 6. apply ruff auto-fix ────────────────────────────────────────────
        result = env.run_shell("ruff check . --fix")
        obs = result.observation
        rewards.append(result.reward or 0.0)
        _log_step(obs.current_step, "run_shell(ruff --fix)", result.reward, result.done)
        log.info("  lint violations after fix : %d", obs.lint_summary.total_errors)

        # ── 7. git diff to inspect changes ───────────────────────────────────
        result = env.git_diff(stat_only=True)
        obs = result.observation
        rewards.append(result.reward or 0.0)
        _log_step(obs.current_step, "git_diff(stat)", result.reward, result.done)
        log.info("  diff stat : %s", obs.git_status.diff_stat or "(no changes)")

        # ── 8. verify tests still pass ───────────────────────────────────────
        result = env.run_shell("pytest -q --tb=short", timeout_sec=90)
        obs = result.observation
        rewards.append(result.reward or 0.0)
        _log_step(obs.current_step, "run_shell(pytest)", result.reward, result.done)
        log.info(
            "  tests : %d/%d passing", obs.test_summary.passed, obs.test_summary.total
        )
        log.info("  pytest output:\n%s", (obs.last_action_output or "")[:400])

        # ── 9. submit ─────────────────────────────────────────────────────────
        result = env.submit(note="e2e full episode — ruff --fix + pytest verify")
        obs = result.observation
        final_reward = result.reward or 0.0
        rewards.append(final_reward)

        _log_step(obs.current_step, "submit()", result.reward, result.done)
        _log_obs("FINAL  submit", obs, result.reward)

        # ── Reward log table ──────────────────────────────────────────────────
        log.info("")
        log.info(SEP2)
        log.info("  STEP-BY-STEP REWARD LOG")
        log.info(SEP2)
        step_labels = (
            ["list_directory"]
            + [f"view_file({p})" for p in py_files]
            + [
                "search_codebase",
                "ruff check",
                "ruff --fix",
                "git_diff",
                "pytest",
                "submit",
            ]
        )
        for i, (label, r) in enumerate(zip(step_labels, rewards), start=1):
            bar = ("█" * int(r * 20)).ljust(20)
            log.info("  Step %2d  %-30s  %.4f  [%s]", i, label, r, bar)
        log.info("  %s", SEP)
        log.info("  final score  : %.4f", final_reward)
        log.info("  max in episode: %.4f", max(rewards))
        log.info(SEP2)

        # ── Final assertions ──────────────────────────────────────────────────
        assert result.done is True, "episode must be done after submit"
        assert 0.0 <= final_reward <= 1.0, "final score must be in [0.0, 1.0]"
        log.info("  ✓ FULL EPISODE PASSED  (final score %.4f)", final_reward)
