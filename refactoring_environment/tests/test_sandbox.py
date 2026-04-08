"""
tests/test_sandbox.py

Full integration test for SandboxEnv and every action type.
Reward/grading is excluded — tested separately once graders land.

Run with:
    pytest tests/test_sandbox.py -v
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from refactoring_environment.environment.registry import Registry
from refactoring_environment.environment.sandbox import SandboxEnv
from refactoring_environment.models import RefactorAction, RefactorObservation
from refactoring_environment.models.actions import ActionType
from refactoring_environment.models.observations import (
    CodebaseContext,
    ExecutionContext,
    GitStatus,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

TASK_NAME = "lint-cleanup"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry()


@pytest.fixture(scope="module")
def sandbox(registry: Registry) -> Generator[SandboxEnv, None, None]:
    """
    One sandbox for the whole module — cheaper than respinning per test.
    Tests are ordered so writes happen after reads.
    """
    scenario = registry.load_scenario(TASK_NAME)
    repo_root = registry.repo_path(TASK_NAME)
    sb = SandboxEnv(repo_root=repo_root, scenario=scenario)
    yield sb
    sb.destroy()


def _step(
    sandbox: SandboxEnv, action_type: str, params: dict, step: int = 1
) -> RefactorObservation:
    """Helper: build a RefactorAction and dispatch through sandbox.act()."""
    action = RefactorAction(action_type=action_type, params=params)
    return sandbox.act(
        action,
        step=step,
        episode_id="test-episode",
        task_id=TASK_NAME,
    )


# ── Init ──────────────────────────────────────────────────────────────────────


class TestSandboxInit:
    def test_root_exists(self, sandbox: SandboxEnv) -> None:
        assert sandbox.root.exists()
        assert sandbox.root.is_dir()

    def test_baseline_commit_set(self, sandbox: SandboxEnv) -> None:
        sha = sandbox.baseline_commit
        assert isinstance(sha, str)
        assert len(sha) == 40  # full git SHA

    def test_git_status_clean_at_reset(self, sandbox: SandboxEnv) -> None:
        status = sandbox.git_manager.status()
        assert isinstance(status, GitStatus)
        assert status.staged_files == []
        assert status.unstaged_files == []
        assert status.untracked_files == []
        assert not status.has_changes

    def test_initial_codebase_context(self, sandbox: SandboxEnv) -> None:
        ctx, _ = sandbox.get_initial_observation()
        assert isinstance(ctx, CodebaseContext)
        assert ctx.active_file is not None
        assert ctx.file_content is not None
        assert ctx.total_file_lines is not None and ctx.total_file_lines > 0
        assert ctx.file_line_start == 1
        assert ctx.file_line_end <= 100

    def test_file_tree_populated(self, sandbox: SandboxEnv) -> None:
        ctx, _ = sandbox.get_initial_observation()
        assert len(ctx.file_tree) > 0
        paths = [e.path for e in ctx.file_tree]
        # active file must appear in the tree
        assert ctx.active_file in paths

    def test_active_file_is_alphabetically_first_root_file(
        self, sandbox: SandboxEnv
    ) -> None:
        ctx, _ = sandbox.get_initial_observation()
        root_files = sorted(
            e.path for e in ctx.file_tree if not e.is_dir and "/" not in e.path
        )
        assert ctx.active_file == root_files[0]


# ── view_file ─────────────────────────────────────────────────────────────────


class TestViewFile:
    def test_view_default_range(self, sandbox: SandboxEnv) -> None:
        ctx, _ = sandbox.get_initial_observation()
        active = ctx.active_file
        obs = _step(sandbox, ActionType.view_file, {"path": active})
        assert obs.codebase.active_file == active
        assert obs.codebase.file_content is not None
        assert obs.codebase.file_line_start == 1

    def test_view_specific_line_range(self, sandbox: SandboxEnv) -> None:
        ctx, _ = sandbox.get_initial_observation()
        active = ctx.active_file
        obs = _step(
            sandbox,
            ActionType.view_file,
            {"path": active, "line_start": 1, "line_end": 5},
        )
        assert obs.codebase.file_line_start == 1
        assert obs.codebase.file_line_end == 5
        lines = obs.codebase.file_content.splitlines()
        assert len(lines) <= 5

    def test_view_updates_active_file(self, sandbox: SandboxEnv) -> None:
        # Pick a different file from the tree
        ctx, _ = sandbox.get_initial_observation()
        other_files = [
            e.path for e in ctx.file_tree if not e.is_dir and e.path != ctx.active_file
        ]
        if not other_files:
            pytest.skip("Only one file in repo — cannot test active file switch.")
        target = other_files[0]
        obs = _step(sandbox, ActionType.view_file, {"path": target})
        assert obs.codebase.active_file == target

    def test_view_nonexistent_file_raises(self, sandbox: SandboxEnv) -> None:
        action = RefactorAction(
            action_type=ActionType.view_file,
            params={"path": "does_not_exist.py"},
        )
        with pytest.raises(FileNotFoundError):
            sandbox.act(action, step=1, episode_id="test", task_id=TASK_NAME)


# ── list_directory ────────────────────────────────────────────────────────────


class TestListDirectory:
    def test_list_root(self, sandbox: SandboxEnv) -> None:
        obs = _step(sandbox, ActionType.list_directory, {"path": "."})
        assert len(obs.codebase.file_tree) > 0

    def test_list_non_recursive(self, sandbox: SandboxEnv) -> None:
        obs = _step(
            sandbox, ActionType.list_directory, {"path": ".", "recursive": False}
        )
        # Non-recursive should not contain paths with more than one separator
        deep = [e.path for e in obs.codebase.file_tree if e.path.count("/") > 1]
        assert deep == []

    def test_list_recursive(self, sandbox: SandboxEnv) -> None:
        obs_flat = _step(
            sandbox, ActionType.list_directory, {"path": ".", "recursive": False}
        )
        obs_recursive = _step(
            sandbox,
            ActionType.list_directory,
            {"path": ".", "recursive": True, "max_depth": 8},
        )
        # Recursive should return at least as many entries
        assert len(obs_recursive.codebase.file_tree) >= len(obs_flat.codebase.file_tree)

    def test_list_preserves_active_file(self, sandbox: SandboxEnv) -> None:
        ctx, _ = sandbox.get_initial_observation()
        before = ctx.active_file
        obs = _step(sandbox, ActionType.list_directory, {"path": "."})
        assert obs.codebase.active_file == before


# ── search_codebase ───────────────────────────────────────────────────────────


class TestSearchCodebase:
    def test_search_finds_results(self, sandbox: SandboxEnv) -> None:
        obs = _step(
            sandbox,
            ActionType.search_codebase,
            {"query": "import", "file_glob": "*.py"},
        )
        assert obs.codebase.file_content is not None
        assert obs.codebase.file_content != "(no results)"

    def test_search_no_results(self, sandbox: SandboxEnv) -> None:
        obs = _step(
            sandbox,
            ActionType.search_codebase,
            {
                "query": "xyzzy_this_string_will_never_exist_12345",
                "file_glob": "*.py",
            },
        )
        assert obs.codebase.file_content == "(no results)"

    def test_search_case_insensitive(self, sandbox: SandboxEnv) -> None:
        obs_upper = _step(
            sandbox,
            ActionType.search_codebase,
            {"query": "IMPORT", "file_glob": "*.py", "case_insensitive": True},
        )
        assert obs_upper.codebase.file_content != "(no results)"

    def test_search_active_file_is_none(self, sandbox: SandboxEnv) -> None:
        obs = _step(
            sandbox,
            ActionType.search_codebase,
            {"query": "import", "file_glob": "*.py"},
        )
        assert obs.codebase.active_file is None


# ── git_diff ──────────────────────────────────────────────────────────────────


class TestGitDiff:
    def test_diff_empty_at_reset(self, sandbox: SandboxEnv) -> None:
        obs = _step(sandbox, ActionType.git_diff, {})
        # No edits yet — diff should be empty
        assert obs.execution.stdout in (None, "(no diff)")

    def test_diff_stat_only(self, sandbox: SandboxEnv) -> None:
        obs = _step(sandbox, ActionType.git_diff, {"stat_only": True})
        assert isinstance(obs.execution, ExecutionContext)
        assert obs.execution.command == "git diff"


# ── edit_file ─────────────────────────────────────────────────────────────────


class TestEditFile:
    NEW_CONTENT = "# refenv test write\nprint('hello from refenv')\n"

    def test_edit_with_new_content(self, sandbox: SandboxEnv) -> None:
        tree = sandbox.file_handler._build_tree()
        target = next(e.path for e in tree if not e.is_dir)

        obs = _step(
            sandbox,
            ActionType.edit_file,
            {"patch": {"path": target, "new_content": self.NEW_CONTENT}},
            step=2,
        )

        assert obs.codebase.active_file == target
        assert obs.codebase.file_content is not None
        assert "hello from refenv" in obs.codebase.file_content

    def test_edit_commits_to_git(self, sandbox: SandboxEnv) -> None:
        # After edit the current commit should differ from baseline
        assert sandbox.git_manager.current_commit != sandbox.git_manager.baseline_commit

    def test_git_status_clean_after_edit(self, sandbox: SandboxEnv) -> None:
        # edit_file calls git.commit() → working tree should be clean
        status = sandbox.git_manager.status()
        assert not status.has_changes

    def test_diff_shows_change_after_edit(self, sandbox: SandboxEnv) -> None:
        obs = _step(sandbox, ActionType.git_diff, {})
        assert obs.execution.stdout not in (None, "(no diff)")
        assert "hello from refenv" in obs.execution.stdout

    def test_edit_with_unified_diff(self, sandbox: SandboxEnv) -> None:
        tree = sandbox.file_handler._build_tree()
        target = next(e.path for e in tree if not e.is_dir)

        # First write a known file so diff applies cleanly
        _step(
            sandbox,
            ActionType.edit_file,
            {"patch": {"path": target, "new_content": "line1\nline2\nline3\n"}},
            step=3,
        )

        unified = (
            f"--- a/{target}\n"
            f"+++ b/{target}\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+line2_modified\n"
            " line3\n"
        )
        obs = _step(
            sandbox,
            ActionType.edit_file,
            {"patch": {"path": target, "unified_diff": unified}},
            step=4,
        )

        assert "line2_modified" in obs.codebase.file_content

    def test_path_traversal_rejected(self, sandbox: SandboxEnv) -> None:
        action = RefactorAction(
            action_type=ActionType.edit_file,
            params={"patch": {"path": "../../etc/passwd", "new_content": "pwned"}},
        )
        with pytest.raises(PermissionError):
            sandbox.act(action, step=99, episode_id="test", task_id=TASK_NAME)


# ── edit_files ────────────────────────────────────────────────────────────────


class TestEditFiles:
    def test_edit_multiple_files(self, sandbox: SandboxEnv) -> None:
        ctx, _ = sandbox.get_initial_observation()
        files = [e.path for e in ctx.file_tree if not e.is_dir]

        # Create a second file if needed
        if len(files) < 2:
            # Try to include test files
            files = [e.path for e in ctx.file_tree if not e.is_dir or 'test' in e.path.lower()]

        # If still not enough, create a temporary test file
        if len(files) < 2:
            # Create a temporary test file
            new_file = "temp_test_file.py"
            action = RefactorAction(
                action_type=ActionType.edit_file.value,
                params={"patch": {"path": new_file, "new_content": "# temporary file\n"}},
            )
            sandbox.act(
                action,
                step=1,
                episode_id="test",
                task_id="lint-cleanup",
            )
            # Use the first file and the new file
            files = [files[0], new_file]

        patches = [
            {"path": files[0], "new_content": "# patched file 0\n"},
            {"path": files[1], "new_content": "# patched file 1\n"},
        ]
        obs = _step(sandbox, ActionType.edit_files, {"patches": patches}, step=5)

        # active file should be the last successfully patched one
        assert obs.codebase.active_file == files[1]
        assert "patched file 1" in obs.codebase.file_content

    def test_edit_files_commits(self, sandbox: SandboxEnv) -> None:
        status = sandbox.git_manager.status()
        assert not status.has_changes


# ── run_shell ─────────────────────────────────────────────────────────────────


class TestRunShell:
    def test_run_basic_command(self, sandbox: SandboxEnv) -> None:
        obs = _step(
            sandbox,
            ActionType.run_shell,
            {"command": "echo hello_refenv", "timeout_sec": 10},
        )
        assert obs.execution.command == "echo hello_refenv"
        assert obs.execution.stdout is not None
        assert "hello_refenv" in obs.execution.stdout
        assert obs.execution.return_code == 0
        assert not obs.execution.timed_out

    def test_run_captures_stderr(self, sandbox: SandboxEnv) -> None:
        obs = _step(
            sandbox,
            ActionType.run_shell,
            {
                "command": "python3 -c \"import sys; sys.stderr.write('err_output')\"",
                "timeout_sec": 10,
            },
        )
        assert obs.execution.stderr is not None
        assert "err_output" in obs.execution.stderr

    def test_run_nonzero_exit(self, sandbox: SandboxEnv) -> None:
        obs = _step(
            sandbox, ActionType.run_shell, {"command": "exit 42", "timeout_sec": 10}
        )
        assert obs.execution.return_code == 42

    def test_run_respects_workdir(self, sandbox: SandboxEnv) -> None:
        obs = _step(
            sandbox,
            ActionType.run_shell,
            {"command": "pwd", "timeout_sec": 10, "workdir": "."},
        )
        assert obs.execution.return_code == 0

    def test_run_path_traversal_rejected(self, sandbox: SandboxEnv) -> None:
        action = RefactorAction(
            action_type=ActionType.run_shell,
            params={"command": "ls", "timeout_sec": 5, "workdir": "../../"},
        )
        with pytest.raises(PermissionError):
            sandbox.act(action, step=99, episode_id="test", task_id=TASK_NAME)

    def test_run_does_not_pollute_git_status(self, sandbox: SandboxEnv) -> None:
        # Shell commands should not auto-commit anything
        before = sandbox.git_manager.current_commit
        _step(sandbox, ActionType.run_shell, {"command": "ls", "timeout_sec": 5})
        assert sandbox.git_manager.current_commit == before
