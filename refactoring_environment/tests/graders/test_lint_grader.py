"""
tests/graders/test_lint_grader.py

Unit and integration tests for LintGrader.
Tests both the _compute_metrics() and grade() methods.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import pytest

from refactoring_environment.environment.graders.types.lint_grader import LintGrader
from refactoring_environment.environment.graders.types.base import GradeResult
from refactoring_environment.models_internal.grader_spec import GraderSpec
from refactoring_environment.models_internal.actions import RunShellParams
from refactoring_environment.environment.sandbox.files import FileHandler
from refactoring_environment.environment.sandbox.runner import ShellExecutor


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_executor() -> Mock:
    """Mock ShellExecutor for testing."""
    return Mock(spec=ShellExecutor)


@pytest.fixture
def mock_file_handler() -> Mock:
    """Mock FileHandler for testing."""
    handler = Mock(spec=FileHandler)
    handler.root = Path("/tmp/test")
    # Setup mock context with file_tree
    mock_context = Mock()
    mock_context.file_tree = []
    handler.context = mock_context
    return handler


@pytest.fixture
def grader_spec() -> GraderSpec:
    """Default grader spec for testing."""
    return GraderSpec(weight=0.5, target_coverage=0.80)


@pytest.fixture
def lint_grader(
    mock_executor: Mock, mock_file_handler: Mock, grader_spec: GraderSpec
) -> LintGrader:
    """Create a LintGrader instance with mocked dependencies."""
    return LintGrader(
        spec=grader_spec, executor=mock_executor, file_handler=mock_file_handler
    )


# ── Helper Functions ──────────────────────────────────────────────────────────


def _mock_file_entry(path: str, is_dir: bool = False) -> Mock:
    """Create a mock file tree entry."""
    entry = Mock()
    entry.path = path
    entry.is_dir = is_dir
    return entry


# ── Unit Tests for _compute_metrics ─────────────────────────────────────────


class TestLintGraderComputeMetrics:
    """Test the _compute_metrics method."""

    def test_no_python_files(
        self, lint_grader: LintGrader, mock_file_handler: Mock
    ) -> None:
        """Test when there are no Python files."""
        # Set empty file tree
        mock_file_handler.context.file_tree = []

        result = lint_grader._compute_metrics()
        assert result == {"count": 0}

    def test_ruff_success_no_violations(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test ruff success with no violations."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        mock_executor.run.return_value = Mock(
            return_code=0, stdout="[]", stderr="", timed_out=False
        )

        result = lint_grader._compute_metrics()
        assert result == {"count": 0}

        # Verify ruff was called correctly (with full path)
        expected_cmd = 'ruff check --output-format json "/tmp/test/test.py"'
        mock_executor.run.assert_called_once_with(
            RunShellParams(command=expected_cmd, timeout_sec=30, workdir=".")
        )

    def test_ruff_success_with_violations(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test ruff success with violations."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        violations_json = '[{"code": "F401", "message": "unused import"}]'
        mock_executor.run.return_value = Mock(
            return_code=0, stdout=violations_json, stderr="", timed_out=False
        )

        result = lint_grader._compute_metrics()
        assert result == {"count": 1}

    def test_ruff_timeout(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test ruff timeout handling."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        mock_executor.run.return_value = Mock(
            return_code=0, stdout="", stderr="", timed_out=True
        )

        result = lint_grader._compute_metrics()
        assert result == {"count": 0, "tool_error": "ruff timed out"}

    def test_ruff_invalid_exit_code(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test ruff invalid exit code handling."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        mock_executor.run.return_value = Mock(
            return_code=2,
            stdout="",
            stderr="Error: something went wrong",
            timed_out=False,
        )

        result = lint_grader._compute_metrics()
        assert result == {"count": 0, "tool_error": "Error: something went wrong"}

    def test_ruff_json_parse_error(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test ruff JSON parsing error handling."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        mock_executor.run.return_value = Mock(
            return_code=0, stdout="invalid json", stderr="", timed_out=False
        )

        result = lint_grader._compute_metrics()
        assert result["count"] == 0
        assert "tool_error" in result
        # The actual error message from json.JSONDecodeError
        assert "Expecting value" in result["tool_error"]

    def test_excludes_test_files(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test that test files are excluded from linting."""
        mock_file_handler.context.file_tree = [
            _mock_file_entry("src/main.py"),
            _mock_file_entry("tests/test_main.py"),
            _mock_file_entry("test_utils.py"),
            _mock_file_entry("module_test.py"),
            _mock_file_entry("tests/integration/test_api.py"),
        ]

        mock_executor.run.return_value = Mock(
            return_code=0, stdout="[]", stderr="", timed_out=False
        )

        lint_grader._compute_metrics()

        # Should only call ruff on src/main.py (with full path)
        mock_executor.run.assert_called_once()
        call_args = mock_executor.run.call_args[0][0]
        assert "src/main.py" in call_args.command
        # Check that test files are not in the command
        assert "test_main.py" not in call_args.command
        assert "module_test.py" not in call_args.command
        assert "integration.py" not in call_args.command


# ── Unit Tests for grade ─────────────────────────────────────────────────────


class TestLintGraderGrade:
    """Test the grade method."""

    def test_baseline_clean_current_clean(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test grading when baseline and current are both clean."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        # Mock baseline metrics
        lint_grader._baseline = {"count": 0}

        # Mock current metrics
        mock_executor.run.return_value = Mock(
            return_code=0, stdout="[]", stderr="", timed_out=False
        )

        result = lint_grader.grade()

        assert isinstance(result, GradeResult)
        assert result.score == 1.0
        assert result.added_violations == 0
        assert any("baseline was already clean" in fb for fb in result.feedbacks)
        assert len(result.errors) == 0
        assert len(result.tool_errors) == 0

    def test_improvement_from_baseline(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test grading when violations are reduced from baseline."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        # Mock baseline metrics
        lint_grader._baseline = {"count": 5}

        # Mock current metrics (improved)
        mock_executor.run.return_value = Mock(
            return_code=0,
            stdout='[{"code": "F401"}]',  # 1 violation
            stderr="",
            timed_out=False,
        )

        result = lint_grader.grade()

        # Score should be proportional improvement: (5-1)/5 = 0.8
        assert result.score == 0.8
        assert result.added_violations == 0
        assert any("↓4" in fb for fb in result.feedbacks)
        assert len(result.errors) == 0

    def test_regression_from_baseline(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test grading when violations increase from baseline (regression)."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        # Mock baseline metrics
        lint_grader._baseline = {"count": 2}

        # Mock current metrics (worse)
        mock_executor.run.return_value = Mock(
            return_code=0,
            stdout='[{"code": "F401"}, {"code": "E501"}, {"code": "F841"}]',  # 3 violations
            stderr="",
            timed_out=False,
        )

        result = lint_grader.grade()

        # Score should be negative: (2-3)/2 = -0.5 -> clamped to 0.0
        assert result.score == 0.0
        assert result.added_violations == 1
        assert any("regression" in err for err in result.errors)
        assert any("↑1" in fb for fb in result.feedbacks)

    def test_no_change_from_baseline(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test grading when violations stay the same."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        # Mock baseline metrics
        lint_grader._baseline = {"count": 3}

        # Mock current metrics (same)
        mock_executor.run.return_value = Mock(
            return_code=0,
            stdout='[{"code": "F401"}, {"code": "E501"}, {"code": "F841"}]',  # 3 violations
            stderr="",
            timed_out=False,
        )

        result = lint_grader.grade()

        assert result.score == 0.0
        assert result.added_violations == 0
        assert any("→0" in fb for fb in result.feedbacks)
        assert len(result.errors) == 0

    def test_baseline_tool_error(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test grading when baseline had tool errors."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        # Mock baseline with tool error
        lint_grader._baseline = {"count": 0, "tool_error": "ruff failed at baseline"}

        # Mock current metrics (clean)
        mock_executor.run.return_value = Mock(
            return_code=0, stdout="[]", stderr="", timed_out=False
        )

        result = lint_grader.grade()

        assert result.score == 1.0
        assert any("ruff failed at baseline" in err for err in result.tool_errors)

    def test_current_tool_error(
        self, lint_grader: LintGrader, mock_executor: Mock, mock_file_handler: Mock
    ) -> None:
        """Test grading when current run has tool errors."""
        mock_file_handler.context.file_tree = [_mock_file_entry("test.py")]

        # Mock baseline metrics
        lint_grader._baseline = {"count": 0}

        # Mock current metrics with tool error
        mock_executor.run.return_value = Mock(
            return_code=2, stdout="", stderr="ruff crashed", timed_out=False
        )

        result = lint_grader.grade()

        assert result.score == 1.0  # Cannot improve from baseline when tool fails
        assert any("ruff failed: ruff crashed" in err for err in result.tool_errors)


# ── Integration Tests ────────────────────────────────────────────────────────


class TestLintGraderIntegration:
    """Integration tests with real file system."""

    def test_source_files_identification(self, tmp_path: Path) -> None:
        """Test that _source_files correctly identifies Python files."""
        # Create test file structure
        src_file = tmp_path / "main.py"
        test_file = tmp_path / "test_main.py"
        nested_test = tmp_path / "tests" / "integration.py"
        nested_test.parent.mkdir()

        src_file.write_text("print('hello')")
        test_file.write_text("def test_something(): pass")
        nested_test.write_text("def test_integration(): pass")

        # Create a mock file handler
        mock_handler = Mock(spec=FileHandler)
        mock_handler.root = tmp_path

        # Create mock file tree entries
        entries = [
            _mock_file_entry("main.py"),
            _mock_file_entry("test_main.py"),
            _mock_file_entry("tests/integration.py"),
        ]
        mock_handler.context.file_tree = entries

        # Test the grader
        spec = GraderSpec(weight=0.5, target_coverage=0.80)
        executor = Mock(spec=ShellExecutor)
        grader = LintGrader(spec=spec, executor=executor, file_handler=mock_handler)

        source_files = grader._source_files()

        # Should only include main.py, exclude test files
        assert len(source_files) == 1
        assert str(tmp_path / "main.py") in source_files
        assert str(tmp_path / "test_main.py") not in source_files
        assert str(tmp_path / "tests" / "integration.py") not in source_files
