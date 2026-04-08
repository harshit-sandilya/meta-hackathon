"""
tests/graders/test_complexity_grader.py

Unit and integration tests for ComplexityGrader.
Tests cyclomatic complexity and Big-O complexity analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from refactoring_environment.environment.graders.types.complexity_grader import (
    ComplexityGrader,
    _cc_cost,
    _collect_cc_metrics,
    _run_radon,
)
from refactoring_environment.environment.graders.types.base import GradeResult
from refactoring_environment.models.grader_spec import GraderSpec
from refactoring_environment.models.observations import GraderContext, RewardContext
from refactoring_environment.environment.registry.scenario import ScenarioSpec
from refactoring_environment.environment.sandbox.files import FileHandler
from refactoring_environment.environment.sandbox.runner import ShellExecutor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_executor() -> Mock:
    """Mock ShellExecutor for testing."""
    return Mock(spec=ShellExecutor)


@pytest.fixture
def mock_file_handler(tmp_path: Path) -> Mock:
    """Mock FileHandler for testing."""
    handler = Mock(spec=FileHandler)
    handler.root = tmp_path
    return handler


@pytest.fixture
def mock_scenario() -> Mock:
    """Mock ScenarioSpec for testing."""
    scenario = Mock(spec=ScenarioSpec)
    scenario.task_id = "test-task"
    scenario.graders = {}
    return scenario


@pytest.fixture
def complexity_grader(mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> ComplexityGrader:
    """Create a ComplexityGrader instance with mocked dependencies."""
    return ComplexityGrader(
        spec=GraderSpec(weight=0.5, target_coverage=0.8),
        executor=mock_executor,
        file_handler=mock_file_handler
    )


# ── Helper Functions ──────────────────────────────────────────────────────────

def _create_test_file(tmp_path: Path, filename: str, content: str) -> Path:
    """Create a test Python file."""
    file_path = tmp_path / filename
    file_path.write_text(content)
    return file_path


# ── Unit Tests for Helper Functions ─────────────────────────────────────────

class TestComplexityHelperFunctions:
    """Test helper functions for complexity analysis."""

    def test_cc_cost_calculation(self) -> None:
        """Test cyclomatic complexity cost calculation."""
        # At threshold, cost should be 0
        assert _cc_cost(10, 10) == 0.0

        # Above threshold, quadratic cost
        assert _cc_cost(15, 10) == 25.0  # (15-10)² = 25
        assert _cc_cost(26, 10) == 256.0  # (26-10)² = 256

        # Below threshold, cost should be 0
        assert _cc_cost(5, 10) == 0.0
        assert _cc_cost(10, 15) == 0.0

    def test_run_radon_with_error(self, tmp_path: Path) -> None:
        """Test radon error handling."""
        # Test with non-existent radon
        with patch('subprocess.run', side_effect=FileNotFoundError("radon not found")):
            by_file, error = _run_radon(tmp_path, 15)

        assert by_file == {}
        assert "radon not found" in error

    def test_run_radon_with_timeout(self, tmp_path: Path) -> None:
        """Test radon timeout handling."""
        import subprocess
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(15, "test")):
            by_file, error = _run_radon(tmp_path, 15)

        assert by_file == {}
        assert "timed out" in error

    def test_collect_cc_metrics_empty(self, tmp_path: Path) -> None:
        """Test CC metrics with empty input."""
        result = _collect_cc_metrics({}, 10, False)

        assert result["cc_cost_sum"] == 0.0
        assert result["cc_avg"] == 0.0
        assert result["cc_total_fns"] == 0
        assert result["cc_worst_fns"] == []

    def test_collect_cc_metrics_with_data(self, tmp_path: Path) -> None:
        """Test CC metrics with sample data."""
        # Create sample radon output
        by_file = {
            "main.py": [
                {"name": "complex_func", "cc": 15, "rank": "C"},
                {"name": "simple_func", "cc": 5, "rank": "A"},
            ],
            "utils.py": [
                {"name": "helper", "cc": 8, "rank": "B"},
            ]
        }

        result = _collect_cc_metrics(by_file, 10, False)

        # complex_func: (15-10)² = 25, simple_func: 0, helper: 0
        assert result["cc_cost_sum"] == 25.0
        assert result["cc_avg"] == round((15 + 5 + 8) / 3, 3)  # 9.333 rounded to 3 decimal places
        assert result["cc_total_fns"] == 3
        # Top 5 worst functions (by cost) - complex_func has cost 25, others have 0
        assert len(result["cc_worst_fns"]) == 3  # All functions since cost threshold is 0
        assert result["cc_worst_fns"][0]["name"] == "complex_func"

    def test_collect_cc_metrics_exclude_tests(self, tmp_path: Path) -> None:
        """Test that test files are excluded when requested."""
        by_file = {
            "main.py": [{"name": "func", "cc": 12, "rank": "B"}],
            "test_main.py": [{"name": "test_func", "cc": 20, "rank": "D"}],
        }

        # With exclude_tests=True, test_main.py should be excluded
        result = _collect_cc_metrics(by_file, 10, True)
        assert result["cc_total_fns"] == 1
        assert result["cc_cost_sum"] == 4.0  # Only main.py


# ── Unit Tests for ComplexityGrader ─────────────────────────────────────────

class TestComplexityGrader:
    """Test the ComplexityGrader class."""

    def test_initialization(self, complexity_grader: ComplexityGrader) -> None:
        """Test basic initialization."""
        assert complexity_grader.grader_id == "complexity"
        assert hasattr(complexity_grader, '_baseline')

    def test_cc_cost_with_config(self, complexity_grader: ComplexityGrader) -> None:
        """Test CC cost calculation with custom config."""
        # Test with hardcoded threshold (default is 10, but we test the function directly)
        cost = _cc_cost(12, 8)  # (12-8)² = 16
        assert cost == 16.0

    def test_compute_metrics_no_files(self, complexity_grader: ComplexityGrader, tmp_path: Path) -> None:
        """Test metrics computation with no Python files."""
        # Use hardcoded defaults (no need to mock config)
        with patch('subprocess.run', return_value=Mock(stdout="{}", returncode=0)):
            result = complexity_grader._compute_metrics()

        assert "cc_cost_sum" in result
        assert "bigO_cost_sum" in result
        assert result["cc_total_fns"] == 0
        assert result["bigO_fn_count"] == 0

    def test_compute_metrics_with_files(self, complexity_grader: ComplexityGrader, tmp_path: Path) -> None:
        """Test metrics computation with sample files."""
        # Create test file
        _create_test_file(tmp_path, "main.py", """
def simple():
    return 42

def complex_func(x, y, z):
    if x > 0:
        if y > 0:
            return x + y
    return 0
""")

        # Mock radon output
        with patch('subprocess.run', return_value=Mock(
            stdout='{"main.py": [{"name": "simple", "complexity": 1, "rank": "A"}, {"name": "complex_func", "complexity": 4, "rank": "B"}]}',
            returncode=0
        )):
            # Use hardcoded defaults (no need to mock config)
            result = complexity_grader._compute_metrics()

        assert result["cc_total_fns"] == 2
        assert "cc_avg" in result
        assert "bigO_fn_count" in result

    def test_grade_with_improvement(self, complexity_grader: ComplexityGrader) -> None:
        """Test grading when complexity improves."""
        # Set baseline with higher complexity
        complexity_grader._baseline = {
            "cc_cost_sum": 100.0,
            "cc_avg": 8.0,
            "bigO_cost_sum": 50.0,
        }

        # Set current with lower complexity
        current_metrics = {
            "cc_cost_sum": 50.0,  # Improved
            "cc_avg": 6.0,
            "bigO_cost_sum": 25.0,  # Improved
            "cc_total_fns": 10,
            "cc_worst_fns": [],
            "bigO_fn_count": 5,
            "bigO_worst_fns": [],
            "bigO_patterns_seen": [],
            "radon_error": "",
        }

        with patch.object(complexity_grader, '_compute_metrics', return_value=current_metrics):
            result = complexity_grader.grade()

        assert isinstance(result, GradeResult)
        assert result.score > 0
        assert result.score <= 1.0
        assert any("Complexity" in feedback for feedback in result.feedbacks)

    def test_grade_with_regression(self, complexity_grader: ComplexityGrader) -> None:
        """Test grading when complexity regresses."""
        # Set baseline with lower complexity
        complexity_grader._baseline = {
            "cc_cost_sum": 50.0,
            "cc_avg": 6.0,
            "bigO_cost_sum": 25.0,
        }

        # Set current with higher complexity
        current_metrics = {
            "cc_cost_sum": 100.0,  # Regressed
            "cc_avg": 8.0,
            "bigO_cost_sum": 50.0,  # Regressed
            "cc_total_fns": 10,
            "cc_worst_fns": [],
            "bigO_fn_count": 5,
            "bigO_worst_fns": [],
            "bigO_patterns_seen": [],
            "radon_error": "",
        }

        with patch.object(complexity_grader, '_compute_metrics', return_value=current_metrics):
            result = complexity_grader.grade()

        assert isinstance(result, GradeResult)
        assert result.score > 0  # Should still be positive, just lower
        assert result.score < 1.0

    def test_radon_error_handling(self, complexity_grader: ComplexityGrader) -> None:
        """Test graceful handling of radon failures."""
        # Set baseline
        complexity_grader._baseline = {
            "cc_cost_sum": 50.0,
            "cc_avg": 6.0,
            "bigO_cost_sum": 25.0,
        }

        # Set current with radon error
        current_metrics = {
            "cc_cost_sum": 0.0,
            "cc_avg": 0.0,
            "bigO_cost_sum": 25.0,
            "cc_total_fns": 0,
            "cc_worst_fns": [],
            "bigO_fn_count": 5,
            "bigO_worst_fns": [],
            "bigO_patterns_seen": [],
            "radon_error": "radon not found",
        }

        with patch.object(complexity_grader, '_compute_metrics', return_value=current_metrics):
            result = complexity_grader.grade()

        # Should fall back to using cc_avg for delta
        assert isinstance(result, GradeResult)
        assert any("radon" in feedback.lower() for feedback in result.feedbacks)


# ── Integration Tests ────────────────────────────────────────────────────────

class TestComplexityGraderIntegration:
    """Integration tests with realistic scenarios."""

    def test_complexity_with_realistic_codebase(self, complexity_grader: ComplexityGrader, tmp_path: Path) -> None:
        """Test with a realistic multi-file codebase."""
        # Create multiple files with varying complexity
        _create_test_file(tmp_path, "main.py", """
def simple():
    return 42

def medium(x, y):
    if x > y:
        return x
    return y

def complex_func(a, b, c):
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
            return a + b
        return a
    return 0
""")

        _create_test_file(tmp_path, "utils.py", """
def helper():
    return "helper"

def process_data(data):
    result = []
    for item in data:
        if item:
            result.append(item * 2)
    return result
""")

        # Mock radon output
        radon_output = {
            "main.py": [
                {"name": "simple", "complexity": 1, "rank": "A"},
                {"name": "medium", "complexity": 3, "rank": "B"},
                {"name": "complex_func", "complexity": 12, "rank": "C"},  # Above threshold
            ],
            "utils.py": [
                {"name": "helper", "complexity": 1, "rank": "A"},
                {"name": "process_data", "complexity": 15, "rank": "D"},  # Above threshold
            ]
        }

        with patch('subprocess.run', return_value=Mock(
            stdout=json.dumps(radon_output),
            returncode=0
        )):
            # Use hardcoded defaults (no need to mock config)
            result = complexity_grader._compute_metrics()

        # Should detect multiple functions with various complexity levels
        assert result["cc_total_fns"] == 5
        assert result["cc_cost_sum"] > 0  # Some functions above threshold
        assert result["bigO_fn_count"] >= 0

    def test_config_parsing(self, complexity_grader: ComplexityGrader) -> None:
        """Test that config values are parsed correctly."""
        # The complexity grader uses hardcoded defaults, so this test just verifies
        # it can run without errors
        result = complexity_grader._compute_metrics()
        # Should work without errors
        assert result is not None
