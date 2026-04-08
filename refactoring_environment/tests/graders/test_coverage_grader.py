"""
tests/graders/test_coverage_grader.py

Unit and integration tests for CoverageGrader.
Tests test execution and coverage measurement functionality.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from refactoring_environment.environment.graders.types.coverage_grader import (
    CoverageGrader,
    _empty_metrics,
    _parse_counts,
    _parse_coverage,
)
from refactoring_environment.environment.graders.types.base import GradeResult
from refactoring_environment.models.grader_spec import GraderSpec
from refactoring_environment.environment.sandbox.files import FileHandler
from refactoring_environment.environment.sandbox.runner import ShellExecutor
from refactoring_environment.models.actions import RunShellParams


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
def grader_spec() -> GraderSpec:
    """Default grader spec for testing."""
    return GraderSpec(weight=0.5, target_coverage=0.80)


@pytest.fixture
def coverage_grader(mock_executor: Mock, mock_file_handler: Mock, grader_spec: GraderSpec) -> CoverageGrader:
    """Create a CoverageGrader instance with mocked dependencies."""
    return CoverageGrader(spec=grader_spec, executor=mock_executor, file_handler=mock_file_handler)


# ── Helper Functions ──────────────────────────────────────────────────────────

def _create_test_file(tmp_path: Path, filename: str, content: str) -> Path:
    """Create a test Python file."""
    file_path = tmp_path / filename
    file_path.write_text(content)
    return file_path


def _create_json_file(tmp_path: Path, filename: str, content: dict) -> Path:
    """Create a JSON file with given content."""
    import json
    file_path = tmp_path / filename
    file_path.write_text(json.dumps(content))
    return file_path


# ── Unit Tests for Helper Functions ─────────────────────────────────────────

class TestCoverageHelperFunctions:
    """Test helper functions for coverage parsing."""

    def test_empty_metrics_function(self) -> None:
        """Test _empty_metrics function."""
        result = _empty_metrics()

        assert result["passed"] == 0
        assert result["failed"] == 0
        assert result["errors"] == 0
        assert result["total"] == 0
        assert result["pass_rate"] == 0.0
        assert result["line_coverage"] == 0.0
        assert result["branch_coverage"] == 0.0
        assert result["per_file"] == {}
        assert result["timed_out"] == False
        assert result["run_error"] == ""

    def test_empty_metrics_with_timeout(self) -> None:
        """Test _empty_metrics with timeout."""
        result = _empty_metrics(timed_out=True, run_error="Timeout occurred")

        assert result["timed_out"] == True
        assert result["run_error"] == "Timeout occurred"

    def test_parse_counts_from_stdout(self, tmp_path: Path) -> None:
        """Test _parse_counts with stdout parsing."""
        stdout = "10 passed, 2 failed, 1 error"
        pytest_json = tmp_path / "pytest.json"
        # No JSON file, should fall back to regex

        passed, failed, errors = _parse_counts(pytest_json, stdout)

        assert passed == 10
        assert failed == 2
        assert errors == 1

    def test_parse_coverage_from_json(self, tmp_path: Path) -> None:
        """Test _parse_coverage with valid JSON."""
        cov_data = {
            "totals": {
                "covered_lines": 80,
                "num_statements": 100,
                "covered_branches": 40,
                "num_branches": 50,
            },
            "files": {
                "main.py": {
                    "summary": {
                        "covered_lines": 40,
                        "num_statements": 50,
                        "covered_branches": 20,
                        "num_branches": 25,
                    }
                }
            }
        }

        cov_json = _create_json_file(tmp_path, "coverage.json", cov_data)

        line_cov, branch_cov, per_file = _parse_coverage(cov_json)

        assert line_cov == 0.8
        assert branch_cov == 0.8
        assert "main.py" in per_file
        assert per_file["main.py"]["line"] == 0.8
        assert per_file["main.py"]["branch"] == 0.8

    def test_parse_coverage_missing_file(self, tmp_path: Path) -> None:
        """Test _parse_coverage with missing file."""
        cov_json = tmp_path / "missing_coverage.json"

        line_cov, branch_cov, per_file = _parse_coverage(cov_json)

        assert line_cov == 0.0
        assert branch_cov == 0.0
        assert per_file == {}


# ── Unit Tests for _compute_metrics ─────────────────────────────────────────

class TestCoverageGraderComputeMetrics:
    """Test the _compute_metrics method."""

    def test_empty_repository_no_tests(self, coverage_grader: CoverageGrader, tmp_path: Path) -> None:
        """Test with empty repository (no test files)."""
        # Mock executor to simulate pytest running with no tests found
        mock_result = Mock()
        mock_result.timed_out = False
        mock_result.run_error = ""
        mock_result.stdout = "No tests found"

        with patch.object(coverage_grader.executor, 'run', return_value=mock_result):
            result = coverage_grader._compute_metrics()

        assert result["passed"] == 0
        assert result["failed"] == 0
        assert result["errors"] == 0
        assert result["total"] == 0
        assert result["pass_rate"] == 0.0

    def test_pytest_timeout(self, coverage_grader: CoverageGrader, tmp_path: Path) -> None:
        """Test timeout handling."""
        mock_result = Mock()
        mock_result.timed_out = True
        mock_result.run_error = ""
        mock_result.stdout = ""

        with patch.object(coverage_grader.executor, 'run', return_value=mock_result):
            result = coverage_grader._compute_metrics()

        assert result["timed_out"] == True
        assert result["run_error"] == "pytest timed out after 60s"

    def test_pytest_run_error(self, coverage_grader: CoverageGrader, tmp_path: Path) -> None:
        """Test pytest run error handling."""
        mock_result = Mock()
        mock_result.timed_out = False
        mock_result.run_error = "ModuleNotFoundError: pytest"
        mock_result.stdout = ""

        with patch.object(coverage_grader.executor, 'run', return_value=mock_result):
            result = coverage_grader._compute_metrics()

        assert result["run_error"] == "ModuleNotFoundError: pytest"

    def test_successful_pytest_run(self, coverage_grader: CoverageGrader, tmp_path: Path) -> None:
        """Test successful pytest run with coverage."""
        # Create mock pytest and coverage JSON files
        pytest_data = {
            "summary": {
                "passed": 8,
                "failed": 2,
                "errors": 0,
            }
        }
        cov_data = {
            "totals": {
                "covered_lines": 160,
                "num_statements": 200,
                "covered_branches": 80,
                "num_branches": 100,
            },
            "files": {
                "main.py": {
                    "summary": {
                        "covered_lines": 40,
                        "num_statements": 50,
                        "covered_branches": 20,
                        "num_branches": 25,
                    }
                }
            }
        }

        pytest_json = _create_json_file(tmp_path, "pytest_report.json", pytest_data)
        cov_json = _create_json_file(tmp_path, "coverage.json", cov_data)

        mock_result = Mock()
        mock_result.timed_out = False
        mock_result.run_error = ""
        mock_result.stdout = "8 passed, 2 failed"
        mock_result.stderr = ""
        mock_result.return_code = 0

        with patch.object(coverage_grader.executor, 'run', return_value=mock_result):
            # Mock the tmp directory behavior
            with patch('tempfile.TemporaryDirectory') as mock_tmp:
                mock_tmp.return_value.__enter__.return_value = str(tmp_path)
                result = coverage_grader._compute_metrics()

        assert result["passed"] == 8
        assert result["failed"] == 2
        assert result["errors"] == 0
        assert result["total"] == 10
        assert result["pass_rate"] == 0.8
        assert result["line_coverage"] == 0.8
        assert result["branch_coverage"] == 0.8
        assert "main.py" in result["per_file"]

    def test_mode_resolution_constraint(self, coverage_grader: CoverageGrader) -> None:
        """Test mode resolution for constraint mode."""
        # Default should be constraint mode
        mode = coverage_grader._resolve_mode()
        assert mode == "constraint"

    def test_mode_resolution_objective(self, coverage_grader: CoverageGrader) -> None:
        """Test mode resolution for objective mode."""
        # After removing config dependency, mode always defaults to constraint
        # This test validates the current behavior
        mode = coverage_grader._resolve_mode()
        assert mode == "constraint"  # Updated to reflect current implementation

    def test_mode_resolution_scenario_type(self, coverage_grader: CoverageGrader) -> None:
        """Test mode resolution based on scenario type."""
        # After removing config dependency, scenario type detection is no longer supported
        # This test validates that the method works without config
        mode = coverage_grader._resolve_mode()
        assert mode == "constraint"  # Updated to reflect current implementation


# ── Unit Tests for grade methods ─────────────────────────────────────────

class TestCoverageGraderGrade:
    """Test the grade methods."""

    def test_constraint_mode_improvement(self, coverage_grader: CoverageGrader) -> None:
        """Test constraint mode when coverage improves."""
        # Set baseline metrics
        coverage_grader._baseline = {
            "passed": 5,
            "failed": 1,
            "errors": 0,
            "total": 6,
            "pass_rate": 5/6,
            "line_coverage": 0.7,
            "branch_coverage": 0.6,
            "per_file": {},
        }

        # Set current metrics (improved)
        current_metrics = {
            "passed": 8,
            "failed": 0,
            "errors": 0,
            "total": 8,
            "pass_rate": 1.0,
            "line_coverage": 0.8,  # Improved
            "branch_coverage": 0.7,
            "per_file": {},
            "timed_out": False,
            "run_error": "",
        }

        with patch.object(coverage_grader, '_compute_metrics', return_value=current_metrics):
            result = coverage_grader.grade()

        assert isinstance(result, GradeResult)
        assert result.score > 0
        assert result.score <= 1.0
        assert "constraint" in str(result.feedbacks)

    def test_constraint_mode_regression(self, coverage_grader: CoverageGrader) -> None:
        """Test constraint mode when coverage regresses."""
        # Set baseline metrics
        coverage_grader._baseline = {
            "passed": 10,
            "failed": 0,
            "errors": 0,
            "total": 10,
            "pass_rate": 1.0,
            "line_coverage": 0.8,
            "branch_coverage": 0.7,
            "per_file": {},
        }

        # Set current metrics (regressed)
        current_metrics = {
            "passed": 8,
            "failed": 0,
            "errors": 0,
            "total": 8,
            "pass_rate": 1.0,
            "line_coverage": 0.6,  # Regressed below tolerance
            "branch_coverage": 0.5,
            "per_file": {},
            "timed_out": False,
            "run_error": "",
        }

        with patch.object(coverage_grader, '_compute_metrics', return_value=current_metrics):
            result = coverage_grader.grade()

        assert isinstance(result, GradeResult)
        # Score should be halved due to regression
        assert result.score == 0.5  # 1.0 pass_rate * 0.5 coverage_multiplier
        # Check that feedback indicates coverage drop
        assert any("dropped beyond" in fb.lower() for fb in result.feedbacks)

    def test_objective_mode_improvement(self, coverage_grader: CoverageGrader) -> None:
        """Test objective mode when working toward target."""
        # Mock config for objective mode
        with patch.object(coverage_grader.spec, 'config', {"coverage": {"mode": "objective", "target_coverage": 0.9}}):
            # Set baseline metrics
            coverage_grader._baseline = {
                "passed": 5,
                "failed": 1,
                "errors": 0,
                "total": 6,
                "pass_rate": 5/6,
                "line_coverage": 0.7,
                "branch_coverage": 0.6,
                "per_file": {},
            }

            # Set current metrics (improved)
            current_metrics = {
                "passed": 8,
                "failed": 0,
                "errors": 0,
                "total": 8,
                "pass_rate": 1.0,
                "line_coverage": 0.8,  # Improved toward target of 0.9
                "branch_coverage": 0.7,
                "per_file": {},
                "timed_out": False,
                "run_error": "",
            }

            with patch.object(coverage_grader, '_compute_metrics', return_value=current_metrics):
                result = coverage_grader.grade()

            assert isinstance(result, GradeResult)
            assert result.score > 0
            assert result.score < 1.0  # Not yet at target
            assert "objective" in str(result.feedbacks)

    def test_objective_mode_target_reached(self, coverage_grader: CoverageGrader) -> None:
        """Test objective mode when target is reached."""
        with patch.object(coverage_grader.spec, 'config', {"coverage": {"mode": "objective", "target_coverage": 0.8}}):
            # Set baseline metrics
            coverage_grader._baseline = {
                "passed": 5,
                "failed": 0,
                "errors": 0,
                "total": 5,
                "pass_rate": 1.0,
                "line_coverage": 0.6,
                "branch_coverage": 0.5,
                "per_file": {},
            }

            # Set current metrics (reached target)
            current_metrics = {
                "passed": 8,
                "failed": 0,
                "errors": 0,
                "total": 8,
                "pass_rate": 1.0,
                "line_coverage": 0.8,  # Reached target
                "branch_coverage": 0.7,
                "per_file": {},
                "timed_out": False,
                "run_error": "",
            }

            with patch.object(coverage_grader, '_compute_metrics', return_value=current_metrics):
                result = coverage_grader.grade()

            assert isinstance(result, GradeResult)
            assert result.score == 1.0  # Target reached
            assert "Target reached" in str(result.feedbacks)

    def test_objective_mode_regression(self, coverage_grader: CoverageGrader) -> None:
        """Test objective mode when coverage regresses."""
        with patch.object(coverage_grader.spec, 'config', {"coverage": {"mode": "objective"}}):
            # Set baseline metrics
            coverage_grader._baseline = {
                "passed": 10,
                "failed": 0,
                "errors": 0,
                "total": 10,
                "pass_rate": 1.0,
                "line_coverage": 0.8,
                "branch_coverage": 0.7,
                "per_file": {},
            }

            # Set current metrics (regressed)
            current_metrics = {
                "passed": 8,
                "failed": 0,
                "errors": 0,
                "total": 8,
                "pass_rate": 1.0,
                "line_coverage": 0.7,  # Regressed
                "branch_coverage": 0.6,
                "per_file": {},
                "timed_out": False,
                "run_error": "",
            }

            with patch.object(coverage_grader, '_compute_metrics', return_value=current_metrics):
                result = coverage_grader.grade()

            assert isinstance(result, GradeResult)
            assert result.score == 0.0  # Regression gives 0 score
            assert "regressed" in str(result.feedbacks).lower()

    def test_pytest_timeout_handling(self, coverage_grader: CoverageGrader) -> None:
        """Test grade method with pytest timeout."""
        current_metrics = {
            "timed_out": True,
            "run_error": "pytest timed out after 60s",
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "total": 0,
            "pass_rate": 0.0,
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
            "per_file": {},
        }

        with patch.object(coverage_grader, '_compute_metrics', return_value=current_metrics):
            result = coverage_grader.grade()

        assert result.score == 0.0
        assert "did not complete" in str(result.feedbacks)


# ── Integration Tests ────────────────────────────────────────────────────────

class TestCoverageGraderIntegration:
    """Integration tests for realistic scenarios."""

    def test_constraint_mode_with_config(self, coverage_grader: CoverageGrader, tmp_path: Path) -> None:
        """Test constraint mode with explicit configuration."""
        with patch.object(coverage_grader.spec, 'config', {
            "coverage": {
                "mode": "constraint",
                "coverage_tolerance": 0.05,  # 5% tolerance
                "test_paths": ["tests/"],
                "source_paths": ["."],
            }
        }):
            mode = coverage_grader._resolve_mode()
            assert mode == "constraint"

            # Set baseline
            coverage_grader._baseline = {
                "passed": 10,
                "failed": 0,
                "errors": 0,
                "total": 10,
                "pass_rate": 1.0,
                "line_coverage": 0.8,
                "branch_coverage": 0.7,
                "per_file": {},
            }

            # Set current metrics (within tolerance)
            current_metrics = {
                "passed": 12,
                "failed": 0,
                "errors": 0,
                "total": 12,
                "pass_rate": 1.0,
                "line_coverage": 0.78,  # Small drop within 5% tolerance
                "branch_coverage": 0.68,
                "per_file": {},
                "timed_out": False,
                "run_error": "",
            }

            with patch.object(coverage_grader, '_compute_metrics', return_value=current_metrics):
                result = coverage_grader.grade()

            # Should get full score since within tolerance
            assert result.score > 0.9
            assert result.score <= 1.0

    def test_forced_constraint_mode(self, coverage_grader: CoverageGrader) -> None:
        """Test forcing constraint mode via grade_as_constraint."""
        # Set baseline
        coverage_grader._baseline = {
            "passed": 5,
            "failed": 1,
            "errors": 0,
            "total": 6,
            "pass_rate": 5/6,
            "line_coverage": 0.7,
            "branch_coverage": 0.6,
            "per_file": {},
        }

        # Set current metrics
        current_metrics = {
            "passed": 8,
            "failed": 0,
            "errors": 0,
            "total": 8,
            "pass_rate": 1.0,
            "line_coverage": 0.8,
            "branch_coverage": 0.7,
            "per_file": {},
            "timed_out": False,
            "run_error": "",
        }

        with patch.object(coverage_grader, '_compute_metrics', return_value=current_metrics):
            result = coverage_grader.grade_as_constraint()

        assert isinstance(result, GradeResult)
        assert "constraint" in str(result.feedbacks)

    def test_forced_objective_mode(self, coverage_grader: CoverageGrader) -> None:
        """Test forcing objective mode via grade_as_objective."""
        with patch.object(coverage_grader.spec, 'config', {
            "coverage": {"target_coverage": 0.9}
        }):
            # Set baseline
            coverage_grader._baseline = {
                "passed": 5,
                "failed": 0,
                "errors": 0,
                "total": 5,
                "pass_rate": 1.0,
                "line_coverage": 0.6,
                "branch_coverage": 0.5,
                "per_file": {},
            }

            # Set current metrics
            current_metrics = {
                "passed": 10,
                "failed": 0,
                "errors": 0,
                "total": 10,
                "pass_rate": 1.0,
                "line_coverage": 0.8,
                "branch_coverage": 0.7,
                "per_file": {},
                "timed_out": False,
                "run_error": "",
            }

            with patch.object(coverage_grader, '_compute_metrics', return_value=current_metrics):
                result = coverage_grader.grade_as_objective()

            assert isinstance(result, GradeResult)
            assert "objective" in str(result.feedbacks)