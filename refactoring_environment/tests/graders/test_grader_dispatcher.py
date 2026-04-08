"""
tests/graders/test_grader_dispatcher.py

Unit and integration tests for GraderDispatcher.
Tests the dispatcher's ability to manage and run multiple graders.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from refactoring_environment.environment.graders import GraderDispatcher
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
def mock_file_handler() -> Mock:
    """Mock FileHandler for testing."""
    return Mock(spec=FileHandler)


@pytest.fixture
def mock_scenario() -> Mock:
    """Mock ScenarioSpec for testing."""
    scenario = Mock(spec=ScenarioSpec)
    scenario.task_id = "test-task"
    scenario.graders = {}
    return scenario


@pytest.fixture
def grader_dispatcher(mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> GraderDispatcher:
    """Create a GraderDispatcher instance with mocked dependencies."""
    return GraderDispatcher(
        scenario=mock_scenario,
        executor=mock_executor,
        file_handler=mock_file_handler
    )


# ── Unit Tests for GraderDispatcher ─────────────────────────────────────────

class TestGraderDispatcher:
    """Test the GraderDispatcher class."""

    def test_initialization_empty_graders(self, grader_dispatcher: GraderDispatcher) -> None:
        """Test initialization with no graders."""
        assert len(grader_dispatcher._graders) == 0

    def test_initialization_with_graders(self, mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> None:
        """Test initialization with configured graders."""
        # Set up mock scenario with graders
        mock_scenario.graders = {
            "lint": {"weight": 0.5},
            "coverage": {"weight": 0.3},
        }

        with patch('refactoring_environment.environment.graders.registry.build_grader') as mock_build:
            # Mock grader instances
            mock_lint_grader = Mock()
            mock_lint_grader.grader_id = "lint"
            mock_lint_grader.spec = GraderSpec(weight=0.5, target_coverage=0.8)

            mock_coverage_grader = Mock()
            mock_coverage_grader.grader_id = "coverage"
            mock_coverage_grader.spec = GraderSpec(weight=0.3, target_coverage=0.8)

            # Set up side effect for build_grader
            mock_build.side_effect = [mock_lint_grader, mock_coverage_grader]

            dispatcher = GraderDispatcher(
                scenario=mock_scenario,
                executor=mock_executor,
                file_handler=mock_file_handler
            )

        assert len(dispatcher._graders) == 2
        assert dispatcher._graders[0].grader_id == "lint"
        assert dispatcher._graders[1].grader_id == "coverage"

    def test_grade_no_graders(self, grader_dispatcher: GraderDispatcher) -> None:
        """Test grading with no graders configured."""
        grader_ctx, reward_ctx = grader_dispatcher.grade(step=1, cumulative_penalty=0.5)

        assert isinstance(grader_ctx, GraderContext)
        assert isinstance(reward_ctx, RewardContext)

        # Should have empty results
        assert grader_ctx.scores == {}
        assert grader_ctx.feedbacks == []
        assert grader_ctx.errors == []
        assert grader_ctx.tool_errors == []
        assert grader_ctx.penalties == []
        assert grader_ctx.is_regression == False

        assert reward_ctx.step_score == 0.0
        assert reward_ctx.cumulative_penalty == 0.5

    def test_grade_with_single_grader(self, mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> None:
        """Test grading with a single grader."""
        # Set up mock scenario with one grader
        mock_scenario.graders = {"lint": {"weight": 0.8}}

        with patch('refactoring_environment.environment.graders.registry.build_grader') as mock_build:
            # Mock grader instance
            mock_lint_grader = Mock()
            mock_lint_grader.grader_id = "lint"
            mock_lint_grader.spec = GraderSpec(weight=0.8, target_coverage=0.8)

            # Mock grade result
            mock_lint_grader.grade.return_value = GradeResult(
                score=0.9,
                feedbacks=["Lint: 2 violations fixed"],
                errors=[],
                tool_errors=[],
                added_violations=0
            )

            mock_build.return_value = mock_lint_grader

            dispatcher = GraderDispatcher(
                scenario=mock_scenario,
                executor=mock_executor,
                file_handler=mock_file_handler
            )

            grader_ctx, reward_ctx = dispatcher.grade(step=1, cumulative_penalty=0.2)

        assert len(grader_ctx.scores) == 1
        assert "lint" in grader_ctx.scores
        assert grader_ctx.scores["lint"] == 0.9
        assert len(grader_ctx.feedbacks) == 1
        assert "Lint: 2 violations fixed" in grader_ctx.feedbacks[0]
        assert grader_ctx.is_regression == False

        # Step score should be weight * score = 0.8 * 0.9 = 0.72
        assert reward_ctx.step_score == 0.72
        assert reward_ctx.cumulative_penalty == 0.2

    def test_grade_with_multiple_graders(self, mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> None:
        """Test grading with multiple graders of different weights."""
        # Set up mock scenario with multiple graders
        mock_scenario.graders = {
            "lint": {"weight": 0.5},
            "coverage": {"weight": 0.3},
            "style": {"weight": 0.2},
        }

        with patch('refactoring_environment.environment.graders.registry.build_grader') as mock_build:
            # Mock grader instances
            mock_lint_grader = Mock()
            mock_lint_grader.grader_id = "lint"
            mock_lint_grader.spec = GraderSpec(weight=0.5, target_coverage=0.8)
            mock_lint_grader.grade.return_value = GradeResult(
                score=0.8,
                feedbacks=["Lint: good"],
                errors=[],
                tool_errors=[],
                added_violations=0
            )

            mock_coverage_grader = Mock()
            mock_coverage_grader.grader_id = "coverage"
            mock_coverage_grader.spec = GraderSpec(weight=0.3, target_coverage=0.8)
            mock_coverage_grader.grade.return_value = GradeResult(
                score=0.9,
                feedbacks=["Coverage: excellent"],
                errors=[],
                tool_errors=[],
                added_violations=0
            )

            mock_style_grader = Mock()
            mock_style_grader.grader_id = "style"
            mock_style_grader.spec = GraderSpec(weight=0.2, target_coverage=0.8)
            mock_style_grader.grade.return_value = GradeResult(
                score=0.7,
                feedbacks=["Style: needs work"],
                errors=[],
                tool_errors=[],
                added_violations=2  # Some violations
            )

            mock_build.side_effect = [mock_lint_grader, mock_coverage_grader, mock_style_grader]

            dispatcher = GraderDispatcher(
                scenario=mock_scenario,
                executor=mock_executor,
                file_handler=mock_file_handler
            )

            grader_ctx, reward_ctx = dispatcher.grade(step=2, cumulative_penalty=0.1)

        # Check grader context
        assert len(grader_ctx.scores) == 3
        assert grader_ctx.scores["lint"] == 0.8
        assert grader_ctx.scores["coverage"] == 0.9
        assert grader_ctx.scores["style"] == 0.7
        assert len(grader_ctx.feedbacks) == 3
        assert grader_ctx.is_regression == True  # style grader has added_violations
        assert grader_ctx.penalties == ["2"]  # style grader violations (as strings)

        # Check reward context
        # Step score = (0.5 * 0.8) + (0.3 * 0.9) + (0.2 * 0.7) = 0.4 + 0.27 + 0.14 = 0.81
        assert reward_ctx.step_score == 0.81
        assert reward_ctx.cumulative_penalty == 0.1

    def test_grade_with_regression(self, mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> None:
        """Test that regression detection works correctly."""
        mock_scenario.graders = {"lint": {"weight": 1.0}}

        with patch('refactoring_environment.environment.graders.registry.build_grader') as mock_build:
            mock_lint_grader = Mock()
            mock_lint_grader.grader_id = "lint"
            mock_lint_grader.spec = GraderSpec(weight=1.0, target_coverage=0.8)
            mock_lint_grader.grade.return_value = GradeResult(
                score=0.6,
                feedbacks=["Lint: regression detected"],
                errors=[],
                tool_errors=[],
                added_violations=5  # Multiple new violations
            )

            mock_build.return_value = mock_lint_grader

            dispatcher = GraderDispatcher(
                scenario=mock_scenario,
                executor=mock_executor,
                file_handler=mock_file_handler
            )

            grader_ctx, reward_ctx = dispatcher.grade(step=1, cumulative_penalty=0.0)

        assert grader_ctx.is_regression == True
        assert grader_ctx.penalties == ["5"]  # Penalties are strings
        assert reward_ctx.step_score == 0.6

    def test_grade_with_tool_errors(self, mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> None:
        """Test handling of tool errors from graders."""
        mock_scenario.graders = {"lint": {"weight": 1.0}}

        with patch('refactoring_environment.environment.graders.registry.build_grader') as mock_build:
            mock_lint_grader = Mock()
            mock_lint_grader.grader_id = "lint"
            mock_lint_grader.spec = GraderSpec(weight=1.0, target_coverage=0.8)
            mock_lint_grader.grade.return_value = GradeResult(
                score=0.5,
                feedbacks=["Lint: some issues"],
                errors=["SyntaxError in test.py"],
                tool_errors=["ruff not found"],
                added_violations=0
            )

            mock_build.return_value = mock_lint_grader

            dispatcher = GraderDispatcher(
                scenario=mock_scenario,
                executor=mock_executor,
                file_handler=mock_file_handler
            )

            grader_ctx, reward_ctx = dispatcher.grade(step=1, cumulative_penalty=0.0)

        assert len(grader_ctx.errors) == 1
        assert "SyntaxError in test.py" in grader_ctx.errors
        assert len(grader_ctx.tool_errors) == 1
        assert "ruff not found" in grader_ctx.tool_errors


# ── Integration Tests ────────────────────────────────────────────────────────

class TestGraderDispatcherIntegration:
    """Integration tests for realistic scenarios."""

    def test_unknown_grader_handling(self, mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> None:
        """Test that unknown grader IDs are handled gracefully."""
        mock_scenario.graders = {
            "lint": {"weight": 0.5},
            "unknown_grader": {"weight": 0.3},  # This should be skipped
            "coverage": {"weight": 0.2},
        }

        with patch('refactoring_environment.environment.graders.registry.build_grader') as mock_build, \
             patch('warnings.warn') as mock_warn:

            mock_lint_grader = Mock()
            mock_lint_grader.grader_id = "lint"
            mock_lint_grader.spec = GraderSpec(weight=0.5, target_coverage=0.8)

            mock_coverage_grader = Mock()
            mock_coverage_grader.grader_id = "coverage"
            mock_coverage_grader.spec = GraderSpec(weight=0.2, target_coverage=0.8)

            # Simulate KeyError for unknown grader
            def build_side_effect(grader_id, spec, exec, handler):
                if grader_id == "unknown_grader":
                    raise KeyError(f"Unknown grader_id '{grader_id}'")
                elif grader_id == "lint":
                    return mock_lint_grader
                else:
                    return mock_coverage_grader

            mock_build.side_effect = build_side_effect

            dispatcher = GraderDispatcher(
                scenario=mock_scenario,
                executor=mock_executor,
                file_handler=mock_file_handler
            )

            # Should warn about unknown grader
            assert mock_warn.called
            assert len(dispatcher._graders) == 2  # Only lint and coverage

    def test_grader_ordering_by_weight(self, mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> None:
        """Test that graders are sorted by weight in descending order."""
        mock_scenario.graders = {
            "lint": {"weight": 0.2},     # Should be last
            "coverage": {"weight": 0.5},  # Should be second
            "style": {"weight": 0.8},    # Should be first
        }

        with patch('refactoring_environment.environment.graders.registry.build_grader') as mock_build:
            # Create mock graders
            graders = []
            for grader_id, weight in [("lint", 0.2), ("coverage", 0.5), ("style", 0.8)]:
                grader = Mock()
                grader.grader_id = grader_id
                grader.spec = GraderSpec(weight=weight, target_coverage=0.8)
                graders.append(grader)

            mock_build.side_effect = graders

            dispatcher = GraderDispatcher(
                scenario=mock_scenario,
                executor=mock_executor,
                file_handler=mock_file_handler
            )

            # Check order - should be sorted by weight descending
            assert dispatcher._graders[0].grader_id == "style"   # weight 0.8
            assert dispatcher._graders[1].grader_id == "coverage"  # weight 0.5
            assert dispatcher._graders[2].grader_id == "lint"     # weight 0.2

    def test_empty_grader_config(self, mock_scenario: Mock, mock_executor: Mock, mock_file_handler: Mock) -> None:
        """Test handling of empty grader configuration."""
        # Empty graders dict
        mock_scenario.graders = {}

        dispatcher = GraderDispatcher(
            scenario=mock_scenario,
            executor=mock_executor,
            file_handler=mock_file_handler
        )

        grader_ctx, reward_ctx = dispatcher.grade(step=1, cumulative_penalty=0.0)

        # Should work fine with no graders
        assert len(grader_ctx.scores) == 0
        assert reward_ctx.step_score == 0.0