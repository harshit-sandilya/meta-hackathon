"""
Comprehensive test suite for all refactoring scenarios.
Tests scenario loading, grader configuration, and basic functionality.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from refactoring_environment.environment.registry.scenario import ScenarioSpec


# Test scenarios
SCENARIO_PATHS = [
    "tasks/lint-cleanup/scenario.yaml",
    "tasks/style-enforcement/scenario.yaml",
    "tasks/module-decompose/scenario.yaml",
]


def test_all_scenarios_load_correctly():
    """Test that all scenario YAML files can be loaded without errors."""
    for scenario_path in SCENARIO_PATHS:
        try:
            spec = ScenarioSpec.from_yaml(scenario_path)
            print(f"✓ Loaded {spec.slug}: {spec.name}")

            # Basic validation
            assert spec.slug, f"Scenario {scenario_path} missing slug"
            assert spec.name, f"Scenario {scenario_path} missing name"
            assert spec.graders, f"Scenario {scenario_path} missing graders"

        except Exception as e:
            pytest.fail(f"Failed to load scenario {scenario_path}: {e}")


def test_style_enforcement_scenario_configuration():
    """Test that the style-enforcement scenario has correct configuration."""
    spec = ScenarioSpec.from_yaml("tasks/style-enforcement/scenario.yaml")

    # Verify scenario identity
    assert spec.slug == "style-enforcement"
    assert "Style" in spec.name
    assert spec.max_steps == 25

    # Verify grader weights - style should be highest
    graders = spec.active_graders()
    assert "style" in graders
    assert graders["style"].weight == 0.60  # Highest weight

    # Verify other graders have lower weights
    assert graders["lint"].weight == 0.20
    assert graders["coverage"].weight == 0.15
    assert graders["symbol"].weight == 0.05

    print("✓ Style-enforcement scenario configuration validated")


def test_module_decompose_scenario_configuration():
    """Test that the module-decompose scenario has correct configuration."""
    spec = ScenarioSpec.from_yaml("tasks/module-decompose/scenario.yaml")

    # Verify scenario identity
    assert spec.slug == "module-decompose"
    assert "Module" in spec.name
    assert spec.max_steps == 30

    # Verify grader weights - complexity should be highest
    graders = spec.active_graders()
    assert "complexity" in graders
    assert graders["complexity"].weight == 0.65  # Highest weight

    # Verify other graders have lower weights
    assert graders["coverage"].weight == 0.20
    assert graders["lint"].weight == 0.10
    assert graders["symbol"].weight == 0.05

    # Verify higher test regression penalty
    assert spec.penalties.test_regression == 0.30

    print("✓ Module-decompose scenario configuration validated")


def test_lint_cleanup_scenario_configuration():
    """Test that the lint-cleanup scenario has correct configuration."""
    spec = ScenarioSpec.from_yaml("tasks/lint-cleanup/scenario.yaml")

    # Verify scenario identity
    assert spec.slug == "lint-cleanup"
    assert "Lint" in spec.name
    assert spec.max_steps == 15

    # Verify grader weights - lint should be highest
    graders = spec.active_graders()
    assert "lint" in graders
    assert graders["lint"].weight == 0.50  # Highest weight

    # Verify other graders have lower weights
    assert graders["symbol"].weight == 0.30
    assert graders["style"].weight == 0.10
    assert graders["coverage"].weight == 0.10

    print("✓ Lint-cleanup scenario configuration validated")


def test_scenario_file_structure():
    """Test that scenario directories have the expected file structure."""
    scenarios = [
        ("style-enforcement", ["repo/data_processor.py", "repo/tests/test_data_processor.py"]),
        ("module-decompose", ["repo/analysis_module.py", "repo/tests/test_analysis_module.py"]),
        ("lint-cleanup", ["repo/utils.py", "repo/tests/test_utils.py"]),
    ]

    for scenario_slug, expected_files in scenarios:
        scenario_dir = Path(f"tasks/{scenario_slug}")
        assert scenario_dir.exists(), f"Scenario directory {scenario_slug} not found"

        for expected_file in expected_files:
            file_path = scenario_dir / expected_file
            assert (
                file_path.exists()
            ), f"Expected file {expected_file} not found in {scenario_slug}"
            print(f"✓ Found {expected_file} in {scenario_slug}")


def test_scenario_invariants():
    """Test that scenarios have appropriate invariants configured."""
    scenarios_to_test = ["style-enforcement", "module-decompose", "lint-cleanup"]

    for scenario_slug in scenarios_to_test:
        spec = ScenarioSpec.from_yaml(f"tasks/{scenario_slug}/scenario.yaml")

        # Should have test directory protection invariant
        has_test_invariant = any(
            inv.type == "no_edit" and any("test" in path for path in inv.paths)
            for inv in spec.invariants
        )
        assert (
            has_test_invariant
        ), f"Scenario {scenario_slug} missing test protection invariant"

        # Should have main file existence invariant
        has_file_invariant = any(inv.type == "file_exists" for inv in spec.invariants)
        assert (
            has_file_invariant
        ), f"Scenario {scenario_slug} missing file existence invariant"

        print(f"✓ {scenario_slug} has proper invariants")


def test_grader_weight_distribution():
    """Test that grader weights sum to reasonable values and prioritize correctly."""
    test_cases = [
        ("style-enforcement", "style"),
        ("module-decompose", "complexity"),
        ("lint-cleanup", "lint"),
    ]

    for scenario_slug, expected_primary_grader in test_cases:
        spec = ScenarioSpec.from_yaml(f"tasks/{scenario_slug}/scenario.yaml")
        graders = spec.active_graders()

        # Find the grader with highest weight
        primary_grader = max(graders.items(), key=lambda x: x[1].weight)
        assert (
            primary_grader[0] == expected_primary_grader
        ), f"Expected {expected_primary_grader} to be primary grader in {scenario_slug}, got {primary_grader[0]}"

        # Verify weights sum to 1.0 (approximately)
        total_weight = sum(grader.weight for grader in graders.values())
        assert (
            abs(total_weight - 1.0) < 0.01
        ), f"Grader weights in {scenario_slug} don't sum to 1.0, got {total_weight}"

        print(f"✓ {scenario_slug} has correct grader weight distribution")


def test_scenario_descriptions_are_comprehensive():
    """Test that scenarios have comprehensive descriptions."""
    for scenario_path in SCENARIO_PATHS:
        spec = ScenarioSpec.from_yaml(scenario_path)

        # Description should be substantial
        description = spec.description or ""
        assert (
            len(description) > 100
        ), f"Scenario {spec.slug} has too short description (length: {len(description)})"

        # Description should mention key aspects
        if spec.slug == "style-enforcement":
            assert "style" in description.lower() or "compliance" in description.lower()
        elif spec.slug == "module-decompose":
            assert (
                "complexity" in description.lower()
                or "decompose" in description.lower()
            )
        elif spec.slug == "lint-cleanup":
            assert "lint" in description.lower() or "violation" in description.lower()

        print(f"✓ {spec.slug} has comprehensive description")


def test_scenario_efficiency_config():
    """Test that scenarios have appropriate efficiency configurations."""
    test_cases = [
        ("style-enforcement", 0.4),
        ("module-decompose", 0.3),
        ("lint-cleanup", 0.5),
    ]

    for scenario_slug, expected_decay_rate in test_cases:
        spec = ScenarioSpec.from_yaml(f"tasks/{scenario_slug}/scenario.yaml")
        assert spec.eff_config.decay_rate == expected_decay_rate, \
            f"Expected decay_rate {expected_decay_rate} for {scenario_slug}, got {spec.eff_config.decay_rate}"

        print(f"✓ {scenario_slug} has correct efficiency configuration")


if __name__ == "__main__":
    # Run all tests
    print("Testing all refactoring scenarios...")

    test_all_scenarios_load_correctly()
    test_style_enforcement_scenario_configuration()
    test_module_decompose_scenario_configuration()
    test_lint_cleanup_scenario_configuration()
    test_scenario_file_structure()
    test_scenario_invariants()
    test_grader_weight_distribution()
    test_scenario_descriptions_are_comprehensive()
    test_scenario_efficiency_config()

    print("\n✅ All scenario tests passed!")