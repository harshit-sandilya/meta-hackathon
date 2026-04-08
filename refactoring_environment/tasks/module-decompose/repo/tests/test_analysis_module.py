"""Tests for the analysis module - these tests should remain green during refactoring."""

import pytest
from analysis_module import DataAnalyzer, validate_and_normalize_data


def test_data_analyzer_initialization():
    """Test DataAnalyzer initialization."""
    analyzer = DataAnalyzer([])
    assert analyzer.data == []
    assert analyzer.preprocessed == False


def test_preprocess_data_empty():
    """Test preprocessing with empty data."""
    analyzer = DataAnalyzer([])
    analyzer.preprocess_data()
    assert analyzer.data == []
    assert analyzer.preprocessed == True


def test_preprocess_data_basic():
    """Test basic data preprocessing."""
    test_data = [
        {'value': 10, 'category': 'A'},
        {'value': -5, 'keep_zeros': True},
        {'has_alt_value': True, 'alt_value': 20}
    ]
    analyzer = DataAnalyzer(test_data)
    analyzer.preprocess_data()

    assert len(analyzer.data) == 3
    assert analyzer.data[0]['processed_value'] == 20  # 10 * 2 for category A
    assert analyzer.data[1]['value'] == -5  # Kept because keep_zeros=True


def test_analyze_complexity_empty():
    """Test complexity analysis with empty data."""
    analyzer = DataAnalyzer([])
    result = analyzer.analyze_complexity()
    assert result['complexity_score'] == 0.0


def test_find_outliers_minimum_data():
    """Test outlier detection with minimum data."""
    analyzer = DataAnalyzer([{'processed_value': 10}])  # Single item
    outliers = analyzer.find_outliers()
    assert len(outliers) == 0


def test_string_concatenation_analysis_basic():
    """Test string concatenation analysis."""
    analyzer = DataAnalyzer([{'processed_value': 60}])  # High value triggers extra details
    result = analyzer.string_concatenation_analysis()
    assert "Item 0: 60" in result
    assert "High value detected: 60" in result


def test_recursive_analysis_base_case():
    """Test recursive analysis at max depth."""
    analyzer = DataAnalyzer([])
    result = analyzer.recursive_analysis(depth=5, max_depth=5)
    assert result['depth'] == 5
    assert result['analysis'] == 'max_depth_reached'


def test_validate_and_normalize_data_empty():
    """Test validation with empty data."""
    result = validate_and_normalize_data([])
    assert result is None


def test_validate_and_normalize_data_valid():
    """Test validation with valid data."""
    test_data = [
        {'id': 1, 'value': 10.5, 'category': 'A'},
        {'id': 2, 'value': '20', 'category': 'B'},
        {'id': 3, 'value': -5, 'category': 'X'}  # Unknown category
    ]
    result = validate_and_normalize_data(test_data)
    assert result is not None
    assert len(result) == 3
    assert result[0]['value'] == 10.5
    assert result[1]['value'] == 20.0  # Converted from string
    assert result[2]['category'] == 'UNKNOWN'  # Normalized unknown category


def test_validate_and_normalize_data_invalid():
    """Test validation with invalid data."""
    # Missing 'id' field
    invalid_data = [{'value': 10}]
    result = validate_and_normalize_data(invalid_data)
    assert result is None

    # Non-dict items
    invalid_data2 = ["not", "dicts"]
    result = validate_and_normalize_data(invalid_data2)
    assert result is None


def test_complex_data_transformation_empty():
    """Test complex transformation with empty data."""
    analyzer = DataAnalyzer([])
    result = analyzer.complex_data_transformation()
    assert result == []


@pytest.mark.parametrize("data_points,expected_size", [
    ([], 0),  # Empty data
    ([1.0], 1),  # Single point
    ([1.0, 2.0, 3.0], 3),  # Multiple points
])
def test_analyze_data_correlation_matrix_sizes(data_points, expected_size):
    """Test correlation matrix size."""
    matrix = DataAnalyzer.analyze_data_correlation_matrix(data_points)
    assert len(matrix) == expected_size
    if expected_size > 0:
        assert all(len(row) == expected_size for row in matrix)