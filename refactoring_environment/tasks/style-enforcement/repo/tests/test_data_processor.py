"""Tests for the data processor module - these tests should remain green during refactoring."""

import pytest
from data_processor import dataProcessor, CalculateStats, validate_data, DataExporter


def test_data_processor_initialization():
    """Test that data processor initializes correctly."""
    processor = dataProcessor("test_data.json")
    assert processor.file_path == "test_data.json"
    assert processor.data is None


def test_calculate_stats_basic():
    """Test basic statistics calculation."""
    test_data = [
        {'value': 10, 'active': True},
        {'value': 20, 'active': True},
        {'value': 30, 'active': False}
    ]
    stats = CalculateStats(test_data)

    assert stats['total'] == 30  # 10 + 20
    assert stats['count'] == 3
    assert stats['mean'] == 10  # 30 / 3
    assert stats['above_mean_count'] == 1  # Only 20 is above mean of 10


def test_calculate_stats_empty():
    """Test statistics calculation with empty data."""
    stats = CalculateStats([])
    assert stats['total'] == 0
    assert stats['count'] == 0
    assert stats['mean'] == 0
    assert stats['above_mean_count'] == 0


def test_validate_data_valid():
    """Test data validation with valid data."""
    valid_data = [
        {'id': 1, 'name': 'test1'},
        {'id': 2, 'name': 'test2'}
    ]
    assert validate_data(valid_data) == True


def test_validate_data_invalid():
    """Test data validation with invalid data."""
    # Missing 'id' field
    invalid_data = [{'name': 'test1'}]
    assert validate_data(invalid_data) == False

    # Non-dict item
    invalid_data2 = ['not', 'a', 'dict']
    assert validate_data(invalid_data2) == False


def test_data_exporter_export():
    """Test data export functionality."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = DataExporter(temp_dir)
        test_data = [{'id': 1, 'value': 100}]

        result = exporter.export_to_json(test_data, "test_output.json")
        assert result == True

        # Verify file was created
        output_path = os.path.join(temp_dir, "test_output.json")
        assert os.path.exists(output_path)


def test_process_batch_basic():
    """Test basic batch processing."""
    from data_processor import process_batch

    test_items = [{'id': i} for i in range(10)]
    batches = process_batch(test_items, batch_size=3)

    # Should create 4 batches: [0-2], [3-5], [6-8], [9]
    assert len(batches) == 4
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    assert len(batches[3]) == 1


@pytest.mark.parametrize("data,expected", [
    ([{'id': 1}], True),
    ([{'value': 10}], False),  # Missing 'id'
    ([1, 2, 3], False),  # Not dicts
    ([], True),  # Empty list is considered valid
])
def test_validate_data_parametrized(data, expected):
    """Parametrized test for data validation."""
    assert validate_data(data) == expected