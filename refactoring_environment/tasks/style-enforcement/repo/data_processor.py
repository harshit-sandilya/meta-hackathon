"""
Data processing module with intentional style violations for the style-enforcement scenario.
This module contains violations across all style dimensions to be fixed by the agent.
"""

import os, sys  # I003: Multiple imports on one line
from typing import List, Dict, Optional, Union  # I005: Import ordering issue
import pandas as pd
import numpy as np
from collections import defaultdict
import json

# Naming violations
camelCaseVar = "bad_name"  # N005: CamelCase variable at module scope
ClassNameVar = 42  # N005: CamelCase variable

class dataProcessor:  # N002: Class name should be CapWords
    """Processes data files."""

    def __init__(self, file_path):  # T002: Missing type annotation for file_path
        self.file_path = file_path
        self.data = None

    def read_data(self):  # D001: Missing docstring on public method
        try:
            with open(self.file_path) as f:
                self.data = json.load(f)
        except:  # T003: Bare except clause
            return False
        return True

    def processRecords(self, records):  # N003: Method name should be snake_case
        # D001: Missing docstring
        result = []
        for i, record in enumerate(records):  # N007: Single letter variable 'i' is ok, but 'record' could be better
            if record.get('active', False):
                processed = self._transform(record)  # Complex nested logic
                if processed:
                    result.append(processed)
        return result

    def _transform(self, item):  # No docstring needed for private method (but could have one)
        """Transform a single data item."""
        if not item:
            return None

        new_item = {}
        for key, value in item.items():
            if key.startswith('_'):
                continue
            if isinstance(value, (int, float)):
                new_item[key] = value * 2  # Some transformation
            else:
                new_item[key] = str(value).upper()
        return new_item

# Function with multiple style issues
def CalculateStats(data: List[Dict]) -> Dict:  # N003: Function name should be snake_case
    """Calculate various statistics from the data."""  # D004: Summary line too long for Google style (should be <80 chars)
    total = sum(item.get('value', 0) for item in data)
    count = len(data)
    mean_val = total / count if count > 0 else 0

    # Complex nested comprehension - could be simplified
    filtered = [item for item in data if item.get('active', False) and item.get('value', 0) > mean_val]

    return {
        'total': total,
        'count': count,
        'mean': mean_val,
        'above_mean_count': len(filtered)
    }

# Missing docstring for public function
def validate_data(data):  # D001: Missing docstring, T001: Missing return type, T002: Missing param types
    if not isinstance(data, list):
        return False

    for item in data:
        if not isinstance(item, dict):
            return False
        if 'id' not in item:
            return False

    return True

# Constant naming violation
MaxItems = 100  # N004: Constant should be UPPER_CASE

class DataExporter:  # Missing docstring for public class - D002 violation
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def export_to_json(self, data, filename):  # Missing type annotations, missing docstring
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True

# Module-level function with formatting issues
def  process_batch(  items,  batch_size=10  ):  # F002: Missing two blank lines before function, N003: bad name
    """Process items in batches."""  # D005: Missing blank line after summary
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        processed = []
        for item in batch:
            if isinstance(item, dict):
                processed.append(item.copy())
        batches.append(processed)
    return batches

# This line is intentionally too long to trigger line length violation - it goes way beyond the 80 character limit that is standard in Google Python Style Guide and should be wrapped to multiple lines or refactored to be more concise and readable (C002)

# Test function with naming issue
def TestDataValidation():  # N010: Test function should start with 'test_'
    """Test the data validation function."""
    test_data = [{'id': 1, 'value': 10}, {'id': 2, 'value': 20}]
    result = validate_data(test_data)
    assert result == True, "Validation should pass for proper data"

    bad_data = [{'value': 10}, {'id': 2}]  # Missing 'id' in first item
    result = validate_data(bad_data)
    assert result == False, "Validation should fail for missing 'id'"

if __name__ == "__main__":
    # Example usage
    processor = dataProcessor("data.json")  # N002: Class name violation
    if processor.read_data():
        stats = CalculateStats(processor.data)  # N003: Function name violation
        print(f"Processed {stats['count']} items with mean {stats['mean']:.2f}")