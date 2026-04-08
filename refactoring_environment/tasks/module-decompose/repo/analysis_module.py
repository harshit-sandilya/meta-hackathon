"""
Complex data analysis module with high cyclomatic complexity and poor Big-O complexity.
This module contains multiple anti-patterns that need to be refactored by the agent.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import time
import math
from collections import defaultdict


class DataAnalyzer:
    """Performs complex data analysis with intentionally high complexity."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.preprocessed = False

    def preprocess_data(self) -> None:
        """Preprocess data with nested conditions and loops - high cyclomatic complexity."""
        if not self.data:
            return

        processed_data = []
        for i, item in enumerate(self.data):
            # Multiple nested conditions increase CC
            if 'value' in item:
                if item['value'] > 0:
                    if isinstance(item['value'], (int, float)):
                        # More nested conditions
                        if 'category' in item:
                            if item['category'] in ['A', 'B', 'C']:
                                new_value = item['value'] * 2
                            elif item['category'] == 'D':
                                new_value = item['value'] * 1.5
                            else:
                                new_value = item['value']
                        else:
                            new_value = item['value'] * 1.1

                        new_item = item.copy()
                        new_item['processed_value'] = new_value
                        processed_data.append(new_item)
                    else:
                        # Handle string values differently
                        try:
                            num_val = float(item['value'])
                            new_item = item.copy()
                            new_item['processed_value'] = num_val * 1.5
                            processed_data.append(new_item)
                        except (ValueError, TypeError):
                            # Skip invalid values
                            continue
                else:
                    # Handle non-positive values
                    if item.get('keep_zeros', False):
                        processed_data.append(item.copy())
            else:
                # Handle items without value field
                if item.get('has_alt_value', False):
                    alt_val = item.get('alt_value', 0)
                    new_item = item.copy()
                    new_item['processed_value'] = alt_val * 0.8
                    processed_data.append(new_item)

        self.data = processed_data
        self.preprocessed = True

    def analyze_complexity(self) -> Dict[str, float]:
        """Analyze data complexity with nested loops - O(n^3) complexity."""
        if not self.preprocessed:
            self.preprocess_data()

        if not self.data:
            return {'complexity_score': 0.0}

        # O(n^2) - nested loop over data
        correlation_matrix = []
        for i, item1 in enumerate(self.data):
            row = []
            for j, item2 in enumerate(self.data):
                # Simulate some correlation calculation
                val1 = item1.get('processed_value', 0)
                val2 = item2.get('processed_value', 0)
                correlation = 0.0

                # More nested conditions
                if val1 > 0 and val2 > 0:
                    if abs(val1 - val2) < 10:
                        correlation = 0.9
                    elif abs(val1 - val2) < 50:
                        correlation = 0.6
                    else:
                        correlation = 0.3
                elif val1 == 0 or val2 == 0:
                    correlation = 0.1
                else:
                    correlation = -0.5

                row.append(correlation)
            correlation_matrix.append(row)

        # O(n^2) - another nested loop
        total_correlation = 0.0
        count = 0
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix[i])):
                if i != j:
                    total_correlation += correlation_matrix[i][j]
                    count += 1

        avg_correlation = total_correlation / count if count > 0 else 0.0

        # O(n^3) - triple nested loop for some complex pattern detection
        pattern_score = 0.0
        for i in range(len(self.data)):
            for j in range(i+1, len(self.data)):
                for k in range(j+1, len(self.data)):
                    # Simulate some pattern detection logic
                    val_i = self.data[i].get('processed_value', 0)
                    val_j = self.data[j].get('processed_value', 0)
                    val_k = self.data[k].get('processed_value', 0)

                    if val_i > 0 and val_j > 0 and val_k > 0:
                        if abs(val_i - val_j) < 5 and abs(val_j - val_k) < 5:
                            pattern_score += 1.0
                        elif (val_i > val_j and val_j > val_k) or (val_i < val_j and val_j < val_k):
                            pattern_score += 0.5

        complexity_score = (avg_correlation * 0.7) + (pattern_score * 0.3)

        return {
            'complexity_score': complexity_score,
            'avg_correlation': avg_correlation,
            'pattern_score': pattern_score
        }

    def find_outliers(self, threshold: float = 3.0) -> List[Dict[str, Any]]:
        """Find outliers using inefficient nested computations - O(n^2) with sort-in-loop."""
        if not self.preprocessed:
            self.preprocess_data()

        if not self.data or len(self.data) < 3:
            return []

        outliers = []
        values = []

        # Extract values - O(n)
        for item in self.data:
            if 'processed_value' in item:
                values.append(item['processed_value'])

        if not values:
            return []

        # Inefficient outlier detection with nested loops
        for i, item in enumerate(self.data):
            if 'processed_value' not in item:
                continue

            current_val = item['processed_value']
            distances = []

            # O(n^2) - nested loop to calculate distances to all other points
            for j, other_item in enumerate(self.data):
                if i == j or 'processed_value' not in other_item:
                    continue

                other_val = other_item['processed_value']
                distance = abs(current_val - other_val)
                distances.append(distance)

            # Sort distances (O(n log n)) inside loop (O(n^2 log n) total)
            distances.sort()

            # Calculate median distance
            median_distance = distances[len(distances) // 2] if distances else 0

            # Determine if outlier based on threshold
            if median_distance > threshold * 10:
                outliers.append(item)

        return outliers

    def recursive_analysis(self, depth: int = 0, max_depth: int = 5) -> Dict[str, Any]:
        """Recursive analysis with exponential complexity - O(2^n) without memoization."""
        if depth >= max_depth:
            return {'depth': depth, 'analysis': 'max_depth_reached'}

        if not self.data:
            return {'depth': depth, 'analysis': 'no_data'}

        # Split data recursively without memoization
        left_data = []
        right_data = []

        for i, item in enumerate(self.data):
            val = item.get('processed_value', 0)
            if i % 2 == 0:
                left_data.append(item)
            else:
                right_data.append(item)

        # Two recursive calls - exponential growth
        left_result = self.recursive_analysis(depth + 1, max_depth)
        right_result = self.recursive_analysis(depth + 1, max_depth)

        return {
            'depth': depth,
            'left': left_result,
            'right': right_result,
            'data_count': len(self.data),
            'analysis': f'analyzed_{depth}'
        }

    def string_concatenation_analysis(self) -> str:
        """Inefficient string concatenation in loop - O(n^2) string building."""
        result = ""

        for i, item in enumerate(self.data):
            if 'processed_value' in item:
                # String concatenation in loop - very inefficient
                result += f"Item {i}: {item['processed_value']}\n"

                # Even more inefficient - nested string operations
                if item['processed_value'] > 50:
                    result += f"  -> High value detected: {item['processed_value']}\n"
                    for j in range(3):  # Unnecessary nested loop
                        result += f"    - Detail {j+1}\n"

        return result

    def complex_data_transformation(self) -> List[Dict[str, Any]]:
        """Complex transformation with multiple nested conditions and loops."""
        transformed = []

        # Unnecessarily complex transformation logic
        for i, item in enumerate(self.data):
            if 'processed_value' in item:
                base_value = item['processed_value']

                # Multiple nested conditions
                if base_value < 10:
                    multiplier = 2.0
                elif base_value < 50:
                    multiplier = 1.5
                    if 'category' in item and item['category'] == 'A':
                        multiplier = 1.8
                elif base_value < 100:
                    multiplier = 1.2
                    if 'category' in item:
                        if item['category'] == 'B':
                            multiplier = 1.3
                        elif item['category'] == 'C':
                            multiplier = 1.4
                else:
                    multiplier = 1.1
                    if 'category' in item:
                        if item['category'] == 'D':
                            multiplier = 1.05
                        elif item['category'] in ['E', 'F']:
                            multiplier = 1.0

                # More nested conditions for additional transformations
                transformed_value = base_value * multiplier
                if transformed_value > 100:
                    if 'priority' in item:
                        if item['priority'] == 'high':
                            transformed_value *= 1.1
                        elif item['priority'] == 'low':
                            transformed_value *= 0.95
                    else:
                        transformed_value *= 0.98

                new_item = item.copy()
                new_item['transformed_value'] = transformed_value
                new_item['multiplier_used'] = multiplier
                transformed.append(new_item)

        return transformed


# Utility function with high cyclomatic complexity
def validate_and_normalize_data(data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Validate and normalize data with many nested conditions."""
    if not data:
        return None

    normalized = []
    error_count = 0

    for i, item in enumerate(data):
        # Multiple validation checks
        if not isinstance(item, dict):
            error_count += 1
            continue

        if 'id' not in item:
            error_count += 1
            continue

        if not isinstance(item['id'], (int, str)):
            error_count += 1
            continue

        # Complex normalization logic
        new_item = {'id': item['id']}

        if 'value' in item:
            if isinstance(item['value'], (int, float)):
                new_item['value'] = float(item['value'])
            elif isinstance(item['value'], str):
                try:
                    new_item['value'] = float(item['value'])
                except ValueError:
                    new_item['value'] = 0.0
                    error_count += 1
            else:
                new_item['value'] = 0.0
                error_count += 1
        else:
            new_item['value'] = 0.0

        if 'category' in item:
            if item['category'] in ['A', 'B', 'C', 'D', 'E']:
                new_item['category'] = item['category']
            else:
                new_item['category'] = 'UNKNOWN'
        else:
            new_item['category'] = 'DEFAULT'

        # Additional complex conditions
        if new_item['value'] > 0:
            if new_item['category'] == 'A':
                new_item['priority'] = 'high'
            elif new_item['category'] in ['B', 'C']:
                new_item['priority'] = 'medium'
            else:
                new_item['priority'] = 'low'
        else:
            new_item['priority'] = 'none'

        normalized.append(new_item)

    if error_count > len(data) * 0.3:  # More nested conditions
        return None

    return normalized if normalized else None


# Function demonstrating multiple complexity anti-patterns
def analyze_data_correlation_matrix(data_points: List[float]) -> List[List[float]]:
    """Generate correlation matrix with O(n^3) complexity and nested loops."""
    n = len(data_points)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    # O(n^3) triple nested loop
    for i in range(n):
        for j in range(n):
            # Inner loop for complex correlation calculation
            correlation = 0.0
            for k in range(min(i, j) + 1):
                # This nested loop creates O(n^3) complexity
                if i != j:
                    diff = abs(data_points[i] - data_points[j])
                    if diff < 1.0:
                        correlation += 0.1
                    elif diff < 5.0:
                        correlation += 0.05
                    else:
                        correlation -= 0.01

                # Even more nested conditions
                if k % 2 == 0:
                    correlation *= 1.01
                else:
                    correlation *= 0.99

            matrix[i][j] = correlation

    return matrix