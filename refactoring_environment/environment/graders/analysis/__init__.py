"""
graders/analysis/__init__.py

Shared analysis layer for all graders. Parse once at reset(), reuse everywhere.

Typical usage inside a grader:

    from .analysis import (
        parse_repository,
        collect_definitions,
        collect_usages,
        DependencyGraph,
        estimate_repo_complexity,
        ComplexityClass,
        COMPLEXITY_COST,
    )
"""

from __future__ import annotations

# --- AST utilities -----------------------------------------------------------
from .ast_utils import (
    AnnotationStats,
    ClassInfo,
    FunctionInfo,
    # Data models
    SymbolDef,
    SymbolRef,
    UnreachableBlock,
    # Definition / usage collection
    collect_definitions,
    collect_usages,
    # Unreachable code detection
    detect_unreachable_blocks,
    get_annotation_density,
    hash_subtree,
    iter_classes,
    # Repository iterators
    iter_functions,
    # Normalisation (duplication grader)
    normalize_ast_subtree,
    # Parsing
    parse_file_safe,
    parse_repository,
)

# --- Complexity estimation ----------------------------------------------------
from .complexity_analysis import (
    COMPLEXITY_COST,
    # Enums / cost table
    ComplexityClass,
    # Data model
    ComplexityEstimate,
    # Repository-wide scan (used by ComplexityGrader)
    estimate_repo_complexity,
    estimate_space_complexity,
    # Per-function estimation
    estimate_time_complexity,
)

# --- Dependency graph --------------------------------------------------------
from .dep_graph import (
    # Edge constants
    EDGE_CALLS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
    DeadSymbol,
    # Main graph class
    DependencyGraph,
    # Data models
    Edge,
)

__all__ = [
    # ast_utils
    "SymbolDef",
    "SymbolRef",
    "FunctionInfo",
    "ClassInfo",
    "AnnotationStats",
    "UnreachableBlock",
    "parse_file_safe",
    "parse_repository",
    "collect_definitions",
    "collect_usages",
    "iter_functions",
    "iter_classes",
    "get_annotation_density",
    "detect_unreachable_blocks",
    "normalize_ast_subtree",
    "hash_subtree",
    # dep_graph
    "EDGE_CALLS",
    "EDGE_IMPORTS",
    "EDGE_INHERITS",
    "Edge",
    "DeadSymbol",
    "DependencyGraph",
    # complexity
    "ComplexityClass",
    "COMPLEXITY_COST",
    "ComplexityEstimate",
    "estimate_time_complexity",
    "estimate_space_complexity",
    "estimate_repo_complexity",
]
