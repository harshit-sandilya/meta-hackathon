"""
invariant.py  —  Typed invariant checkers for refactor-env.

Each invariant in scenario.yaml becomes an Invariant dataclass instance.
check(sandbox_root) returns a list of violation strings (empty = pass).
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import List, Literal, Union


InvariantType = Literal["no_edit", "file_exists", "no_delete", "symbol_present"]


@dataclass(frozen=True)
class NoEditInvariant:
    """Ensures given paths were not modified relative to the baseline."""

    type: Literal["no_edit"]
    paths: List[str]
    message: str

    def check(self, sandbox_root: str, git_diff_output: str) -> List[str]:
        violations = []
        for path_prefix in self.paths:
            # Any line in git diff touching this path prefix = violation
            for line in git_diff_output.splitlines():
                if (
                    line.startswith(("---", "+++", "diff"))
                    and path_prefix.rstrip("/") in line
                ):
                    violations.append(self.message)
                    break
        return violations


@dataclass(frozen=True)
class FileExistsInvariant:
    """Ensures given files were not deleted or renamed."""

    type: Literal["file_exists"]
    paths: List[str]
    message: str

    def check(self, sandbox_root: str, git_diff_output: str) -> List[str]:
        violations = []
        for rel_path in self.paths:
            abs_path = os.path.join(sandbox_root, rel_path)
            if not os.path.isfile(abs_path):
                violations.append(self.message)
        return violations


@dataclass(frozen=True)
class NoDeleteInvariant:
    """Alias for file_exists — kept separate for semantic clarity."""

    type: Literal["no_delete"]
    paths: List[str]
    message: str

    def check(self, sandbox_root: str, git_diff_output: str) -> List[str]:
        violations = []
        for rel_path in self.paths:
            abs_path = os.path.join(sandbox_root, rel_path)
            if not os.path.exists(abs_path):
                violations.append(self.message)
        return violations


@dataclass(frozen=True)
class SymbolPresentInvariant:
    """Ensures a named Python symbol still exists in a file."""

    type: Literal["symbol_present"]
    paths: List[str]  # [file_path, symbol_name]
    message: str

    def check(self, sandbox_root: str, git_diff_output: str) -> List[str]:
        if len(self.paths) < 2:
            return [f"Malformed symbol_present invariant: {self.paths}"]
        file_path, symbol_name = self.paths[0], self.paths[1]
        abs_path = os.path.join(sandbox_root, file_path)
        if not os.path.isfile(abs_path):
            return [self.message]
        try:
            tree = ast.parse(open(abs_path).read())
        except SyntaxError:
            return [self.message]
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        if symbol_name not in names:
            return [self.message]
        return []


AnyInvariant = Union[
    NoEditInvariant, FileExistsInvariant, NoDeleteInvariant, SymbolPresentInvariant
]


def build_invariant(raw: dict) -> AnyInvariant:
    """
    Factory: parse a raw YAML invariant dict into the correct typed dataclass.
    Raises ValueError for unknown types.
    """
    t = raw.get("type")
    paths = raw.get("paths", [])
    message = raw.get("message", f"Invariant '{t}' violated.")

    if t == "no_edit":
        return NoEditInvariant(type="no_edit", paths=paths, message=message)
    elif t == "file_exists":
        return FileExistsInvariant(type="file_exists", paths=paths, message=message)
    elif t == "no_delete":
        return NoDeleteInvariant(type="no_delete", paths=paths, message=message)
    elif t == "symbol_present":
        return SymbolPresentInvariant(
            type="symbol_present", paths=paths, message=message
        )
    else:
        raise ValueError(f"Unknown invariant type: '{t}'")
