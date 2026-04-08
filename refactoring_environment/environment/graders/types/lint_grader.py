"""
refactoring_environment/environment/graders/lint_grader.py
"""

from __future__ import annotations

import json
import logging

from ....models_internal.actions import RunShellParams
from .base import BaseGrader, GradeResult

logger = logging.getLogger(__name__)

_RUFF_OK_EXIT_CODES = {0, 1}


class LintGrader(BaseGrader):
    grader_id = "lint"

    def _compute_metrics(self) -> dict:
        py_files = self._source_files()
        if not py_files:
            return {"count": 0}

        cmd = "ruff check --output-format json " + " ".join(f'"{f}"' for f in py_files)
        result = self.executor.run(
            RunShellParams(command=cmd, timeout_sec=30, workdir=".")
        )

        if result.timed_out:
            return {"count": 0, "tool_error": "ruff timed out"}

        if result.return_code not in _RUFF_OK_EXIT_CODES:
            stderr = result.stderr or ""
            logger.warning(
                "ruff exited with code %d — treating as 0 violations.\nstderr: %s",
                result.return_code,
                stderr[:400],
            )
            return {"count": 0, "tool_error": stderr.strip()}

        stdout = result.stdout or ""
        try:
            violations: list[dict] = json.loads(stdout) if stdout.strip() else []
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse ruff JSON output: %s", exc)
            return {"count": 0, "tool_error": str(exc)}

        return {"count": len(violations)}

    def grade(self) -> GradeResult:
        current = self._compute_metrics()

        baseline_count: int = self._baseline.get("count", 0)
        current_count: int = current.get("count", 0)

        score = self._delta_score(
            baseline=float(baseline_count),
            current=float(current_count),
        )
        added = max(0, current_count - baseline_count)

        feedbacks: list[str] = []
        errors: list[str] = []
        tool_errors: list[str] = []

        if "tool_error" in current:
            tool_errors.append(f"[lint] ruff failed: {current['tool_error']}")
        if "tool_error" in self._baseline:
            tool_errors.append(
                f"[lint] ruff failed at baseline: {self._baseline['tool_error']}"
            )

        if baseline_count == 0:
            feedbacks.append("Lint: baseline was already clean (0 violations).")
        else:
            arrow = (
                f"↓{baseline_count - current_count}"
                if current_count < baseline_count
                else (
                    f"↑{current_count - baseline_count}"
                    if current_count > baseline_count
                    else "→0"
                )
            )
            feedbacks.append(
                f"Lint: {baseline_count} → {current_count} violations "
                f"({arrow})  score={score:.2f}"
            )

        if added > 0:
            errors.append(
                f"[lint] regression: {added} new violation(s) introduced vs baseline."
            )

        return GradeResult(
            score=score,
            feedbacks=feedbacks,
            errors=errors,
            tool_errors=tool_errors,
            added_violations=added,
        )

    def _source_files(self) -> list[str]:
        root = self.file_handler.root
        return [
            str(root / entry.path)
            for entry in self.file_handler.context.file_tree
            if not entry.is_dir
            and entry.path.endswith(".py")
            and not _is_test_file(entry.path)
        ]


def _is_test_file(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    return (
        any(p.startswith("test") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
    )
