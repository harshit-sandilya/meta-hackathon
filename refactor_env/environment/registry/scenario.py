"""
scenario.py  —  Typed ScenarioSpec dataclass.

A ScenarioSpec is the immutable description of one task:
  - metadata (slug, name, difficulty, language, max_steps)
  - reward weights (acc, qual, eff, fmt)
  - penalty magnitudes
  - invariants (list of typed Invariant objects)
  - repo_path (relative path from project root to seed files)
  - task_files (populated by TaskRegistry after loading)

ScenarioSpec is constructed via ScenarioSpec.from_yaml(path).
It is frozen (immutable) once built — the environment never mutates it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .invariant import AnyInvariant, build_invariant


@dataclass(frozen=True)
class GraderConfig:
    """
    Maps each active reward component to a grader name.
    Components absent from this dict use the default grader for that type.
    """

    acc: str = "pytest_grader"
    qual: str = "ruff_grader"
    eff: str = "step_efficiency"
    fmt: str = "fmt_grader"  # only used if fmt weight > 0

    @classmethod
    def from_dict(cls, d: dict) -> "GraderConfig":
        return cls(
            acc=d.get("acc", "pytest_grader"),
            qual=d.get("qual", "ruff_grader"),
            eff=d.get("eff", "step_efficiency"),
            fmt=d.get("fmt", "fmt_grader"),
        )


@dataclass(frozen=True)
class LintConfig:
    """Ruff rule selection and baseline for the qual grader."""

    select: List[str] = field(default_factory=lambda: ["E", "F", "B"])
    ignore: List[str] = field(default_factory=list)
    baseline_violations: int = 0

    @classmethod
    def from_dict(cls, d: dict, baseline: int = 0) -> "LintConfig":
        return cls(
            select=d.get("select", ["E", "F", "B"]),
            ignore=d.get("ignore", []),
            baseline_violations=baseline,
        )


@dataclass(frozen=True)
class EffConfig:
    """Parameters for the step-efficiency decay formula."""

    decay_rate: float = 0.5  # eff = max(0, 1 - (steps/max_steps) * decay_rate)

    @classmethod
    def from_dict(cls, d: dict) -> "EffConfig":
        return cls(decay_rate=float(d.get("decay_rate", 0.5)))


@dataclass(frozen=True)
class RewardWeights:
    acc: float = 0.5
    qual: float = 0.3
    eff: float = 0.1
    fmt: float = 0.1

    def __post_init__(self) -> None:
        total = self.acc + self.qual + self.eff + self.fmt
        if not (0.999 < total < 1.001):
            raise ValueError(
                f"reward_weights must sum to 1.0, got {total:.4f} "
                f"(acc={self.acc}, qual={self.qual}, eff={self.eff}, fmt={self.fmt})"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "RewardWeights":
        return cls(
            acc=float(d.get("acc", 0.5)),
            qual=float(d.get("qual", 0.3)),
            eff=float(d.get("eff", 0.1)),
            fmt=float(d.get("fmt", 0.1)),
        )

    def active_components(self) -> List[str]:
        """Return names of components with weight > 0. Used for parallel grader dispatch."""
        return [name for name, w in self._as_dict().items() if w > 0.0]

    def _as_dict(self) -> Dict[str, float]:
        return {"acc": self.acc, "qual": self.qual, "eff": self.eff, "fmt": self.fmt}


@dataclass(frozen=True)
class PenaltyConfig:
    syntax_error: float = 0.30
    repeated_noop: float = 0.10
    broken_import: float = 0.20

    @classmethod
    def from_dict(cls, d: dict) -> "PenaltyConfig":
        return cls(
            syntax_error=float(d.get("syntax_error", 0.30)),
            repeated_noop=float(d.get("repeated_noop", 0.10)),
            broken_import=float(d.get("broken_import", 0.20)),
        )


@dataclass(frozen=True)
class ScenarioSpec:
    # ── Identity ──────────────────────────────────────────────────────────────
    slug: str
    name: str
    description: str
    difficulty: str  # "easy" | "medium" | "hard"
    language: str  # "python"
    max_steps: int
    style_guide: str  # "pep8" | "google" | etc.

    # ── Scoring ───────────────────────────────────────────────────────────────
    reward_weights: RewardWeights
    penalties: PenaltyConfig

    # ── Constraints ───────────────────────────────────────────────────────────
    invariants: List[AnyInvariant]  # ordered list, checked after every submit

    # ── Grader routing ────────────────────────────────────────────────────────
    graders: GraderConfig  # which grader handles each component
    lint_config: LintConfig  # ruff rules + baseline for qual grader
    test_paths: List[str]  # pytest discovery paths (for acc grader)
    eff_config: EffConfig  # efficiency decay params

    # ── File loading ──────────────────────────────────────────────────────────
    repo_path: str  # relative path from project root to seed repo
    task_files: Dict[str, str]  # {relative_path: content} — populated by registry

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def task_id(self) -> str:
        """Canonical task identifier used in RefactorState and observations."""
        return self.slug

    def check_invariants(
        self,
        sandbox_root: str,
        git_diff_output: str,
    ) -> List[str]:
        """
        Run all invariants and return aggregated violation messages.
        Empty list = all pass.

        Invariants are checked sequentially (not parallel) because they
        are cheap string/filesystem checks, not graders.
        """
        violations: List[str] = []
        for inv in self.invariants:
            violations.extend(inv.check(sandbox_root, git_diff_output))
        return violations

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(
        cls, yaml_path: str, project_root: Optional[str] = None
    ) -> "ScenarioSpec":
        """
        Load and validate a scenario.yaml file.

        Parameters
        ----------
        yaml_path:
            Absolute or relative path to the scenario.yaml.
        project_root:
            Root of the project (used to resolve repo_path for file loading).
            Defaults to the directory containing the yaml file's parent.
        """
        yaml_path = os.path.realpath(yaml_path)
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"scenario.yaml not found at: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"scenario.yaml at {yaml_path} did not parse to a dict.")

        # Resolve project root (default: two levels up from scenario.yaml)
        # tasks/lint_cleanup/scenario.yaml → project root is ../../
        if project_root is None:
            project_root = str(Path(yaml_path).parent.parent.parent)

        # Parse reward weights — validate they sum to 1.0
        rw = RewardWeights.from_dict(raw.get("reward_weights", {}))

        # Parse penalties
        pc = PenaltyConfig.from_dict(raw.get("penalties", {}))

        # Parse invariants
        invariants = [build_invariant(inv_raw) for inv_raw in raw.get("invariants", [])]

        graders = (GraderConfig.from_dict(raw.get("graders", {})),)
        lint_config = (
            LintConfig.from_dict(
                raw.get("lint_rules", {}),
                baseline=int(raw.get("baseline_violations", 0)),
            ),
        )
        test_paths = (raw.get("test_paths", ["tests/"]),)
        eff_config = (EffConfig.from_dict(raw.get("eff_config", {})),)

        # Resolve repo_path and load task files
        repo_path_raw = raw.get("repo_path", "")
        abs_repo = os.path.join(project_root, repo_path_raw)
        task_files = _load_task_files(abs_repo)

        return cls(
            slug=raw["slug"],
            name=raw["name"],
            description=raw.get("description", ""),
            difficulty=raw.get("difficulty", "medium"),
            language=raw.get("language", "python"),
            max_steps=int(raw.get("max_steps", 20)),
            style_guide=raw.get("style_guide", "pep8"),
            reward_weights=rw,
            penalties=pc,
            invariants=invariants,
            graders=graders,
            lint_config=lint_config,
            test_paths=test_paths,
            eff_config=eff_config,
            repo_path=repo_path_raw,
            task_files=task_files,
        )


# ── Private helpers ────────────────────────────────────────────────────────────


def _load_task_files(abs_repo_path: str) -> Dict[str, str]:
    """
    Recursively read all files under abs_repo_path.
    Returns {relative_path: content} mapping.

    Skips: .git/, __pycache__/, *.pyc, .DS_Store
    """
    _SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".ruff_cache"}
    _SKIP_EXTS = {".pyc", ".pyo"}

    if not os.path.isdir(abs_repo_path):
        raise FileNotFoundError(
            f"Task repo directory not found: '{abs_repo_path}'\n"
            f"Check repo_path in scenario.yaml points to an existing directory."
        )

    files: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(abs_repo_path):
        # Prune skipped dirs in-place so os.walk doesn't descend
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

        for fname in filenames:
            _, ext = os.path.splitext(fname)
            if ext in _SKIP_EXTS or fname == ".DS_Store":
                continue
            abs_file = os.path.join(dirpath, fname)
            rel_file = os.path.relpath(abs_file, abs_repo_path)
            try:
                content = Path(abs_file).read_text(encoding="utf-8", errors="replace")
            except OSError:
                content = ""
            files[rel_file] = content

    if not files:
        raise ValueError(f"Task repo at '{abs_repo_path}' contains no readable files.")

    return files
