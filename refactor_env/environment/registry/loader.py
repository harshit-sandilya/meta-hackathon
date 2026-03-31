"""
loader.py  —  TaskRegistry: discovers, loads, and caches ScenarioSpec objects.

Usage
-----
    registry = TaskRegistry.from_tasks_dir("tasks/")
    spec = registry.get("lint-cleanup")
    task_files = spec.task_files   # ready to pass to sandbox.init()

Design decisions
----------------
  - TaskRegistry is a plain class (not a true singleton) so tests can
    construct isolated instances with temporary directories.
  - Scenarios are loaded lazily on first get() and then cached — avoids
    paying the YAML parse + file I/O cost for unused tasks at startup.
  - Parallel loading (ThreadPoolExecutor) is available via load_all()
    for cases where startup latency matters (e.g. inference.py pre-warm).
  - The registry holds ScenarioSpec objects which are frozen dataclasses,
    so sharing them across threads is always safe.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .scenario import ScenarioSpec


_SCENARIO_FILENAME = "scenario.yaml"


class TaskRegistry:
    """
    Discovers task directories under a root folder and provides
    ScenarioSpec objects by slug.

    Directory convention
    --------------------
        tasks/
        ├── lint_cleanup/
        │   ├── scenario.yaml
        │   └── repo/
        │       ├── utils.py
        │       └── tests/
        │           └── test_utils.py
        └── api_rename/
            ├── scenario.yaml
            └── repo/
                └── ...

    The registry scans for any subdirectory containing a scenario.yaml.
    The slug in scenario.yaml is the canonical key — NOT the directory name.
    """

    def __init__(
        self,
        tasks_root: str,
        project_root: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        tasks_root:
            Path to the directory containing task subdirectories.
        project_root:
            Project root passed to ScenarioSpec.from_yaml().
            Defaults to the parent of tasks_root.
        """
        self._tasks_root = os.path.realpath(tasks_root)
        self._project_root = project_root or str(Path(self._tasks_root).parent)
        self._cache: Dict[str, ScenarioSpec] = {}  # slug → spec
        self._yaml_index: Dict[str, str] = {}  # slug → yaml_path (lazy discovery)
        self._discovered: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def from_tasks_dir(
        cls,
        tasks_root: str,
        project_root: Optional[str] = None,
    ) -> "TaskRegistry":
        """Convenience constructor — equivalent to TaskRegistry(...) but more readable."""
        return cls(tasks_root=tasks_root, project_root=project_root)

    def get(self, slug: str) -> ScenarioSpec:
        """
        Return the ScenarioSpec for *slug*.

        Loads and caches on first call.
        Raises KeyError if the slug is not found in the tasks directory.
        """
        if slug in self._cache:
            return self._cache[slug]

        self._ensure_discovered()

        if slug not in self._yaml_index:
            available = sorted(self._yaml_index.keys())
            raise KeyError(
                f"Task '{slug}' not found in registry.\n"
                f"Available tasks: {available}"
            )

        spec = self._load_one(self._yaml_index[slug])
        self._cache[slug] = spec
        return spec

    def load_all(self, max_workers: int = 4) -> Dict[str, ScenarioSpec]:
        """
        Load ALL discovered tasks in parallel and return {slug: spec} mapping.

        Used by inference.py to pre-warm the registry, and by the test suite
        to validate all scenarios parse correctly.
        """
        self._ensure_discovered()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._load_one, yaml_path): slug
                for slug, yaml_path in self._yaml_index.items()
                if slug not in self._cache
            }
            for future in as_completed(futures):
                slug = futures[future]
                try:
                    spec = future.result()
                    self._cache[slug] = spec
                except Exception as exc:
                    # Surface the failing task path for easier debugging
                    yaml_path = self._yaml_index[slug]
                    raise RuntimeError(
                        f"Failed to load task '{slug}' from '{yaml_path}': {exc}"
                    ) from exc

        return dict(self._cache)

    def list_slugs(self) -> List[str]:
        """Return sorted list of all discovered task slugs."""
        self._ensure_discovered()
        return sorted(self._yaml_index.keys())

    def list_specs(self) -> List[ScenarioSpec]:
        """Return all loaded ScenarioSpec objects (triggers full load)."""
        self.load_all()
        return list(self._cache.values())

    def __iter__(self) -> Iterator[ScenarioSpec]:
        """Iterate over all specs (triggers full load if not already done)."""
        self.load_all()
        return iter(self._cache.values())

    def __len__(self) -> int:
        self._ensure_discovered()
        return len(self._yaml_index)

    def __repr__(self) -> str:
        self._ensure_discovered()
        return (
            f"TaskRegistry(tasks_root='{self._tasks_root}', "
            f"tasks={self.list_slugs()}, "
            f"loaded={sorted(self._cache.keys())})"
        )

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _ensure_discovered(self) -> None:
        """Scan tasks_root for scenario.yaml files if not done yet."""
        if self._discovered:
            return
        self._yaml_index = self._discover(self._tasks_root)
        self._discovered = True

    @staticmethod
    def _discover(tasks_root: str) -> Dict[str, str]:
        """
        Walk tasks_root one level deep to find scenario.yaml files.
        Returns {slug: absolute_yaml_path}.

        We only go one level deep (task_dir/scenario.yaml) — no nesting.
        """
        index: Dict[str, str] = {}

        if not os.path.isdir(tasks_root):
            raise FileNotFoundError(f"Tasks root directory not found: '{tasks_root}'")

        for entry in os.scandir(tasks_root):
            if not entry.is_dir():
                continue
            yaml_path = os.path.join(entry.path, _SCENARIO_FILENAME)
            if not os.path.isfile(yaml_path):
                continue  # skip dirs without scenario.yaml

            # Peek at YAML to extract slug without full parse
            try:
                import yaml

                with open(yaml_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                slug = raw.get("slug")
                if not slug:
                    raise ValueError(f"Missing 'slug' field in {yaml_path}")
                if slug in index:
                    raise ValueError(
                        f"Duplicate slug '{slug}' found in:\n"
                        f"  {index[slug]}\n  {yaml_path}"
                    )
                index[slug] = yaml_path
            except Exception as exc:
                # Log warning but don't crash the whole registry for one bad task
                import warnings

                warnings.warn(
                    f"Skipping task at '{yaml_path}': {exc}",
                    stacklevel=2,
                )

        return index

    def _load_one(self, yaml_path: str) -> ScenarioSpec:
        """Load a single ScenarioSpec from its yaml_path."""
        return ScenarioSpec.from_yaml(
            yaml_path=yaml_path,
            project_root=self._project_root,
        )
