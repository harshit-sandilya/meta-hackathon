"""
environment/registry/__init__.py — Task registry for the refactoring environment.
Scans the tasks/ directory at package root and builds a flat { task_name: absolute_task_path } dict.
"""

from pathlib import Path

from .loader import load_task_registry
from .scenario import ScenarioSpec


class Registry:

    def __init__(self) -> None:
        self._registry: dict[str, str] | None = None

    def get(self, task_name: str) -> str:
        """
        Return absolute path for task_name.
        Raises KeyError with available task names if task_name is not found.
        """
        reg = self.registry
        if task_name not in reg:
            raise KeyError(
                f"Task '{task_name}' not in registry. " f"Available: {sorted(reg)}"
            )
        return reg[task_name]

    def load_scenario(self, task_name: str) -> ScenarioSpec:
        task_root = Path(self.get(task_name))
        scenario_path = task_root / "scenario.yaml"
        return ScenarioSpec.from_yaml(scenario_path)

    def repo_path(self, task_name: str) -> Path:
        return Path(self.get(task_name)) / "repo"

    def __contains__(self, task_name: str) -> bool:
        return task_name in self.registry

    def __repr__(self) -> str:
        tasks = sorted(self.registry)
        return f"RegistryHelper({len(tasks)} tasks: {tasks})"

    @property
    def registry(self) -> dict[str, str]:
        """
        Lazy-loaded registry dict.  Loaded once, cached for the lifetime
        of this instance.  Call load_task_registry() to force a re-scan.
        """
        if self._registry is None:
            self._registry = load_task_registry()
        return self._registry

    def get_all_task_names(self) -> list[str]:
        """
        Return a list of all available task names in the registry.
        """
        return list(self.registry.keys())
