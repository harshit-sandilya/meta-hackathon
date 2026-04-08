from pathlib import Path

# Resolve tasks/ relative to this file:
#   environment/registry/loader.py
#   └─ parent  → environment/registry/
#   └─ parent  → environment/
#   └─ parent  → refactoring_environment/
#   └─ / tasks → refactoring_environment/tasks/
_TASKS_ROOT = Path(__file__).parent.parent.parent / "tasks"


def load_task_registry() -> dict[str, str]:
    """Scan tasks/ and return {folder_name: absolute_path}.

    Raises FileNotFoundError if the tasks/ directory does not exist.
    Re-scans on every explicit call (use .registry for cached access).
    """
    if not _TASKS_ROOT.exists():
        raise FileNotFoundError(
            f"Tasks directory not found at: {_TASKS_ROOT}. "
            "Create it or check your project layout."
        )

    registry = {
        entry.name: str(entry.resolve())
        for entry in sorted(_TASKS_ROOT.iterdir())
        if entry.is_dir()
    }
    return registry
