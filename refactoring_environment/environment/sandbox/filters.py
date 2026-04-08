"""
Build a copytree ignore-callable from the repo's .gitignore + .git dir.
No hardcoded skip lists — we parse whatever .gitignore is present.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path


def _parse_gitignore(repo_root: Path) -> list[str]:
    """
    Return a list of gitignore patterns from repo_root/.gitignore.
    Strips comments, blank lines, and negation patterns (! prefix).
    """
    gitignore = repo_root / ".gitignore"
    if not gitignore.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        patterns.append(line.rstrip("/"))
    return patterns


def make_ignore_fn(repo_root: Path):
    """
    Returns a shutil.copytree-compatible ignore callable.
    Always skips .git regardless of .gitignore contents.
    """
    patterns = _parse_gitignore(repo_root)

    def _ignore(src: str, names: list[str]) -> list[str]:
        ignored: list[str] = []
        for name in names:
            if name == ".git":
                ignored.append(name)
                continue
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    ignored.append(name)
                    break
        return ignored

    return _ignore
