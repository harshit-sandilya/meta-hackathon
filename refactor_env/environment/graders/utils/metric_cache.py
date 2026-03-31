"""
File-hash-keyed in-memory cache for expensive tool results.

Lives for exactly one episode — created at reset(), discarded when the
next reset() creates a new environment instance. Never writes to disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional


def _hash_file(path: Path) -> str:
    """Return MD5 hash of a file's contents. Returns '' on read error."""
    try:
        content = path.read_bytes()
        return hashlib.md5(content).hexdigest()
    except OSError:
        return ""


def _hash_directory(repo_path: Path, rel_files: Optional[list[str]] = None) -> str:
    """
    Return a combined hash representing the current state of a set of files.
    If rel_files is None, hashes all .py files in the repo.
    """
    if rel_files is not None:
        paths = sorted(repo_path / f for f in rel_files)
    else:
        paths = sorted(repo_path.rglob("*.py"))

    h = hashlib.md5()
    for p in paths:
        h.update(p.name.encode())
        h.update(_hash_file(p).encode())
    return h.hexdigest()


class MetricCache:
    """
    In-memory cache keyed by (tool_name, file_content_hash).

    Usage pattern:
        cache = MetricCache(repo_path)
        result = cache.get_or_compute("ruff", ["utils.py"], compute_fn)
        cache.invalidate("utils.py")   # called after edit_file
    """

    def __init__(self, repo_path: Path) -> None:
        self._repo_path = repo_path
        # (tool_name, content_hash) → Any
        self._store: dict[tuple[str, str], Any] = {}
        # rel_path → last known hash
        self._file_hashes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_compute(
        self,
        tool_name: str,
        rel_files: Optional[list[str]],
        compute_fn: Callable[[], Any],
    ) -> Any:
        """
        Return cached result if the relevant files haven't changed.
        Otherwise call compute_fn(), cache, and return the result.

        Parameters
        ----------
        tool_name  : Logical name of the tool (e.g. "ruff", "radon_cc").
        rel_files  : List of relative file paths this computation depends on.
                     If None, the cache key covers the entire repo.
        compute_fn : Zero-argument callable that computes and returns the result.
        """
        content_hash = _hash_directory(self._repo_path, rel_files)
        cache_key = (tool_name, content_hash)

        if cache_key in self._store:
            return self._store[cache_key]

        result = compute_fn()
        self._store[cache_key] = result
        # Update stored file hashes for invalidation tracking
        if rel_files:
            for f in rel_files:
                self._file_hashes[f] = _hash_file(self._repo_path / f)
        return result

    def invalidate(self, rel_path: str) -> None:
        """
        Invalidate all cache entries that involved rel_path.

        Called by the environment after every edit_file action on a file.
        Entries are evicted lazily: we mark the file as dirty so the next
        get_or_compute sees a new hash and recomputes.
        """
        # Evict any entry whose hash would now be stale
        # Since the key includes the content hash, a changed file will
        # naturally produce a different hash and miss the cache.
        # We only need to clear any entry that was keyed to the old hash.
        old_hash = self._file_hashes.get(rel_path, "")
        if old_hash:
            stale_keys = [k for k in self._store if old_hash in k[1]]
            for k in stale_keys:
                del self._store[k]
            del self._file_hashes[rel_path]

    def invalidate_many(self, rel_paths: list[str]) -> None:
        """Invalidate cache for multiple files at once."""
        for path in rel_paths:
            self.invalidate(path)

    def clear(self) -> None:
        """Clear the entire cache. Called at the start of reset()."""
        self._store.clear()
        self._file_hashes.clear()

    def stats(self) -> dict[str, int]:
        """Return cache statistics for debugging/logging."""
        return {
            "entries": len(self._store),
            "tracked_files": len(self._file_hashes),
        }

    def warm(self, tool_name: str, rel_files: Optional[list[str]], result: Any) -> None:
        """
        Pre-populate the cache with a known result.

        Used by the environment at reset() to store baseline metrics
        immediately after computing them — so the first step doesn't
        re-run all tools unnecessarily.
        """
        content_hash = _hash_directory(self._repo_path, rel_files)
        self._store[(tool_name, content_hash)] = result
        if rel_files:
            for f in rel_files:
                self._file_hashes[f] = _hash_file(self._repo_path / f)
