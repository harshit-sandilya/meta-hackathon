"""
refactoring_environment/environment/sandbox/_files.py

FileHandler
-----------
Owns all filesystem operations inside the sandbox.
Every public method accepts the corresponding action Params directly
and returns an updated CodebaseContext (also cached as self._context).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from unidiff import PatchSet

from ...models_internal.actions import (
    EditFileParams,
    EditFilesParams,
    ListDirectoryParams,
    RunShellParams,
    SearchCodebaseParams,
    ViewFileParams,
)
from ...models_internal.observations import CodebaseContext
from ...models_internal.primitives import FilePatch, FileTreeEntry
from .runner import ShellExecutor

_CONTEXT_LINES = 100


class FileHandler:
    def __init__(self, root: Path, executor: ShellExecutor) -> None:
        """
        Build the initial CodebaseContext:
        - full file tree from git ls-files
        - active file = alphabetically first root-level file
        - content = first 100 lines of that file
        """
        self._root = root.resolve()
        self._executor = executor
        self._context = CodebaseContext()
        tree = self._build_tree()

        root_files = sorted(e.path for e in tree if not e.is_dir and "/" not in e.path)
        active = root_files[0] if root_files else None
        content, line_start, line_end, total = self._read_lines(active, None, None)

        self._context = CodebaseContext(
            file_tree=tree,
            active_file=active,
            file_content=content,
            file_line_start=line_start,
            file_line_end=line_end,
            total_file_lines=total,
        )

    # ── Action handlers ───────────────────────────────────────────────

    def view_file(self, params: ViewFileParams) -> CodebaseContext:
        content, line_start, line_end, total = self._read_lines(
            params.path, params.line_start, params.line_end
        )
        self._context = CodebaseContext(
            file_tree=self._context.file_tree,  # preserve existing tree
            active_file=params.path,
            file_content=content,
            file_line_start=line_start,
            file_line_end=line_end,
            total_file_lines=total,
        )
        return self._context

    def list_dir(self, params: ListDirectoryParams) -> CodebaseContext:
        entries = self._list_entries(params.path, params.recursive, params.max_depth)
        self._context = CodebaseContext(
            file_tree=entries,
            # preserve current active file view unchanged
            active_file=self._context.active_file,
            file_content=self._context.file_content,
            file_line_start=self._context.file_line_start,
            file_line_end=self._context.file_line_end,
            total_file_lines=self._context.total_file_lines,
        )
        return self._context

    def search(self, params: SearchCodebaseParams) -> CodebaseContext:
        output = self._grep(params)
        # search results surface in file_content; no single active file
        self._context = CodebaseContext(
            file_tree=self._context.file_tree,
            active_file=None,
            file_content=output or "(no results)",
            file_line_start=None,
            file_line_end=None,
            total_file_lines=None,
        )
        return self._context

    def apply_patch(self, params: EditFileParams) -> CodebaseContext:
        self._apply_single(params.patch)
        # re-read the patched file as the new active view
        return self.view_file(ViewFileParams(path=params.patch.path))

    def apply_patches(self, params: EditFilesParams) -> CodebaseContext:
        errors: list[str] = []
        last_ok: str | None = None

        for patch in params.patches:
            try:
                self._apply_single(patch)
                last_ok = patch.path
            except Exception as exc:
                errors.append(f"{patch.path}: {exc}")

        # re-read last successfully patched file
        if last_ok:
            return self.view_file(ViewFileParams(path=last_ok))
        return self._context

    # ── Property ──────────────────────────────────────────────────────

    @property
    def context(self) -> CodebaseContext:
        return self._context

    @property
    def root(self) -> Path:
        return self._root

    # ── Private: tree ─────────────────────────────────────────────────

    def _build_tree(self) -> list[FileTreeEntry]:
        result = self._executor.run(
            RunShellParams(command="git ls-files", timeout_sec=10, workdir=".")
        )
        files = (result.stdout or "").splitlines()
        entries: dict[str, FileTreeEntry] = {}

        for rel_path in files:
            # infer parent directory entries
            parts = Path(rel_path).parts
            for depth in range(1, len(parts)):
                dir_path = str(Path(*parts[:depth]))
                if dir_path not in entries:
                    entries[dir_path] = FileTreeEntry(
                        path=dir_path,
                        is_dir=True,
                        size_bytes=0,
                        last_modified=None,
                    )

            # file entry
            abs_path = self._root / rel_path
            try:
                stat = abs_path.stat()
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
            except OSError:
                size, mtime = 0, None

            entries[rel_path] = FileTreeEntry(
                path=rel_path,
                is_dir=False,
                size_bytes=size,
                last_modified=mtime,
            )

        return sorted(entries.values(), key=lambda e: (not e.is_dir, e.path))

    def _list_entries(
        self, path: str, recursive: bool, max_depth: int
    ) -> list[FileTreeEntry]:
        base = self._safe_path(path)
        entries: list[FileTreeEntry] = []
        self._walk(base, base, recursive, max_depth, 0, entries)
        return sorted(entries, key=lambda e: (not e.is_dir, e.path))

    def _walk(
        self,
        base: Path,
        current: Path,
        recursive: bool,
        max_depth: int,
        depth: int,
        out: list[FileTreeEntry],
    ) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(
                (c for c in current.iterdir() if c.name != ".git"),
                key=lambda p: (p.is_file(), p.name),
            )
        except PermissionError:
            return
        for child in children:
            rel = child.relative_to(self._root)
            if child.is_dir():
                out.append(FileTreeEntry(path=str(rel), is_dir=True))
                if recursive:
                    self._walk(base, child, recursive, max_depth, depth + 1, out)
            else:
                try:
                    stat = child.stat()
                    size = stat.st_size
                    mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                except OSError:
                    size, mtime = 0, None
                out.append(
                    FileTreeEntry(
                        path=str(rel),
                        is_dir=False,
                        size_bytes=size,
                        last_modified=mtime,
                    )
                )

    # ── Private: read ─────────────────────────────────────────────────

    def _read_lines(
        self,
        path: str | None,
        line_start: int | None,
        line_end: int | None,
    ) -> tuple[str | None, int | None, int | None, int | None]:
        if path is None:
            return None, None, None, None

        abs_path = self._safe_path(path)
        if not abs_path.is_file():
            raise FileNotFoundError(f"File not found in sandbox: {path}")

        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)

        # convert to 0-indexed slice; params are 1-indexed
        start_0 = (line_start - 1) if line_start else 0
        end_0 = min(line_end, total) if line_end else min(_CONTEXT_LINES, total)

        content = "\n".join(lines[start_0:end_0])
        # return 1-indexed back to caller
        return content, start_0 + 1, end_0, total

    # ── Private: search ───────────────────────────────────────────────

    def _grep(self, params: SearchCodebaseParams) -> str:
        flags = "-rn"
        if params.case_insensitive:
            flags += "i"

        cmd = (
            f"grep {flags} "
            f"--include='{params.file_glob}' "
            f"-C {params.context_lines} "
            f"-m {params.max_results} "
            f"'{params.query}' ."
        )
        result = self._executor.run(
            RunShellParams(command=cmd, timeout_sec=30, workdir=".")
        )
        return (result.stdout or "").strip()

    # ── Private: patch ────────────────────────────────────────────────

    def _apply_single(self, patch: FilePatch) -> None:
        target = self._safe_path(patch.path)

        if patch.new_content is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(patch.new_content, encoding="utf-8")
            return

        # unified_diff path
        if not target.is_file():
            raise FileNotFoundError(f"Cannot apply diff — file not found: {patch.path}")
        original = target.read_text(encoding="utf-8", errors="replace")
        patched = self._apply_unified_diff(original, patch.unified_diff)
        target.write_text(patched, encoding="utf-8")

    @staticmethod
    def _apply_unified_diff(original: str, diff: str) -> str:
        patch_set = PatchSet(diff)
        orig_lines = original.splitlines(keepends=True)
        result = list(orig_lines)
        offset = 0

        for patched_file in patch_set:
            for hunk in patched_file:
                start = hunk.source_start - 1 + offset
                old = [str(l.value) for l in hunk if not l.is_added]
                new = [str(l.value) for l in hunk if not l.is_removed]
                result[start : start + len(old)] = new
                offset += len(new) - len(old)

        return "".join(result)

    # ── Private: safety ───────────────────────────────────────────────

    def _safe_path(self, rel_path: str) -> Path:
        target = (self._root / rel_path).resolve()
        if not str(target).startswith(str(self._root.resolve())):
            raise PermissionError(f"Path '{rel_path}' escapes sandbox root.")
        return target

    def __repr__(self) -> str:
        return f"FileHandler(root={self._root}, active={self._context.active_file!r})"

    # ─── File listing ──────────────────────────────────────────────────────

    def read(self, path: str) -> str:
        """Read the content of a file."""
        abs_path = self._safe_path(path)
        return abs_path.read_text(encoding="utf-8", errors="replace")

    def list_python_files(self, exclude_patterns: list[str] | None = None) -> list[str]:
        """List all Python files in the repository, filtering by exclude patterns."""
        result = self._executor.run(
            RunShellParams(command="git ls-files", timeout_sec=10, workdir=".")
        )
        files = (result.stdout or "").splitlines()

        # Filter Python files
        py_files = [f for f in files if f.endswith(".py")]

        # Apply exclude patterns if provided
        if exclude_patterns:
            import fnmatch

            filtered = []
            for f in py_files:
                excluded = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(f, pattern):
                        excluded = True
                        break
                if not excluded:
                    filtered.append(f)
            py_files = filtered

        return py_files
