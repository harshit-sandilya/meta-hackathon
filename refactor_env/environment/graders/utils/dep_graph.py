"""
Cross-file dependency and call graph.

Built once at reset(), partially rebuilt on step() for changed files only.
All public methods are deterministic given the same input modules dict.
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ast_utils import (
    collect_definitions,
    collect_usages,
    _module_name_from_path,
    SymbolDef,
)


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

EDGE_CALLS = "CALLS"
EDGE_IMPORTS = "IMPORTS"
EDGE_INHERITS = "INHERITS"


@dataclass(frozen=True)
class Edge:
    src: str  # qualified source symbol
    dst: str  # qualified destination symbol
    kind: str  # EDGE_* constant
    file: str  # file where the edge originates
    line: int


@dataclass(frozen=True)
class DeadSymbol:
    name: str
    qualified_name: str
    kind: str  # "function" | "class" | "import" | "variable"
    file: str
    line: int
    reason: str  # "never_called" | "never_imported" | "unreachable_after_return"


# ---------------------------------------------------------------------------
# Main graph class
# ---------------------------------------------------------------------------


class DependencyGraph:
    """
    Directed multi-edge graph over qualified symbol names.

    Nodes: qualified_name strings.
    Edges: typed directed edges (see EDGE_* constants).
    """

    def __init__(self) -> None:
        # node_meta: qualified_name → SymbolDef
        self._nodes: dict[str, SymbolDef] = {}
        # adjacency: src → list[Edge]
        self._out_edges: dict[str, list[Edge]] = defaultdict(list)
        # reverse adjacency: dst → set[src] (for in-degree / callers lookup)
        self._in_edges: dict[str, set[str]] = defaultdict(set)
        # module_name → list of top-level symbol qualified_names it defines
        self._module_symbols: dict[str, list[str]] = defaultdict(list)
        # public_api: entry points that are always considered live
        self._public_api: set[str] = set()

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------

    def build(
        self,
        modules: dict[str, ast.Module],
        public_api: Optional[list[str]] = None,
    ) -> None:
        """
        Build the full graph from a {rel_path: ast.Module} dict.
        Clears any previously built state.
        """
        self._nodes.clear()
        self._out_edges.clear()
        self._in_edges.clear()
        self._module_symbols.clear()
        self._public_api = set(public_api or [])

        # Pass 1: collect all definitions (needed for import resolution)
        all_defs: dict[str, SymbolDef] = {}
        for rel_path, module in modules.items():
            defs = collect_definitions(module, rel_path)
            mod_name = _module_name_from_path(rel_path)
            for d in defs:
                all_defs[d.qualified_name] = d
                self._nodes[d.qualified_name] = d
                self._module_symbols[mod_name].append(d.qualified_name)

        # Pass 2: collect usages and build edges
        for rel_path, module in modules.items():
            self._build_edges_for_file(rel_path, module, modules, all_defs)

    def rebuild_partial(
        self,
        changed_files: list[str],
        modules: dict[str, ast.Module],
    ) -> None:
        """
        Rebuild only the nodes and edges originating from changed_files.
        Nodes from unchanged files are preserved. Used for incremental updates
        after an edit_file step — much faster than full rebuild for large repos.
        """
        for rel_path in changed_files:
            mod_name = _module_name_from_path(rel_path)

            # Remove stale nodes from this file
            stale_qnames = [
                qn for qn, sym in self._nodes.items() if sym.file == rel_path
            ]
            for qn in stale_qnames:
                del self._nodes[qn]
                self._out_edges.pop(qn, None)
                # Clean in-edges pointing to this node
                for srcs in self._in_edges.values():
                    srcs.discard(qn)
            self._module_symbols.pop(mod_name, None)

            # Remove stale out-edges originating from this file
            stale_edge_keys = [
                src
                for src, edges in self._out_edges.items()
                if any(e.file == rel_path for e in edges)
            ]
            for key in stale_edge_keys:
                self._out_edges[key] = [
                    e for e in self._out_edges[key] if e.file != rel_path
                ]

            # Re-add definitions from updated file
            module = modules.get(rel_path)
            if module is None:
                continue
            defs = collect_definitions(module, rel_path)
            for d in defs:
                self._nodes[d.qualified_name] = d
                self._module_symbols[mod_name].append(d.qualified_name)

        # Re-build edges only for changed files
        for rel_path in changed_files:
            module = modules.get(rel_path)
            if module:
                self._build_edges_for_file(rel_path, module, modules, self._nodes)

    def _build_edges_for_file(
        self,
        rel_path: str,
        module: ast.Module,
        modules: dict[str, ast.Module],
        all_defs: dict[str, SymbolDef],
    ) -> None:
        """Internal: build CALLS, IMPORTS, INHERITS edges for one file."""
        mod_name = _module_name_from_path(rel_path)
        usages = collect_usages(module, rel_path)

        # Build alias map: local_name → qualified_name (from imports in this file)
        alias_map: dict[str, str] = {}
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    alias_map[local] = alias.name
            elif isinstance(node, ast.ImportFrom):
                src_mod = node.module or ""
                # Resolve relative imports
                if node.level and node.level > 0:
                    parts = mod_name.split(".")
                    parent = ".".join(parts[: max(0, len(parts) - node.level)])
                    src_mod = f"{parent}.{src_mod}".strip(".")
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    alias_map[local] = f"{src_mod}.{alias.name}"

        # Emit IMPORTS edges (module-level)
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dst_mod = alias.name
                    _add_edge = Edge(
                        src=mod_name,
                        dst=dst_mod,
                        kind=EDGE_IMPORTS,
                        file=rel_path,
                        line=node.lineno,
                    )
                    self._out_edges[mod_name].append(_add_edge)
                    self._in_edges[dst_mod].add(mod_name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                _add_edge = Edge(
                    src=mod_name,
                    dst=node.module,
                    kind=EDGE_IMPORTS,
                    file=rel_path,
                    line=node.lineno,
                )
                self._out_edges[mod_name].append(_add_edge)
                self._in_edges[node.module].add(mod_name)

        # Emit CALLS edges using usage sites + alias resolution
        for ref in usages:
            if ref.kind not in ("call", "load"):
                continue
            # Try to resolve via alias map
            resolved = alias_map.get(ref.name, ref.qualified_name)
            # Find the caller: the function definition that contains this line
            caller_qname = (
                self._find_enclosing_function(module, rel_path, ref.line) or mod_name
            )
            edge = Edge(
                src=caller_qname,
                dst=resolved,
                kind=EDGE_CALLS if ref.kind == "call" else EDGE_IMPORTS,
                file=rel_path,
                line=ref.line,
            )
            self._out_edges[caller_qname].append(edge)
            self._in_edges[resolved].add(caller_qname)

        # Emit INHERITS edges
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                cls_qname = f"{mod_name}.{node.name}"
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = alias_map.get(base.id, base.id)
                    elif isinstance(base, ast.Attribute) and isinstance(
                        base.value, ast.Name
                    ):
                        base_name = f"{base.value.id}.{base.attr}"
                    if base_name:
                        edge = Edge(
                            src=cls_qname,
                            dst=base_name,
                            kind=EDGE_INHERITS,
                            file=rel_path,
                            line=node.lineno,
                        )
                        self._out_edges[cls_qname].append(edge)
                        self._in_edges[base_name].add(cls_qname)

    @staticmethod
    def _find_enclosing_function(
        module: ast.Module, rel_path: str, target_line: int
    ) -> Optional[str]:
        """Return qualified name of the innermost function containing target_line."""
        mod_name = _module_name_from_path(rel_path)
        best: Optional[tuple[int, str]] = None

        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start + len(node.body))
                if start <= target_line <= end:
                    depth = sum(
                        1
                        for p in ast.walk(module)
                        if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and p.lineno <= start
                        and getattr(p, "end_lineno", p.lineno) >= end
                        and p is not node
                    )
                    if best is None or depth > best[0]:
                        best = (depth, f"{mod_name}.{node.name}")

        return best[1] if best else None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_dead_symbols(self, exclude_tests: bool = True) -> list[DeadSymbol]:
        """
        Return all symbols unreachable from any public_api entry point or
        module-level entry (i.e., in-degree 0 after BFS from live set).

        Parameters
        ----------
        exclude_tests : bool
            If True, symbols defined in files matching test_*.py or *_test.py
            are never reported as dead.
        """
        # Seed the live set with public_api + module-level __main__ blocks
        live: set[str] = set(self._public_api)
        # Also treat any name defined in an __main__ guard as live
        for qn in self._nodes:
            if qn.endswith(".__main__") or ".__main__." in qn:
                live.add(qn)

        # BFS forward from live set
        queue: deque[str] = deque(live)
        visited: set[str] = set(live)
        while queue:
            node = queue.popleft()
            for edge in self._out_edges.get(node, []):
                if edge.dst not in visited:
                    visited.add(edge.dst)
                    queue.append(edge.dst)

        dead: list[DeadSymbol] = []
        for qn, sym in self._nodes.items():
            if qn in visited:
                continue
            if sym.kind == "import":
                # Unused import = in-degree 0 for its local name in this file
                pass
            if exclude_tests:
                fname = sym.file
                base = Path(fname).name
                if base.startswith("test_") or base.endswith("_test.py"):
                    continue
            # Determine reason
            if sym.kind == "import":
                reason = "never_imported"
            elif sym.kind in ("function", "async_function", "class"):
                reason = "never_called"
            else:
                reason = "never_used"

            dead.append(
                DeadSymbol(
                    name=sym.name,
                    qualified_name=qn,
                    kind=sym.kind,
                    file=sym.file,
                    line=sym.line,
                    reason=reason,
                )
            )

        return dead

    def get_import_cycles(self) -> list[list[str]]:
        """
        Return all import cycles using Tarjan's SCC algorithm.
        Only returns SCCs of size ≥ 2 (actual cycles).
        Considers only EDGE_IMPORTS edges.
        """
        # Collect module-level nodes only
        mod_nodes = list(
            {
                e.src
                for edges in self._out_edges.values()
                for e in edges
                if e.kind == EDGE_IMPORTS
            }
            | {
                e.dst
                for edges in self._out_edges.values()
                for e in edges
                if e.kind == EDGE_IMPORTS
            }
        )

        # Build adjacency for import edges only
        import_adj: dict[str, list[str]] = defaultdict(list)
        for src, edges in self._out_edges.items():
            for e in edges:
                if e.kind == EDGE_IMPORTS:
                    import_adj[src].append(e.dst)

        # Tarjan's SCC
        index_counter = [0]
        stack: list[str] = []
        lowlink: dict[str, int] = {}
        index: dict[str, int] = {}
        on_stack: dict[str, bool] = {}
        sccs: list[list[str]] = []

        def strongconnect(v: str) -> None:
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in import_adj.get(v, []):
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink.get(w, lowlink[v]))
                elif on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) >= 2:
                    sccs.append(scc)

        for node in mod_nodes:
            if node not in index:
                strongconnect(node)

        return sccs

    def get_callers(self, qualified_name: str) -> set[str]:
        """Return all symbols that directly call or import qualified_name."""
        return set(self._in_edges.get(qualified_name, set()))

    def get_transitive_dependencies(self, module_name: str) -> set[str]:
        """Return all modules transitively imported by module_name (DFS on import edges)."""
        visited: set[str] = set()
        stack = [module_name]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for e in self._out_edges.get(cur, []):
                if e.kind == EDGE_IMPORTS and e.dst not in visited:
                    stack.append(e.dst)
        visited.discard(module_name)
        return visited

    def get_module_fanout(self, module_path: str) -> int:
        """Return number of distinct modules that module_path imports from."""
        mod_name = _module_name_from_path(module_path)
        imported = {
            e.dst for e in self._out_edges.get(mod_name, []) if e.kind == EDGE_IMPORTS
        }
        return len(imported)

    def all_symbols(self) -> list[SymbolDef]:
        return list(self._nodes.values())
