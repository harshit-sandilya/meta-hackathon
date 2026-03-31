"""
Duplication Grader — code clone detection and DRY principle enforcement.

Gold standard : zero duplicated code blocks across all non-test .py files.

Clone taxonomy (Type I → Type IV, industry standard)
─────────────────────────────────────────────────────
  Type I   — Exact copies (whitespace/comment differences only)
  Type II  — Syntactically identical after identifier renaming
  Type III — Semantically similar with added/removed statements
  Type IV  — Semantically equivalent but structurally different
             (not detected — requires semantic analysis beyond AST)

This grader implements Types I, II, and a lightweight III detector.
Each clone pair is scored by severity:
  exact     (Type I)  → severity 1.0
  renamed   (Type II) → severity 0.8
  similar   (Type III)→ severity 0.5

Score formula
─────────────
  total_severity   = Σ severity_i for all clone pairs
  baseline_severity = total_severity at episode reset
  score = clamp(1 - total_severity / max(baseline_severity, ε))

  Improvement is delta-proportional so partial deduplication is rewarded.

Sub-signals (for diagnostics in sub_scores)
────────────────────────────────────────────
  exact_clones    — Type I count
  renamed_clones  — Type II count
  similar_clones  — Type III count
  clone_coverage  — fraction of logical LOC that is duplicated

Scenario config keys (under config["graders"]["duplication"]):
  weight              : float       — contribution to overall reward (default 0.10)
  min_lines           : int         — minimum block size to consider (default 6)
  min_tokens          : int         — minimum token count per block  (default 50)
  similarity_threshold: float       — Jaccard threshold for Type III (default 0.75)
  max_file_pairs      : int         — cap file-pair comparisons for speed (default 200)
  exclude_patterns    : list[str]   — file globs to skip
  ignore_types        : list[str]   — clone types to skip ["exact","renamed","similar"]
  severity_weights    : dict        — override per-type severity values
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base_grader import BaseGrader, GradeResult, _clamp, already_gold, empty_grade
from .utils import MetricCache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MIN_LINES = 6
_DEFAULT_MIN_TOKENS = 50
_DEFAULT_SIMILARITY_THRESH = 0.75
_DEFAULT_MAX_FILE_PAIRS = 200

_DEFAULT_SEVERITY: dict[str, float] = {
    "exact": 1.0,
    "renamed": 0.8,
    "similar": 0.5,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeBlock:
    """A contiguous block of AST statements extracted from a file."""

    file: str
    start_line: int
    end_line: int
    node_types: tuple[str, ...]  # AST node type sequence (structure fingerprint)
    token_seq: tuple[str, ...]  # normalised token sequence
    raw_hash: str  # hash of exact text (Type I)
    norm_hash: str  # hash of name-normalised text (Type II)
    loc: int  # logical lines


@dataclass
class ClonePair:
    block_a: CodeBlock
    block_b: CodeBlock
    clone_type: str  # "exact" | "renamed" | "similar"
    severity: float
    similarity: float  # Jaccard score (1.0 for exact/renamed)

    def to_dict(self) -> dict:
        return {
            "clone_type": self.clone_type,
            "severity": self.severity,
            "similarity": round(self.similarity, 3),
            "a": {
                "file": self.block_a.file,
                "start_line": self.block_a.start_line,
                "end_line": self.block_a.end_line,
                "loc": self.block_a.loc,
            },
            "b": {
                "file": self.block_b.file,
                "start_line": self.block_b.start_line,
                "end_line": self.block_b.end_line,
                "loc": self.block_b.loc,
            },
        }


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# Token types that carry semantic meaning (strip whitespace, comments, strings)
_STRIP_RE = re.compile(r"#.*")
_IDENTIFIER_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_STRING_RE = re.compile(r"(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|\".*?\"|\'.*?\')", re.DOTALL)
_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?\b")

# Python keywords — kept verbatim during normalisation
_KEYWORDS = frozenset(
    {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }
)


def _normalise_source(source: str) -> str:
    """
    Type II normalisation: replace all non-keyword identifiers with a
    canonical placeholder, collapse strings and numbers.
    """
    # Strip comments
    source = _STRIP_RE.sub("", source)
    # Collapse strings
    source = _STRING_RE.sub("__STR__", source)
    # Collapse numbers
    source = _NUMBER_RE.sub("__NUM__", source)

    def _replace_id(m: re.Match) -> str:
        word = m.group(0)
        return word if word in _KEYWORDS else "__ID__"

    source = _IDENTIFIER_RE.sub(_replace_id, source)
    # Collapse whitespace
    return re.sub(r"\s+", " ", source).strip()


def _token_set(normalised: str) -> frozenset[str]:
    """Bag-of-tokens for Jaccard similarity (Type III)."""
    return frozenset(normalised.split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _ast_node_types(stmts: list[ast.stmt]) -> tuple[str, ...]:
    """
    Structural fingerprint: sequence of top-level AST node type names.
    Used for quick pre-filter before full comparison.
    """
    return tuple(type(s).__name__ for s in stmts)


def _block_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------


def _extract_blocks(
    rel_path: str,
    source: str,
    tree: ast.Module,
    min_lines: int,
    min_tokens: int,
) -> list[CodeBlock]:
    """
    Extract sliding-window blocks of statements from a module.

    Blocks are extracted at function-body and class-body granularity
    (not just top-level) to catch intra-function duplication.

    Window size is adaptive: starts at min_lines and grows until the
    block exceeds 3× min_lines, producing overlapping sub-blocks.
    This is a pragmatic O(n²) approach suited to the task scale.
    """
    lines = source.splitlines()
    blocks: list[CodeBlock] = []

    def _process_stmts(stmts: list[ast.stmt]) -> None:
        n = len(stmts)
        if n < 2:
            return

        # Sliding windows of size [min_lines .. min(n, 3*min_lines)]
        max_window = min(n, 3 * min_lines)
        for window in range(min_lines, max_window + 1):
            for i in range(n - window + 1):
                chunk = stmts[i : i + window]

                start = chunk[0].lineno
                end = (
                    chunk[-1].end_lineno
                    if hasattr(chunk[-1], "end_lineno")
                    else chunk[-1].lineno + 1
                )

                if end - start + 1 < min_lines:
                    continue

                # Extract raw text
                raw_lines = lines[start - 1 : end]
                raw_text = "\n".join(raw_lines)

                # Normalise
                norm_text = _normalise_source(raw_text)
                tokens = norm_text.split()

                if len(tokens) < min_tokens:
                    continue

                node_types = _ast_node_types(chunk)
                raw_hash = _block_hash(
                    re.sub(r"\s+", " ", _STRIP_RE.sub("", raw_text)).strip()
                )
                norm_hash = _block_hash(norm_text)
                token_seq = tuple(tokens)
                loc = sum(
                    1 for l in raw_lines if l.strip() and not l.strip().startswith("#")
                )

                blocks.append(
                    CodeBlock(
                        file=rel_path,
                        start_line=start,
                        end_line=end,
                        node_types=node_types,
                        token_seq=token_seq,
                        raw_hash=raw_hash,
                        norm_hash=norm_hash,
                        loc=loc,
                    )
                )

        # Recurse into function/class bodies
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _process_stmts(stmt.body)
            elif isinstance(stmt, ast.ClassDef):
                _process_stmts(stmt.body)

    _process_stmts(tree.body)
    return blocks


# ---------------------------------------------------------------------------
# Clone detection
# ---------------------------------------------------------------------------


def _detect_clones(
    blocks_by_file: dict[str, list[CodeBlock]],
    max_file_pairs: int,
    similarity_threshold: float,
    ignore_types: set[str],
    severity_weights: dict[str, float],
) -> list[ClonePair]:
    """
    Compare blocks across all file pairs and within each file.

    Strategy:
      1. Build hash indexes for Type I (raw_hash) and Type II (norm_hash)
      2. Hash collisions → confirmed clone pairs
      3. For remaining structurally similar blocks (same node_type prefix)
         compute Jaccard → Type III if above threshold

    Deduplication: each (file_a, start_a, file_b, start_b) pair is
    reported at most once, preferring the highest-severity type.
    """
    # Flatten all blocks with file tags
    all_blocks: list[CodeBlock] = [
        b for blocks in blocks_by_file.values() for b in blocks
    ]

    # ── Index by hash ──────────────────────────────────────────────────
    raw_index: dict[str, list[CodeBlock]] = defaultdict(list)
    norm_index: dict[str, list[CodeBlock]] = defaultdict(list)

    for block in all_blocks:
        raw_index[block.raw_hash].append(block)
        norm_index[block.norm_hash].append(block)

    reported: set[tuple] = set()  # (file_a, start_a, file_b, start_b)
    pairs: list[ClonePair] = []

    def _key(a: CodeBlock, b: CodeBlock) -> tuple:
        if (a.file, a.start_line) <= (b.file, b.start_line):
            return (a.file, a.start_line, b.file, b.start_line)
        return (b.file, b.start_line, a.file, a.start_line)

    def _add(a: CodeBlock, b: CodeBlock, ctype: str, sim: float) -> None:
        k = _key(a, b)
        if k in reported:
            return
        # Don't report overlapping blocks in the same file as clones
        if a.file == b.file and (
            a.start_line <= b.start_line <= a.end_line
            or b.start_line <= a.start_line <= b.end_line
        ):
            return
        reported.add(k)
        sev = severity_weights.get(ctype, _DEFAULT_SEVERITY.get(ctype, 0.5))
        pairs.append(
            ClonePair(
                block_a=a,
                block_b=b,
                clone_type=ctype,
                severity=sev,
                similarity=sim,
            )
        )

    # ── Type I — exact clones ──────────────────────────────────────────
    if "exact" not in ignore_types:
        for group in raw_index.values():
            if len(group) < 2:
                continue
            for a, b in itertools.combinations(group, 2):
                _add(a, b, "exact", 1.0)

    # ── Type II — renamed clones ───────────────────────────────────────
    if "renamed" not in ignore_types:
        for group in norm_index.values():
            if len(group) < 2:
                continue
            for a, b in itertools.combinations(group, 2):
                k = _key(a, b)
                if k not in reported:
                    _add(a, b, "renamed", 1.0)

    # ── Type III — similar clones (Jaccard) ───────────────────────────
    if "similar" not in ignore_types:
        # Pre-filter: only compare blocks with same leading node type
        # to avoid O(n²) full comparison
        by_first_node: dict[str, list[CodeBlock]] = defaultdict(list)
        for block in all_blocks:
            if block.node_types:
                by_first_node[block.node_types[0]].append(block)

        comparisons = 0
        for group in by_first_node.values():
            if len(group) < 2:
                continue
            # Cap comparisons per group
            sample = group[:50]
            for a, b in itertools.combinations(sample, 2):
                k = _key(a, b)
                if k in reported:
                    continue
                if comparisons >= max_file_pairs * 10:
                    break
                comparisons += 1

                # Token set Jaccard
                tok_a = frozenset(a.token_seq)
                tok_b = frozenset(b.token_seq)
                sim = _jaccard(tok_a, tok_b)

                if sim >= similarity_threshold:
                    _add(a, b, "similar", sim)

    return pairs


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _clone_coverage(
    pairs: list[ClonePair],
    total_loc: int,
) -> float:
    """
    Fraction of logical LOC that appears in at least one clone pair.
    Uses a union of line ranges to avoid double-counting.
    """
    if total_loc <= 0:
        return 0.0

    covered: dict[str, set[int]] = defaultdict(set)
    for pair in pairs:
        for b in (pair.block_a, pair.block_b):
            covered[b.file].update(range(b.start_line, b.end_line + 1))

    total_covered = sum(len(lines) for lines in covered.values())
    return min(1.0, total_covered / total_loc)


def _total_severity(pairs: list[ClonePair]) -> float:
    return sum(p.severity for p in pairs)


def _counts_by_type(pairs: list[ClonePair]) -> dict[str, int]:
    counts: dict[str, int] = {"exact": 0, "renamed": 0, "similar": 0}
    for p in pairs:
        counts[p.clone_type] = counts.get(p.clone_type, 0) + 1
    return counts


def _is_test_file(rel_path: str) -> bool:
    from pathlib import PurePosixPath

    base = PurePosixPath(rel_path).name
    parts = PurePosixPath(rel_path).parts
    return (
        base.startswith("test_")
        or base.endswith("_test.py")
        or "tests" in parts
        or "test" in parts
    )


# ---------------------------------------------------------------------------
# Feedback helpers
# ---------------------------------------------------------------------------


def _build_feedback(
    score: float,
    baseline_score: float,
    is_regression: bool,
    pairs: list[ClonePair],
    counts: dict[str, int],
    coverage: float,
    total_severity: float,
) -> str:
    if score >= 1.0:
        return "[Duplication] Gold standard — zero clone pairs detected."

    if is_regression:
        new_exact = counts.get("exact", 0)
        new_renamed = counts.get("renamed", 0)
        return (
            f"[Duplication] Regression: severity increased "
            f"(baseline score {baseline_score:.0%} → {score:.0%}). "
            f"New clones: {new_exact} exact, {new_renamed} renamed."
        )

    parts: list[str] = []
    if counts.get("exact", 0):
        parts.append(f"{counts['exact']} exact")
    if counts.get("renamed", 0):
        parts.append(f"{counts['renamed']} renamed")
    if counts.get("similar", 0):
        parts.append(f"{counts['similar']} similar")

    clone_str = ", ".join(parts) if parts else "none"

    # Top actionable pair: highest severity
    if pairs:
        top = max(pairs, key=lambda p: p.severity)
        hot = (
            f"Worst: {top.block_a.file}:{top.block_a.start_line}–{top.block_a.end_line} "
            f"↔ {top.block_b.file}:{top.block_b.start_line}–{top.block_b.end_line} "
            f"({top.clone_type}, {top.block_a.loc} LOC)"
        )
    else:
        hot = ""

    return (
        f"[Duplication] {score:.0%} DRY score "
        f"({clone_str} clone pairs, {coverage:.0%} LOC duplicated). "
        f"{hot}"
    )


# ---------------------------------------------------------------------------
# DuplicationGrader
# ---------------------------------------------------------------------------


class DuplicationGrader(BaseGrader):
    """
    Scores the agent on eliminating duplicated code blocks (DRY principle).

    Detection pipeline:
      1. Parse all non-test .py files into AST.
      2. Extract overlapping statement-window blocks (adaptive size).
      3. Hash-index for Type I (exact) and Type II (renamed) clones.
      4. Jaccard similarity for Type III (similar) clones, pre-filtered
         by structural fingerprint to stay O(n·k) not O(n²).

    Scoring:
      score = clamp(1 - total_severity / baseline_severity)
      where severity = Σ per-pair severity weights.

    The score is delta-based: any reduction in total severity earns
    proportional reward. Reaching zero clones = gold standard.
    """

    grader_id = "duplication"

    # ------------------------------------------------------------------
    # compute_metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        repo_path: object,
        config: dict[str, Any],
        cache: object,
    ) -> dict[str, Any]:
        """
        Run clone detection across all non-test .py files.

        Returns
        -------
        {
          "clone_pairs"      : list[dict]        — serialised ClonePair list
          "counts_by_type"   : dict[str, int]    — {type: count}
          "total_severity"   : float
          "clone_coverage"   : float             — fraction of LOC duplicated
          "total_loc"        : int               — total logical LOC scanned
          "files_checked"    : int
          "error"            : str|None
        }
        """
        repo_path = Path(repo_path)
        dup_cfg = config.get("graders", {}).get("duplication", {})

        min_lines = int(dup_cfg.get("min_lines", _DEFAULT_MIN_LINES))
        min_tokens = int(dup_cfg.get("min_tokens", _DEFAULT_MIN_TOKENS))
        similarity_threshold = float(
            dup_cfg.get("similarity_threshold", _DEFAULT_SIMILARITY_THRESH)
        )
        max_file_pairs = int(dup_cfg.get("max_file_pairs", _DEFAULT_MAX_FILE_PAIRS))
        ignore_types: set[str] = set(dup_cfg.get("ignore_types", []))
        severity_weights: dict[str, float] = {
            **_DEFAULT_SEVERITY,
            **dup_cfg.get("severity_weights", {}),
        }
        exclude_patterns: list[str] = dup_cfg.get("exclude_patterns", []) + config.get(
            "exclude_patterns", []
        )

        def _run() -> tuple[list[ClonePair], int, int]:
            blocks_by_file: dict[str, list[CodeBlock]] = {}
            total_loc = 0
            files_checked = 0

            for fpath in sorted(repo_path.rglob("*.py")):
                rel_path = str(fpath.relative_to(repo_path))
                if _is_test_file(rel_path):
                    continue
                if any(
                    fpath.match(p) or rel_path.startswith(p.rstrip("/"))
                    for p in exclude_patterns
                ):
                    continue
                try:
                    source = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                try:
                    tree = ast.parse(source, filename=rel_path)
                except SyntaxError:
                    continue

                blocks = _extract_blocks(rel_path, source, tree, min_lines, min_tokens)
                if blocks:
                    blocks_by_file[rel_path] = blocks

                # Logical LOC
                total_loc += sum(
                    1
                    for line in source.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                )
                files_checked += 1

            pairs = _detect_clones(
                blocks_by_file=blocks_by_file,
                max_file_pairs=max_file_pairs,
                similarity_threshold=similarity_threshold,
                ignore_types=ignore_types,
                severity_weights=severity_weights,
            )
            return pairs, total_loc, files_checked

        try:
            pairs, total_loc, files_checked = cache.get_or_compute(
                "duplication_analysis", None, _run
            )
        except Exception as exc:
            return {
                "clone_pairs": [],
                "counts_by_type": {"exact": 0, "renamed": 0, "similar": 0},
                "total_severity": 0.0,
                "clone_coverage": 0.0,
                "total_loc": 0,
                "files_checked": 0,
                "error": str(exc),
            }

        coverage = _clone_coverage(pairs, total_loc)
        counts = _counts_by_type(pairs)
        severity = _total_severity(pairs)

        return {
            "clone_pairs": [p.to_dict() for p in pairs],
            "counts_by_type": counts,
            "total_severity": severity,
            "clone_coverage": coverage,
            "total_loc": total_loc,
            "files_checked": files_checked,
            "error": None,
        }

    # ------------------------------------------------------------------
    # grade — pure, no side effects
    # ------------------------------------------------------------------

    def grade(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
        config: dict[str, Any],
    ) -> GradeResult:
        if current.get("error"):
            return empty_grade("Duplication", f"analysis failed: {current['error']}")
        if current.get("files_checked", 0) == 0:
            return empty_grade("Duplication", "no Python files found to check")

        b_severity = baseline.get("total_severity", 0.0)
        c_severity = current.get("total_severity", 0.0)

        # Already at gold at baseline
        if b_severity <= 0:
            return already_gold("Duplication", current)

        # Score: fraction of baseline severity eliminated
        final_score = _clamp(1.0 - c_severity / max(b_severity, 1e-9))
        is_regression = c_severity > b_severity + 1e-6

        if is_regression:
            final_score = 0.0

        solved = c_severity <= 0

        b_counts = baseline.get("counts_by_type", {})
        c_counts = current.get("counts_by_type", {})
        b_cover = baseline.get("clone_coverage", 0.0)
        c_cover = current.get("clone_coverage", 0.0)

        # Per-type sub-scores
        sub_scores: dict[str, float] = {}
        for ctype in ("exact", "renamed", "similar"):
            b_c = float(b_counts.get(ctype, 0))
            c_c = float(c_counts.get(ctype, 0))
            if b_c <= 0:
                sub_scores[f"type:{ctype}"] = 1.0
            else:
                sub_scores[f"type:{ctype}"] = _clamp(1.0 - c_c / b_c)

        # Deserialise pairs for feedback
        raw_pairs = current.get("clone_pairs", [])

        feedback = _build_feedback(
            score=final_score,
            baseline_score=_clamp(1.0 - b_severity / max(b_severity, 1e-9)),
            is_regression=is_regression,
            pairs=[],  # feedback uses counts + raw_pairs summary
            counts=c_counts,
            coverage=c_cover,
            total_severity=c_severity,
        )

        delta_counts = {
            ctype: b_counts.get(ctype, 0) - c_counts.get(ctype, 0)
            for ctype in ("exact", "renamed", "similar")
        }

        return GradeResult(
            score=final_score,
            gold_distance=c_severity / max(b_severity, 1e-9),
            raw_baseline={
                "total_severity": b_severity,
                "counts_by_type": b_counts,
                "clone_coverage": b_cover,
                "total_loc": baseline.get("total_loc", 0),
            },
            raw_current={
                "total_severity": c_severity,
                "counts_by_type": c_counts,
                "clone_coverage": c_cover,
                "total_loc": current.get("total_loc", 0),
                "files_checked": current.get("files_checked", 0),
            },
            delta={
                "total_severity": b_severity - c_severity,
                "clone_coverage": b_cover - c_cover,
                "counts_by_type": delta_counts,
            },
            feedback=feedback,
            solved=solved,
            is_regression=is_regression,
            sub_scores={
                "baseline_severity": b_severity,
                "current_severity": c_severity,
                "coverage_delta": b_cover - c_cover,
                **sub_scores,
            },
        )

    # ------------------------------------------------------------------
    # gold_standard
    # ------------------------------------------------------------------

    def gold_standard(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "clone_pairs": [],
            "counts_by_type": {"exact": 0, "renamed": 0, "similar": 0},
            "total_severity": 0.0,
            "clone_coverage": 0.0,
            "total_loc": 0,
            "files_checked": 1,
            "error": None,
        }
