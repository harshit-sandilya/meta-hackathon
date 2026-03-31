# Grading System — RefactorEnv

**Project:** Code Refactoring & Maintenance RL Environment (Meta × PyTorch OpenEnv Hackathon)  
**Package path:** `environment/graders/`  
**Last updated:** March 2026

---

## Overview

The grading system is the analytical core of RefactorEnv. It translates raw repository state — test results, lint output, AST analysis, structural invariants — into a single normalised score per dimension that the reward function can combine into a final `RefactorReward`.

Every grader follows the same contract: **delta-based, improvement-relative scoring**. There is no "correct" code. The baseline is computed once at `reset()` on the raw unmodified repo. Every subsequent step is scored as improvement over that baseline. A score of `1.0` means the agent has fully solved what that grader measures from its starting state.

\[
\text{score} = f(\text{baseline_metrics},\ \text{current_metrics},\ \text{scenario_config}) \in [0.0,\ 1.0]
\]

This design makes the system fair across scenarios: a repo that starts with 2 lint violations and one that starts with 200 are both scored on proportional improvement, not absolute counts.

---

## Package Structure

```
environment/graders/
├── utils
├── __init__.py              ← Public exports + GRADER_REGISTRY + factory helpers
├── base_grader.py           ← Abstract BaseGrader ABC + GradeResult model
├── lint_grader.py           ← Ruff-based weighted violation reduction
├── symbol_grader.py         ← Dead code elimination via dependency graph
├── coverage_grader.py       ← Test pass rate + line/branch coverage
├── structure_grader.py      ← Scenario invariant checklist
├── style_grader.py          ← Naming, docstrings, annotations, complexity proxy
├── complexity_grader.py     ← Cyclomatic complexity via radon + AST heuristics
├── duplication_grader.py    ← AST hash-window structural clone detection
└── production_grader.py     ← 10-check production-hygiene suite
```

Each file is independently importable. The `__init__.py` wires them all together under a single public surface.

---

## Shared Infrastructure (Pre-Grader Layer)

Before any grader runs, two shared analysis modules produce data structures that multiple graders consume. This avoids redundant subprocess calls and ensures all graders reason about the same parsed state.

### `utils/ast_utils.py` — AST Foundation

Every grader that inspects code uses this module. Key functions:

- **`parse_file(path) → ast.Module`** — Error-tolerant parsing. Returns a partial AST on `SyntaxError` rather than crashing, so broken code still receives a score and the agent gets feedback.
- **`collect_definitions(tree) → list[SymbolDef]`** — Walks the AST to extract all `FunctionDef`, `ClassDef`, `Import`, `ImportFrom`, and module-level `Assign` nodes with their qualified names.
- **`collect_usages(tree) → list[SymbolRef]`** — Walks for all `Name(ctx=Load)`, `Attribute`, and `Call` nodes.
- **`normalize_ast_subtree(node) → str`** — Renames all local variable names to `var_0, var_1, ...` and returns a canonical hash string. Used by `DuplicationGrader` to detect structural clones regardless of variable naming.

### `utils/dep_graph.py` — Dependency Graph

The backbone consumed by `SymbolGrader` and `StructureGrader`. Built once per `reset()` and cached in `MetricCache`.

```
Nodes: (qualified_name, type)  →  e.g. ("utils.process_data", "function")
Edges: (user, used)            →  e.g. ("main.run", "utils.process_data")   [call edge]
                               →  e.g. ("api.handler", "utils")             [import edge]
```

Key methods:

- **`build_from_files(file_paths)`** — Parses all `.py` files, resolves relative imports, builds the graph.
- **`get_dead_symbols(public_api: list[str]) → list[str]`** — Returns nodes with in-degree 0, excluding the scenario's declared public API entry points.
- **`get_import_cycles() → list[list[str]]`** — Strongly connected components via Tarjan's algorithm. Used by `StructureGrader` to detect circular imports.
- **`get_callers(symbol) → set[str]`** — Reverse lookup for invariant checking.

---

## Base Contract

### `base_grader.py`

All graders inherit from `BaseGrader` and return `GradeResult`. The separation between `compute_metrics` (side-effectful) and `grade` (pure function) is intentional — it allows unit tests to mock metrics without running `pytest` or `ruff` on every test invocation.

```python
class GradeResult(BaseModel):
    score: float        # 0.0–1.0 final grade for this dimension
    raw_metrics: dict   # e.g. {"violations_now": 4, "violations_baseline": 17}
    feedback: str       # human-readable: "Reduced lint violations by 76%"

class BaseGrader(ABC):
    grader_id: str      # canonical ID — must match scenario.yaml key

    @abstractmethod
    def compute_metrics(self, repo_path: Path, config: dict) -> dict:
        """Run analysis tools, return raw metrics dict. Has side effects."""

    @abstractmethod
    def grade(self, baseline: dict, current: dict, config: dict) -> GradeResult:
        """Compute score from delta. Pure function — no I/O."""
```

Shared helpers on `BaseGrader`:

- **`_clamp(v, lo, hi)`** — Saturates a float to `[lo, hi]`.
- **`_delta_score(baseline_val, current_val)`** — Standard `(baseline - current) / max(baseline, 1)` clamped to `[0, 1]`. Used by lint, symbol, complexity, and duplication graders.
- **`already_gold(baseline, current)`** — Returns `True` if baseline was already 0 (nothing to improve). Score is `1.0` vacuously.
- **`empty_grade(grader_id)`** — Returns a `GradeResult(score=1.0)` for graders with no applicable items.

---

## The 9 Graders

### 1. `LintGrader` — Static Violation Reduction

**`grader_id`:** `"lint"`  
**Tool:** `ruff check --output-format json`  
**Primary scenario:** `lint-cleanup` (weight 0.50), present in all scenarios as a minor signal

Ruff is used over flake8/pylint because it is ~100× faster, produces structured JSON output, and covers `pyflakes (F)`, `pycodestyle (E/W)`, `isort (I)`, `pyupgrade (UP)`, and `flake8-bugbear (B)` in a single deterministic pass.

**Metric computation:**

Violations are weighted by category before counting. Not all violations carry equal engineering cost:

| Category        | Weight | Examples                            |
| --------------- | ------ | ----------------------------------- |
| F (pyflakes)    | 1.5×   | Unused imports, undefined names     |
| B (bugbear)     | 1.2×   | Mutable default args, assert misuse |
| E (pycodestyle) | 1.0×   | Indentation, whitespace             |
| C (complexity)  | 0.8×   | McCabe complexity flags             |
| W (warnings)    | 0.5×   | Deprecated constructs               |
| UP (pyupgrade)  | 0.7×   | Old-style string formatting         |
| I (isort)       | 0.6×   | Import ordering                     |

**Scoring:**

\[
\text{score} = \text{clamp}\!\left(\frac{w*{\text{baseline}} - w*{\text{now}}}{\max(w\_{\text{baseline}},\ 1)},\ 0.0,\ 1.0\right)
\]

Proportional partial credit — removing 5 of 20 violations gives `0.25`. Increasing violations stays at `0.0` (not negative; the penalty layer in `reward.py` handles regression separately).

**Feedback example:** `"Lint: 17 → 4 violations (−76%). Remaining: F401 ×2, E501 ×2"`

---

### 2. `SymbolGrader` — Dead Code Elimination

**`grader_id`:** `"symbol"`  
**Tool:** `ast_utils` + `dep_graph` (internal)  
**Primary scenario:** `lint-cleanup` (weight 0.30), `api-rename` (invariant enforcement)

Detects four classes of dead code using the dependency graph, each with a different detection method and weight:

| Dead Code Class    | Detection Method                                                                     | Weight |
| ------------------ | ------------------------------------------------------------------------------------ | ------ |
| Unused imports     | `import X` but `X` never appears as `Name(Load)` in same file                        | 1.0×   |
| Unused variables   | Assigned via `ast.Assign` / `ast.AnnAssign` but `Name(Load)` count = 0 in same scope | 0.8×   |
| Dead functions     | In-degree 0 in call graph, not in `scenario.public_api`, not in test files           | 1.5×   |
| Unreachable blocks | Statements after `return` / `raise` / `continue` / `break` in same block             | 1.2×   |

**The public API problem:** A function with no internal callers might still be the public interface. `scenario.yaml` declares `public_api: ["MyClass", "run"]` — these symbols are excluded from dead function detection. Everything else is fair game.

**Scoring:**

\[
\text{score} = \text{clamp}\!\left(\frac{w*{\text{dead_baseline}} - w*{\text{dead_now}}}{\max(w\_{\text{dead_baseline}},\ 1)},\ 0.0,\ 1.0\right)
\]

If the agent creates new dead code while removing old dead code (e.g., extracts a helper but never calls it), `weighted_dead_now` may exceed `weighted_dead_baseline`. Score stays at `0.0`; the raw delta is captured in `raw_metrics` for the penalty layer.

---

### 3. `CoverageGrader` — Test Correctness & Coverage

**`grader_id`:** `"coverage"`  
**Tool:** `pytest --tb=no -q --cov --cov-report=json`  
**Primary scenario:** `test-coverage` (objective mode), all other scenarios (constraint mode)

This grader has two distinct operating modes selected by the scenario config:

**Constraint mode** (`lint-cleanup`, `api-rename`, `style-enforce`): Coverage is a guardrail, not a goal. Score is halved if line coverage drops by more than 2% from baseline.

```python
pass_rate   = current["passed"] / max(current["total"], 1)
coverage_ok = current["line_coverage"] >= baseline["line_coverage"] - 0.02
score       = pass_rate if coverage_ok else pass_rate * 0.5
```

**Objective mode** (`test-coverage`): Coverage improvement is the primary goal. Score is computed as progress toward the target threshold declared in `scenario.yaml`.

```python
target   = config.get("target_coverage", 0.80)
progress = (current["line_coverage"] - baseline["line_coverage"]) \
           / max(target - baseline["line_coverage"], 0.01)
score    = clamp(progress, 0.0, 1.0)
```

`raw_metrics` includes: pass count, fail count, error count, line coverage %, and branch coverage % per file. Branch coverage is secondary — weighted at 0.3× relative to line coverage.

---

### 4. `StructureGrader` — Invariant Compliance

**`grader_id`:** `"structure"`  
**Tool:** AST + `pathlib` + `dep_graph` + git diff  
**Primary scenario:** `module-decompose` (weight 0.40), `api-rename` (weight 0.35)

Enforces task-specific structural rules declared in `scenario.yaml` under `invariants:`. Each invariant is a boolean check. Score is the fraction of satisfied invariants.

**Invariant categories:**

| Category          | Key                 | Detection Method                                    |
| ----------------- | ------------------- | --------------------------------------------------- |
| Required symbols  | `required_symbols`  | AST walk — `FunctionDef.name`, `ClassDef.name`      |
| Forbidden symbols | `forbidden_symbols` | AST walk — old name must be absent                  |
| Forbidden imports | `forbidden_imports` | `dep_graph.get_import_cycles()` + direct edge check |
| Required files    | `required_files`    | `pathlib.Path.exists()`                             |
| No-edit files     | `no_edit_files`     | Git diff against baseline snapshot in `sandbox.py`  |
| Max file size     | `max_file_size`     | `len(Path(f).read_text().splitlines())`             |

**Example `scenario.yaml` block for `api-rename`:**

```yaml
invariants:
  required_symbols:
    - file: "utils.py"
      symbol: "normalize_data"
      type: "function"
  forbidden_symbols:
    - file: "*"
      symbol: "process_data"
  no_edit_files:
    - "tests/**/*.py"
```

**Scoring:**

\[
\text{score} = \frac{\text{satisfied invariants}}{\text{total invariants}}
\]

Invariants marked `weight: hard` in `scenario.yaml` apply an additional penalty via `reward.py` when violated — not just a score reduction.

---

### 5. `StyleGrader` — Higher-Order Style Quality

**`grader_id`:** `"style"`  
**Tool:** AST + regex (no external linter)  
**Primary scenario:** `lint-cleanup` (weight 0.10), `style-enforce` (weight 0.25)

Catches quality habits that `ruff` will not flag. Four independently scored sub-dimensions:

**a) Naming conventions** — AST node names matched against regexes:

- `FunctionDef.name`: must match `^[a-z_][a-z0-9_]*$`
- `ClassDef.name`: must match `^[A-Z][a-zA-Z0-9]*$`
- Module-level constants: must match `^[A-Z_][A-Z0-9_]*$`
- Score: `compliant_symbols / total_symbols`

**b) Docstring coverage** — via `ast.get_docstring()`:

- All public (no `_` prefix) `FunctionDef` + `ClassDef` nodes checked
- Bonus `+0.1` for Google-style sections (`Args:`, `Returns:`, `Raises:`) detected via regex
- Score: `documented / total_public_symbols`

**c) Type annotation density** — `ast.FunctionDef.args.annotations` and `returns`:

- Annotated parameters / total parameters across all function definitions
- Score: `annotated / total`

**d) Complexity proxy** — raw AST node counting per function:

- Counts `ast.If`, `ast.For`, `ast.While`, `ast.ExceptHandler`, `ast.BoolOp` nodes
- Functions exceeding `config.max_complexity` (default: 10) are flagged
- Score: `simple_functions / total_functions`

**Final style score (delta from baseline):**

\[
\text{style_raw} = 0.25 \cdot \text{naming} + 0.30 \cdot \text{docstring} + 0.25 \cdot \text{type_annot} + 0.20 \cdot \text{complexity}
\]

\[
\text{score} = \text{clamp}\!\left(\frac{\text{style_raw} - \text{baseline_style_raw}}{\max(1 - \text{baseline_style_raw},\ 0.01)},\ 0.0,\ 1.0\right)
\]

---

### 6. `ComplexityGrader` — Cyclomatic Complexity Reduction

**`grader_id`:** `"complexity"`  
**Tool:** `radon cc --json` + AST heuristics  
**Primary scenario:** `module-decompose` (weight 0.20), `style-enforce` (weight 0.15)

Radon gives per-function McCabe cyclomatic complexity grades (A–F). Score is based on reducing the count of functions graded D, E, or F (complexity > threshold declared in `scenario.yaml`, default 10).

\[
\text{score} = \text{clamp}\!\left(\frac{\text{complex_fns_baseline} - \text{complex_fns_now}}{\max(\text{complex_fns_baseline},\ 1)},\ 0.0,\ 1.0\right)
\]

Radon is preferred over the standalone `mccabe` package because it produces structured JSON output and computes the Maintainability Index in the same pass, giving richer `raw_metrics`.

---

### 7. `DuplicationGrader` — Structural Clone Detection

**`grader_id`:** `"duplication"`  
**Tool:** `ast_utils.normalize_ast_subtree()` (internal)  
**Primary scenario:** `module-decompose` (weight 0.15), `style-enforce` (weight 0.10)

Detects Type I / II / III code clones using a hash-window approach over normalised ASTs. The algorithm:

1. Slide a window of N statements (N=3 by default) over every function body's AST.
2. Normalise each window: rename locals to `var_0, var_1, ...`, strip docstrings, strip comments.
3. Hash the normalised subtree string.
4. Count hash collisions across files → duplicate block count.
5. Score is delta from baseline using the same formula as complexity and lint graders.

This approach is preferred over `pylint --duplicate-code` because pylint's token-based duplicate detector is non-deterministic across runs (sensitive to file ordering). AST hashing is fully deterministic on same input.

**Cross-file vs same-file clones:** Both are counted. Cross-file clones get a 1.2× weight multiplier because they are harder to deduplicate and represent stronger DRY violations.

---

### 8. `ProductionGrader` — Production-Hygiene Suite

**`grader_id`:** `"production"`  
**Tool:** AST + `tokenize` (internal, no subprocess)  
**Primary scenario:** `style-enforce` (weight 0.20)

A 10-check static analysis suite covering production-readiness hygiene. All checks are pure AST + regex — no subprocess calls, so this grader adds negligible latency. Each check produces a ratio `compliant_items / total_items`.

| #   | Check              | What It Detects                                                 | Default Weight |
| --- | ------------------ | --------------------------------------------------------------- | -------------- |
| 1   | `type_annotations` | Public function parameters and return types without annotations | 0.22           |
| 2   | `docstrings`       | Public functions, classes, and modules missing docstrings       | 0.20           |
| 3   | `exception_safety` | Bare `except:` or `except Exception:` without re-raise          | 0.14           |
| 4   | `logging_hygiene`  | `print()` calls in non-test files                               | 0.12           |
| 5   | `no_debug_code`    | `breakpoint()`, `pdb.set_trace()`, `ipdb`, `ic()`               | 0.08           |
| 6   | `no_todos`         | `TODO` / `FIXME` / `HACK` / `XXX` / `NOQA` comments             | 0.07           |
| 7   | `exports_defined`  | Public modules missing `__all__`                                | 0.07           |
| 8   | `resource_safety`  | `open()` / `socket()` calls outside `with` context managers     | 0.05           |
| 9   | `constant_naming`  | Module-level constants not in `UPPER_SNAKE_CASE`                | 0.03           |
| 10  | `error_messages`   | `raise SomeError()` with no message string                      | 0.02           |

**Weighted scoring:**

\[
\text{final_score} = \frac{\sum_i w_i \cdot \text{score}\_i}{\sum_i w_i}
\]

where the sum is taken only over checks with at least one applicable item (vacuously compliant checks do not dilute the denominator).

**Configurability:** `scenario.yaml` can override per-check weights, enable/disable individual checks, set `strict_annotations` (to also require `*args`/`**kwargs` annotations), and declare `exclude_patterns` for files to skip (e.g., generated code).

---

## Registry and Factory

### `__init__.py` — Public Surface

The `__init__.py` exports all types and two factory functions:

```python
from environment.registry.graders import (
    BaseGrader, GradeResult,
    LintGrader, SymbolGrader, CoverageGrader,
    StructureGrader, StyleGrader, ComplexityGrader,
    DuplicationGrader, ProductionGrader,
    GRADER_REGISTRY,
    get_grader,
    get_graders_for_scenario,
)
```

### `GRADER_REGISTRY`

```python
GRADER_REGISTRY: dict[str, type[BaseGrader]] = {
    "lint":        LintGrader,
    "symbol":      SymbolGrader,
    "coverage":    CoverageGrader,
    "structure":   StructureGrader,
    "style":       StyleGrader,
    "complexity":  ComplexityGrader,
    "duplication": DuplicationGrader,
    "production":  ProductionGrader,
}
```

Keys are the canonical grader IDs. They must match the keys used in `scenario.yaml`'s `graders:` block. IDs are defined as class attributes (`grader_id = "lint"`) on each grader class — the registry dict is built by reading those attributes, not hardcoded strings, so renaming a class does not silently break the mapping.

### `get_grader(grader_id: str) → BaseGrader`

Returns a fresh instance of the named grader. Raises `KeyError` with a list of valid IDs on unknown input. Used in tests and utility scripts where a hard failure is the right signal.

### `get_graders_for_scenario(config: dict) → list[BaseGrader]`

Parses `config["graders"]`, instantiates each listed grader, and returns them **sorted by descending weight**. Unknown grader IDs emit a `warnings.warn` and are skipped — this prevents a typo in `scenario.yaml` from crashing an entire episode with a zero reward and no feedback.

The descending-weight ordering is not cosmetic: `CoverageGrader` (which spawns `pytest --cov`) is the most expensive operation. Running it first populates `MetricCache` with the pytest JSON output, allowing cheaper graders (`LintGrader`, `ProductionGrader`) to reuse the cached subprocess results.

---

## Scenario Configuration

Each task's `scenario.yaml` declares which graders are active and their weights. The weights in these blocks feed directly into `compute_reward()` in `reward.py`.

**`lint-cleanup` (easy):**

```yaml
graders:
  lint: { weight: 0.50 }
  symbol: { weight: 0.30 }
  style: { weight: 0.10 }
  coverage: { weight: 0.10, mode: constraint }
```

**`api-rename` (easy–medium):**

```yaml
graders:
  structure: { weight: 0.50 }
  coverage: { weight: 0.30, mode: constraint }
  lint: { weight: 0.20 }
```

**`test-coverage` (medium):**

```yaml
graders:
  coverage: { weight: 0.70, mode: objective, target_coverage: 0.80 }
  lint: { weight: 0.20 }
  structure: { weight: 0.10 }
```

**`module-decompose` (medium–hard):**

```yaml
graders:
  structure: { weight: 0.40 }
  coverage: { weight: 0.30, mode: constraint }
  complexity: { weight: 0.20 }
  duplication: { weight: 0.10 }
```

**`style-enforce` (hard):**

```yaml
graders:
  production: { weight: 0.25 }
  style: { weight: 0.25 }
  coverage: { weight: 0.20, mode: constraint }
  complexity: { weight: 0.15 }
  duplication: { weight: 0.10 }
  lint: { weight: 0.05 }
```

---

## Reward Composition

`compute_reward()` in `reward.py` assembles per-grader scores into the final `RefactorReward`:

\[
\text{qual} = \sum\_{i} w_i \cdot \text{grader}\_i.\text{grade}(\text{baseline},\ \text{current},\ \text{config}).\text{score}
\]

\[
\text{score_raw} = 0.5 \cdot \text{acc} + 0.3 \cdot \text{qual} + 0.1 \cdot \text{eff} + 0.1 \cdot \text{fmt} - \text{pen}
\]

\[
\text{score} = \text{clamp}(\text{score_raw},\ 0.0,\ 1.0)
\]

Where:

- `acc` = `CoverageGrader` pass rate (test correctness — always 50% of reward)
- `qual` = weighted sum of all active graders (the improvement signal — 30% of reward)
- `eff` = step efficiency: `max(0, 1 − step_count / max_steps)` (10%)
- `fmt` = `StructureGrader` binary invariant compliance (10%)
- `pen` = penalty: +0.3 syntax crash, +0.1 per repeated no-op, capped at 0.7

---

## Data Flow

```
reset()
  ├─ copy repo to sandbox
  ├─ get_graders_for_scenario(config) → ordered grader list
  ├─ for each grader: compute_metrics(sandbox_path, config) → baseline_metrics
  └─ return initial RefactorObservation

step(edit_file action)
  ├─ apply diff to sandbox
  ├─ for each grader: compute_metrics(sandbox_path, config) → current_metrics
  ├─ for each grader: grade(baseline_metrics, current_metrics, config) → GradeResult
  ├─ compute_reward() assembles RefactorReward from GradeResult list
  └─ return new observation + reward
```

Full grader stack latency target: **< 5 seconds per step** on the small codebases in `tasks/*/repo/`. `ruff` and all AST-based graders run in milliseconds. `pytest --cov` (2–4 seconds) runs only on explicit `run_tests` actions, not on every `edit_file` step.

---

## Key Design Decisions

| Decision           | Choice                                          | Rationale                                                                                    |
| ------------------ | ----------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Lint tool          | `ruff` (not `flake8` / `pylint`)                | Deterministic, structured JSON, ~100× faster, covers more rules in one pass                  |
| Dead code          | Custom dep graph (not `vulture`)                | `vulture` has false positives on dynamic attributes; custom graph is tunable for small repos |
| Complexity         | `radon cc` (not standalone `mccabe`)            | Radon gives JSON output + Maintainability Index in one pass                                  |
| Duplication        | AST hash-window (not `pylint --duplicate-code`) | pylint's token-based detector is non-deterministic across runs; AST hashing is pure          |
| Scoring model      | Delta from baseline (not absolute thresholds)   | Every repo starts different — relative improvement is the fair signal                        |
| Invariant spec     | `scenario.yaml` (not hardcoded)                 | Graders stay generic; tasks are fully swappable                                              |
| Unknown grader IDs | `warnings.warn` + skip (not `raise`)            | A typo in `scenario.yaml` should not crash an episode with zero reward and no feedback       |
| Grader ordering    | Descending weight (not alphabetical)            | Most expensive grader (coverage) runs first, populates MetricCache for cheaper graders       |
| Production checks  | Pure AST + tokenize (no subprocess)             | 10-check suite adds negligible latency — no external tools required                          |
