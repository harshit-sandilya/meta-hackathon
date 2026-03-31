# Design Plan for a Refactoring & Maintenance OpenEnv Environment

## Overview and Goals

This document defines a full design for a **code refactoring and maintenance** environment built on OpenEnv, targeting the Meta PyTorch OpenEnv hackathon.
The environment simulates a realistic developer workflow: incrementally refactoring and cleaning up small codebases while preserving behavior and improving quality metrics.
The primary goals are:

- Implement a **unified environment pipeline** that can host multiple refactoring scenarios (3–5 tasks across easy/medium/hard).
- Use **deterministic, algorithmic reward functions** based on tests, static analysis, and style rules, with dense partial credit and clear penalties.
- Reuse ideas and assets from established tools and repos (Claude Code, Everything Claude Code, system-prompt collections, Google style guides, OpenSandbox, Agent Lightning, BitNet/AirLLM, OpenRouter) where they help either reward design, agent baselines, or infrastructure.[1][2][3][4][5][6][7][8][9][10]

The environment is intended to be reusable as an open-source benchmark for agentic coding systems, beyond the hackathon.

---

## Unified Environment Pipeline

### High-Level Architecture

The environment follows the standard OpenEnv three-component pattern:

- **Models layer (`models.py`)**
  - `RefactorAction`, `RefactorObservation`, `RefactorState`, and `RefactorReward` implemented as Pydantic models, following the hackathon’s recommended structure for `Action`, `Observation`, `State`, and `Reward`.[11]
- **Server layer (`server/environment.py`)**
  - Implements `Environment` with `reset`, `step`, and `state`, plus internal helpers for sandbox management, patch application, metrics computation, and reward calculation.[11]
- **Client layer (`client.py`)**
  - Defines `RefactorEnvClient` mapping JSON payloads to typed models, similar to the examples in the OpenEnv course and docs.[12][13]

Deployment is via a FastAPI app and Docker container, as recommended by OpenEnv documentation and the hackathon notes.[13][11]

### Scenario Abstraction

The unified pipeline treats each specific refactoring problem as a **scenario**, identified by `task_id` and described in a configuration file.
Each scenario includes:

- A **baseline repository snapshot** (in `data/<scenario>/repo/`).
- A **config file** (`scenario.yaml`) describing:
  - Difficulty: `easy`, `medium`, or `hard`.
  - Objectives: e.g., "reduce lint errors", "extract function X", "module split".
  - Metric weights or thresholds (e.g., minimum acceptable test pass percentage).
- Optional **target invariants** describing structural expectations (e.g., required function names, module layouts).

On `reset`, the environment selects a scenario (based on difficulty setting or a round-robin schedule), copies the baseline repo into a per-episode sandbox, and pre-computes baseline metrics.[11]

### Episode Loop

The episode loop is standard RL style, consistent with the OpenEnv training examples:

1. `reset()`
   - Choose scenario; set up sandbox; compute baseline metrics; return initial `RefactorObservation`.[11]
2. `step(action)`
   - Apply a code action (e.g., patch, run tests, run linter) in the sandbox.
   - Recompute metrics; call `compute_reward`; update `RefactorState`.
   - Return new observation, reward, `done` flag, and `info`.
3. `state()`
   - Return current state summary for debugging and logging.

This loop is compatible with OpenEnv’s TRL integration, where GRPO or similar algorithms can be used to train policy models on environment feedback.[12]

---

## Typed Models: Observation, Action, State, Reward

### Observation Model

`RefactorObservation` includes:

- `task_id: str` – scenario identifier.
- `difficulty: Literal["easy", "medium", "hard"]`.
- `current_step: int` – steps taken so far.
- `max_steps: int` – per-scenario cap.
- `file_list: List[str]` – files eligible for reading/editing.
- `active_file: Optional[str]` – path of file currently in view.
- `file_content: Optional[str]` – content (or windowed excerpt) of `active_file`.
- `test_summary: dict` – e.g., `{ "total": int, "passed": int, "failed": int }` from last test run.
- `lint_summary: dict` – counts of lint/type errors by category.
- `quality_metrics: dict` – optional metrics (cyclomatic complexity, dead-code warnings, etc.).
- `remaining_steps: int` – `max_steps - current_step`.

This matches the hackathon’s guidance that observations should clearly communicate state and progress to the agent.[11]

### Action Model

`RefactorAction` supports a small set of high-level operations:

- `action_type: Literal["view_file", "edit_file", "run_tests", "run_linter", "submit"]`.
- For `view_file`:
  - `path: str`.
- For `edit_file`:
  - `path: str`
  - `diff: str` – unified diff or patch.
- For `run_tests` and `run_linter`:
  - No extra fields.
- For `submit`:
  - Optional `note: Optional[str]`.

This mirrors patch-based workflows commonly used by agentic coding tools (Claude Code, Cursor, etc.), where the agent proposes diffs rather than raw file rewrites.[14][15][1]

### State Model

`RefactorState` represents internal environment state:

- `episode_id: str`.
- `task_id: str`.
- `step_count: int`.
- `sandbox_path: str` – ephemeral directory for this episode.
- `baseline_metrics: dict` – baseline test and lint counts, complexity measures.
- `last_metrics: dict` – metrics computed at last step.
- `violations: List[str]` – invariants violated so far.

### Reward Model

`RefactorReward` follows the hackathon’s recommended pattern:

- `score: float` – overall reward in \([0.0, 1.0]\).
- `partial_credit: float` – interpretable main progress metric (e.g., test pass ratio).
- `penalty: float` – aggregated penalty applied to `score`.
- `feedback: str` – human-friendly explanation of metric changes.
- `done: bool` – whether the episode is complete.

The underlying `compute_reward` function combines accuracy, quality, efficiency, and invariant compliance, closely following the example reward design guidelines in the hackathon notes.[11]

---

## Deterministic Reward Function

### Metrics

For each step, the environment computes:

1. **Test correctness (`acc`)**
   - Run `pytest` (or an equivalent runner) inside the sandbox for the current scenario.
   - Let `p = passed / total` (0 to 1).
   - Define `acc = p`.

2. **Static quality improvement (`qual`)**
   - Run a linter and/or type checker; record `lint_now` and baseline `lint_start`.
   - `lint_improvement = (lint_start - lint_now) / max(lint_start, 1)`, clipped to \([0,1]\).
   - Optionally incorporate complexity or dead-code metrics using AST-based analysis inspired by refactoring benchmarks.[16][17]
   - Set `qual = lint_improvement` or a weighted average of multiple signals.

3. **Invariant / style compliance (`fmt`)**
   - Enforce invariants such as:
     - No edits to tests in easy/medium scenarios.
     - No new files outside allowed directories.
     - Required functions/classes/modules exist after refactor.
   - Integrate style rules informed by Google’s style guides and engineering practices (naming, comments, test presence), and/or language-specific rules like `gts` for TypeScript where applicable.[18][19]
   - `fmt = 1.0` if all invariants and style thresholds pass, else `0.0`.

4. **Efficiency (`eff`)**
   - Use a simple step-based efficiency bonus as in the hackathon’s reward examples:  
     `eff = max(0.0, 1.0 - step_count / max_steps)`.[11]

5. **Penalty (`pen`)**
   - Aggregate penalties for clearly bad behavior:
     - `+0.3` if tests or lints crash due to syntax/runtime errors from an edit.
     - `+0.1` per repeated no-op action (e.g., applying identical diffs, tight action loops).
     - Optional: `+0.2` if code no longer compiles or imports break in a way not covered by tests.
   - Cap `pen` at, for example, `0.7`.

### Combination

The reward is then:

- `score_raw = 0.5 * acc + 0.3 * qual + 0.1 * eff + 0.1 * fmt - pen`.
- `score = clamp(score_raw, 0.0, 1.0)`.
- `partial_credit = acc`.
- `done` is set when either:
  - `acc >= target_threshold` (e.g., tests fully passing), or
  - `step_count >= max_steps`, or
  - Agent calls `submit`.

This mirrors the hackathon’s recommended multi-component reward design (accuracy, efficiency, format compliance, penalties), while remaining fully deterministic.[11]

---

## Scenario Design: 3–5 Concrete Refactoring Tasks

### Scenario 1: Single-File Cleanup (Easy)

**Goal:** Clean up a single Python utility module with style issues and unused code, without breaking tests.

- Baseline repo: a small utility file from a curated or synthetic source, similar to Aider’s single-file refactor tasks.[16]
- Issues include:
  - Unused imports and dead code.
  - Inconsistent naming and docstring patterns.
  - Lint errors but fully passing tests.
- Objectives:
  - Reduce lint errors by at least a defined fraction.
  - Preserve all tests.
  - Optionally improve docstrings/naming in line with style guidelines.
- Difficulty: `easy`.

**Reward behavior:**

- Immediate positive reward for removing unused imports or dead code without breaking tests (increase in `qual`, `acc` unchanged).
- Penalties for syntax errors or breaking tests.

### Scenario 2: API Rename Propagation (Easy–Medium)

**Goal:** Rename a frequently used function or method across a small codebase, updating all call sites and preserving behavior.

- Baseline repo: small multi-file project with a central function and call sites across 3–4 files.
- Provided objective: rename `process_data` → `normalize_data` and ensure all references are updated.
- Difficulty: transitions between `easy` and `medium` depending on codebase size.

**Invariants and checks:**

- Required new function name must exist; old name must not be referenced
  after refactor.
- All tests must pass.

**Reward behavior:**

- `acc` rewards preserved behavior.
- `qual` and `fmt` reward correct API rename and removal of references to the old name.

### Scenario 3: Function Extraction and De-duplication (Medium)

**Goal:** Extract duplicated logic into a shared helper function and reduce code duplication across files.

- Baseline repo: two or more modules sharing nearly identical logic blocks.
- Objective: create a shared helper function and update call sites.
- Difficulty: `medium`.

**Metrics:**

- Use AST or textual similarity to detect duplicated logic; compute `dup_start` and `dup_now`.
- Include `dup_improvement` as part of `qual`.

**Reward behavior:**

- As duplicates decrease and tests remain passing, `qual` improves.
- Penalties if the helper function is created but not used consistently.

### Scenario 4: Module Decomposition (Medium–Hard)

**Goal:** Split a "god module" into smaller modules according to a target structure while preserving functionality.

- Baseline repo: a single large file with multiple logically separable concerns (e.g., parsing, validation, I/O).
- Target invariant: new module layout described in `scenario.yaml`, e.g., `parser.py`, `validator.py`, `io.py`.
- Difficulty: `medium` or `hard` depending on complexity.

**Invariants and checks:**

- Each target module exists and exports expected APIs.
- Imports are updated accordingly.
- Tests continue to pass.

**Reward behavior:**

- Positive reward for matching target module structure (AST checks) and for maintaining passing tests.

### Scenario 5: Performance-Sensitive Refactor (Hard)

**Goal:** Replace a naive implementation of a DSA-style routine with a more efficient variant, while preserving correctness.

- Baseline repo: includes a brute-force implementation of an algorithm (e.g., two-sum, longest-subarray) and a large test harness.
- Objective: implement a more efficient version or refactor the code to meet a time budget on a larger input set.
- Difficulty: `hard`.

**Metrics:**

- Correctness still measured by tests.
- An additional timing or operation-count metric is computed on synthetic benchmarks.
- `qual` can incorporate performance improvements (time or complexity proxies) while `acc` covers correctness.

**Reward behavior:**

- Agent receives reward for achieving or surpassing a performance threshold without breaking tests.

---

## Use of External Repos and Practices

### Claude Code and Everything Claude Code

- **Claude Code repo and best practices** provide patterns for patch-based editing, subagents, and safe git workflows, which inform how `edit_file` actions and diffs can be structured and validated in the environment.[20][1][14]
- **Everything Claude Code** offers a large catalogue of specialized agents (refactorers, TDD agents, code reviewers) and verification loops; its rules can inspire baseline prompts and guidelines in `inference.py`, especially around TDD and verification before `submit`.[21][22][15]

### System Prompts and Agent Design

- The **system-prompts-and-models-of-ai-tools** repository aggregates system prompts from tools like Cursor, v0, and others and can be mined for prompt structures that encourage stepwise reasoning, diff-based workflows, and structured outputs.[23][24]
- These patterns can be used to design the `system` messages and step prompts in `inference.py`, to ensure the baseline agent behaves predictably and returns valid JSON actions.

### OpenSandbox and Safe Execution

- **OpenSandbox** is a general-purpose sandbox for AI agents with FastAPI control plane and Docker/Kubernetes runtimes; its architecture is relevant for thinking about safe, isolated code execution.[25][26][2]
- While OpenEnv already mandates containerization, OpenSandbox’s design (multi-language SDKs, gVisor-based isolation, unified exec API) can inform how the environment manages subprocess calls for tests and linters inside the container.

### Agent Lightning and RL Training

- **Agent Lightning** is a framework for training arbitrary agent frameworks with RL, using trace-based reward logic and a loop that requires minimal changes to agent code.[27][28][3]
- Although the hackathon focuses on building an environment rather than training, designing the environment with clean step-level traces and rich reward signals makes it naturally compatible with tools like Agent Lightning for future RL fine-tuning.

### BitNet, AirLLM, and Efficient Inference

- **BitNet** and **AirLLM** demonstrate efficient inference of large models on constrained hardware, using 1-bit quantization and layer-wise streaming, respectively.[29][30][6]
- For the hackathon baseline `inference.py`, these libraries are not required, but the **design can assume that resource-efficient models or quantized variants (e.g., BitNet on CPU) may be used by future users**, and thus keep prompts compact and context windows manageable.

### OpenRouter Quickstart

- OpenRouter provides an OpenAI-compatible API that can proxy multiple models under a single endpoint.[31][32][33]
- This is directly helpful for `inference.py`: the script can use the OpenAI Python client but point `base_url` to `https://openrouter.ai/api/v1`, reading API key from `OPENROUTER_API_KEY` or environment variables defined by the hackathon (`HFTOKEN` / custom mapping).

### Google Style Guides and Engineering Practices

- Google’s style guides and engineering practices documents (JavaScript/TypeScript style, Go style, and code review guidelines) can be used to define **style and quality invariants** for some scenarios.[34][35][19][18]
- For example, scenarios can include:
  - Enforcing JS/TS style via rules that mirror the Google JS style guide or gts configuration.
  - Using code-review criteria (tests, naming, comments, style, documentation) from Google’s eng-practices as part of the textual `feedback` provided in `RefactorReward`.[35][19]

### Awesome Scalability

- The **awesome-scalability** reading list documents patterns for scalable, reliable systems; it does not directly drive the environment but can inspire **future advanced scenarios** involving refactors that improve scalability or performance characteristics.[36]

---

## Baseline Agent and `inference.py`

### Requirements

The hackathon requires a root-level `inference.py` that:[11]

- Uses an OpenAI-compatible client, reading base URL and model from the environment.
- Runs the environment on all 3 difficulty tasks and logs baseline scores.
- Completes within 20 minutes on a 2 vCPU / 8 GB machine.

### Design

The baseline script can:

- Use the OpenAI Python client with `base_url` pointing to either:
  - A Hugging Face Inference Endpoint (as suggested by OpenEnv docs), or
  - OpenRouter’s API gateway for flexible model selection.[31][13]
- Implement a simple agent loop:
  - Build prompts from `RefactorObservation` (task id, file list, current file contents, test/lint summaries).
  - Use a system prompt inspired by Claude Code and Everything Claude Code, describing:
    - How to propose diffs.
    - How to call `run_tests` / `run_linter` strategically.
    - When to call `submit`.
  - Parse the model’s JSON response into `RefactorAction`.
- Run for a small, fixed number of steps per scenario, then record final scores.

This baseline does not need to be sophisticated; its role is to provide reproducible scores and demonstrate that the environment is usable.

---

## Quality, Testing, and Determinism

### Testing the Environment

High environment quality is crucial for hackathon judging.
Key practices include:

- **Unit tests for environment logic and graders** (e.g., `tests/test_env.py`, `tests/test_graders.py`), mirroring the structure suggested in the hackathon reference notes.[11]
- **Deterministic seeds and randomization control** to ensure the same action sequence produces the same rewards and scores.
- **CI checks** using pytest, linters, and type checkers on the environment code itself, following patterns from projects like Agent Lightning and BitNet, which enforce code quality via automated checks.[37][27]

### Documentation and README

The README should follow the hackathon’s template:[11]

- Environment description and motivation: why refactoring and maintenance matters for real-world agents.
- Observation and action space documentation.
- Detailed descriptions of each scenario (easy/medium/hard).
- Reward function details, including formulas and rationale.
- Setup and usage instructions.
- Baseline scores from `inference.py`.

Providing clear diagrams or tables for metrics and scenarios will help both judges and future users.

---

## Summary

The proposed refactoring and maintenance environment offers:

- A **unified pipeline** for multiple realistic refactor scenarios.
- **Deterministic, shaped rewards** grounded in tests, static analysis, and style invariants.
- Reuse of best practices from **Claude Code**, **Everything Claude Code**, **Google style guides**, **OpenSandbox**, **Agent Lightning**, and **OpenRouter** to strengthen agent interaction patterns, sandbox safety, style checks, and future RL integration.[2][3][15][19][18][21][25][27][1][14][31]

This environment should be both hackathon-ready and valuable as an open-source benchmark for evaluating agentic coding systems.
