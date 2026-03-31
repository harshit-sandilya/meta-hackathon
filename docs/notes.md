# Meta × PyTorch Hackathon — Complete Reference Guide

> **Hackathon Page:** https://www.scaler.com/school-of-technology/meta-pytorch-hackathon  
> **Topic:** Build a real-world OpenEnv environment for AI agent training and evaluation

---

## Table of Contents

1. [Overview & Core Task](#1-overview--core-task)
2. [Key Requirements Summary](#2-key-requirements-summary)
3. [Functional Requirements (Deep Dive)](#3-functional-requirements-deep-dive)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [Evaluation Criteria & Scoring](#5-evaluation-criteria--scoring)
6. [Judging Phases](#6-judging-phases)
7. [Disqualification Criteria](#7-disqualification-criteria)
8. [Pre-Submission Checklist](#8-pre-submission-checklist)
9. [Required Environment Variables](#9-required-environment-variables)
10. [Infrastructure Restrictions](#10-infrastructure-restrictions)
11. [Project Structure & File Requirements](#11-project-structure--file-requirements)
12. [OpenEnv Spec Reference](#12-openenv-spec-reference)
13. [Deployment Guide](#13-deployment-guide)
14. [Recommended Domain Ideas](#14-recommended-domain-ideas)
15. [Reward Function Design Guidelines](#15-reward-function-design-guidelines)
16. [Baseline Inference Script Guidelines](#16-baseline-inference-script-guidelines)

---

## 1. Overview & Core Task

**Goal:** Build a **complete, real-world OpenEnv environment** that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

OpenEnv is a framework (similar in spirit to OpenAI Gym) designed for LLM fine-tuning and agent evaluation using real-world task simulations rather than game-based environments.

---

## 2. Key Requirements Summary

| Requirement      | Details                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------ |
| Task type        | Real-world simulation (NOT games/toys)                                                     |
| API compliance   | Full OpenEnv spec: `step()`, `reset()`, `state()`, typed models, `openenv.yaml`            |
| Minimum tasks    | 3 tasks with agent graders (easy → medium → hard)                                          |
| Scoring range    | 0.0 – 1.0 per task, graders must be deterministic                                          |
| Reward function  | Partial progress signals, not just binary end-of-episode                                   |
| Inference script | Named `inference.py`, placed in root directory                                             |
| LLM client       | Must use **OpenAI Client** for all LLM calls                                               |
| Deployment       | Hugging Face Spaces + working Dockerfile                                                   |
| README           | Full documentation including action/observation spaces, task descriptions, baseline scores |

---

## 3. Functional Requirements (Deep Dive)

### 3.1 Real-World Task Simulation

- The environment **must simulate a task that humans actually perform**.
- **Prohibited:** games, toys, abstract simulations.
- **Allowed examples:**
  - Email triage
  - Code review
  - Data cleaning
  - Scheduling
  - Customer support
  - Content moderation
  - Document summarization
  - Bug report classification
  - Inventory management
  - Medical record parsing

---

### 3.2 OpenEnv Spec Compliance

The environment must implement the **full OpenEnv interface**:

#### Typed Pydantic Models

```python
from pydantic import BaseModel

class Observation(BaseModel):
    # Define the observation space — what the agent sees
    ...

class Action(BaseModel):
    # Define the action space — what the agent can do
    ...

class Reward(BaseModel):
    # Define the reward structure
    score: float           # 0.0 – 1.0
    partial_credit: float  # signal for partial progress
    info: dict             # optional additional info
    ...
```

#### Required API Methods

```python
def reset() -> Observation:
    """
    Resets the environment to a clean initial state.
    Returns the initial observation.
    Called at the start of each episode.
    """

def step(action: Action) -> tuple[Observation, Reward, bool, dict]:
    """
    Executes one action in the environment.
    Returns:
        - observation: next state the agent sees
        - reward: Reward model instance with score
        - done: True if episode is complete
        - info: optional diagnostic dict
    """

def state() -> dict:
    """
    Returns current internal state of the environment.
    Used for debugging, logging, and checkpointing.
    """
```

#### openenv.yaml

Must be present at the project root with metadata:

```yaml
name: your-environment-name
version: "1.0.0"
description: "Short description of what this environment simulates"
author: your-name
tags:
  - real-world
  - <domain-specific-tags>
tasks:
  - name: easy_task
    difficulty: easy
    description: "What the agent must accomplish"
  - name: medium_task
    difficulty: medium
    description: "What the agent must accomplish"
  - name: hard_task
    difficulty: hard
    description: "What the agent must accomplish"
api_version: "1.0"
```

#### Validation

Run before submitting:

```bash
openenv validate
```

This checks spec compliance, YAML correctness, and API method signatures.

---

### 3.3 Minimum 3 Tasks with Agent Graders

- Each task must define a **concrete, well-scoped objective** for the agent.
- Each task must have a **programmatic grader** that:
  - Returns a score between **0.0 and 1.0**
  - Is **deterministic and reproducible** — same inputs → same score every time
  - Has **clear success/failure criteria** (no ambiguity)
- Tasks must range in difficulty: **Easy → Medium → Hard**
- The **Hard task should genuinely challenge frontier models** (GPT-4-class, Claude Opus-class)

#### Example Task Structure

```python
class Task:
    name: str
    difficulty: str  # "easy" | "medium" | "hard"
    description: str

    def grade(self, agent_output: Any, ground_truth: Any) -> float:
        """
        Returns float in [0.0, 1.0].
        Must be deterministic.
        """
        ...
```

---

### 3.4 Meaningful Reward Function

The reward function must:

- ✅ Provide **signal over the full trajectory** (not just sparse end-of-episode reward)
- ✅ Reward **partial progress** toward task completion
- ✅ **Penalize clearly undesirable behavior**, such as:
  - Infinite loops
  - Destructive or irreversible actions
  - Repeatedly calling the same action with no effect
  - Generating outputs that violate task constraints

#### Reward Design Principles

```
Binary (BAD):  reward = 1.0 if done else 0.0
               → Agent gets no signal until episode end

Shaped (GOOD): reward = progress_score * 0.6
                      + quality_score * 0.3
                      + efficiency_bonus * 0.1
                      - penalty_for_bad_actions * 0.2
               → Agent learns at every step
```

---

### 3.5 Baseline Inference Script

- **Filename:** `inference.py`
- **Location:** Project root directory (mandatory)
- Must use the **OpenAI API client** for all LLM calls:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["HF_TOKEN"],
    base_url=os.environ["API_BASE_URL"]
)

response = client.chat.completions.create(
    model=os.environ["MODEL_NAME"],
    messages=[{"role": "user", "content": prompt}]
)
```

- Must read credentials from **environment variables** (never hardcode):
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- Must run against **all 3 tasks** and produce a **reproducible baseline score**
- Must complete **without error**
- Must finish within **20 minutes** runtime

---

## 4. Non-Functional Requirements

### 4.1 Hugging Face Space Deployment

- Must deploy as a containerized **HF Space**
- The Space must be **tagged with `openenv`**
- Must respond to a ping at the Space URL with HTTP **200**
- Must respond to `reset()` API calls

### 4.2 Dockerized Execution

Must include a working `Dockerfile`. The environment must start cleanly with:

```bash
docker build -t my-env .
docker run -p 8080:8080 my-env
```

#### Example Dockerfile Structure

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
```

### 4.3 Documentation (README.md)

The README must include all of the following sections:

| Section                 | Required Content                                              |
| ----------------------- | ------------------------------------------------------------- |
| Environment Description | What real-world task does it model? Why is it valuable?       |
| Motivation              | Why this domain matters for agent evaluation                  |
| Observation Space       | What the agent sees — type, shape, meaning of each field      |
| Action Space            | What actions the agent can take — type and constraints        |
| Task Descriptions       | For each task: name, difficulty, objective, expected behavior |
| Setup Instructions      | How to install dependencies and run locally                   |
| Usage Instructions      | How to run the environment, how to call the API               |
| Baseline Scores         | The scores produced by running `inference.py`                 |
| Dockerfile Instructions | How to build and run the container                            |

---

## 5. Evaluation Criteria & Scoring

### 5.1 Scoring Weights

| Parameter                          | Weight  | Description                                                                                                                        |
| ---------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Real-world utility**             | **30%** | Does the environment model a genuine task? Would someone actually use this to train or evaluate agents?                            |
| **Task & grader quality**          | **25%** | Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression? |
| **Environment design**             | **20%** | Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries.                        |
| **Code quality & spec compliance** | **15%** | Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works.                                 |
| **Creativity & novelty**           | **10%** | Novel problem domain, interesting mechanics, clever reward design, original approach.                                              |

---

### 5.2 Detailed Scoring Rubrics

#### Real-world Utility (30 points)

| Score | Meaning                                                                  |
| ----- | ------------------------------------------------------------------------ |
| 0–5   | Toy or artificial problem with no practical application                  |
| 6–15  | Valid domain but shallow modeling of the real task                       |
| 16–25 | Good domain modeling, would be useful for agent evaluation               |
| 26–30 | Excellent — fills a real gap, immediate value for the RL/agent community |

#### Task & Grader Quality (25 points) — Checklist

- [ ] 3+ tasks with difficulty range (easy → medium → hard)?
- [ ] Graders produce scores between 0.0 – 1.0?
- [ ] Graders are deterministic and reproducible?
- [ ] Hard task genuinely challenges frontier models?

#### Environment Design (20 points) — Checklist

- [ ] `reset()` produces a clean, well-defined initial state?
- [ ] Action and observation types are well-designed and documented?
- [ ] Reward function provides useful, varying signal (not just sparse)?
- [ ] Episode boundaries are sensible and well-defined?

#### Code Quality & Spec Compliance (15 points) — Checklist

- [ ] `openenv validate` passes without errors?
- [ ] `docker build && docker run` works?
- [ ] HF Space deploys and responds correctly?
- [ ] Baseline script runs and reproduces scores?

#### Creativity & Novelty (10 points) — Checklist

- [ ] Domain not seen in OpenEnv before?
- [ ] Reward design has interesting or novel properties?
- [ ] Clever mechanics that make the environment engaging?

---

## 6. Judging Phases

### Phase 1: Automated Validation (Pass/Fail Gate)

All of these must pass to proceed to Phase 2:

- [ ] HF Space deploys and is reachable
- [ ] OpenEnv spec compliance verified
- [ ] Dockerfile builds successfully
- [ ] Baseline score reproduces without error
- [ ] 3+ tasks with graders returning valid scores

### Phase 2: Agentic Evaluation (Scored)

- Baseline agent is re-run by judges
- A standard Open LLM agent (e.g., **Nemotron 3 Super**) is run against all environments
- Score variance check is performed to detect exploits or inconsistencies

### Phase 3: Human Review

- Top submissions reviewed by **Meta and Hugging Face engineers**
- Reviewers assess: real-world utility, creativity, and exploit checks
- Manual inspection for edge cases and grader fairness

---

## 7. Disqualification Criteria

Your submission will be **automatically disqualified** if:

| Reason                                 | Description                                                                |
| -------------------------------------- | -------------------------------------------------------------------------- |
| Environment does not deploy or respond | HF Space is down or returns errors                                         |
| Plagiarism                             | Trivially modified copy of an existing environment                         |
| Broken graders                         | Graders that always return the same score (e.g., always 0.0 or always 1.0) |
| Missing inference script               | `inference.py` is absent from the root directory                           |

---

## 8. Pre-Submission Checklist

Run through every item before submitting:

### Infrastructure

- [ ] HF Space is live and returns HTTP 200
- [ ] `reset()` endpoint is responsive on the Space URL
- [ ] `docker build` completes without errors
- [ ] `docker run` starts the environment cleanly

### Spec Compliance

- [ ] `openenv validate` passes
- [ ] `openenv.yaml` is present and complete
- [ ] Typed Pydantic models for `Observation`, `Action`, `Reward` are defined
- [ ] `step()`, `reset()`, `state()` are all implemented

### Tasks & Graders

- [ ] At least 3 tasks are defined
- [ ] Each task has a programmatic grader
- [ ] All grader scores are in range [0.0, 1.0]
- [ ] Graders are deterministic (same input = same output)
- [ ] Easy → Medium → Hard difficulty progression exists

### Inference Script

- [ ] `inference.py` is in the root directory
- [ ] Script uses OpenAI Client for all LLM calls
- [ ] Script reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment
- [ ] Script runs all 3 tasks without error
- [ ] Script completes in under 20 minutes
- [ ] Baseline scores are documented in README

### Environment Variables

- [ ] `API_BASE_URL` is configured
- [ ] `MODEL_NAME` is configured
- [ ] `HF_TOKEN` is configured

### Documentation

- [ ] README describes the real-world task and motivation
- [ ] Observation space is fully documented
- [ ] Action space is fully documented
- [ ] All 3 task descriptions are in README
- [ ] Setup and usage instructions are clear
- [ ] Baseline scores are included in README

---

## 9. Required Environment Variables

These **must** be defined in the environment configuration and readable from `inference.py`:

| Variable       | Purpose                                                                              |
| -------------- | ------------------------------------------------------------------------------------ |
| `API_BASE_URL` | The API endpoint for the LLM (base URL for OpenAI-compatible API)                    |
| `MODEL_NAME`   | The model identifier to use for inference (e.g., `meta-llama/Llama-3.1-8B-Instruct`) |
| `HF_TOKEN`     | Your Hugging Face token / API key used for authentication                            |

### Usage in inference.py

```python
import os
from openai import OpenAI

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ["HF_TOKEN"]

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)
```

---

## 10. Infrastructure Restrictions

| Constraint               | Limit            |
| ------------------------ | ---------------- |
| Inference script runtime | **< 20 minutes** |
| Target machine vCPUs     | 2                |
| Target machine RAM       | 8 GB             |

Design your environment and inference script to run efficiently within these constraints. Avoid:

- Loading large models locally
- Making excessive or slow API calls
- Memory-heavy data structures for task state

---

## 11. Project Structure & File Requirements

Recommended project layout:

```
your-env-name/
├── openenv.yaml             # Required: OpenEnv metadata and task definitions
├── inference.py             # Required: Baseline inference script (root level)
├── Dockerfile               # Required: Container definition
├── README.md                # Required: Full documentation
├── requirements.txt         # Python dependencies
├── app.py                   # Main entry point for the HF Space / API server
├── environment/
│   ├── __init__.py
│   ├── env.py               # Core environment logic (step, reset, state)
│   ├── models.py            # Pydantic models: Observation, Action, Reward
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── easy_task.py     # Easy task definition + grader
│   │   ├── medium_task.py   # Medium task definition + grader
│   │   └── hard_task.py     # Hard task definition + grader
│   └── data/
│       └── ...              # Task data, fixtures, templates
└── tests/
    ├── test_env.py           # Unit tests for environment
    └── test_graders.py       # Unit tests for graders
```

---

## 12. OpenEnv Spec Reference

### Full Environment Class Template

```python
from pydantic import BaseModel
from typing import Any

# ── Typed Models ──────────────────────────────────────────────
class Observation(BaseModel):
    """What the agent receives as input."""
    task_id: str
    context: str
    current_step: int
    # ... domain-specific fields

class Action(BaseModel):
    """What the agent sends as output."""
    action_type: str
    content: str
    # ... domain-specific fields

class Reward(BaseModel):
    """Structured reward signal."""
    score: float           # Primary score [0.0, 1.0]
    partial_credit: float  # Progress toward goal [0.0, 1.0]
    penalty: float         # Penalty for bad actions [0.0, 1.0]
    feedback: str          # Human-readable explanation
    done: bool             # Episode completion flag


# ── Environment ───────────────────────────────────────────────
class MyEnvironment:
    def __init__(self):
        self._state = {}
        self._current_task = None
        self._step_count = 0

    def reset(self) -> Observation:
        """Reset to clean state, return initial observation."""
        self._state = self._initialize_state()
        self._step_count = 0
        return Observation(**self._build_observation())

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Execute action, return (obs, reward, done, info)."""
        # 1. Apply action to state
        self._apply_action(action)
        self._step_count += 1

        # 2. Compute reward
        reward = self._compute_reward(action)

        # 3. Check if done
        done = self._check_done()

        # 4. Build next observation
        obs = Observation(**self._build_observation())

        # 5. Build info dict
        info = {"step": self._step_count, "state_summary": self.state()}

        return obs, reward, done, info

    def state(self) -> dict:
        """Return current internal state for inspection."""
        return {
            "step_count": self._step_count,
            "task_id": self._current_task,
            **self._state
        }

    def _initialize_state(self) -> dict:
        raise NotImplementedError

    def _apply_action(self, action: Action):
        raise NotImplementedError

    def _compute_reward(self, action: Action) -> Reward:
        raise NotImplementedError

    def _check_done(self) -> bool:
        raise NotImplementedError

    def _build_observation(self) -> dict:
        raise NotImplementedError
```

---

## 13. Deployment Guide

### Step 1: Create HF Space

```bash
# Install huggingface_hub CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create a new Space
huggingface-cli repo create your-env-name --type space --space_sdk docker
```

### Step 2: Push Your Code

```bash
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/your-env-name
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 3: Tag with `openenv`

In your HF Space settings, add the tag: `openenv`

### Step 4: Verify Deployment

```bash
# Ping the Space
curl https://YOUR_USERNAME-your-env-name.hf.space/

# Test reset()
curl -X POST https://YOUR_USERNAME-your-env-name.hf.space/reset
```

### Step 5: Set Environment Variables in HF Space

In HF Space settings → "Repository secrets", add:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

### Step 6: Verify Locally with Docker

```bash
docker build -t your-env-name .
docker run \
  -e API_BASE_URL="your_base_url" \
  -e MODEL_NAME="your_model" \
  -e HF_TOKEN="your_token" \
  -p 8080:8080 \
  your-env-name
```

---

## 14. Recommended Domain Ideas

These are strong candidates for high real-world utility scores:

| Domain                       | Description                                                  | Why It's Strong                                      |
| ---------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **Email Triage**             | Classify, prioritize, and draft responses to emails          | Very common human task; clear metrics                |
| **Code Review**              | Identify bugs, style issues, security flaws in code snippets | Well-defined ground truth; LLMs trained for this     |
| **Data Cleaning**            | Detect and fix malformed records in tabular data             | Messy real-world data; partial credit is natural     |
| **Customer Support Routing** | Route and respond to support tickets                         | High business value; easy to create gold labels      |
| **Content Moderation**       | Classify posts for policy violations                         | Critical task; multi-label scoring possible          |
| **Meeting Scheduling**       | Parse requests, find conflicts, propose times                | Constraint satisfaction; natural difficulty gradient |
| **Bug Report Triage**        | Classify severity, assign team, extract repro steps          | Engineering value; structured output scoring         |
| **Legal Document Review**    | Flag clauses, summarize obligations, identify risks          | High-stakes real task; expert knowledge required     |
| **Medical Record Parsing**   | Extract structured info from unstructured clinical notes     | High value; clear difficulty levels                  |
| **Inventory Reorder**        | Decide when and how much to reorder based on data            | Business simulation; quantitative reward             |

---

## 15. Reward Function Design Guidelines

### Principles

1. **Dense > Sparse** — reward at every step, not just at episode end
2. **Partial Credit** — if the agent is 70% correct, reward 0.7, not 0.0
3. **Penalize Waste** — penalize inefficiency (extra steps, repeated actions)
4. **Penalize Harm** — penalize destructive or invalid actions more heavily
5. **Normalize** — always return scores in [0.0, 1.0]

### Example Reward Components

```python
def compute_reward(action, ground_truth, step_count, max_steps) -> Reward:
    # 1. Accuracy score (primary)
    accuracy = compute_similarity(action.output, ground_truth)  # 0.0–1.0

    # 2. Efficiency bonus (secondary)
    efficiency = max(0, 1.0 - (step_count / max_steps))        # 0.0–1.0

    # 3. Format compliance
    format_ok = validate_format(action.output)                  # bool → 0 or 1

    # 4. Penalty for invalid/harmful actions
    penalty = 0.3 if is_invalid_action(action) else 0.0

    # Weighted combination
    score = (
        accuracy * 0.6
        + efficiency * 0.2
        + format_ok * 0.2
        - penalty
    )

    return Reward(
        score=max(0.0, min(1.0, score)),  # Clamp to [0.0, 1.0]
        partial_credit=accuracy,
        penalty=penalty,
        feedback=f"Accuracy: {accuracy:.2f}, Efficiency: {efficiency:.2f}",
        done=(accuracy >= 0.95 or step_count >= max_steps)
    )
```

---

## 16. Baseline Inference Script Guidelines

### Full Template for `inference.py`

```python
"""
inference.py — Baseline inference script for <Your Environment Name>
Reads API credentials from environment variables.
Runs a model against all 3 tasks and reports baseline scores.
Must complete in < 20 minutes on 2 vCPU / 8 GB RAM.
"""

import os
import json
from openai import OpenAI
from environment.env import MyEnvironment
from environment.models import Action

# ── Configuration ─────────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
HF_TOKEN     = os.environ["HF_TOKEN"]

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

# ── Agent Logic ───────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    """Call the LLM via OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0  # Use 0.0 for reproducibility
    )
    return response.choices[0].message.content.strip()


def run_task(env: MyEnvironment, task_name: str, max_steps: int = 10) -> float:
    """Run agent on a single task, return final score."""
    obs = env.reset()
    total_score = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        # Build prompt from observation
        prompt = f"""
You are an AI agent. Complete the following task:

Task: {obs.task_id}
Context: {obs.context}
Step: {obs.current_step}

Respond with your action in JSON format:
{{"action_type": "...", "content": "..."}}
"""
        # Get LLM response
        raw_response = call_llm(prompt)

        try:
            action_dict = json.loads(raw_response)
            action = Action(**action_dict)
        except Exception as e:
            print(f"  [!] Parse error at step {step}: {e}")
            action = Action(action_type="skip", content="")

        # Step the environment
        obs, reward, done, info = env.step(action)
        total_score = reward.score
        step += 1

        print(f"  Step {step}: score={reward.score:.3f}, partial={reward.partial_credit:.3f}")

    return total_score


# ── Main Runner ───────────────────────────────────────────────
def main():
    env = MyEnvironment()
    tasks = ["easy_task", "medium_task", "hard_task"]
    results = {}

    for task_name in tasks:
        print(f"\n{'='*50}")
        print(f"Running task: {task_name}")
        print(f"{'='*50}")

        env.set_task(task_name)
        score = run_task(env, task_name)
        results[task_name] = score
        print(f"Final score for {task_name}: {score:.4f}")

    print(f"\n{'='*50}")
    print("BASELINE SCORES SUMMARY")
    print(f"{'='*50}")
    for task, score in results.items():
        print(f"  {task}: {score:.4f}")
    overall = sum(results.values()) / len(results)
    print(f"  Overall average: {overall:.4f}")

    # Save results to file for reproducibility
    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

---

## Quick Reference Card

```
HACKATHON ESSENTIALS
════════════════════════════════════════════════════════

✅ Simulate a REAL task (not games)
✅ Implement: step() / reset() / state()
✅ Typed Pydantic models: Observation, Action, Reward
✅ openenv.yaml with metadata
✅ 3 tasks: easy → medium → hard
✅ Graders: deterministic, 0.0–1.0, programmatic
✅ Rewards: partial credit, not binary
✅ inference.py in ROOT directory
✅ Use OpenAI Client for ALL LLM calls
✅ Env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
✅ Dockerfile: docker build + docker run must work
✅ Deploy to HF Spaces, tagged "openenv"
✅ README: full docs + baseline scores
✅ Runtime < 20 min on 2vCPU / 8GB RAM
✅ Run: openenv validate before submitting

════════════════════════════════════════════════════════
SCORING WEIGHTS
  Real-world utility        30%
  Task & grader quality     25%
  Environment design        20%
  Code quality & compliance 15%
  Creativity & novelty      10%

════════════════════════════════════════════════════════
JUDGING PHASES
  Phase 1: Automated validation (pass/fail gate)
  Phase 2: Agentic evaluation (scored)
  Phase 3: Human review by Meta + HF engineers

════════════════════════════════════════════════════════
DISQUALIFIED IF:
  ❌ Space doesn't deploy
  ❌ Plagiarized environment
  ❌ Graders always return same score
  ❌ No inference.py in root
```

---
