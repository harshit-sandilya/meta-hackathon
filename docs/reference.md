# Meta × PyTorch OpenEnv Hackathon — Reference Doc

> **Hackathon Page:** https://www.scaler.com/school-of-technology/meta-pytorch-hackathon  
> **OpenEnv GitHub:** https://github.com/meta-pytorch/OpenEnv  
> **HuggingFace Course:** https://github.com/huggingface/openenv-course  
> **Discord:** https://discord.gg/Dedhy5pkWD

---

## What Is This?

India's biggest AI hackathon, hosted by **Scaler School of Technology**, sponsored by **Meta** and **Hugging Face**. You build a real-world OpenEnv environment — a standardized, containerized AI training/evaluation environment using Meta's OpenEnv framework.

**OpenEnv** is Meta & Hugging Face's open-source framework for creating isolated, reusable environments for training and deploying AI agents. Think of it as the universal standard for AI training environments — like OpenAI Gym, but for real-world agentic tasks. It uses a Gymnasium-style API (`step / reset / state`), runs in Docker containers, and environments are hosted on Hugging Face Spaces.

---

## Key Dates

| Event                                        | Date                       |
| -------------------------------------------- | -------------------------- |
| Registration closes                          | **Friday, 3rd April 2026** |
| Round 1 (build phase)                        | **25th March – 8th April** |
| Round 1 results                              | **10th April**             |
| Advanced RL Bootcamp (online, for finalists) | **18th–19th April**        |
| Grand Finale (48hr, in-person, Bangalore)    | **25th–26th April**        |

---

## Prizes

| Position       | Prize                              |
| -------------- | ---------------------------------- |
| 🥇 1st         | Major prize (exact amount on site) |
| 🥈 2nd         | Major prize                        |
| 🥉 3rd         | Prize                              |
| 4th–8th        | $2,000 each                        |
| 9th–15th       | $650 each                          |
| **Total pool** | **$30,000**                        |

**Bonus for finalists:** Direct interview opportunity at Meta & Hugging Face AI teams. Code reviewed by Meta engineers. Exclusive merch. All meals covered during the Bangalore finale.

---

## Who Can Participate

- Developers, ML engineers, CS students in **India**
- Solo or teams of **up to 3 members** (can be from different colleges/companies)
- **No prior RL experience required** — free prep courses are provided
- Pre-requisites: basic Python, some ML familiarity, comfortable with GitHub

---

## What You're Building (The Task)

Build a **complete, real-world OpenEnv environment** that an AI agent can learn from. Not a game. Not a toy. A simulation of something humans actually do.

**Examples of valid domains:** email triage, code review, data cleaning, scheduling, customer support, content moderation, bug report classification, document review.

**Examples from the site:** Autonomous Traffic Control, Customer Service Agents, Email Triage System.

---

## Core Technical Requirements

### The OpenEnv API You Must Implement

Your environment must expose three methods:

- **`reset()`** — returns the initial observation (start of episode, clean state)
- **`step(action)`** — takes an action, returns observation + reward + done flag + info
- **`state()`** — returns current internal state (for debugging/logging)

All inputs/outputs must use **typed Pydantic models** for `Observation`, `Action`, and `Reward`.

### The openenv.yaml

A metadata file at the root of your project describing your environment, its tasks, and version info. Required for `openenv validate` to pass.

### 3 Tasks with Graders

- Easy, medium, and hard tasks — each with a **programmatic grader**
- Graders score **0.0 to 1.0**, must be **deterministic** (same input → same output always)
- Hard task should genuinely challenge frontier-level models

### Reward Function

Must give **partial credit** throughout the episode — not just a binary score at the end. Should also penalize bad behavior (loops, invalid actions, etc.).

### inference.py (Baseline Script)

- **Must be named `inference.py`** and placed in the **root directory**
- Must use the **OpenAI Client** for all LLM calls
- Reads credentials from environment variables — never hardcoded
- Runs all 3 tasks and produces scores
- Must finish in **under 20 minutes** on a 2 vCPU / 8 GB RAM machine

### Required Environment Variables

| Variable       | Purpose                          |
| -------------- | -------------------------------- |
| `API_BASE_URL` | Base URL of the LLM API endpoint |
| `MODEL_NAME`   | Model identifier for inference   |
| `HF_TOKEN`     | Hugging Face / API key           |

### Deployment

- Must deploy to a **Hugging Face Space** tagged with `openenv`
- Must include a working **Dockerfile** (`docker build` + `docker run` must work)
- Space must respond with HTTP 200 and handle `reset()` calls

### README

Must include: environment description & motivation, observation space, action space, all 3 task descriptions with difficulty, setup instructions, usage instructions, baseline scores.

---

## Evaluation Criteria

| Criterion                      | Weight  | What They're Looking For                                                 |
| ------------------------------ | ------- | ------------------------------------------------------------------------ |
| Real-world utility             | **30%** | Does it model a genuine task? Would someone actually use this?           |
| Task & grader quality          | **25%** | Well-defined objectives, fair graders, meaningful difficulty range       |
| Environment design             | **20%** | Clean state management, good reward shaping, sensible episode boundaries |
| Code quality & spec compliance | **15%** | OpenEnv spec passes, Docker works, typed models, clean code              |
| Creativity & novelty           | **10%** | Novel domain, interesting reward design, original approach               |

### Real-world Utility Scoring (30 pts)

- 0–5: Toy problem, no real application
- 6–15: Valid domain but shallow modeling
- 16–25: Good modeling, useful for agent evaluation
- **26–30: Fills a real gap, immediate value for the RL/agent community**

---

## Judging Phases

**Phase 1 — Automated Validation (Pass/Fail gate)**  
HF Space must deploy and respond, Dockerfile must build, `openenv validate` must pass, baseline script must run, 3+ tasks with valid graders.

**Phase 2 — Agentic Evaluation (Scored)**  
Judges re-run your baseline agent. A standard open LLM (e.g., Nemotron 3 Super) is run against your environment. Score variance check to catch exploits.

**Phase 3 — Human Review**  
Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and grader fairness.

---

## Disqualification — Automatic

- Environment doesn't deploy or respond
- Plagiarized or trivially copied environment
- Graders that always return the same score
- No `inference.py` in the root directory

---

## Pre-Submission Checklist

- [ ] `openenv validate` passes
- [ ] `openenv.yaml` is present and complete
- [ ] Typed models for Observation, Action, Reward defined
- [ ] `step()`, `reset()`, `state()` all implemented
- [ ] 3 tasks defined (easy → medium → hard)
- [ ] All graders return scores in [0.0, 1.0] and are deterministic
- [ ] Reward function gives partial credit, not just binary
- [ ] `inference.py` is in root directory
- [ ] inference.py uses OpenAI Client for all LLM calls
- [ ] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` read from env vars
- [ ] Baseline script finishes in < 20 minutes
- [ ] Baseline scores are in README
- [ ] Dockerfile builds and runs cleanly
- [ ] HF Space is live, tagged `openenv`, returns HTTP 200
- [ ] README covers all required sections

---

## Resources & Links

| Resource                      | Link                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| Hackathon page                | https://www.scaler.com/school-of-technology/meta-pytorch-hackathon                                      |
| OpenEnv GitHub (Meta-PyTorch) | https://github.com/meta-pytorch/OpenEnv                                                                 |
| HuggingFace OpenEnv Course    | https://github.com/huggingface/openenv-course                                                           |
| OpenEnv Docs                  | https://meta-pytorch.org/OpenEnv/                                                                       |
| OpenEnv on PyPI               | https://pypi.org/project/openenv/                                                                       |
| TRL integration example       | https://huggingface.co/docs/trl/openenv                                                                 |
| OpenEnv Colab tutorial        | https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb |
| Discord community             | https://discord.gg/Dedhy5pkWD                                                                           |
| Register                      | https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/register                             |

---

## Quick Notes on OpenEnv CLI

```
pip install openenv-core

openenv init my_env        # scaffold a new environment
openenv validate           # validate before submitting
openenv push               # deploy to Hugging Face Spaces
```

The `openenv init` command creates the full folder structure: models, server, client, Dockerfile, README, openenv.yaml — you fill in the logic.

---

## Support Available

- Free prep courses from Hugging Face & PyTorch (accessible after registration)
- Meta-led deep dive sessions on OpenEnv before each round
- Weekend online bootcamp for Round 2 finalists (18–19 April)
- Direct access to Meta engineers during the Bangalore finale
- Discord community from day one
