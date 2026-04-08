# OpenEnv: A Complete Implementation Guide

> **OpenEnv** is a framework for building, deploying, and using Reinforcement Learning environments as production-grade microservices. It replaces the traditional in-process environment model (Gymnasium) with isolated, type-safe, HTTP/WebSocket-based services that run in Docker containers.

---

## Table of Contents

1. [The RL Loop & Why OpenEnv Exists](#1-the-rl-loop--why-openenv-exists)
2. [Architecture Overview](#2-architecture-overview)
3. [Using Existing Environments](#3-using-existing-environments)
4. [Deploying Environments](#4-deploying-environments)
5. [Building Your Own Environment](#5-building-your-own-environment)
6. [Training LLMs with OpenEnv + TRL](#6-training-llms-with-openenv--trl)

---

## 1. The RL Loop & Why OpenEnv Exists

### The RL Loop in 60 Seconds

Every Reinforcement Learning system is built on a single loop:

```python
while not done:
    observation = environment.observe()
    action = policy.choose(observation)
    reward = environment.step(action)
    policy.learn(reward)
```

Observe → Act → Reward → Repeat. The agent interacts with an environment, receives feedback, and improves over time. This pattern applies everywhere — from classic game-playing bots to fine-tuning LLMs with GRPO.

### Why Gymnasium Falls Short for Production

OpenAI Gym (now Gymnasium) is the standard for RL research and works well for toy problems like CartPole. However, when you try to use it for production-grade LLM training, several fundamental problems emerge:

| Challenge       | Gymnasium                         | What you actually need             |
| --------------- | --------------------------------- | ---------------------------------- |
| **Type Safety** | `obs[0][3]` — what is this?       | `obs.info_state` — IDE knows       |
| **Isolation**   | Same process (can crash training) | Docker containers (fully isolated) |
| **Deployment**  | "Works on my machine"             | Same container everywhere          |
| **Scaling**     | Hard to distribute                | Deploy to Kubernetes               |
| **Language**    | Python only                       | Any language via HTTP              |
| **Debugging**   | Cryptic numpy errors              | Clear type errors                  |

The root cause of all these issues: Gymnasium assumes your environment runs **in the same process** as your training code. That's acceptable for research; it's a liability in production.

### The OpenEnv Philosophy

**RL environments should be microservices.**

Just as you don't run your database in the same process as your web server, you shouldn't run your RL environment in the same process as your training loop. OpenEnv enforces this separation:

- **Isolated** — Runs in containers for security and stability
- **Standard** — HTTP/WebSocket API accessible from any language
- **Versioned** — Docker images make every deployment fully reproducible
- **Scalable** — Deploy to Kubernetes or Hugging Face Spaces with a single command
- **Type-safe** — Pydantic models catch bugs before they crash your training run

---

## 2. Architecture Overview

### System Diagram

```
┌────────────────────────────────────────────────────────────┐
│  YOUR TRAINING CODE                                        │
│                                                            │
│  env = EchoEnv(base_url="https://...")                     │
│  result = env.reset()           ← Type-safe!               │
│  result = env.step(action)      ← Type-safe!               │
│                                                            │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  │  WebSocket / HTTP  (Language-Agnostic)
                  │
┌─────────────────▼──────────────────────────────────────────┐
│  DOCKER CONTAINER (HF Space, local, cloud)                 │
│                                                            │
│  ┌──────────────────────────────────────────────┐          │
│  │  FastAPI Server                              │          │
│  │  └─ Environment (reset, step, state)         │          │
│  │     └─ Your Game/Simulation Logic            │          │
│  └──────────────────────────────────────────────┘          │
│                                                            │
│  Isolated • Reproducible • Secure                          │
└────────────────────────────────────────────────────────────┘
```

Your training code communicates with the environment container over WebSockets. The client library abstracts this entirely — you just call Python methods:

```python
env.reset()    # Under the hood: WebSocket message
env.step(...)  # Under the hood: WebSocket message
env.state()    # Under the hood: WebSocket message
```

### The 3-Method Interface

Every OpenEnv environment exposes exactly three methods, regardless of what the environment does internally:

| Method         | What it does         | Returns                                  |
| -------------- | -------------------- | ---------------------------------------- |
| `reset()`      | Start a new episode  | `StepResult` (observation, reward, done) |
| `step(action)` | Take an action       | `StepResult` (observation, reward, done) |
| `state()`      | Get episode metadata | `State` (episode_id, step_count, etc.)   |

This uniformity means the same client code works for Catch, Wordle, Tic-Tac-Toe, or any custom environment you build.

### The 3-Component Pattern

Every OpenEnv environment is composed of three parts:

```
my_env/
├── models.py              ← Type-safe contracts (Action, Observation, State)
├── client.py              ← What you import in training code
└── server/
    ├── environment.py     ← Game/simulation logic
    ├── app.py             ← FastAPI server
    └── Dockerfile         ← Container definition
```

**Server side** — the game logic that runs inside Docker:

```python
class Environment(ABC):
    def reset(self) -> Observation: ...
    def step(self, action: Action) -> Observation: ...
    @property
    def state(self) -> State: ...
```

**Client side** — what you import in your training code:

```python
class EnvClient(ABC):
    async def reset(self, **kwargs) -> StepResult: ...
    async def step(self, action) -> StepResult: ...
    async def state(self) -> State: ...
    def sync(self) -> SyncEnvClient: ...  # Sync wrapper for notebooks/scripts
```

The interface is identical on both sides — the only thing between them is a WebSocket connection.

> **Note:** For simple MCP-based environments (like the Echo environment), the interface is tool-based instead: `env.list_tools()` and `env.call_tool(name, **kwargs)`.

---

## 3. Using Existing Environments

### The Environment Hub

OpenEnv environments are hosted on Hugging Face Spaces. The [Environment Hub collection](https://huggingface.co/collections/openenv/environment-hub) provides ready-to-use environments you can connect to immediately — no setup required.

Every Space gives you three access methods out of the box:

| Component      | What it provides             | How to access                                           |
| -------------- | ---------------------------- | ------------------------------------------------------- |
| **Server**     | Running environment endpoint | `https://<username>-<space-name>.hf.space`              |
| **Repository** | Installable Python package   | `pip install git+https://huggingface.co/spaces/<space>` |
| **Registry**   | Docker container image       | `docker pull registry.hf.space/<space>:latest`          |

You don't need to build anything to start using an environment. Install the client package, point it at the server URL, and begin interacting.

### Type-Safe Models

Every OpenEnv environment defines typed Pydantic models for its actions, observations, and state. These are not just documentation — they are runtime-validated Python classes that your IDE can autocomplete and your type checker can validate.

For OpenSpiel environments, the models look like this (note that `done` and `reward` are inherited from the `Observation` base class):

```python
from openenv.core.env_server import Action, Observation, State
from pydantic import Field
from typing import Any, Dict, List, Optional

class OpenSpielAction(Action):
    action_id: int                              # Which action to take
    game_name: str = "catch"                   # Which game
    game_params: Dict[str, Any] = Field(default_factory=dict)  # Game config

class OpenSpielObservation(Observation):
    # done: bool and reward: Optional[float] are inherited from Observation
    info_state: List[float]      # Game state as a vector
    legal_actions: List[int]     # Valid actions this step
    game_phase: str = "playing"  # Current phase
    current_player_id: int = 0   # Whose turn
    opponent_last_action: Optional[int] = None
```

This eliminates the cryptic `obs[0][3]` indexing of raw numpy arrays — you get named fields, IDE autocompletion, and type-level validation.

### Available OpenSpiel Games

OpenEnv wraps 6 games from DeepMind's OpenSpiel library, all sharing the same `OpenSpielEnv` client and the same `OpenSpielAction`/`OpenSpielObservation` types:

| Single-Player                 | Multi-Player                |
| ----------------------------- | --------------------------- |
| Catch — catch falling ball    | Tic-Tac-Toe — classic 3×3   |
| Cliff Walking — navigate grid | Kuhn Poker — imperfect info |
| 2048 — tile puzzle            |                             |
| Blackjack — card game         |                             |

The only difference between games is the `game_name` parameter in your action.

### Writing Policies

A policy is a function that maps an observation to an action. Below are four example policies for the Catch game, ranging from naive to optimal:

**Random** — baseline, ~20% success rate:

```python
def random_policy(obs):
    return random.choice(obs.legal_actions)
```

**Always Stay** — ignores the game state entirely, ~20% success rate:

```python
def stay_policy(obs):
    return 1  # STAY
```

**Smart Heuristic** — reads ball and paddle positions from the observation vector, achieves 100% success:

```python
def smart_policy(obs):
    ball_col = find_ball(obs.info_state)
    paddle_col = find_paddle(obs.info_state)
    if paddle_col < ball_col: return 2  # RIGHT
    if paddle_col > ball_col: return 0  # LEFT
    return 1  # STAY
```

**Epsilon-Greedy** — explores initially, then exploits, reaches ~85% success:

```python
def learning_policy(obs, step):
    epsilon = max(0.1, 1.0 - step / 100)
    if random.random() < epsilon:
        return random.choice(obs.legal_actions)
    return smart_policy(obs)
```

All four policies work against the same `OpenSpielObservation` type. Swapping from Catch to Tic-Tac-Toe doesn't change the observation schema — only the game logic differs.

### Switching Games

Because all OpenSpiel games share the same client interface, switching between them is a one-line change — just update the `base_url`:

```python
# Catch
with OpenSpielEnv(base_url="https://openenv-openspiel-catch.hf.space").sync() as env:
    result = env.reset()

# Tic-Tac-Toe — same client, different URL
with OpenSpielEnv(base_url="https://openenv-openspiel-tictactoe.hf.space").sync() as env:
    result = env.reset()
```

Your policy code requires no changes. The observation has the same fields. You only need a new strategy for the new game's rules.

---

## 4. Deploying Environments

### Local Development with Uvicorn

The fastest development loop is to clone an existing Space and run it locally. The `--reload` flag restarts the server automatically on every code change:

```bash
# Clone from HF Space
git clone https://huggingface.co/spaces/openenv/echo-env
cd echo-env

# Install and run
uv sync
uv run server
```

Alternatively, invoke uvicorn directly:

```bash
uvicorn echo_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

Verify the server is healthy and connect from Python:

```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

```python
with EchoEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
```

### Docker Deployment

Docker provides full process isolation and ensures your environment behaves identically across machines.

**Pull a pre-built image from a Space's registry:**

```bash
docker pull registry.hf.space/openenv-echo-env:latest
docker run -d -p 8000:8000 registry.hf.space/openenv-echo-env:latest
```

**Build from source:**

```bash
git clone https://huggingface.co/spaces/openenv/echo-env
cd echo-env
docker build -t my-echo-env:latest -f server/Dockerfile .
docker run -d -p 8000:8000 my-echo-env:latest
```

**Pass runtime configuration via environment variables:**

```bash
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    my-echo-env:latest
```

### Deploying to Hugging Face Spaces

The `openenv push` command is the fastest path from local code to a live public endpoint:

```bash
cd my_env
openenv push --repo-id username/my-env
```

Once deployed, your environment is available at four URLs:

- **API endpoint:** `https://username-my-env.hf.space`
- **Web UI:** `https://username-my-env.hf.space/web`
- **API docs:** `https://username-my-env.hf.space/docs`
- **Health check:** `https://username-my-env.hf.space/health`

#### The `openenv.yaml` Manifest

This file controls Space metadata and settings:

```yaml
name: my_env
version: "1.0.0"
description: My custom environment
```

#### Environment Variables

Configure the server behaviour via Space Settings → Variables:

| Variable              | Default | Description              |
| --------------------- | ------- | ------------------------ |
| `WORKERS`             | 4       | Uvicorn worker processes |
| `PORT`                | 8000    | Server port              |
| `HOST`                | 0.0.0.0 | Bind address             |
| `MAX_CONCURRENT_ENVS` | 100     | Max WebSocket sessions   |

#### Hardware Tiers

| Tier             | vCPU | RAM  | Cost     |
| ---------------- | ---- | ---- | -------- |
| CPU Basic (Free) | 2    | 16GB | Free     |
| CPU Upgrade      | 8    | 32GB | $0.03/hr |

The free tier supports approximately 128 concurrent WebSocket sessions — sufficient for development and demos.

### Choosing Your Access Method

| Method            | Use when                     | Pros                     | Cons            |
| ----------------- | ---------------------------- | ------------------------ | --------------- |
| **Remote Space**  | Quick testing, low volume    | Zero setup               | Network latency |
| **Local Docker**  | Development, high throughput | Full control, no network | Requires Docker |
| **Local Uvicorn** | Fast iteration               | Fastest reload           | No isolation    |

### The End-to-End Workflow

```
1. openenv init my_env        # Scaffold the project structure
2. Edit server/environment.py # Implement your game/simulation logic
3. uv run server              # Test locally with live reload
4. openenv push               # Deploy to HF Spaces
5. pip install git+https://huggingface.co/spaces/username/my-env  # Install client anywhere
```

---

## 5. Building Your Own Environment

### Project Structure

Every OpenEnv environment follows the same file layout:

```
my_env/
├── models.py              ← Types: Action, Observation, State
├── client.py              ← HTTP/WebSocket client (what users import)
├── server/
│   ├── environment.py     ← Game logic (reset, step, state)
│   ├── app.py             ← FastAPI server
│   └── Dockerfile         ← Container definition
├── openenv.yaml           ← Manifest
└── pyproject.toml         ← Package metadata
```

The following walkthrough builds a complete word-guessing game in approximately 100 lines of meaningful code.

---

### Step 1: Define Your Types (`models.py`)

Start by defining the data contracts — what an action looks like, what an observation contains, and what state the server tracks. These are Pydantic models; no `@dataclass` decorator is needed:

```python
from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

# Action, Observation, State are Pydantic BaseModel subclasses —
# no @dataclass decorator needed; define fields directly as class attributes.

class WordGameAction(Action):
    guess: str  # The player's guessed letter

class WordGameObservation(Observation):
    # done: bool and reward: Optional[float] are already in Observation base
    masked_word: str           # e.g., "h_ll_"
    guessed_letters: List[str] # Letters tried so far
    attempts_remaining: int
    message: str               # Feedback message

class WordGameState(State):
    # episode_id: Optional[str] and step_count: int are already in State base
    target_word: str = ""
    max_attempts: int = 10
```

These models serve three purposes simultaneously:

1. **Document the API** — anyone reading `models.py` immediately understands the interface
2. **Enable IDE autocomplete** — `obs.masked_word` instead of `obs["masked_word"]`
3. **Catch bugs at type-check time** — misspell a field and your linter tells you before runtime

---

### Step 2: Implement the Environment (`server/environment.py`)

This is where your game logic lives. Subclass `Environment` and implement `reset()`, `step()`, and the `state` property:

```python
import random
import uuid
from openenv.core.env_server import Environment
from .models import WordGameAction, WordGameObservation, WordGameState

WORDS = ["python", "neural", "tensor", "matrix", "vector",
         "kernel", "lambda", "signal", "binary", "cipher"]

class WordGameEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True  # Allow multiple simultaneous clients

    MAX_ATTEMPTS = 10

    def __init__(self):
        self._state = WordGameState()
        self._target = ""
        self._guessed = set()
        self._remaining = self.MAX_ATTEMPTS

    def reset(self, seed=None, episode_id=None, **kwargs) -> WordGameObservation:
        self._target = random.choice(WORDS)
        self._guessed = set()
        self._remaining = self.MAX_ATTEMPTS
        self._state = WordGameState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            target_word=self._target,
            max_attempts=self.MAX_ATTEMPTS,
        )
        return WordGameObservation(
            done=False,
            reward=None,
            masked_word=self._mask(),
            guessed_letters=[],
            attempts_remaining=self._remaining,
            message=f"Guess letters in a {len(self._target)}-letter word!",
        )

    def step(self, action: WordGameAction, timeout_s=None, **kwargs) -> WordGameObservation:
        letter = action.guess.lower().strip()
        self._state.step_count += 1
        self._guessed.add(letter)

        if letter in self._target:
            message = f"'{{letter}}' is in the word!"
        else:
            self._remaining -= 1
            message = f"'{{letter}}' is not in the word."

        # Check win/lose
        masked = self._mask()
        won = "_" not in masked
        lost = self._remaining <= 0
        done = won or lost

        if won:
            reward = 1.0
            message = f"You got it! The word was '{{self._target}}'."
        elif lost:
            reward = 0.0
            message = f"Out of attempts. The word was '{{self._target}}'."
        else:
            reward = 0.0

        return WordGameObservation(
            done=done,
            reward=reward,
            masked_word=masked,
            guessed_letters=sorted(self._guessed),
            attempts_remaining=self._remaining,
            message=message,
        )

    @property
    def state(self) -> WordGameState:
        return self._state

    def _mask(self) -> str:
        return "".join(c if c in self._guessed else "_" for c in self._target)
```

---

### Step 3: Create the Client (`client.py`)

The client is the interface your training code imports. It translates between your typed Pydantic models and the raw WebSocket wire format. You only need to implement three methods — the `EnvClient` base class handles all connection management:

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import WordGameAction, WordGameObservation, WordGameState

class WordGameEnv(EnvClient[WordGameAction, WordGameObservation, WordGameState]):
    def _step_payload(self, action: WordGameAction) -> dict:
        return {"guess": action.guess}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=WordGameObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                masked_word=obs_data.get("masked_word", ""),
                guessed_letters=obs_data.get("guessed_letters", []),
                attempts_remaining=obs_data.get("attempts_remaining", 0),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> WordGameState:
        return WordGameState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            target_word=payload.get("target_word", ""),
            max_attempts=payload.get("max_attempts", 6),
        )
```

---

### Step 4: Wire Up the FastAPI Server (`server/app.py`)

The server registration is a single function call. `create_fastapi_app()` automatically generates all required endpoints: `/ws`, `/reset`, `/step`, `/state`, `/health`, `/web`, and `/docs`:

```python
from openenv.core.env_server import create_fastapi_app
from environment import WordGameEnvironment

app = create_fastapi_app(WordGameEnvironment)
```

---

### Step 5: Containerize (`server/Dockerfile`)

A minimal Dockerfile that packages the server into a portable container:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### Using `openenv init` to Scaffold

If you don't want to write the boilerplate by hand, the CLI generates the full project structure with placeholder code:

```bash
openenv init word_game
cd word_game
```

You then fill in:

1. Your types in `models.py`
2. Your game logic in `server/environment.py`
3. Your client parsing in `client.py`

Then test locally and deploy:

```bash
uv run server                          # Test locally
openenv push --repo-id user/word-game  # Deploy to HF Spaces
```

---

## 6. Training LLMs with OpenEnv + TRL

### What is GRPO?

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm designed for fine-tuning language models. The core idea:

1. Generate a **group** of completions for the same prompt
2. Score each completion using one or more reward functions
3. Use the **relative ranking within the group** as the gradient signal — no value model needed

This makes GRPO simpler and more memory-efficient than PPO, and it works well for any task where you can define a reward function: games, code generation, mathematical reasoning, structured output.

### The TRL + OpenEnv Integration

[TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl) provides `GRPOTrainer` with native OpenEnv support. The central abstraction is the **rollout function** — a user-defined function that describes how the model interacts with the environment during training.

The training loop works as follows:

1. `GRPOTrainer` calls your rollout function with a batch of prompts
2. Your function generates completions using the current model
3. Each completion is sent as an action to the OpenEnv environment
4. The environment returns observations and rewards
5. TRL uses those rewards to update the model weights

```python
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_correct, reward_greens, reward_yellows],
    rollout_func=rollout_func,   # Your environment interaction
    train_dataset=dataset,
    args=grpo_config,
)
trainer.train()
```

### Example: Training on Wordle

The following pipeline trains Qwen3-1.7B to play Wordle using the TextArena environment.

#### Connecting to the Environment

```python
from envs.textarena_env import TextArenaEnv

env = TextArenaEnv(base_url="https://burtenshaw-textarena.hf.space")
```

The TextArena Wordle environment accepts guesses in the format `[word]` (a 5-letter word in square brackets) and returns per-letter feedback: **G** (green — correct position), **Y** (yellow — wrong position), **X** (gray — not in word). The episode ends after 6 attempts or a correct guess.

#### System Prompt

The system prompt defines the model's strategy at inference time:

```python
system_prompt = """
You are an expert Wordle solver.

RULES:
- Guess a 5-letter English word
- Feedback: GREEN (correct position), YELLOW (wrong position), GRAY (not in word)
- 6 attempts maximum

RESPONSE FORMAT:
Only respond with your guess in square brackets, e.g., [crane]

STRATEGY:
- Start with vowel-rich words: CRANE, SLATE, STARE
- Use GREEN letters in their positions
- Move YELLOW letters to new positions
- Eliminate GRAY letters
"""
```

#### Reward Functions

Using multiple reward signals gives the model richer gradient information at each step, not just at the end of the episode:

| Reward              | What it measures          | Range      |
| ------------------- | ------------------------- | ---------- |
| `reward_correct`    | Did the model solve it?   | 0.0 or 1.0 |
| `reward_greens`     | How many green letters?   | 0.0 to 1.0 |
| `reward_yellows`    | How many yellow letters?  | 0.0 to 1.0 |
| `reward_repetition` | Penalize repeated guesses | 0.0 to 1.0 |

`reward_greens` and `reward_yellows` provide shaped signal even when the model doesn't win the game. `reward_repetition` discourages the model from guessing the same word multiple times — a common failure mode in early training.

#### The Rollout Function

The rollout function plays one complete Wordle game. It is called once per training example:

```python
def rollout_once(trainer, env, tokenizer, prompt, system_prompt, max_turns):
    result = env.reset()
    observation = result.observation

    for turn in range(max_turns):
        if result.done:
            break

        # Build prompt from game state
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_game_state(observation)},
        ]

        # Generate with the model
        rollout = generate_rollout_completions(trainer, [messages])

        # Parse guess and send to environment
        guess = extract_guess(rollout["text"])
        result = env.step(TextArenaAction(message=guess))
        observation = result.observation

    return {
        "prompt_ids": ..., "completion_ids": ..., "logprobs": ...,
        "correct_reward": ..., "green_reward": ...,
    }
```

#### GRPO Configuration

```python
grpo_config = GRPOConfig(
    num_train_epochs=1,
    learning_rate=5e-6,
    gradient_accumulation_steps=64,
    per_device_train_batch_size=1,
    num_generations=2,
    max_completion_length=8,
    max_prompt_length=1400,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.1,
    gradient_checkpointing=True,
    report_to="trackio",
)
```

Key configuration decisions:

- **`vllm_mode="colocate"`** — generation and training share a single GPU, avoiding the need for a separate inference server
- **`gradient_accumulation_steps=64`** — achieves a large effective batch size without running out of GPU memory
- **`max_completion_length=8`** — Wordle guesses are short; this keeps generation fast

#### Hardware Requirements

| Resource        | Requirement           |
| --------------- | --------------------- |
| GPU             | A100 40GB (or equiv.) |
| Training time   | ~90 minutes           |
| Peak VRAM usage | ~37GB                 |

### What the Model Learns

After training, the model demonstrably improves its Wordle strategy:

- Opens with high-information starter words (CRANE, SLATE)
- Uses green-letter feedback to fix positions in subsequent guesses
- Uses yellow-letter feedback to eliminate wrong positions
- Still occasionally repeats guesses — a common RL challenge that can be addressed with stronger `reward_repetition` weighting or longer training

### Swapping Environments

The key design insight is that the training pipeline is **environment-agnostic**. The `rollout_func` interface is identical regardless of which OpenEnv environment is connected. To train on a different task, you swap the environment URL and update the system prompt — the GRPO training loop requires no changes:

- Replace Wordle with the word game you built in Module 5
- Replace it with a code execution environment
- Replace it with a math problem solver

The OpenEnv client interface remains the same throughout.

---

_Built with OpenEnv — RL environments as microservices._
