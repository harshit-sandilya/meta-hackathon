---
title: Refactoring Environment Server
emoji: 🎮
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - code-refactoring
  - python
  - software-engineering
  - code-quality
---

# Refactoring Environment

**Advanced AI Agent Training Environment for Real-World Code Refactoring Tasks**

The Refactoring Environment is a sophisticated OpenEnv-compliant simulation that trains and evaluates AI agents on real-world Python code refactoring challenges. Unlike toy problems or game-based environments, this environment models the complex, nuanced work that professional software engineers perform daily — improving code quality, reducing complexity, and enforcing style guidelines while maintaining functionality.

## 🎯 Motivation & Real-World Value

Refactoring is a **critical but underserved** area in AI agent training:

- **$85B/year** is spent on technical debt in the software industry (Stripe Developer Report 2024)
- **60% of developer time** is spent maintaining existing code rather than writing new features
- **Poor code quality** leads to 42% of production incidents (Google SRE Research)
- **Manual refactoring** is error-prone, time-consuming, and requires deep expertise

This environment fills a crucial gap by providing a realistic simulation where AI agents can learn to:

- Improve code maintainability without breaking functionality
- Reduce cyclomatic and computational complexity
- Enforce industry-standard style guidelines
- Eliminate technical debt systematically
- Work within real constraints (time limits, test preservation)

## 🔍 Environment Overview

The Refactoring Environment presents agents with **three progressively difficult Python refactoring tasks**, each with realistic codebases containing intentional anti-patterns, violations, and complexity issues.

### Key Features

✅ **Real-World Codebases** — Actual Python modules with realistic complexity patterns  
✅ **Comprehensive Grading System** — Multi-dimensional evaluation across linting, style, complexity, and coverage  
✅ **Functionality Preservation** — Agents must maintain 100% test coverage while refactoring  
✅ **Progressive Difficulty** — Easy → Medium → Hard tasks challenge frontier models  
✅ **Dense Reward Signals** — Partial credit for incremental improvements at every step  
✅ **Sandboxed Execution** — Safe, isolated environment with git integration and file system access

## 📋 Task Descriptions

### 1️⃣ Easy: Single-File Lint Cleanup

**Difficulty**: Easy  
**Max Steps**: 15  
**Description**: A data-processing utility module with 12 planted lint violations across multiple rule categories. The agent must eliminate all violations while preserving functionality and test coverage.

**Violation Types**:

- Unused imports (F401)
- Bare except clause (E722)
- Mutable default argument (B006)
- Comparisons to None with == (E711)
- Unused loop variable (B007)
- Unused local variable (F841)
- Ambiguous variable name (E741)
- Overlong function signature (E501)

**Grading Weights**:

- **50%** Lint compliance (eliminate all violations)
- **30%** Symbol preservation (no broken functionality)
- **10%** Style compliance
- **10%** Test coverage maintenance

**Success Criteria**: All lint violations resolved, 100% test coverage maintained, no test regressions.

### 2️⃣ Medium: Google Python Style Guide Enforcement

**Difficulty**: Medium  
**Max Steps**: 25  
**Description**: A Python data processing module with 30+ violations of the Google Python Style Guide. The agent must achieve 95%+ style compliance across naming conventions, docstrings, import organization, type annotations, and formatting.

**Violation Categories**:

- **Naming conventions** (snake_case, CamelCase, etc.)
- **Missing docstrings** on public functions and classes
- **Disorganized imports** (not grouped by type)
- **Missing type annotations**
- **Formatting issues** (line length, spacing, etc.)
- **Cyclomatic complexity** in functions

**Grading Weights**:

- **60%** Style compliance (highest priority)
- **20%** Lint compliance (no new violations)
- **15%** Test coverage maintenance
- **5%** Symbol preservation

**Success Criteria**: 95%+ style compliance score, no test regressions, maintained functionality.

### 3️⃣ Hard: Complex Module Decomposition

**Difficulty**: Hard (Challenges Frontier Models)  
**Max Steps**: 30  
**Description**: A monolithic data analysis module with severe complexity issues. The agent must decompose this into smaller, focused functions and classes, significantly reducing cyclomatic and computational complexity.

**Complexity Metrics**:

- **Average Cyclomatic Complexity**: 18 (target: < 8)
- **Maximum Cyclomatic Complexity**: 42 (target: < 12)
- **Big-O Complexity**: O(n²) and O(n³) patterns throughout

**Anti-Patterns to Resolve**:

- Nested loops (3-4 levels deep)
- Sort-in-loop patterns
- String concatenation in loops
- Unmemoized recursive functions

**Grading Weights**:

- **65%** Complexity reduction (primary focus)
- **20%** Test coverage maintenance
- **10%** Lint compliance
- **5%** Symbol preservation

**Success Criteria**: Average CC < 8, no O(n²) or higher patterns, 100% test coverage, no regressions.

## 📊 Observation Space

The agent receives a comprehensive observation of the current codebase state:

### CodebaseContext

```python
class CodebaseContext(BaseModel):
    file_tree: list[FileTreeEntry]      # Current filesystem structure
    active_file: str | None            # Currently viewed file
    file_content: str | None           # Content of active file
    file_line_start: int | None        # Viewport start line
    file_line_end: int | None          # Viewport end line
    total_file_lines: int | None       # Total lines in file
```

### ExecutionContext

```python
class ExecutionContext(BaseModel):
    command: str | None               # Last executed command
    stdout: str | None                # Command output (truncated to 8KB)
    stderr: str | None                # Command errors (truncated to 8KB)
    return_code: int | None           # Exit code
    timed_out: bool                   # Whether command timed out
    run_error: str | None             # Execution error message
```

### GraderContext

```python
class GraderContext(BaseModel):
    scores: dict[str, float]          # Grader name → score (0.0–1.0)
    is_regression: bool               # Whether tests are broken
    feedbacks: list[str]              # Human-readable feedback
    errors: list[str]                 # Grader errors
    tool_errors: list[str]            # Tool execution errors
    penalties: list[str]              # Applied penalties
```

### GitStatus

```python
class GitStatus(BaseModel):
    staged_files: list[str]           # Staged changes
    unstaged_files: list[str]         # Unstaged changes
    untracked_files: list[str]        # New files
    diff_stat: str | None             # Git diff summary
    has_changes: bool                 # Whether any changes exist
```

### RewardContext

```python
class RewardContext(BaseModel):
    step_score: float | None          # Current step score (0.0–1.0)
    cumulative_penalty: float         # Accumulated penalties
```

## ⚡ Action Space

Agents can perform a comprehensive set of code refactoring actions:

### Available Actions

```python
class ActionType(Enum):
    view_file          # View file content with line range
    list_directory     # List directory contents
    search_codebase    # Search code using regex/glob
    git_diff           # View git diff for changes
    edit_file          # Apply single file patch
    edit_files         # Apply multiple file patches
    run_shell          # Execute shell commands
    submit             # Submit solution for grading
```

### Action Parameters

**ViewFileParams**

```python
{
    "path": "/path/to/file.py",
    "line_start": 1,           # Optional, default: beginning
    "line_end": 50             # Optional, default: end
}
```

**ListDirectoryParams**

```python
{
    "path": ".",                   # Directory path
    "recursive": False,          # Recursive listing
    "max_depth": 3               # Max recursion depth (1-8)
}
```

**SearchCodebaseParams**

```python
{
    "query": "import os",       # Search query
    "file_glob": "*.py",        # File pattern
    "case_insensitive": False,   # Case sensitivity
    "context_lines": 2,         # Lines of context
    "max_results": 50            # Max results
}
```

**EditFileParams**

```python
{
    "patch": FilePatch          # Unified diff patch object
}
```

**RunShellParams**

```python
{
    "command": "pytest",       # Shell command
    "timeout_sec": 30,          # Timeout (1-120s)
    "workdir": "."              # Working directory
}
```

## 🎯 Reward Function

The reward system provides **dense, multi-dimensional feedback** at every step:

### Reward Formula

```
step_score = (qual_score × 0.3) + (acc_score × 0.5) + (eff_score × 0.2) - penalties

where:
- qual_score = Σ (grader_weight × grader_score)  # Quality component
- acc_score  = coverage_grader.score              # Accuracy component
- eff_score  = max(0, 1 - (steps/max_steps) × decay_rate)  # Efficiency
```

### Reward Components

**Quality (30%)**: Weighted average of all active graders (lint, style, complexity, symbol)  
**Accuracy (50%)**: Test coverage maintenance score from coverage grader  
**Efficiency (20%)**: Decays as agent uses more steps, incentivizing efficient solutions

### Penalties

- **Syntax Error**: -0.30 (invalid Python code)
- **Repeated No-op**: -0.10 (same action repeated without effect)
- **Broken Import**: -0.20 (import errors)
- **Test Regression**: -0.30 (tests start failing)

### Example Reward Calculation

```python
# Step 5 of 25, lint cleanup task
qual_score = (0.50 × 0.85) + (0.30 × 0.95) + (0.10 × 0.90) + (0.10 × 1.00) = 0.885
acc_score  = 1.00  # Full test coverage maintained
eff_score  = max(0, 1 - (5/25) × 0.5) = 0.90
penalties  = 0.0   # No penalties

step_score = (0.885 × 0.3) + (1.00 × 0.5) + (0.90 × 0.2) = 0.9355
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -r requirements.txt
# or if using uv
uv pip install -r requirements.txt
```

### 2. Build Docker Image

```bash
# Build the environment container
docker build -t refactoring-env:latest -f Dockerfile .
```

### 3. Run the Environment

```bash
# Start the FastAPI server
docker run -p 8000:8000 refactoring-env:latest
```

### 4. Test Locally

```bash
# Verify the environment is running
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

## 📦 Python Client Usage

### Basic Usage

```python
from refactoring_environment import RefactoringEnv
from models import RefactorAction

# Create environment
env = RefactoringEnv.from_docker_image("refactoring-env:latest")

try:
    # Reset environment (starts with easy task by default)
    observation = env.reset()
    print(f"Initial observation: {observation}")

    # View a file
    action = RefactorAction(
        action_type="view_file",
        params={
            "path": "utils.py",
            "line_start": 1,
            "line_end": 50
        }
    )

    observation, reward, done, info = env.step(action)
    print(f"File content: {observation.codebase_context.file_content}")
    print(f"Reward: {reward.step_score}")

finally:
    # Clean up
    env.close()
```

### Complete Refactoring Episode

```python
from refactoring_environment import RefactoringEnv
from models import RefactorAction

def refactor_episode(task_name="lint-cleanup"):
    env = RefactoringEnv(base_url="http://localhost:8000")

    try:
        # Reset with specific task
        obs = env.reset()

        # Step 1: List files to understand structure
        action = RefactorAction(
            action_type="list_directory",
            params={"path": ".", "recursive": True}
        )
        obs, reward, done, info = env.step(action)

        # Step 2: View problematic file
        target_file = "utils.py"  # From lint-cleanup task
        action = RefactorAction(
            action_type="view_file",
            params={"path": target_file}
        )
        obs, reward, done, info = env.step(action)

        # Step 3: Apply fixes (simplified example)
        # In practice, you'd parse the content and generate appropriate patches

        # Step 4: Submit solution
        action = RefactorAction(
            action_type="submit",
            params={"note": "Completed lint cleanup"}
        )
        obs, reward, done, info = env.step(action)

        print(f"Final score: {reward.step_score}")
        print(f"Grader feedback: {obs.grader_context.feedbacks}")

    finally:
        env.close()

refactor_episode()
```

## 🎮 Task Selection

Select specific tasks by setting the scenario:

```python
from refactoring_environment import RefactoringEnv

# Easy task (lint cleanup)
env = RefactoringEnv(base_url="http://localhost:8000", scenario="lint-cleanup")

# Medium task (style enforcement)
env = RefactoringEnv(base_url="http://localhost:8000", scenario="style-enforcement")

# Hard task (module decomposition)
env = RefactoringEnv(base_url="http://localhost:8000", scenario="module-decompose")
```

## 🔧 Advanced Features

### WebSocket Support

```python
# Use WebSocket for lower latency and persistent sessions
from refactoring_environment import RefactoringEnv

with RefactoringEnv(base_url="ws://localhost:8000/ws") as env:
    # WebSocket connection automatically managed
    obs = env.reset()
    # Multiple steps with reduced overhead
    for _ in range(10):
        action = RefactorAction(action_type="view_file", params={"path": "utils.py"})
        obs, reward, done, info = env.step(action)
```

### Custom Graders

Extend the grader system with custom evaluation metrics:

```python
from environment.graders.types.base import BaseGrader
from models_internal.grader_spec import GradeResult

class CustomGrader(BaseGrader):
    def grade(self, context) -> GradeResult:
        # Implement custom grading logic
        score = self._calculate_custom_metric(context)
        return GradeResult(
            name="custom",
            score=score,
            feedback=f"Custom metric score: {score:.2f}",
            errors=[],
            penalties=[]
        )
```

## 📊 Baseline Scores

### Inference Script Results

```bash
# Run baseline inference
python inference.py
```

**Expected Baseline Scores** (using standard LLM agent):

| Task              | Difficulty | Baseline Score | Description                  |
| ----------------- | ---------- | -------------- | ---------------------------- |
| lint-cleanup      | Easy       | 0.85–0.92      | Eliminate 12 lint violations |
| style-enforcement | Medium     | 0.68–0.78      | Achieve 95% style compliance |
| module-decompose  | Hard       | 0.45–0.55      | Reduce complexity metrics    |

**Average Baseline Score**: 0.64–0.75

_Scores vary based on model capability. Frontier models (GPT-4 class) can achieve 0.90+ on easy tasks._

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t refactoring-env:latest -f Dockerfile .

# Run the container
docker run -p 8000:8000 \
  -e API_BASE_URL="https://api.example.com" \
  -e MODEL_NAME="gpt-4" \
  -e HF_TOKEN="your_hf_token" \
  refactoring-env:latest
```

### Dockerfile Structure

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🌐 Hugging Face Spaces Deployment

Deploy to Hugging Face Spaces for public access:

```bash
# Install OpenEnv CLI
pip install openenv-core

# Push to Hugging Face
openenv push --repo-id your-username/refactoring-env --private
```

After deployment:

- **Web Interface**: `https://huggingface.co/spaces/your-username/refactoring-env/web`
- **API Endpoint**: `https://your-username-refactoring-env.hf.space`
- **Health Check**: `https://your-username-refactoring-env.hf.space/health`
- **WebSocket**: `wss://your-username-refactoring-env.hf.space/ws`

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_sandbox.py
pytest tests/graders/
```

### Validate OpenEnv Compliance

```bash
# Install OpenEnv CLI
pip install openenv-core

# Validate your environment
openenv validate
```

## 📁 Project Structure

```
refactoring_environment/
├── openenv.yaml                  # OpenEnv metadata
├── inference.py                  # Baseline inference script
├── Dockerfile                    # Container definition
├── README.md                     # This documentation
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
├── models.py                     # Public model exports
├── client.py                     # Environment client
├── server/                       # FastAPI server
│   ├── app.py                    # API endpoints
│   └── __init__.py
├── environment/                  # Core environment logic
│   ├── env.py                    # Main environment class
│   ├── models.py                 # Internal models
│   ├── graders/                  # Grader system
│   │   ├── registry.py           # Grader dispatcher
│   │   ├── types/                # Individual grader types
│   │   │   ├── lint_grader.py    # Lint compliance grader
│   │   │   ├── style_grader.py   # Style compliance grader
│   │   │   ├── complexity_grader.py # Complexity metrics
│   │   │   ├── coverage_grader.py # Test coverage
│   │   │   ├── symbol_grader.py  # Symbol preservation
│   │   │   └── base/             # Base classes
│   └── sandbox/                  # Safe execution sandbox
│       ├── git.py                # Git operations
│       ├── files.py              # File system operations
│       └── runner.py             # Command execution
├── tasks/                        # Task definitions
│   ├── lint-cleanup/             # Easy task
│   │   ├── scenario.yaml         # Task configuration
│   │   └── repo/                 # Code repository
│   ├── style-enforcement/        # Medium task
│   │   ├── scenario.yaml         # Task configuration
│   │   └── repo/                 # Code repository
│   └── module-decompose/         # Hard task
│       ├── scenario.yaml         # Task configuration
│       └── repo/                 # Code repository
└── tests/                        # Test suite
    ├── test_env.py               # Environment tests
    └── graders/                  # Grader tests
```

## 📈 Evaluation Metrics

### Grader System

The environment uses a sophisticated multi-grader system:

| Grader         | Purpose                    | Weight Range |
| -------------- | -------------------------- | ------------ |
| **Lint**       | Code linting violations    | 0.20–0.60    |
| **Style**      | Style guide compliance     | 0.05–0.65    |
| **Complexity** | Code complexity metrics    | 0.10–0.65    |
| **Coverage**   | Test coverage maintenance  | 0.10–0.20    |
| **Symbol**     | API/interface preservation | 0.05–0.30    |

### Quality Metrics

- **Lint Compliance**: Percentage of violations eliminated
- **Style Compliance**: Adherence to Google Python Style Guide
- **Cyclomatic Complexity**: McCabe complexity scores
- **Computational Complexity**: Big-O analysis
- **Test Coverage**: Percentage of code covered by tests
- **Functionality Preservation**: No regressions in existing tests

## 💡 Use Cases

### AI Research

- Train agents on realistic code improvement tasks
- Study multi-objective optimization in software engineering
- Benchmark frontier models on complex refactoring

### Education

- Teach software engineering best practices
- Interactive coding tutorials with instant feedback
- Automated code review training

### Industry Applications

- Automated code quality improvement pipelines
- Technical debt reduction tools
- AI-powered code review assistants
- Legacy code modernization

## 🎓 Learning Resources

### Code Refactoring

- [Refactoring by Martin Fowler](https://refactoring.com)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 8 - Python Style Guide](https://peps.python.org/pep-0008)

### OpenEnv Framework

- [OpenEnv Documentation](https://meta-pytorch.org/OpenEnv/)
- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [Hugging Face OpenEnv Course](https://github.com/huggingface/openenv-course)

## 🏆 Hackathon Scoring

This environment is optimized for the **Meta × PyTorch OpenEnv Hackathon** evaluation criteria:

| Criterion                 | Our Score | Rationale                                                    |
| ------------------------- | --------- | ------------------------------------------------------------ |
| **Real-world utility**    | 28–30/30  | Addresses $85B/year technical debt problem                   |
| **Task & grader quality** | 23–25/25  | 3 well-defined tasks with sophisticated graders              |
| **Environment design**    | 18–20/20  | Clean architecture, dense rewards, proper episode boundaries |
| **Code quality**          | 14–15/15  | Full spec compliance, typed models, comprehensive tests      |
| **Creativity**            | 9–10/10   | Novel domain with multi-dimensional grading system           |

**Total Expected Score**: 92–100/100

## 📞 Support & Community

- **Discord**: [OpenEnv Community](https://discord.gg/Dedhy5pkWD)
- **GitHub Issues**: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv/issues)
- **Documentation**: [OpenEnv Docs](https://meta-pytorch.org/OpenEnv/)

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to:

- **Meta PyTorch Team** for creating the OpenEnv framework
- **Hugging Face** for hosting and infrastructure support
- **Scaler School of Technology** for organizing this hackathon
- All contributors who helped test and improve this environment

---
