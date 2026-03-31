.
├── inference.py
├── requirements.txt
│
├── docs/
│
├── tests/
│ ├── **init**.py
│ ├── test_env_contract.py
│ ├── test_registry.py
│ ├── test_executor.py
│ ├── test_graders.py
│ └── test_reward.py
│
└── refactor_env/
├── README.md
├── **init**.py
├── openenv.yaml
├── pyproject.toml
├── uv.lock
│
├── models.py ← Public Pydantic contract
├── client.py ← HTTP client wrapper
│
├── environment/ ← The core engine
│ ├── **init**.py ← ONLY exports: reset(), step(), state()
│ ├── core/
│ │ ├── **init**.py
│ │ ├── environment.py ← RefactorEnvironment orchestrator
│ │ ├── sandbox.py ← Git-backed episode workspace
│ │ ├── executor.py ← Safe subprocess runner
│ │ ├── metrics.py ← pytest, ruff, radon runners
│ │ └── reward.py ← Pure compute_reward() function
│ └── registry/
│ ├── **init**.py
│ ├── task_registry.py ← Slug → Scenario loader
│ ├── scenario_loader.py ← scenario.yaml parser + validator
│ └── graders/
│ ├── **init**.py ← Explicit registry dict
│ ├── base_grader.py
│ ├── lint_grader.py
│ ├── symbol_grader.py
│ ├── coverage_grader.py
│ ├── structure_grader.py
│ └── style_grader.py
│
├── tasks/
│ ├── lint-cleanup/
│ │ ├── scenario.yaml
│ │ └── repo/
│ ├── api-rename/
│ │ ├── scenario.yaml
│ │ └── repo/
│ ├── test-coverage/
│ │ ├── scenario.yaml
│ │ └── repo/
│ ├── module-decompose/
│ │ ├── scenario.yaml
│ │ └── repo/
│ └── style-enforce/
│ ├── scenario.yaml
│ └── repo/
│
└── server/
├── **init**.py
├── Dockerfile
├── requirements.txt
├── app.py
└── environment.py ← Thin OpenEnv adapter
