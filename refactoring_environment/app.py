"""
FastAPI application for the Refactoring Environment.

This module creates an HTTP server that exposes the RefactorEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m app
"""

import argparse
import os

from openenv.core.env_server.http_server import create_app

from .environment import RefactorEnvironment
from .models_internal import RefactorAction, RefactorObservation

app = create_app(
    RefactorEnvironment,
    RefactorAction,
    RefactorObservation,
    env_name="refactoring_environment",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", 4)),
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m refactoring_environment.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn refactoring_environment.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
