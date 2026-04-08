ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=refactoring_environment

# Copy environment code (always at root of build context)
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies (excluding project itself first for better layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-install-project --no-editable; \
    else \
    uv sync --no-install-project --no-editable; \
    fi

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-editable; \
    else \
    uv sync --no-editable; \
    fi


# ── Final runtime stage ────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

# app.py lives at the repo root, which was copied to /app/env
WORKDIR /app/env

# Copy the virtual environment built by the builder stage
COPY --from=builder /app/env/.venv /app/env/.venv

# Copy the full environment/application code
COPY --from=builder /app/env /app/env

# Activate the venv
ENV PATH="/app/env/.venv/bin:$PATH"

# Make all packages importable (environment/, models/, tasks/, etc.)
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# app.py is at the root of the repo → /app/env/app.py → module name is just `app`
CMD ["uv", "run", "server"]