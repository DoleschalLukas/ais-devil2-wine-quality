ARG PYTHON_VERSION=3.13

# Builder stage: Install dependencies
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

# Install curl for uv
RUN apt-get update && apt-get install -y curl

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files needed for dependency installation
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies (creates .venv with Python and packages)
RUN uv sync --frozen

# Runtime stage: Clean image with only runtime dependencies
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Copy Python environment and application from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY wine_quality_api.py ./
COPY wine_quality_model.pkl ./

# Set the entrypoint
ENTRYPOINT ["/app/.venv/bin/fastapi", "run", "wine_quality_api.py"]