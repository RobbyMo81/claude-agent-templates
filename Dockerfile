# Multi-stage build for Agent Army
# Stage 1: Builder

FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies (without dev dependencies for production)
RUN poetry install --no-dev --no-root

# =============================================================================
# Stage 2: Runtime
# =============================================================================

FROM python:3.12-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r agent && useradd -r -g agent agent

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ /app/src/
COPY api/ /app/api/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Change ownership to non-root user
RUN chown -R agent:agent /app

# Switch to non-root user
USER agent

# Expose port
EXPOSE 8000

# Run application with Gunicorn
CMD ["gunicorn", \
     "src.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]

# =============================================================================
# Build Instructions
# =============================================================================
# Build:
#   docker build -t agent-army:latest .
#
# Build with build args:
#   docker build --build-arg APP_VERSION=1.0.0 -t agent-army:1.0.0 .
#
# Run:
#   docker run -p 8000:8000 \
#     -e DATABASE_URL=postgresql://... \
#     -e REDIS_URL=redis://... \
#     -e ANTHROPIC_API_KEY=sk-... \
#     agent-army:latest
#
# Multi-platform build (ARM64 + AMD64):
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t ghcr.io/robbym081/claude-agent-templates:latest \
#     --push .
