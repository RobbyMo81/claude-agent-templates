# =============================================================================
# STAGE 1: Builder image to build dependencies
# =============================================================================
FROM python:3.11-slim-bullseye AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .

# Corrected version
# First, upgrade pip in its standard location
RUN python -m pip install --no-cache-dir --upgrade pip && \
    # Second, install all Python dependencies into the separate /install directory
    python -m pip install \
        --no-cache-dir \
        --prefix=/install \
        -r requirements.txt \
        tensorflow-cpu==2.15.0 \
        tensorflow-probability==0.21.0
# =============================================================================
# STAGE 2: Final lightweight image
# =============================================================================
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup --no-create-home appuser

# Copy only the built packages from the builder stage
COPY --from=builder /install /usr/local

# Create directories with proper permissions
RUN mkdir -p data models && \
    chown -R appuser:appgroup /app

# Copy application code with proper ownership
COPY --chown=appuser:appgroup . .

# Switch to non-root user for better security
USER appuser

# Expose port for Streamlit
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Health check to verify application is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1
