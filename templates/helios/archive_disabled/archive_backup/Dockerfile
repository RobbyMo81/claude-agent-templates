# ==============================================================================
# HELIOS UNIFIED DOCKERFILE - Frontend + Backend in Single Container
# ==============================================================================
# This Dockerfile builds both the React frontend and Python backend into 
# a single production container for Google Cloud Run deployment.
# ==============================================================================

# ==============================================================================
# STAGE 1: Frontend Builder
# ==============================================================================
FROM node:22.13.1-alpine3.21 AS frontend-builder

WORKDIR /app

# Install security updates
RUN apk update && apk upgrade && apk add --no-cache dumb-init

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install && npm audit fix

# Copy frontend source code including environment files
COPY . .

# Build arguments for API - should match runtime port
ARG VITE_GEMINI_API_KEY
ARG VITE_API_HOST=localhost
ARG VITE_API_PORT=8080  # ← Changed from 5001 to 8080
ARG VITE_API_PROTOCOL=http

# Set environment variables for Vite build
ENV VITE_GEMINI_API_KEY=$VITE_GEMINI_API_KEY
ENV VITE_API_HOST=$VITE_API_HOST  
ENV VITE_API_PORT=$VITE_API_PORT
ENV VITE_API_PROTOCOL=$VITE_API_PROTOCOL

# Build static files with environment variables baked in
RUN npm run build

# ==============================================================================
# STAGE 2: Production Container with Python + Frontend
# ==============================================================================
FROM python:3.11-slim

# Install system dependencies for Python packages and security updates
RUN apt-get update && apt-get install -y \
        gcc \
        build-essential \
        curl \
        dumb-init \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create directories for both frontend and backend
RUN mkdir -p /app/backend /app/static /app/data

# Copy Python requirements and install dependencies
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy Python backend code
COPY backend/ /app/backend/

# Copy built frontend files from builder stage
COPY --from=frontend-builder /app/dist /app/static

# Copy startup script
COPY start-server.sh /app/start-server.sh

# Create non-root user for security (Debian style)
RUN groupadd -g 1001 helios && \
    useradd -r -u 1001 -g helios helios && \
    chown -R helios:helios /app

# Make startup script executable
RUN chmod +x /app/start-server.sh

# Switch to non-root user
USER helios

# Expose the port (Cloud Run will override with PORT env var)
EXPOSE 8080

# Health check for both frontend and backend
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/ && \
        curl -f http://localhost:${PORT:-8080}/api/health || exit 1

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Start the unified server
CMD ["/app/start-server.sh"]
