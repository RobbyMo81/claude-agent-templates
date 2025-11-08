#!/bin/sh
echo " Starting Helios Unified Server..."
echo " Static files location: /app/static"
echo " Backend location: /app/backend"
echo " Port: ${PORT:-8080}"

cd /app/backend

# Start Gunicorn with custom configuration for serving static files + API
exec gunicorn \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --timeout 300 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    server:app
