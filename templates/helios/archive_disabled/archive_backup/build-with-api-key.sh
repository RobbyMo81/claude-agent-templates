#!/bin/bash

# Helios Docker Build Script with API Key Support
# Usage: ./build-with-api-key.sh

set -e

echo " Building Helios with API Key Integration..."

# Check if .env.local exists and has API key
if [ -f ".env.local" ]; then
    echo " Found .env.local file"
    
    # Extract API key from .env.local
    API_KEY=$(grep "VITE_GEMINI_API_KEY=" .env.local | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    
    if [ "$API_KEY" = "PLACEHOLDER_API_KEY" ] || [ -z "$API_KEY" ]; then
        echo "  WARNING: Using placeholder API key. Gemini features will be disabled."
        echo "   To fix: Add your real Gemini API key to .env.local"
        API_KEY="PLACEHOLDER_API_KEY"
    else
        echo " Found valid API key (${API_KEY:0:10}...)"
    fi
else
    echo "  No .env.local found. Creating from template..."
    cp .env.example .env.local
    API_KEY="PLACEHOLDER_API_KEY"
fi

# Build Docker image with API key as build argument
echo " Building Docker image..."
docker build \
    --build-arg VITE_GEMINI_API_KEY="$API_KEY" \
    --build-arg VITE_API_HOST="localhost" \
    --build-arg VITE_API_PORT="8080" \
    --build-arg VITE_API_PROTOCOL="http" \
    -t helios-image .

echo " Build complete!"
echo ""
echo " To run the container:"
echo "   docker run -d --name helios-container -p 3000:8080 helios-image"
echo ""
echo " Then open: http://localhost:3000"
