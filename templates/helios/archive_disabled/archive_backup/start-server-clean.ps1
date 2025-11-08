#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Helios Unified Server Startup - Direct Mirror of Docker start-server.sh
    
.DESCRIPTION
    This script is the PowerShell equivalent of the Docker start-server.sh.
    It assumes frontend is already built and starts only the backend server
    configured to serve both static files and API endpoints.
    
.PARAMETER Port  
    Port to run the server on (default: 5001, Docker uses 8080)
#>

param(
    [int]$Port = 5001
)

Write-Host "Starting Helios Unified Server..." -ForegroundColor Cyan
Write-Host "Static files location: ../dist (mirrors Docker /app/static)" -ForegroundColor White  
Write-Host "Backend location: backend/" -ForegroundColor White
Write-Host "Port: $Port" -ForegroundColor White
Write-Host ""

# Change to backend directory (mirrors Docker WORKDIR /app/backend)
Set-Location backend

# Activate virtual environment
$venvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating Python virtual environment..." -ForegroundColor Cyan
    & $venvPath
} else {
    Write-Host "Warning: Virtual environment not found at $venvPath" -ForegroundColor Yellow
}


# Set environment variables (mirrors Docker environment)
$env:PORT = $Port.ToString()
$env:FLASK_APP = "server.py"
$env:FLASK_ENV = "production"
$env:HELIOS_PORT = $Port.ToString()
$env:HELIOS_MODE = "production"

# Use virtual environment's Python
$pythonExe = if (Test-Path ".\venv\Scripts\python.exe") { 
    ".\venv\Scripts\python.exe" 
} else { 
    "python" 
}

Write-Host "Using Python: $pythonExe" -ForegroundColor Gray

# Check if Gunicorn is available and compatible (Linux/Docker only)
$gunicornExe = if (Test-Path ".\venv\Scripts\gunicorn.exe") { 
    ".\venv\Scripts\gunicorn.exe" 
} else { 
    $null 
}

# Note: Gunicorn doesn't work on Windows due to fcntl dependency
# But we can simulate the Docker behavior with Flask development server
Write-Host "Starting Flask server (Windows equivalent of Docker Gunicorn)..." -ForegroundColor Green
Write-Host "Note: Gunicorn requires Linux/Unix (used in Docker), Flask dev server used on Windows" -ForegroundColor Gray
Write-Host "Configuration mirrors Docker: bind=0.0.0.0:$Port, debug=on" -ForegroundColor Gray

# Use Flask development server with Docker-like configuration
& $pythonExe server.py
