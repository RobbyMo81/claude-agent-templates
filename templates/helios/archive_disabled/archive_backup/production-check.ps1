#!/usr/bin/env pwsh
# Helios Production-Readiness Validation Script

Write-Host "================== HELIOS PRODUCTION-READINESS CHECK ==================" -ForegroundColor Cyan
# Force the script to use the same `python` that's in your PATH:
$pythonExe = (Get-Command python).Source
Write-Host "Using Python for checks: $pythonExe"

# 1. Python Environment
Test-Path "venv/Scripts/python.exe" | Out-Null
$python = if (Test-Path "venv/Scripts/python.exe") { ".\venv\Scripts\python.exe" } else { "python" }
$pyVersion = & $python --version 2>&1
Write-Host "Python: $pyVersion" -ForegroundColor White


# Improved Python package check using import-based try/catch logic
$packages = @('torch','numpy','flask','flask_cors','sqlalchemy','alembic')
foreach ($pkg in $packages) {
    try {
        & $pythonExe -c "import $pkg" 2>$null
        Write-Host "Python package '$pkg' found." -ForegroundColor Green
    }
    catch {
        Write-Host "Python package '$pkg' MISSING!" -ForegroundColor Red
    }
}

# 2. Node.js Environment
$nodeVersion = node --version 2>&1
$npmVersion = npm --version 2>&1
Write-Host "Node.js: $nodeVersion, npm: $npmVersion" -ForegroundColor White

# 3. Backend Files & Directories
$backendPath = "backend"
$serverFile = Join-Path $backendPath "server.py"
$modelsDir = Join-Path $backendPath "models"
$dbFile = Join-Path $backendPath "helios_memory.db"

Write-Host "Checking backend files and directories..." -ForegroundColor Cyan
if (Test-Path $serverFile) { Write-Host "server.py found." -ForegroundColor Green } else { Write-Host "server.py MISSING!" -ForegroundColor Red }
if (Test-Path $modelsDir) { Write-Host "models directory found." -ForegroundColor Green } else { Write-Host "models directory MISSING!" -ForegroundColor Red }
if (Test-Path $dbFile) { Write-Host "Database file found." -ForegroundColor Green } else { Write-Host "Database file MISSING!" -ForegroundColor Yellow }

# 4. Environment Variables
if ($env:HELIOS_MODE -eq "production") {
    Write-Host "HELIOS_MODE=production is set." -ForegroundColor Green
} else {
    Write-Host "HELIOS_MODE is NOT set to 'production'!" -ForegroundColor Red
}

# 5. Frontend Build
if (Test-Path "dist/index.html") {
    Write-Host "Frontend build found (dist/index.html)." -ForegroundColor Green
} else {
    Write-Host "Frontend build missing (dist/index.html not found)." -ForegroundColor Red
}

# 6. Permissions
try {
    $testFile = Join-Path $modelsDir "write_test.txt"
    Set-Content $testFile "test" -ErrorAction Stop
    Remove-Item $testFile
    Write-Host "Write permissions to models directory: OK" -ForegroundColor Green
} catch {
    Write-Host "No write permissions to models directory!" -ForegroundColor Red
}

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Manual checks: Run the backend, verify API endpoints, and check browser for frontend errors." -ForegroundColor Yellow
Write-Host "=======================================================================" -ForegroundColor Cyan