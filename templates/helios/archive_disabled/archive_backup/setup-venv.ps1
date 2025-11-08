# Helios Python Virtual Environment Setup Script for Windows PowerShell
# This script creates and configures a Python virtual environment for the backend

Write-Host " Helios Python Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] Python is available: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "[ERROR] Python is not available or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://python.org and add it to your PATH" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue..."
    exit 1
}

# Check if we're in the correct directory
if (-not (Test-Path "backend")) {
    Write-Host "[ERROR] Backend directory not found" -ForegroundColor Red
    Write-Host "Please run this script from the Helios project root directory" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue..."
    exit 1
}

Write-Host "[INFO] Backend directory found" -ForegroundColor Green
Write-Host ""

# Check if virtual environment already exists
if (Test-Path "backend\venv") {
    Write-Host "[INFO] Virtual environment already exists" -ForegroundColor Yellow
    $choice = Read-Host "Do you want to recreate it? (y/N)"
    
    if ($choice -eq "y" -or $choice -eq "Y") {
        Write-Host "[INFO] Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force "backend\venv"
    } else {
        Write-Host "[INFO] Using existing virtual environment" -ForegroundColor Green
        goto ActivateExisting
    }
}

# Create virtual environment
Write-Host "[INFO] Creating Python virtual environment..." -ForegroundColor Yellow
try {
    & python -m venv backend\venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment"
    }
    Write-Host "[SUCCESS] Virtual environment created successfully" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
    Write-Host "Try using: python3 -m venv backend\venv" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue..."
    exit 1
}

:ActivateExisting

# Activate virtual environment
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
try {
    & backend\venv\Scripts\Activate.ps1
    Write-Host "[SUCCESS] Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "You may need to change PowerShell execution policy:" -ForegroundColor Yellow
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to continue..."
    exit 1
}

# Upgrade pip
Write-Host "[INFO] Upgrading pip..." -ForegroundColor Yellow
try {
    & python -m pip install --upgrade pip
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Pip upgraded successfully" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Failed to upgrade pip, continuing anyway..." -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] Failed to upgrade pip, continuing anyway..." -ForegroundColor Yellow
}

# Install dependencies
if (Test-Path "backend\requirements.txt") {
    Write-Host "[INFO] Installing Python dependencies..." -ForegroundColor Yellow
    try {
        & pip install -r backend\requirements.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] Dependencies installed successfully" -ForegroundColor Green
        } else {
            throw "Failed to install dependencies"
        }
    } catch {
        Write-Host "[ERROR] Failed to install some dependencies" -ForegroundColor Red
        Write-Host "Check the output above for details" -ForegroundColor Yellow
        Write-Host ""
        Read-Host "Press Enter to continue..."
        exit 1
    }
} else {
    Write-Host "[WARNING] No requirements.txt found, skipping dependency installation" -ForegroundColor Yellow
}

# Verify installation
Write-Host ""
Write-Host "[INFO] Verifying installation..." -ForegroundColor Yellow
Write-Host "Python version:" -ForegroundColor Cyan
& python --version
Write-Host ""
Write-Host "Pip version:" -ForegroundColor Cyan
& pip --version
Write-Host ""
Write-Host "Installed packages:" -ForegroundColor Cyan
& pip list
Write-Host ""

Write-Host "[SUCCESS] Virtual environment setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host " Next Steps:" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host "1. To activate the virtual environment manually:" -ForegroundColor White
Write-Host "   backend\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. To start the backend server:" -ForegroundColor White
Write-Host "   cd backend" -ForegroundColor Yellow
Write-Host "   python server.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. To deactivate when done:" -ForegroundColor White
Write-Host "   deactivate" -ForegroundColor Yellow
Write-Host ""
Write-Host "4. Or use npm scripts:" -ForegroundColor White
Write-Host "   npm run python:dev    # Start backend" -ForegroundColor Yellow
Write-Host "   npm run dev          # Start frontend" -ForegroundColor Yellow
Write-Host ""

# Create activation shortcut
Write-Host "[INFO] Creating activation shortcut..." -ForegroundColor Yellow
$activationScript = @"
# Helios Virtual Environment Activation Script
Write-Host "Activating Helios Python environment..." -ForegroundColor Cyan
& backend\venv\Scripts\Activate.ps1
Write-Host ""
Write-Host "[SUCCESS] Virtual environment activated!" -ForegroundColor Green
Write-Host "To start the backend server: cd backend; python server.py" -ForegroundColor Yellow
Write-Host "To deactivate: deactivate" -ForegroundColor Yellow
Write-Host ""
"@

$activationScript | Out-File -FilePath "activate-venv.ps1" -Encoding UTF8
Write-Host "[INFO] Created 'activate-venv.ps1' for easy activation" -ForegroundColor Green

Write-Host ""
$testServer = Read-Host "Would you like to test the backend server now? (y/N)"

if ($testServer -eq "y" -or $testServer -eq "Y") {
    Write-Host "[INFO] Starting backend server test..." -ForegroundColor Yellow
    Set-Location backend
    & python server.py
    # Note: The server will run until manually stopped with Ctrl+C
}
