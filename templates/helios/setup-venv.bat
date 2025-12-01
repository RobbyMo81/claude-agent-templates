@echo off
REM Helios Python Virtual Environment Setup Script for Windows
REM This script creates and configures a Python virtual environment for the backend

echo 🐍 Helios Python Virtual Environment Setup
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not available or not in PATH
    echo Please install Python from https://python.org and add it to your PATH
    echo.
    pause
    exit /b 1
)

echo [INFO] Python is available
python --version

REM Check if we're in the correct directory
if not exist "backend" (
    echo [ERROR] Backend directory not found
    echo Please run this script from the Helios project root directory
    echo.
    pause
    exit /b 1
)

echo [INFO] Backend directory found
echo.

REM Check if virtual environment already exists
if exist "backend\venv" (
    echo [INFO] Virtual environment already exists
    choice /c YN /m "Do you want to recreate it? (Y/N)"
    if errorlevel 2 goto activate_existing

    echo [INFO] Removing existing virtual environment...
    rmdir /s /q "backend\venv"
)

REM Create virtual environment
echo [INFO] Creating Python virtual environment...
python -m venv backend\venv

if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    echo Try using: python3 -m venv backend\venv
    echo.
    pause
    exit /b 1
)

echo [SUCCESS] Virtual environment created successfully

:activate_existing
REM Activate virtual environment
echo [INFO] Activating virtual environment...
call backend\venv\Scripts\activate.bat

if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    echo Try running: backend\venv\Scripts\activate.bat
    echo.
    pause
    exit /b 1
)

echo [SUCCESS] Virtual environment activated

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)

REM Install dependencies
if exist "backend\requirements.txt" (
    echo [INFO] Installing Python dependencies...
    pip install -r backend\requirements.txt

    if errorlevel 1 (
        echo [ERROR] Failed to install some dependencies
        echo Check the output above for details
        echo.
        pause
        exit /b 1
    )

    echo [SUCCESS] Dependencies installed successfully
) else (
    echo [WARNING] No requirements.txt found, skipping dependency installation
)

REM Verify installation
echo.
echo [INFO] Verifying installation...
echo Python version:
python --version
echo.
echo Pip version:
pip --version
echo.
echo Installed packages:
pip list
echo.

echo [SUCCESS] ✅ Virtual environment setup completed!
echo.
echo 📋 Next Steps:
echo ================
echo 1. To activate the virtual environment manually:
echo    backend\venv\Scripts\activate
echo.
echo 2. To start the backend server:
echo    cd backend
echo    python server.py
echo.
echo 3. To deactivate when done:
echo    deactivate
echo.
echo 4. Or use npm scripts:
echo    npm run python:dev    # Start backend
echo    npm run dev          # Start frontend
echo.

REM Create activation shortcut
echo [INFO] Creating activation shortcut...
(
echo @echo off
echo echo Activating Helios Python environment...
echo call backend\venv\Scripts\activate.bat
echo echo.
echo echo [SUCCESS] Virtual environment activated!
echo echo To start the backend server: cd backend ^&^& python server.py
echo echo To deactivate: deactivate
echo echo.
) > activate-venv.bat

echo [INFO] Created 'activate-venv.bat' for easy activation

echo.
echo Press any key to continue...
pause >nul

echo [INFO] Starting backend server test...
cd backend
python server.py

REM Note: The server will run until manually stopped with Ctrl+C
