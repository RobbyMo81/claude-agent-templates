@echo off
:: =============================================================================
::  Powerball-Insights launcher
::  • Creates/activates a virtual-env once
::  • Installs / upgrades Python packages listed in requirements.txt
::  • Starts the Streamlit front-end (app.py)
:: =============================================================================
::  CONFIG — edit these four lines if you ever move files around
:: -----------------------------------------------------------------------------
set "PYTHON=python"            :: change to py, py -3.12, etc. if you wish
set "VENV_DIR=.venv"           :: relative or absolute path to the venv
set "REQ_FILE=requirements.txt"
set "APP_ENTRY=app.py"
:: =============================================================================
setlocal EnableDelayedExpansion
pushd "%~dp0"

:: -----------------------------------------------------------------------------
::  1) Create the virtual-env once (idempotent)
:: -----------------------------------------------------------------------------
if not exist "%VENV_DIR%\Scripts\activate" (
    echo [+] Creating virtual environment "%VENV_DIR%" …
    %PYTHON% -m venv "%VENV_DIR%" || goto :fail
)

:: -----------------------------------------------------------------------------
::  2) Activate the venv
:: -----------------------------------------------------------------------------
call "%VENV_DIR%\Scripts\activate"

:: -----------------------------------------------------------------------------
::  3) Upgrade pip (safe + quick) and install deps
:: -----------------------------------------------------------------------------
echo [+] Upgrading pip …
%PYTHON% -m pip install --upgrade pip

echo [+] Installing / upgrading dependencies …
%PYTHON% -m pip install --upgrade --requirement "%REQ_FILE%" || goto :fail

:: -----------------------------------------------------------------------------
::  4) Launch Streamlit
:: -----------------------------------------------------------------------------
echo.
echo [+] Starting Streamlit. The UI will open at http://localhost:8501
streamlit run "%APP_ENTRY%"
goto :eof

:fail
echo.
echo [!] Setup failed — check the log above for the exact error.
exit /b 1
