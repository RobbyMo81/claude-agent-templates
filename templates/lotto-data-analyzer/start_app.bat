@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM --- Navigate to the script's directory to ensure paths are correct ---
pushd "%~dp0"

REM --- Configuration ---
SET "VENV_NAME=venv"
SET "REQUIREMENTS_FILE=requirements.txt"
SET "APP_FILE=app.py"

REM --- Script Header ---
ECHO ======================================================
ECHO  Powerball Insights Setup & Launch
ECHO ======================================================
ECHO This script will prepare the environment and run the application.
ECHO It will:
ECHO   1. Check for Python.
ECHO   2. Create a Virtual Environment (if it doesn't exist).
ECHO   3. Install required dependencies.
ECHO   4. Launch the Streamlit application.
ECHO.
ECHO Press any key to begin...
pause >nul

::================================================================================
:: 1. CHECK FOR PYTHON
::================================================================================
ECHO [1/4] Checking for Python installation...
python -V >nul 2>&1 || (
    ECHO.
    ECHO [ERROR] Python is not found in your system's PATH.
    ECHO Please install Python (3.11+) and ensure it's added to the PATH.
    GOTO :error
)
ECHO      ...Python found.
ECHO.

::================================================================================
:: 2. SETUP VIRTUAL ENVIRONMENT
::================================================================================
ECHO [2/4] Setting up virtual environment...
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    ECHO      ...Creating new virtual environment in '%VENV_NAME%'. This may take a moment.
    python -m venv %VENV_NAME% || (
        ECHO.
        ECHO [ERROR] Failed to create the virtual environment.
        GOTO :error
    )
) else (
    ECHO      ...Existing virtual environment found.
)
ECHO      ...Activating virtual environment.
call "%VENV_NAME%\Scripts\activate.bat" || (
    ECHO.
    ECHO [ERROR] Failed to activate the virtual environment.
    GOTO :error
)
ECHO.

::================================================================================
:: 3. INSTALL DEPENDENCIES
::================================================================================
ECHO [3/4] Installing dependencies from '%REQUIREMENTS_FILE%'...
pip install -r %REQUIREMENTS_FILE% --quiet --disable-pip-version-check || (
    ECHO.
    ECHO [ERROR] Failed to install dependencies. Try running this command manually:
    ECHO pip install -r %REQUIREMENTS_FILE%
    GOTO :error
)
ECHO      ...Dependencies are up to date.
ECHO.

::================================================================================
:: 4. LAUNCH APPLICATION
::================================================================================
ECHO [4/4] Launching the Streamlit application...
ECHO      You can close this window to stop the application.
ECHO.
python -m streamlit run %APP_FILE% || (
    ECHO.
    ECHO [ERROR] Failed to launch the Streamlit application.
    ECHO Ensure Streamlit is in your %REQUIREMENTS_FILE% and the file '%APP_FILE%' exists.
    GOTO :error
)

GOTO :end

:error
ECHO.
ECHO ======================================================
ECHO  AN ERROR OCCURRED. SCRIPT HALTED.
ECHO ======================================================
ECHO Please review the error message above.
pause

:end
popd
ENDLOCAL