@echo off
REM Helios System Stress Test Script for Windows
REM Simple and reliable version for Windows compatibility

echo 🚀 Helios System Stress Test Suite
echo ==================================
echo.

echo [INFO] Starting stress test runner...
echo.

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not available. Please install Node.js first.
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)

REM Run the JavaScript stress test runner
echo [INFO] Running comprehensive stress tests...
node tests/runStressTest.js

REM Capture the exit code
set TEST_EXIT_CODE=%errorlevel%

echo.
echo [INFO] Stress test completed with exit code: %TEST_EXIT_CODE%

if %TEST_EXIT_CODE% equ 0 (
    echo [SUCCESS] ✅ All tests completed successfully!
) else (
    echo [WARNING] ⚠️ Some tests failed or had warnings. Review the output above.
)

echo.
echo Press any key to continue...
pause >nul

exit /b %TEST_EXIT_CODE%
