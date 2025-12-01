# Helios System Stress Test Script for Windows PowerShell
# Alternative PowerShell version for better Windows compatibility

Write-Host "🚀 Helios System Stress Test Suite" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[INFO] Starting PowerShell stress test runner..." -ForegroundColor Yellow
Write-Host ""

# Check if Node.js is available
try {
    $nodeVersion = node --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Node.js is available: $nodeVersion" -ForegroundColor Green
    } else {
        throw "Node.js not found"
    }
} catch {
    Write-Host "[ERROR] Node.js is not available. Please install Node.js first." -ForegroundColor Red
    Write-Host "Download from: https://nodejs.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to continue..."
    exit 1
}

Write-Host ""
Write-Host "[INFO] Running comprehensive stress tests using JavaScript runner..." -ForegroundColor Yellow

# Run the JavaScript stress test runner
try {
    & node tests/runStressTest.js
    $testExitCode = $LASTEXITCODE
} catch {
    Write-Host "[ERROR] Failed to run stress test: $_" -ForegroundColor Red
    $testExitCode = 1
}

Write-Host ""
Write-Host "[INFO] Stress test completed with exit code: $testExitCode" -ForegroundColor Yellow

if ($testExitCode -eq 0) {
    Write-Host "[SUCCESS] ✅ All tests completed successfully!" -ForegroundColor Green
} else {
    Write-Host "[WARNING] ⚠️ Some tests failed or had warnings. Review the output above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

exit $testExitCode
