# =================================================================
#  Powerball Insights Quick Start
# =================================================================
# This script provides a simplified startup option that just runs the cleanup and launch.

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " Powerball Insights - Quick Start" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "This script will stop any existing instances and start a fresh one."

# First run the stop script to clean up any existing processes
Write-Host "Stopping any existing processes..." -ForegroundColor Yellow
& "$PSScriptRoot\stop_app.ps1"

# Wait a moment to ensure everything is stopped
Start-Sleep -Seconds 2

# Try to determine an available port
function Test-PortAvailable {
    param($port)
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Any, $port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

# Find available port
$port = 5050  # Start with default
$maxPort = 5100
while (-not (Test-PortAvailable -port $port) -and $port -lt $maxPort) {
    $port++
}

if ($port -ge $maxPort) {
    Write-Host "ERROR: Could not find an available port between 5050 and $maxPort." -ForegroundColor Red
    Write-Host "Please run the stop_app.ps1 script or restart your computer before trying again." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Update config file with the available port
$configPath = "$PSScriptRoot\.streamlit\config.toml"
if (Test-Path $configPath) {
    $content = Get-Content $configPath -Raw
    $pattern = '(?ms)(\[server\][^\[]*port\s*=\s*)(\d+)'
    $newContent = $content -replace $pattern, "`$1$port"
    Set-Content -Path $configPath -Value $newContent
    Write-Host "Updated Streamlit config to use port $port." -ForegroundColor Gray
}

Write-Host ""
Write-Host "Starting Powerball Insights on port $port..." -ForegroundColor Green

# Activate virtualenv if it exists
$venvPath = "$PSScriptRoot\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Gray
    & $venvPath
}

# Start the app
Write-Host "Launching Streamlit application..." -ForegroundColor Cyan
Write-Host "The app will be available at http://localhost:$port" -ForegroundColor Cyan
Write-Host ""

try {
    streamlit run "$PSScriptRoot\app.py"
}
catch {
    Write-Host "Error starting Streamlit: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Trying alternative method..." -ForegroundColor Yellow
    
    try {
        python -m streamlit run "$PSScriptRoot\app.py"
    }
    catch {
        Write-Host "Failed to start Streamlit. Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# This will only execute if Streamlit exits
Write-Host ""
Write-Host "Streamlit application has closed." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
