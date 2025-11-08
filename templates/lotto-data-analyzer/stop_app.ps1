# =================================================================
#  Powerball Insights Application Termination Script
# =================================================================
# This script stops any running instances of the Streamlit application.

Write-Host "======================================================" -ForegroundColor Yellow
Write-Host " Powerball Insights - Application Shutdown" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Yellow
Write-Host "This script will terminate any running Streamlit processes."
Write-Host ""

$foundProcesses = $false

# First try to find Streamlit processes
$streamlitProcs = Get-Process -Name "streamlit" -ErrorAction SilentlyContinue
if ($streamlitProcs) {
    $foundProcesses = $true
    Write-Host "Found Streamlit processes:" -ForegroundColor Cyan
    foreach ($proc in $streamlitProcs) {
        try {
            Write-Host "  - PID: $($proc.Id), Started: $($proc.StartTime)" -ForegroundColor Gray
            Stop-Process -Id $proc.Id -Force
            Write-Host "    ✓ Process terminated." -ForegroundColor Green
        } catch {
            Write-Host "    ✗ Failed to terminate process: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# Then look for python processes running Streamlit
try {
    $pythonProcs = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | 
                  Where-Object { $_.CommandLine -like "*streamlit*" -or $_.CommandLine -like "*app.py*" }
    
    if ($pythonProcs) {
        $foundProcesses = $true
        Write-Host "Found Python processes running Streamlit:" -ForegroundColor Cyan
        foreach ($proc in $pythonProcs) {
            try {
                Write-Host "  - PID: $($proc.ProcessId), Command: $($proc.CommandLine)" -ForegroundColor Gray
                Stop-Process -Id $proc.ProcessId -Force
                Write-Host "    ✓ Process terminated." -ForegroundColor Green
            } catch {
                Write-Host "    ✗ Failed to terminate process: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
} catch {
    # CIM query might fail on some systems - fallback to basic check
    $pythonProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue
    if ($pythonProcs) {
        Write-Host "Found Python processes (checking manually):" -ForegroundColor Cyan
        foreach ($proc in $pythonProcs) {
            try {
                # Only display processes related to this app's directory
                $processPath = $proc.Path
                if ($processPath -like "*LottoDataAnalyzer*") {
                    $foundProcesses = $true
                    Write-Host "  - PID: $($proc.Id), Path: $processPath" -ForegroundColor Gray
                    Stop-Process -Id $proc.Id -Force
                    Write-Host "    ✓ Process terminated." -ForegroundColor Green
                }
            } catch {
                # Skip processes we can't access
            }
        }
    }
}

# Check if any ports are being used that might be related to our app
$ports = @(5000, 5050, 5051, 5052, 5053, 5054, 5055, 8501, 8502)  # Added standard Streamlit ports
foreach ($port in $ports) {
    try {
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($connections) {
            $foundProcesses = $true
            Write-Host "Found processes using port $port:" -ForegroundColor Cyan
            foreach ($conn in $connections) {
                try {
                    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
                    if ($proc) {
                        Write-Host "  - PID: $($proc.Id), Process: $($proc.ProcessName)" -ForegroundColor Gray
                        Stop-Process -Id $proc.Id -Force
                        Write-Host "    ✓ Process terminated." -ForegroundColor Green
                    }
                } catch {
                    Write-Host "    ✗ Failed to terminate process: $($_.Exception.Message)" -ForegroundColor Red
                }
            }
        }
    } catch {
        # Skip if we can't check a port
    }
}

if (-not $foundProcesses) {
    Write-Host "No Streamlit or related processes found running." -ForegroundColor Green
} else {
    # Wait a moment to allow processes to fully terminate
    Start-Sleep -Seconds 2
    Write-Host ""
    Write-Host "Processes have been terminated. You can now restart the application." -ForegroundColor Green
}

# Verify ports are now free
$portCheck = $false
foreach ($port in $ports) {
    if ((Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
        Write-Host "WARNING: Port $port is still in use after termination attempts." -ForegroundColor Red
        $portCheck = $true
    }
}

if ($foundProcesses -and -not $portCheck) {
    Write-Host "All processes successfully terminated and ports freed." -ForegroundColor Green
} elseif ($portCheck) {
    Write-Host "Some ports may still be in use. You might need to restart your computer if you continue to have issues." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done. Press Enter to exit." -ForegroundColor Cyan
Read-Host
