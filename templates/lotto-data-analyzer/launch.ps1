# =================================================================
#  Powerball Insights Setup & Launch (PowerShell Version)
# =================================================================
# This script prepares the environment and runs the Streamlit application.
# It should be run from a PowerShell terminal.

# --- Configuration ---
$VenvName = "venv"
$RequirementsFile = "requirements.txt"
$AppFile = "app.py"
$DefaultPort = 5050  # Changed from 5000 to avoid common conflicts
$StreamlitConfigPath = ".streamlit\config.toml"

# --- Function for Error Handling ---
# RENAMED to use an approved PowerShell verb 'Invoke'.
function Invoke-ErrorHandler {
    param($message)
    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Red
    Write-Host "  AN ERROR OCCURRED. SCRIPT HALTED." -ForegroundColor Red
    Write-Host "======================================================" -ForegroundColor Red
    Write-Host $message -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Function to check if a port is in use ---
function Test-PortInUse {
    param($port)
    $result = $null
    try {
        $result = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    } catch {
        # Try an alternative method if Get-NetTCPConnection fails
        try {
            $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Any, $port)
            $listener.Start()
            $listener.Stop()
            return $false
        } catch {
            # If we can't create a listener, the port is likely in use
            return $true
        }
    }
    return ($null -ne $result)
}

# --- Function to find next available port ---
function Find-AvailablePort {
    param($startPort)
    $port = $startPort
    while (Test-PortInUse -port $port) {
        $port++
        if ($port -gt ($startPort + 100)) {
            # Safety limit - don't search forever
            throw "Unable to find an available port after checking 100 ports"
        }
    }
    return $port
}

# --- Function to update Streamlit config with new port ---
function Update-StreamlitConfig {
    param($port)
    if (Test-Path $StreamlitConfigPath) {
        $content = Get-Content $StreamlitConfigPath -Raw
        $pattern = '(?ms)(\[server\][^\[]*port\s*=\s*)(\d+)'
        $newContent = $content -replace $pattern, "`$1$port"
        Set-Content -Path $StreamlitConfigPath -Value $newContent
        Write-Host "     ...Updated Streamlit config to use port $port." -ForegroundColor Gray
    }
}

# --- Function to kill previous Streamlit instances ---
function Stop-StreamlitProcesses {
    # First, try to stop streamlit processes directly
    $streamlitProcs = Get-Process -Name "streamlit" -ErrorAction SilentlyContinue
    if ($streamlitProcs) {
        Write-Host "     ...Stopping Streamlit processes." -ForegroundColor Gray
        foreach ($proc in $streamlitProcs) {
            try {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                Write-Host "     ...Stopped streamlit process ID: $($proc.Id)" -ForegroundColor Gray
            } catch {
                # Continue even if we can't stop a process
            }
        }
    }
    
    # Try to find Python processes running Streamlit
    try {
        $pythonProcs = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | 
                      Where-Object { $_.CommandLine -like "*streamlit*" -or $_.CommandLine -like "*app.py*" }
        
        if ($pythonProcs) {
            Write-Host "     ...Stopping Python processes running Streamlit." -ForegroundColor Gray
            foreach ($proc in $pythonProcs) {
                try {
                    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
                    Write-Host "     ...Stopped python process ID: $($proc.ProcessId)" -ForegroundColor Gray
                } catch {
                    # Continue even if we can't stop a process
                }
            }
        }
    } catch {
        # CIM query might fail - fallback to basic check
        $pythonProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue
        if ($pythonProcs) {
            foreach ($proc in $pythonProcs) {
                try {
                    # Only check processes in this app's directory
                    if ($proc.Path -like "*LottoDataAnalyzer*") {
                        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                        Write-Host "     ...Stopped python process ID: $($proc.Id)" -ForegroundColor Gray
                    }
                } catch {
                    # Skip processes we can't access
                }
            }
        }
    }
    
    # Short delay to ensure processes are fully terminated
    Start-Sleep -Seconds 3
}

Clear-Host
Write-Host "======================================================" -ForegroundColor Green
Write-Host " Powerball Insights Setup & Launch" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host "This script will:"
Write-Host "  1. Check for Python."
Write-Host "  2. Create a Virtual Environment (if it doesn't exist)."
Write-Host "  3. Install required dependencies."
Write-Host "  4. Launch the Streamlit application."
Write-Host ""
Read-Host "Press Enter to begin"

try {
    # ===============================================================
    # 1. CHECK FOR PYTHON
    # ===============================================================
    Write-Host "[1/4] Checking for Python installation..."
    Get-Command python -ErrorAction Stop > $null
    Write-Host "     ...Python found." -ForegroundColor Gray
    Write-Host ""


    # ===============================================================
    # 2. SETUP VIRTUAL ENVIRONMENT
    # ===============================================================
    Write-Host "[2/4] Setting up virtual environment..."
    $ActivateScript = Join-Path -Path "." -ChildPath ($VenvName + "\Scripts\Activate.ps1")

    if (-not (Test-Path $ActivateScript)) {
        Write-Host "     ...Creating new virtual environment in '$VenvName'. This may take a moment." -ForegroundColor Gray
        python -m venv $VenvName
        if (-not $?) { throw "Failed to create the Python virtual environment. Please check your Python installation." }

    } else {
        Write-Host "     ...Existing virtual environment found." -ForegroundColor Gray
    }

    Write-Host "     ...Activating virtual environment." -ForegroundColor Gray
    . $ActivateScript
    Write-Host ""


    # ===============================================================
    # 3. INSTALL DEPENDENCIES
    # ===============================================================
    Write-Host "[3/4] Installing dependencies from '$RequirementsFile'..."
    pip install -r $RequirementsFile --quiet --disable-pip-version-check
    if (-not $?) { throw "Failed to install dependencies from $RequirementsFile" }
    Write-Host "     ...Dependencies are up to date." -ForegroundColor Gray
    Write-Host ""


    # ===============================================================
    # 4. LAUNCH APPLICATION
    # ===============================================================
    Write-Host "[4/4] Launching the Streamlit application..." -ForegroundColor Cyan
    
    # Stop any existing Streamlit processes
    Stop-StreamlitProcesses
    
    # Check if the default port is in use
    $port = $DefaultPort
    if (Test-PortInUse -port $port) {
        Write-Host "     ...Default port $port is in use. Finding an alternative port." -ForegroundColor Yellow
        $port = Find-AvailablePort -startPort ($port + 1)
        Update-StreamlitConfig -port $port
        Write-Host "     ...Will use port $port instead." -ForegroundColor Cyan
    } else {
        # Ensure config has the correct port even if default is available
        Update-StreamlitConfig -port $port
    }
    
    # Double check that the port is really free now
    if (Test-PortInUse -port $port) {
        Write-Host "     ...WARNING: Port $port still appears to be in use despite our attempts to free it." -ForegroundColor Red
        Write-Host "     ...Attempting to forcibly close the port..." -ForegroundColor Yellow
        
        # Try to forcibly close any processes using this port
        try {
            $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
            if ($connections) {
                foreach ($conn in $connections) {
                    try {
                        $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
                        if ($proc) {
                            Write-Host "     ...Terminating process $($proc.ProcessName) (ID: $($proc.Id)) that's using port $port" -ForegroundColor Yellow
                            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                        }
                    } catch {
                        # Skip if we can't stop a specific process
                    }
                }
            }
            # Wait for processes to terminate
            Start-Sleep -Seconds 2
        } catch {
            # Continue even if this fails
        }
        
        # If port is still in use after all our efforts, try another port
        if (Test-PortInUse -port $port) {
            $port = Find-AvailablePort -startPort ($port + 1)
            Update-StreamlitConfig -port $port
            Write-Host "     ...FALLBACK: Will use port $port instead." -ForegroundColor Yellow
        }
    }
    
    Write-Host "     ...Starting application on port $port" -ForegroundColor Gray
    Write-Host "     ...You can stop the application by pressing Ctrl+C in this window or by running stop_app.ps1" -ForegroundColor Gray
    Write-Host ""
    
    try {
        streamlit run $AppFile
        if (-not $?) { throw "Failed to launch the Streamlit application." }
    } catch {
        Write-Host "     ...Error launching Streamlit: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "     ...Trying an alternative launch method..." -ForegroundColor Yellow
        
        # Alternative launch method
        try {
            & python -m streamlit run $AppFile
        } catch {
            throw "Failed to launch the Streamlit application using alternative method: $($_.Exception.Message)"
        }
    }

}
catch {
    # UPDATED to call the renamed function.
    Invoke-ErrorHandler -message $_.Exception.Message
}