# Watchdog.ps1 - Monitors training log and restarts if stalled

$logPath = "C:\Users\RobMo\OneDrive\Documents\helios\backend\training_log.txt"
$trainingScript = "C:\Users\RobMo\OneDrive\Documents\helios\backend\train_model.ps1"
$timeoutMinutes = 30

function Get-LastLogTime {
    if (Test-Path $logPath) {
        $lastLine = Get-Content $logPath | Select-Object -Last 1
        if ($lastLine -match "\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}") {
            return [datetime]::Parse($lastLine.Substring(0, 19))
        }
    }
    return $null
}

function RestartTraining {
    Write-Host "Training appears stalled. Restarting..."
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File '$trainingScript'"
}

while ($true) {
    $lastLogTime = Get-LastLogTime
    if ($lastLogTime -ne $null) {
        $elapsed = (Get-Date) - $lastLogTime
        if ($elapsed.TotalMinutes -gt $timeoutMinutes) {
            RestartTraining
        }
    }
    Start-Sleep -Seconds 300  # Check every 5 minutes
}
