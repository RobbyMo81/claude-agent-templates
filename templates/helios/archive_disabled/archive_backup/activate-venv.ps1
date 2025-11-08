# Helios Virtual Environment Activation Script
Write-Host "Activating Helios Python environment..." -ForegroundColor Cyan
& backend\venv\Scripts\Activate.ps1
Write-Host ""
Write-Host "[SUCCESS] Virtual environment activated!" -ForegroundColor Green
Write-Host "To start the backend server: cd backend; python server.py" -ForegroundColor Yellow
Write-Host "To deactivate: deactivate" -ForegroundColor Yellow
Write-Host ""
