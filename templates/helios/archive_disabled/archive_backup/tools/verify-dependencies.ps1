# Helios Virtual Environment and Dependency Verification Script
# This script activates the virtual environment and verifies ML dependencies

# Set the project directory (adjust if needed)
$ProjectDir = "C:\Users\RobMo\OneDrive\Documents\helios"
$VirtualEnvDir = "backend\venv"  # Corrected path

# Navigate to the project directory
Write-Host "Navigating to project directory: $ProjectDir"
try {
    Set-Location -Path $ProjectDir -ErrorAction Stop
    Write-Host "✅ Successfully navigated to: $(Get-Location)"
} catch {
    Write-Error "❌ Failed to navigate to project directory: $($_.Exception.Message)"
    Exit 1
}

# Check if virtual environment exists
$VenvPath = Join-Path $ProjectDir $VirtualEnvDir "Scripts" "Activate.ps1"
Write-Host "Checking for virtual environment at: $VenvPath"

if (-not (Test-Path $VenvPath)) {
    Write-Error "❌ Virtual environment not found at: $VenvPath"
    Write-Host "Available directories in backend:"
    Get-ChildItem -Path "backend" -Directory | ForEach-Object { Write-Host "  - $($_.Name)" }
    Write-Host ""
    Write-Host "🔧 To create a virtual environment, run:"
    Write-Host "   cd backend"
    Write-Host "   python -m venv venv"
    Write-Host "   .\venv\Scripts\Activate.ps1"
    Write-Host "   pip install -r requirements.txt"
    Exit 1
}

# Activate the virtual environment
Write-Host "Activating virtual environment..."
try {
    & $VenvPath
    if ($env:VIRTUAL_ENV) {
        Write-Host "✅ Virtual environment activated successfully"
        Write-Host "   Virtual Environment: $env:VIRTUAL_ENV"
    } else {
        Write-Warning "⚠️  Virtual environment activation script executed, but VIRTUAL_ENV variable not set"
        Write-Host "Continuing with checks..."
    }
} catch {
    Write-Error "❌ Failed to activate virtual environment: $($_.Exception.Message)"
    Write-Host "🔧 Ensure the execution policy allows local scripts:"
    Write-Host "   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    Exit 1
}

# Verify Python version
Write-Host "`n📋 Checking Python version..."
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python Version: $pythonVersion"
} catch {
    Write-Error "❌ Python not found or not accessible"
    Exit 1
}

# Verify PyTorch (torch) installation
Write-Host "`n🔥 Verifying PyTorch (torch) installation..."
try {
    $torchResult = python -c "import torch; print(f'PyTorch {torch.__version__} is installed and can be imported.')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $torchResult"
    } else {
        throw "PyTorch import failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Error "❌ PyTorch (torch) import failed: $($_.Exception.Message)"
    Write-Host "🔧 To install PyTorch, run:"
    Write-Host "   pip install torch torchvision torchaudio"
    Write-Host "   # Or visit https://pytorch.org/get-started/locally/ for specific installation commands"
    $torchFailed = $true
}

# Verify NumPy installation and address the source directory issue
Write-Host "`n🔢 Verifying NumPy installation..."
try {
    $numpyResult = python -c "import numpy as np; print(f'NumPy {np.__version__} is installed and can be imported.')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $numpyResult"
    } else {
        throw "NumPy import failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Error "❌ NumPy import failed: $($_.Exception.Message)"
    if ($_.Exception.Message -like "*source directory*") {
        Write-Host "🚨 CRITICAL: You appear to be importing NumPy from its source directory!"
        Write-Host "🔧 Solutions:"
        Write-Host "   1. Change to a different directory before running this script"
        Write-Host "   2. Reinstall NumPy: pip install numpy --upgrade --force-reinstall"
        Write-Host "   3. Check that you're not in a numpy source directory"
    } else {
        Write-Host "🔧 To install/fix NumPy, run:"
        Write-Host "   pip install numpy --upgrade --force-reinstall"
    }
    $numpyFailed = $true
}

# Verify other key dependencies
Write-Host "`n📊 Verifying other ML dependencies..."
$dependencies = @("pandas", "sklearn", "matplotlib")
$failedDeps = @()

foreach ($dep in $dependencies) {
    try {
        $result = python -c "import $dep; print(f'$dep is installed')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $result"
        } else {
            throw "Import failed"
        }
    } catch {
        Write-Host "⚠️  $dep import failed"
        $failedDeps += $dep
    }
}

# Summary and recommendations
Write-Host "`n" + "="*60
Write-Host "🎯 DEPENDENCY CHECK SUMMARY"
Write-Host "="*60

$allPassed = $true

if ($torchFailed) {
    Write-Host "❌ PyTorch: FAILED"
    $allPassed = $false
} else {
    Write-Host "✅ PyTorch: PASSED"
}

if ($numpyFailed) {
    Write-Host "❌ NumPy: FAILED" 
    $allPassed = $false
} else {
    Write-Host "✅ NumPy: PASSED"
}

if ($failedDeps.Count -gt 0) {
    Write-Host "⚠️  Other dependencies: $($failedDeps.Count) failed ($($failedDeps -join ', '))"
} else {
    Write-Host "✅ Other dependencies: All passed"
}

Write-Host "`n🎯 EXPECTED HELIOS SERVER BEHAVIOR:"
if ($allPassed) {
    Write-Host "   DEPENDENCIES_AVAILABLE = True"
    Write-Host "   trainer = ModelTrainer instance"
    Write-Host "   Mock mode = Disabled (full ML training available)"
} else {
    Write-Host "   DEPENDENCIES_AVAILABLE = False"
    Write-Host "   trainer = None"
    Write-Host "   Mock mode = Enabled"
}

if (-not $allPassed) {
    Write-Host "`n🔧 RECOMMENDED ACTIONS:"
    if ($torchFailed) {
        Write-Host "   1. Install PyTorch: pip install torch torchvision torchaudio"
    }
    if ($numpyFailed) {
        Write-Host "   2. Fix NumPy: pip install numpy --upgrade --force-reinstall"
    }
    if ($failedDeps.Count -gt 0) {
        Write-Host "   3. Install missing deps: pip install $($failedDeps -join ' ')"
    }
    Write-Host "   4. Then restart the Helios server"
}

Write-Host "`n✔️  All dependency checks completed."
Write-Host "Exit code: $(if ($allPassed) { '0 (Success)' } else { '1 (Issues found)' })"

if (-not $allPassed) {
    Exit 1
}
