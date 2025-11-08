#Requires -Version 5.1
<#
.SYNOPSIS
    Helios System Validation Framework - Modular QA/QC Workflow
    
.DESCRIPTION
    Production-ready validation script with structured logging, phase separation,
    virtual environment awareness, and audit-ready output formats.
    
.PARAMETER Quick
    Run only critical validation phases (skip integration tests)
    
.PARAMETER OutputFormat
    Output format: Console, JSON, CSV, XML, or All
    
.PARAMETER LogLevel
    Logging verbosity: Silent, Minimal, Normal, Verbose, Debug
    
.PARAMETER ExportPath
    Custom path for exported logs and reports
    
.EXAMPLE
    .\test-system-components.ps1 -Quick -OutputFormat JSON -LogLevel Verbose
    
.NOTES
    Author: Helios QA/QC Framework
    Version: 2.0.0
    Compatible: PowerShell 5.1+, Windows 10+
#>

[CmdletBinding()]
param(
    [switch]$Quick,
    [ValidateSet("Console", "JSON", "CSV", "XML", "All")]
    [string]$OutputFormat = "Console",
    [ValidateSet("Silent", "Minimal", "Normal", "Verbose", "Debug")]
    [string]$LogLevel = "Normal",
    [string]$ExportPath = $null
)

# ============================================================================
# PHASE 0: FRAMEWORK INITIALIZATION
# ============================================================================

class ValidationResult {
    [string]$Timestamp
    [string]$Phase
    [string]$Component
    [string]$Status
    [string]$Message
    [string]$Details
    [int]$Duration
    [hashtable]$Metadata
    
    ValidationResult([string]$phase, [string]$component, [string]$status, [string]$message) {
        $this.Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
        $this.Phase = $phase
        $this.Component = $component
        $this.Status = $status
        $this.Message = $message
        $this.Details = ""
        $this.Duration = 0
        $this.Metadata = @{}
    }
}

class HeliosValidator {
    [string]$ProjectPath
    [string]$VenvPath
    [string]$PythonExe
    [string]$LogLevel
    [System.Collections.ArrayList]$Results
    [hashtable]$Configuration
    [System.Diagnostics.Stopwatch]$SessionTimer
    [System.Diagnostics.Stopwatch]$PhaseTimer
    
    HeliosValidator([string]$projectPath, [string]$logLevel) {
        $this.ProjectPath = $projectPath
        $this.LogLevel = $logLevel
        $this.Results = [System.Collections.ArrayList]::new()
        $this.SessionTimer = [System.Diagnostics.Stopwatch]::StartNew()
        $this.PhaseTimer = [System.Diagnostics.Stopwatch]::new()
        
        # Auto-detect virtual environment
        $this.DetectEnvironment()
        
        # Load configuration
        $this.LoadConfiguration()
    }
    
    [void]DetectEnvironment() {
        # Primary: Root-level venv
        $rootVenv = Join-Path $this.ProjectPath "venv"
        if (Test-Path $rootVenv) {
            $this.VenvPath = $rootVenv
            $this.PythonExe = Join-Path $rootVenv "Scripts\python.exe"
            return
        }
        
        # Fallback: Backend venv (legacy)
        $backendVenv = Join-Path $this.ProjectPath "backend\venv"
        if (Test-Path $backendVenv) {
            $this.VenvPath = $backendVenv
            $this.PythonExe = Join-Path $backendVenv "Scripts\python.exe"
            $this.LogResult("Environment", "Warning", "Using legacy backend venv location", "Consider migrating to root-level venv")
            return
        }
        
        # System Python fallback
        $this.VenvPath = $null
        $this.PythonExe = "python"
        $this.LogResult("Environment", "Warning", "No virtual environment detected", "Using system Python")
    }
    
    [void]LoadConfiguration() {
        $this.Configuration = @{
            # Phase configuration
            Phases = @{
                Environment = @{ Enabled = $true; Critical = $true }
                Dependencies = @{ Enabled = $true; Critical = $true }
                Database = @{ Enabled = $true; Critical = $false }
                Syntax = @{ Enabled = $true; Critical = $true }
                Integration = @{ Enabled = $true; Critical = $false }
                Performance = @{ Enabled = $false; Critical = $false }
            }
            
            # Timeout configuration (seconds)
            Timeouts = @{
                Python = 10
                Node = 5
                Database = 15
                Syntax = 30
                Integration = 120
            }
            
            # Expected versions
            MinVersions = @{
                Python = "3.8.0"
                Node = "16.0.0"
                NPM = "8.0.0"
            }
            
            # Critical files to validate
            CriticalFiles = @(
                "backend\server.py"
                "backend\__init__.py"
                "package.json"
                "vite.config.ts"
            )
        }
    }
    
    [ValidationResult]LogResult([string]$phase, [string]$status, [string]$message, [string]$details = "") {
        $result = [ValidationResult]::new($phase, $this.GetCurrentComponent(), $status, $message)
        $result.Details = $details
        $result.Duration = if ($this.PhaseTimer.IsRunning) { $this.PhaseTimer.ElapsedMilliseconds } else { 0 }
        
        # Add contextual metadata
        $result.Metadata = @{
            PythonExe = $this.PythonExe
            VenvPath = $this.VenvPath
            SessionDuration = $this.SessionTimer.ElapsedMilliseconds
        }
        
        $this.Results.Add($result) | Out-Null
        $this.WriteLog($result)
        return $result
    }
    
    [string]GetCurrentComponent() {
        return "SystemValidation"
    }
    
    [void]WriteLog([ValidationResult]$result) {
        switch ($this.LogLevel) {
            "Silent" { return }
            "Minimal" { 
                if ($result.Status -eq "FAILED") {
                    Write-Host "[$($result.Timestamp)] [FAIL] $($result.Phase): $($result.Message)" -ForegroundColor Red
                }
            }
            "Normal" {
                $color = switch ($result.Status) {
                    "PASSED" { "Green" }
                    "FAILED" { "Red" }
                    "WARNING" { "Yellow" }
                    "INFO" { "Cyan" }
                    default { "White" }
                }
                $icon = switch ($result.Status) {
                    "PASSED" { "[PASS]" }
                    "FAILED" { "[FAIL]" }
                    "WARNING" { "[WARN]" }
                    "INFO" { "[INFO]" }
                    default { "[TEST]" }
                }
                Write-Host "[$($result.Timestamp)] $icon $($result.Phase): $($result.Message)" -ForegroundColor $color
            }
            "Verbose" {
                # Use Normal format first
                $color = switch ($result.Status) {
                    "PASSED" { "Green" }
                    "FAILED" { "Red" }
                    "WARNING" { "Yellow" }
                    "INFO" { "Cyan" }
                    default { "White" }
                }
                $icon = switch ($result.Status) {
                    "PASSED" { "[PASS]" }
                    "FAILED" { "[FAIL]" }
                    "WARNING" { "[WARN]" }
                    "INFO" { "[INFO]" }
                    default { "[TEST]" }
                }
                Write-Host "[$($result.Timestamp)] $icon $($result.Phase): $($result.Message)" -ForegroundColor $color
                
                if ($result.Details) {
                    Write-Host "   Details: $($result.Details)" -ForegroundColor Gray
                }
                if ($result.Duration -gt 0) {
                    Write-Host "   Duration: $($result.Duration)ms" -ForegroundColor Gray
                }
            }
            "Debug" {
                # Use Verbose format first
                $color = switch ($result.Status) {
                    "PASSED" { "Green" }
                    "FAILED" { "Red" }
                    "WARNING" { "Yellow" }
                    "INFO" { "Cyan" }
                    default { "White" }
                }
                $icon = switch ($result.Status) {
                    "PASSED" { "[PASS]" }
                    "FAILED" { "[FAIL]" }
                    "WARNING" { "[WARN]" }
                    "INFO" { "[INFO]" }
                    default { "[TEST]" }
                }
                Write-Host "[$($result.Timestamp)] $icon $($result.Phase): $($result.Message)" -ForegroundColor $color
                
                if ($result.Details) {
                    Write-Host "   Details: $($result.Details)" -ForegroundColor Gray
                }
                if ($result.Duration -gt 0) {
                    Write-Host "   Duration: $($result.Duration)ms" -ForegroundColor Gray
                }
                Write-Host "   Component: $($result.Component)" -ForegroundColor DarkGray
                Write-Host "   Metadata: $($result.Metadata | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
            }
        }
    }
    
    [void]RunDiagnostics() {
        Write-Host "`nDIAGNOSTIC INFORMATION:" -ForegroundColor Magenta
        Write-Host "Project Path: $($this.ProjectPath)" -ForegroundColor Gray
        Write-Host "Virtual Environment: $($this.VenvPath)" -ForegroundColor Gray
        Write-Host "Python Executable: $($this.PythonExe)" -ForegroundColor Gray
        Write-Host "Python Path Exists: $(Test-Path $this.PythonExe)" -ForegroundColor Gray
        
        if (Test-Path $this.PythonExe) {
            try {
                $version = & $this.PythonExe --version 2>&1
                Write-Host "Python Version: $version" -ForegroundColor Gray
            } catch {
                Write-Host "Python Version: ERROR - $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
    
    [void]StartPhase([string]$phaseName) {
        $this.PhaseTimer.Restart()
        if ($this.LogLevel -in @("Verbose", "Debug")) {
            Write-Host "`nStarting Phase: $phaseName" -ForegroundColor Magenta
            Write-Host ("-" * 60) -ForegroundColor DarkMagenta
        }
    }
    
    [void]EndPhase([string]$phaseName) {
        $this.PhaseTimer.Stop()
        if ($this.LogLevel -in @("Verbose", "Debug")) {
            Write-Host "Completed Phase: $phaseName ($($this.PhaseTimer.ElapsedMilliseconds)ms)" -ForegroundColor Magenta
        }
    }
}

# ============================================================================
# PHASE 1: ENVIRONMENT VALIDATION
# ============================================================================

function Test-PythonEnvironment {
    param([HeliosValidator]$validator)
    
    $validator.StartPhase("Python Environment")
    
    try {
        # Test Python executable existence
        if (-not (Test-Path $validator.PythonExe)) {
            $validator.LogResult("Python Environment", "FAILED", "Python executable not found", "Path: $($validator.PythonExe)")
            return $false
        }
        
        # Test Python version
        $versionOutput = & $validator.PythonExe --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            $validator.LogResult("Python Environment", "FAILED", "Python execution failed", $versionOutput)
            return $false
        }
        
        # Parse and validate version
        if ($versionOutput -match "Python (\d+\.\d+\.\d+)") {
            $currentVersion = [Version]$matches[1]
            $minVersion = [Version]$validator.Configuration.MinVersions.Python
            
            if ($currentVersion -lt $minVersion) {
                $validator.LogResult("Python Environment", "WARNING", "Python version below minimum", "Current: $currentVersion, Required: $minVersion")
            } else {
                $validator.LogResult("Python Environment", "PASSED", "Python environment validated", "Version: $currentVersion")
            }
        } else {
            $validator.LogResult("Python Environment", "WARNING", "Could not parse Python version", $versionOutput)
        }
        
        # Test virtual environment status
        if ($validator.VenvPath) {
            $validator.LogResult("Python Environment", "INFO", "Virtual environment detected", "Path: $($validator.VenvPath)")
        }
        
        return $true
        
    } catch {
        $validator.LogResult("Python Environment", "FAILED", "Exception during Python validation", $_.Exception.Message)
        return $false
    } finally {
        $validator.EndPhase("Python Environment")
    }
}

function Test-NodeEnvironment {
    param([HeliosValidator]$validator)
    
    $validator.StartPhase("Node.js Environment")
    
    try {
        # Test Node.js
        $nodeVersion = node --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            $validator.LogResult("Node.js Environment", "FAILED", "Node.js not found or execution failed", $nodeVersion)
            return $false
        }
        
        # Test NPM
        $npmVersion = npm --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            $validator.LogResult("Node.js Environment", "FAILED", "NPM not found or execution failed", $npmVersion)
            return $false
        }
        
        $validator.LogResult("Node.js Environment", "PASSED", "Node.js environment validated", "Node: $nodeVersion, NPM: $npmVersion")
        return $true
        
    } catch {
        $validator.LogResult("Node.js Environment", "FAILED", "Exception during Node.js validation", $_.Exception.Message)
        return $false
    } finally {
        $validator.EndPhase("Node.js Environment")
    }
}

# ============================================================================
# PHASE 2: DEPENDENCY VALIDATION
# ============================================================================

function Test-PythonDependencies {
    param([HeliosValidator]$validator)
    
    $validator.StartPhase("Python Dependencies")
    
    try {
        # Method 1: Try live pip scan first (most reliable)
        $validator.LogResult("Python Dependencies", "INFO", "Scanning installed packages via pip", "")
        
        Push-Location (Join-Path $validator.ProjectPath "backend")
        try {
            $pipOutput = & $validator.PythonExe -m pip list --format=json 2>&1
            if ($LASTEXITCODE -eq 0) {
                # Verify output is valid JSON before parsing
                $rawOutput = $pipOutput -join "`n"
                if ([string]::IsNullOrWhiteSpace($rawOutput) -or $rawOutput.Trim() -eq '.' -or $rawOutput.Trim() -eq '') {
                    throw "Empty or invalid pip output"
                }
                
                $installedPackages = $rawOutput | ConvertFrom-Json
                $validator.LogResult("Python Dependencies", "INFO", "Pip scan successful", "Found $($installedPackages.Count) packages")
                
                # Check critical dependencies
                $critical_deps = @('flask', 'numpy', 'pandas', 'scikit-learn')
                $optional_deps = @('torch', 'matplotlib', 'scipy')
                
                $packageLookup = @{}
                foreach ($pkg in $installedPackages) {
                    $packageLookup[$pkg.name.ToLower()] = $pkg.version
                }
                
                $missing_critical = @()
                $missing_optional = @()
                $installed = @()
                
                foreach ($dep in $critical_deps) {
                    if ($packageLookup.ContainsKey($dep)) {
                        $installed += "$dep ($($packageLookup[$dep]))"
                    } else {
                        $missing_critical += $dep
                    }
                }
                
                foreach ($dep in $optional_deps) {
                    if ($packageLookup.ContainsKey($dep)) {
                        $installed += "$dep ($($packageLookup[$dep]))"
                    } else {
                        $missing_optional += $dep
                    }
                }
                
                if ($missing_critical.Count -gt 0) {
                    $validator.LogResult("Python Dependencies", "FAILED", "Critical dependencies missing", "Missing: $($missing_critical -join ', ')")
                    return $false
                }
                
                if ($missing_optional.Count -gt 0) {
                    $validator.LogResult("Python Dependencies", "WARNING", "Optional dependencies missing", "Missing: $($missing_optional -join ', ')")
                }
                
                $validator.LogResult("Python Dependencies", "PASSED", "Dependencies validated via pip", "Installed: $($installed.Count), Critical: $($critical_deps.Count - $missing_critical.Count)/$($critical_deps.Count)")
                return $true
            }
        } catch {
            $validator.LogResult("Python Dependencies", "WARNING", "Pip scan failed, trying importlib method", $_.Exception.Message)
        }
        
        # Method 2: Fallback to importlib check
        $validator.LogResult("Python Dependencies", "INFO", "Using importlib validation method", "")
        
        $testScript = @"
import sys
import importlib
import json

results = {}
critical_deps = ['flask', 'numpy', 'pandas', 'sklearn']
optional_deps = ['torch', 'matplotlib', 'scipy']

for dep in critical_deps + optional_deps:
    try:
        # Handle scikit-learn special case
        module_name = 'sklearn' if dep == 'scikit-learn' else dep
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        results[dep] = {'status': 'installed', 'version': version, 'critical': dep in critical_deps}
    except ImportError:
        results[dep] = {'status': 'missing', 'version': None, 'critical': dep in critical_deps}

print(json.dumps(results, indent=2))
"@
        
        # Write script to temp file to avoid shell escaping issues
        $tempScript = Join-Path ([System.IO.Path]::GetTempPath()) "helios_dep_check.py"
        $testScript | Out-File -FilePath $tempScript -Encoding UTF8
        
        try {
            $output = & $validator.PythonExe $tempScript 2>&1
            Remove-Item $tempScript -Force -ErrorAction SilentlyContinue
            
            if ($LASTEXITCODE -ne 0) {
                $validator.LogResult("Python Dependencies", "FAILED", "Dependency check script failed", $output)
                return $false
            }
            
            # Verify output before parsing
            $rawOutput = ($output | Out-String).Trim()
            if ([string]::IsNullOrWhiteSpace($rawOutput) -or $rawOutput -eq '.' -or $rawOutput -eq '') {
                $validator.LogResult("Python Dependencies", "FAILED", "Empty output from dependency check", "No valid JSON returned")
                return $false
            }
            
            $validator.LogResult("Python Dependencies", "INFO", "Raw dependency output received", "Length: $($rawOutput.Length) chars")
            
            try {
                $deps = $rawOutput | ConvertFrom-Json
                $missing_critical = @()
                $missing_optional = @()
                $installed = @()
                
                foreach ($dep in $deps.PSObject.Properties) {
                    $name = $dep.Name
                    $info = $dep.Value
                    
                    if ($info.status -eq "installed") {
                        $installed += "$name ($($info.version))"
                    } elseif ($info.critical) {
                        $missing_critical += $name
                    } else {
                        $missing_optional += $name
                    }
                }
                
                if ($missing_critical.Count -gt 0) {
                    $validator.LogResult("Python Dependencies", "FAILED", "Critical dependencies missing", "Missing: $($missing_critical -join ', ')")
                    return $false
                }
                
                if ($missing_optional.Count -gt 0) {
                    $validator.LogResult("Python Dependencies", "WARNING", "Optional dependencies missing", "Missing: $($missing_optional -join ', ')")
                }
                
                $validator.LogResult("Python Dependencies", "PASSED", "Dependencies validated via importlib", "Installed: $($installed.Count), Missing optional: $($missing_optional.Count)")
                return $true
                
            } catch {
                $validator.LogResult("Python Dependencies", "FAILED", "Failed to parse dependency JSON", "Error: $($_.Exception.Message), Raw output: $($rawOutput.Substring(0, [Math]::Min(200, $rawOutput.Length)))")
                return $false
            }
        } finally {
            Remove-Item $tempScript -Force -ErrorAction SilentlyContinue
        }
        
    } catch {
        $validator.LogResult("Python Dependencies", "FAILED", "Exception during dependency validation", $_.Exception.Message)
        return $false
    } finally {
        Pop-Location
        $validator.EndPhase("Python Dependencies")
    }
}

# ============================================================================
# PHASE 3: DATABASE CONNECTIVITY
# ============================================================================

function Test-DatabaseConnectivity {
    param([HeliosValidator]$validator)
    
    $validator.StartPhase("Database Connectivity")
    
    try {
        $testScript = @"
import sys
import os
sys.path.insert(0, '.')

try:
    import sqlite3
    # Test basic SQLite functionality
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE test (id INTEGER PRIMARY KEY)')
    cursor.execute('INSERT INTO test (id) VALUES (1)')
    result = cursor.fetchone()
    conn.close()
    print('SUCCESS: Database connectivity verified')
except Exception as e:
    print(f'ERROR: {str(e)}')
    sys.exit(1)
"@
        
        Push-Location (Join-Path $validator.ProjectPath "backend")
        try {
            $output = $testScript | & $validator.PythonExe 2>&1
            if ($LASTEXITCODE -eq 0 -and $output -match "SUCCESS") {
                $validator.LogResult("Database Connectivity", "PASSED", "Database connectivity verified", $output)
                return $true
            } else {
                $validator.LogResult("Database Connectivity", "FAILED", "Database connectivity test failed", $output)
                return $false
            }
        } finally {
            Pop-Location
        }
        
    } catch {
        $validator.LogResult("Database Connectivity", "FAILED", "Exception during database validation", $_.Exception.Message)
        return $false
    } finally {
        $validator.EndPhase("Database Connectivity")
    }
}

# ============================================================================
# PHASE 4: SYNTAX VALIDATION
# ============================================================================

function Test-ServerSyntax {
    param([HeliosValidator]$validator)
    
    $validator.StartPhase("Server Syntax Check")
    
    try {
        $serverPath = Join-Path $validator.ProjectPath "backend\server.py"
        
        if (-not (Test-Path $serverPath)) {
            $validator.LogResult("Server Syntax Check", "FAILED", "Server file not found", "Path: $serverPath")
            return $false
        }
        
        Push-Location (Join-Path $validator.ProjectPath "backend")
        try {
            $output = & $validator.PythonExe -m py_compile server.py 2>&1
            if ($LASTEXITCODE -eq 0) {
                $validator.LogResult("Server Syntax Check", "PASSED", "Server syntax validated", "No syntax errors found")
                return $true
            } else {
                $validator.LogResult("Server Syntax Check", "FAILED", "Server syntax errors detected", $output)
                return $false
            }
        } finally {
            Pop-Location
        }
        
    } catch {
        $validator.LogResult("Server Syntax Check", "FAILED", "Exception during syntax validation", $_.Exception.Message)
        return $false
    } finally {
        $validator.EndPhase("Server Syntax Check")
    }
}

# ============================================================================
# PHASE 5: INTEGRATION TESTING
# ============================================================================

function Test-Integration {
    param([HeliosValidator]$validator)
    
    $validator.StartPhase("Integration Tests")
    
    try {
        $testPath = Join-Path $validator.ProjectPath "backend\test_integration_basic.py"
        
        if (-not (Test-Path $testPath)) {
            $validator.LogResult("Integration Tests", "WARNING", "Integration test file not found", "Path: $testPath")
            return $true  # Not critical
        }
        
        Push-Location (Join-Path $validator.ProjectPath "backend")
        try {
            $output = & $validator.PythonExe test_integration_basic.py 2>&1 | Out-String
            
            if ($LASTEXITCODE -eq 0 -and $output -match "SUCCESS|OK") {
                # Parse test results
                if ($output -match "Ran (\d+) tests.*OK") {
                    $testCount = $matches[1]
                    $validator.LogResult("Integration Tests", "PASSED", "Integration tests completed successfully", "Tests run: $testCount")
                } else {
                    $validator.LogResult("Integration Tests", "PASSED", "Integration tests completed", "Output indicates success")
                }
                return $true
            } else {
                $validator.LogResult("Integration Tests", "FAILED", "Integration tests failed", $output.Substring(0, [Math]::Min(500, $output.Length)))
                return $false
            }
        } finally {
            Pop-Location
        }
        
    } catch {
        $validator.LogResult("Integration Tests", "FAILED", "Exception during integration testing", $_.Exception.Message)
        return $false
    } finally {
        $validator.EndPhase("Integration Tests")
    }
}

# ============================================================================
# REPORTING AND EXPORT FUNCTIONS
# ============================================================================

function Export-ValidationResults {
    param(
        [HeliosValidator]$validator,
        [string]$format,
        [string]$exportPath
    )
    
    if (-not $exportPath) {
        $exportPath = $validator.ProjectPath
    }
    
    $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    
    switch ($format) {
        "JSON" {
            $jsonPath = Join-Path $exportPath "validation-results_$timestamp.json"
            $exportData = @{
                Session = @{
                    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                    Duration = $validator.SessionTimer.ElapsedMilliseconds
                    ProjectPath = $validator.ProjectPath
                    VenvPath = $validator.VenvPath
                    PythonExe = $validator.PythonExe
                }
                Results = $validator.Results
                Summary = Get-ValidationSummary $validator
            }
            $exportData | ConvertTo-Json -Depth 5 | Out-File $jsonPath -Encoding UTF8
            Write-Host "JSON report exported: $jsonPath" -ForegroundColor Green
        }
        
        "CSV" {
            $csvPath = Join-Path $exportPath "validation-results_$timestamp.csv"
            $validator.Results | Export-Csv $csvPath -NoTypeInformation -Encoding UTF8
            Write-Host "CSV report exported: $csvPath" -ForegroundColor Green
        }
        
        "XML" {
            $xmlPath = Join-Path $exportPath "validation-results_$timestamp.xml"
            $validator.Results | Export-Clixml $xmlPath
            Write-Host "XML report exported: $xmlPath" -ForegroundColor Green
        }
        
        "All" {
            Export-ValidationResults $validator "JSON" $exportPath
            Export-ValidationResults $validator "CSV" $exportPath
            Export-ValidationResults $validator "XML" $exportPath
        }
    }
}

function Get-ValidationSummary {
    param([HeliosValidator]$validator)
    
    $total = $validator.Results.Count
    $passed = ($validator.Results | Where-Object { $_.Status -eq "PASSED" }).Count
    $failed = ($validator.Results | Where-Object { $_.Status -eq "FAILED" }).Count
    $warnings = ($validator.Results | Where-Object { $_.Status -eq "WARNING" }).Count
    
    return @{
        Total = $total
        Passed = $passed
        Failed = $failed
        Warnings = $warnings
        SuccessRate = if ($total -gt 0) { [Math]::Round(($passed / $total) * 100, 2) } else { 0 }
        Duration = $validator.SessionTimer.ElapsedMilliseconds
    }
}

function Write-ValidationSummary {
    param([HeliosValidator]$validator)
    
    $summary = Get-ValidationSummary $validator
    
    Write-Host "`n" + "=" * 80 -ForegroundColor Cyan
    Write-Host "HELIOS SYSTEM VALIDATION SUMMARY" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Cyan
    
    Write-Host "Session Duration: $($summary.Duration)ms" -ForegroundColor White
    Write-Host "Total Validations: $($summary.Total)" -ForegroundColor White
    Write-Host "Passed: $($summary.Passed)" -ForegroundColor Green
    Write-Host "Failed: $($summary.Failed)" -ForegroundColor Red
    Write-Host "Warnings: $($summary.Warnings)" -ForegroundColor Yellow
    Write-Host "Success Rate: $($summary.SuccessRate)%" -ForegroundColor $(if ($summary.SuccessRate -ge 90) { "Green" } elseif ($summary.SuccessRate -ge 70) { "Yellow" } else { "Red" })
    
    # Overall assessment
    if ($summary.Failed -eq 0) {
        Write-Host "`nALL VALIDATIONS PASSED!" -ForegroundColor Green
        Write-Host "System is ready for deployment." -ForegroundColor Green
    } elseif ($summary.Failed -le 2) {
        Write-Host "`nMINOR ISSUES DETECTED" -ForegroundColor Yellow
        Write-Host "Review failed validations before deployment." -ForegroundColor Yellow
    } else {
        Write-Host "`nCRITICAL ISSUES DETECTED" -ForegroundColor Red
        Write-Host "System requires attention before deployment." -ForegroundColor Red
    }
    
    Write-Host "=" * 80 -ForegroundColor Cyan
}

# ============================================================================
# MAIN EXECUTION FLOW
# ============================================================================

function Main {
    param(
        [bool]$Quick,
        [string]$OutputFormat,
        [string]$LogLevel,
        [string]$ExportPath
    )
    
    $projectPath = $PSScriptRoot
    if (-not $projectPath) {
        $projectPath = "C:\Users\RobMo\OneDrive\Documents\helios"
    }
    
    # Initialize validator
    $validator = [HeliosValidator]::new($projectPath, $LogLevel)
    
    Write-Host "HELIOS SYSTEM VALIDATION FRAMEWORK v2.0.0" -ForegroundColor Magenta
    Write-Host "Project: $projectPath" -ForegroundColor Gray
    Write-Host "Mode: $(if ($Quick) { 'Quick' } else { 'Full' })" -ForegroundColor Gray
    Write-Host "Log Level: $LogLevel" -ForegroundColor Gray
    Write-Host "=" * 80 -ForegroundColor Magenta
    
    # Execute validation phases
    $phases = @(
        @{ Name = "Python Environment"; Function = { Test-PythonEnvironment $validator } },
        @{ Name = "Node.js Environment"; Function = { Test-NodeEnvironment $validator } },
        @{ Name = "Python Dependencies"; Function = { Test-PythonDependencies $validator } },
        @{ Name = "Database Connectivity"; Function = { Test-DatabaseConnectivity $validator } },
        @{ Name = "Server Syntax Check"; Function = { Test-ServerSyntax $validator } }
    )
    
    if (-not $Quick) {
        $phases += @{ Name = "Integration Tests"; Function = { Test-Integration $validator } }
    }
    
    # Run all phases
    foreach ($phase in $phases) {
        try {
            & $phase.Function
        } catch {
            $validator.LogResult($phase.Name, "FAILED", "Phase execution failed", $_.Exception.Message)
        }
    }
    
    # Generate summary and reports
    Write-ValidationSummary $validator
    
    if ($OutputFormat -ne "Console") {
        Export-ValidationResults $validator $OutputFormat $ExportPath
    }
    
    # Set exit code based on results
    $summary = Get-ValidationSummary $validator
    exit $(if ($summary.Failed -eq 0) { 0 } else { 1 })
}

# Execute main function
Main -Quick:$Quick -OutputFormat $OutputFormat -LogLevel $LogLevel -ExportPath $ExportPath
