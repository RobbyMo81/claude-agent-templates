#!/usr/bin/env pwsh
# Helios System Validation - Clean Version

param([switch]$Quick)

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "HELIOS SYSTEM VALIDATION" -ForegroundColor Cyan  
Write-Host "================================================================================" -ForegroundColor Cyan

$testsPassed = 0
$testsFailed = 0

function Test-Component($Name, $TestCode) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Testing: $Name" -ForegroundColor Cyan
    
    try {
        $result = & $TestCode
        if ($result) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] PASSED: $Name" -ForegroundColor Green
            $script:testsPassed++
        } else {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] FAILED: $Name" -ForegroundColor Red
            $script:testsFailed++
        }
    }
    catch {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] FAILED: $Name - $($_.Exception.Message)" -ForegroundColor Red
        $script:testsFailed++
    }
}

# Test 1: Python Environment
Test-Component "Python Environment" {
    try {
        $version = python --version 2>&1
        Write-Host "Python version: $version" -ForegroundColor White
        return $true
    }
    catch {
        return $false
    }
}

# Test 2: Node.js Environment
Test-Component "Node.js Environment" {
    try {
        $nodeVersion = node --version
        $npmVersion = npm --version
        Write-Host "Node.js: $nodeVersion, npm: $npmVersion" -ForegroundColor White
        return $true
    }
    catch {
        return $false
    }
}

# Test 3: Backend Files
Test-Component "Backend Files Check" {
    $backendPath = Join-Path $PSScriptRoot "backend"
    $serverFile = Join-Path $backendPath "server.py"
    $memoryFile = Join-Path $backendPath "memory_store.py"
    
    $serverExists = Test-Path $serverFile
    $memoryExists = Test-Path $memoryFile
    
    Write-Host "Server.py: $serverExists, MemoryStore: $memoryExists" -ForegroundColor White
    return $serverExists -and $memoryExists
}

# Test 4: Python Import Test
Test-Component "Python Import Test" {
    $backendPath = Join-Path $PSScriptRoot "backend"
    Push-Location $backendPath
    
    try {
        $result = python -c "import memory_store; print('SUCCESS')" 2>&1
        Pop-Location
        return $result -match "SUCCESS"
    }
    catch {
        Pop-Location
        return $false
    }
}

if (-not $Quick) {
    # Test 5: Integration Tests (Full mode only)
    Test-Component "Integration Tests" {
        $backendPath = Join-Path $PSScriptRoot "backend"
        Push-Location $backendPath
        
        try {
            $output = python test_integration_basic.py | Out-String
            Pop-Location
            
            if ($output -match "Success Rate: ([0-9.]+)%") {
                $successRate = [float]($matches[1])
                Write-Host "Success Rate: $successRate%" -ForegroundColor White
                return $successRate -ge 70
            }
            return $false
        }
        catch {
            Pop-Location
            return $false
        }
    }
}

Write-Host "" 
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "RESULTS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$totalTests = $testsPassed + $testsFailed
Write-Host "Total Tests: $totalTests" -ForegroundColor White
Write-Host "Passed: $testsPassed" -ForegroundColor Green
Write-Host "Failed: $testsFailed" -ForegroundColor Red

if ($testsFailed -eq 0) {
    Write-Host "ALL TESTS PASSED - SYSTEM READY!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "TESTS FAILED - CHECK SYSTEM" -ForegroundColor Red
    exit 1
}
