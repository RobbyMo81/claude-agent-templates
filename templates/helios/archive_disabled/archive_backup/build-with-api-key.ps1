# Helios Docker Build Script with API Key Su    Write-Host ""
    Write-Host " To run the container:" -ForegroundColor Green
    Write-Host "   docker run -d --name helios-container -p 3000:8080 helios-image" -ForegroundColor White
    Write-Host ""
    Write-Host " Then open: http://localhost:3000" -ForegroundColor Cyan (PowerShell)
# Usage: .\build-with-api-key.ps1

Write-Host " Building Helios with API Key Integration..." -ForegroundColor Green

# Check if .env.local exists and has API key
if (Test-Path ".env.local") {
    Write-Host " Found .env.local file" -ForegroundColor Blue
    
    # Extract API key from .env.local
    $envContent = Get-Content ".env.local"
    $apiKeyLine = $envContent | Where-Object { $_ -like "VITE_GEMINI_API_KEY=*" }
    
    if ($apiKeyLine) {
        $apiKey = ($apiKeyLine -split '=', 2)[1].Trim('"').Trim("'")
        
        if ($apiKey -eq "PLACEHOLDER_API_KEY" -or [string]::IsNullOrEmpty($apiKey)) {
            Write-Host "  WARNING: Using placeholder API key. Gemini features will be disabled." -ForegroundColor Yellow
            Write-Host "   To fix: Add your real Gemini API key to .env.local" -ForegroundColor Yellow
            $apiKey = "PLACEHOLDER_API_KEY"
        } else {
            $maskedKey = $apiKey.Substring(0, [Math]::Min(10, $apiKey.Length)) + "..."
            Write-Host " Found valid API key ($maskedKey)" -ForegroundColor Green
        }
    } else {
        $apiKey = "PLACEHOLDER_API_KEY"
    }
} else {
    Write-Host "  No .env.local found. Creating from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env.local"
    $apiKey = "PLACEHOLDER_API_KEY"
}

# Build Docker image with API key as build argument
Write-Host " Building Docker image..." -ForegroundColor Blue
docker build `
    --build-arg VITE_GEMINI_API_KEY="$apiKey" `
    --build-arg VITE_API_HOST="localhost" `
    --build-arg VITE_API_PORT="8080" `
    --build-arg VITE_API_PROTOCOL="http" `
    -t helios-image .

if ($LASTEXITCODE -eq 0) {
    Write-Host " Build complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host " To run the container:" -ForegroundColor Cyan
    Write-Host "   docker run -d --name helios-container -p 8080:8080 helios-image" -ForegroundColor White
    Write-Host ""
    Write-Host " Then open: http://localhost:8080" -ForegroundColor Cyan
} else {
    Write-Host " Build failed!" -ForegroundColor Red
    exit 1
}
