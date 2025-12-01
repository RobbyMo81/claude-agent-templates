@echo off
REM Helios Docker Management Script for Windows

setlocal enabledelayedexpansion

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    exit /b 1
)

REM Parse command line arguments
set "command=%1"
set "service=%2"

if "%command%"=="" set "command=help"

if "%command%"=="start-prod" (
    echo [INFO] Starting Helios in production mode...
    docker-compose up -d
    echo [SUCCESS] Helios is running in production mode
    echo [INFO] Frontend: http://localhost
    echo [INFO] Backend API: http://localhost:5001
    goto :end
)

if "%command%"=="start-dev" (
    echo [INFO] Starting Helios in development mode...
    docker-compose --profile dev up -d
    echo [SUCCESS] Helios is running in development mode
    echo [INFO] Frontend: http://localhost:5173
    echo [INFO] Backend API: http://localhost:5001
    goto :end
)

if "%command%"=="stop" (
    echo [INFO] Stopping Helios services...
    docker-compose down
    echo [SUCCESS] All services stopped
    goto :end
)

if "%command%"=="restart" (
    echo [INFO] Restarting Helios services...
    docker-compose down
    docker-compose up -d
    echo [SUCCESS] Services restarted
    goto :end
)

if "%command%"=="logs" (
    if "%service%"=="" (
        echo [INFO] Showing logs for all services...
        docker-compose logs -f
    ) else (
        echo [INFO] Showing logs for %service%...
        docker-compose logs -f %service%
    )
    goto :end
)

if "%command%"=="status" (
    echo [INFO] Service Status:
    docker-compose ps
    echo.
    echo [INFO] Resource Usage:
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    goto :end
)

if "%command%"=="rebuild" (
    if "%service%"=="" (
        echo [INFO] Rebuilding all services...
        docker-compose build --no-cache
    ) else (
        echo [INFO] Rebuilding %service%...
        docker-compose build --no-cache %service%
    )
    echo [SUCCESS] Rebuild completed
    goto :end
)

if "%command%"=="cleanup" (
    echo [WARNING] This will remove all containers, volumes, and images.
    set /p "response=Are you sure? (y/N): "
    if /i "!response!"=="y" (
        echo [INFO] Cleaning up...
        docker-compose down -v --rmi all
        docker system prune -f
        echo [SUCCESS] Cleanup completed
    ) else (
        echo [INFO] Cleanup cancelled
    )
    goto :end
)

if "%command%"=="health" (
    echo [INFO] Checking service health...

    REM Check frontend
    curl -s http://localhost/ >nul 2>&1
    if %errorlevel% equ 0 (
        echo [SUCCESS] Frontend is healthy
    ) else (
        echo [ERROR] Frontend is not responding
    )

    REM Check backend
    curl -s http://localhost:5001/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo [SUCCESS] Backend is healthy
    ) else (
        echo [ERROR] Backend is not responding
    )
    goto :end
)

REM Default: show help
echo Helios Docker Management Script
echo.
echo Usage: %0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   start-prod          Start production environment
echo   start-dev           Start development environment
echo   stop                Stop all services
echo   restart             Restart all services
echo   logs [service]      View logs (optionally for specific service)
echo   status              Show service status and resource usage
echo   rebuild [service]   Rebuild services (optionally specific service)
echo   cleanup             Remove all containers, volumes, and images
echo   health              Check service health
echo   help                Show this help message
echo.
echo Examples:
echo   %0 start-prod       # Start in production mode
echo   %0 start-dev        # Start in development mode
echo   %0 logs frontend    # View frontend logs
echo   %0 rebuild backend  # Rebuild only backend

:end
endlocal
