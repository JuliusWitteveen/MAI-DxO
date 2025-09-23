@echo off
echo.
echo ========================================
echo   MAI-DxO Web Server - Quick Start
echo ========================================
echo.
echo Starting server on http://127.0.0.1:8000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the server directly without dependency checks
python -m uvicorn webui.server:app --host 127.0.0.1 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Server failed to start!
    echo.
    echo Possible issues:
    echo 1. Python not found - check Python is installed
    echo 2. Missing packages - run: pip install fastapi uvicorn
    echo 3. Port 8000 in use - close other applications
    echo ========================================
    pause
)
