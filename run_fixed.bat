@echo off
cls
echo.
echo ==============================================================
echo            MAI-DxO Web UI - FIXED for Windows
echo ==============================================================
echo.
echo This version fixes the hang/crash issue on Windows by delaying
echo the mai_dx module import until it's actually needed.
echo.
echo Starting server...
echo ==============================================================
echo.

python start_fixed.py

if errorlevel 1 (
    echo.
    echo ==============================================================
    echo Server failed to start. See error above.
    echo ==============================================================
    pause
)
