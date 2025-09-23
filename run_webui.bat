@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Quick start the MAI-DxO Web UI
set ROOT=%~dp0

if not exist "%ROOT%webui\server.py" (
  echo webui\server.py not found. Are you running from the project root?
  pause
  exit /b 1
)

REM Run the new startup script that handles Windows compatibility
python "%ROOT%start_webui.py"

if errorlevel 1 (
  pause
)

endlocal

