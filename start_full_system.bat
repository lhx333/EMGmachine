@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Project virtual environment was not found: %~dp0.venv
    echo Create it and install dependencies first. See README.md.
    pause
    exit /b 1
)

set "PYTHONPATH=%~dp0src"
".venv\Scripts\python.exe" -m acquisition.app
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo [ERROR] Full System exited with code %APP_EXIT%.
    pause
)

exit /b %APP_EXIT%
