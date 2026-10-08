@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo Python is missing.
            echo Install Python 3.11+ from https://www.python.org/downloads/windows/
            pause
            exit /b 1
        )
        python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Failed to create the Python environment.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Setup complete.
echo First transcription may still download the selected Whisper model.
pause
exit /b 0

:failed
echo.
echo Setup failed. Check your internet connection and try again.
pause
exit /b 1
