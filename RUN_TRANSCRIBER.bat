@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo ============================================================
echo  Lecture Transcriber - Local
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating the local Python environment...

    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo.
            echo Python is not installed or not on PATH.
            echo Install Python 3.11 or newer from:
            echo https://www.python.org/downloads/windows/
            echo During installation, enable "Add python.exe to PATH".
            echo.
            pause
            exit /b 1
        )
        python -m venv .venv
    )

    if not exist ".venv\Scripts\python.exe" (
        echo Failed to create the Python environment.
        pause
        exit /b 1
    )

    echo Installing required packages...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto :install_failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :install_failed
    echo.
)

echo Checking Python dependencies...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade -r requirements.txt
if errorlevel 1 goto :install_failed
echo Dependencies ready.
echo.

".venv\Scripts\python.exe" transcribe.py %*
set CODE=%errorlevel%

echo.
if %CODE%==0 (
    echo Finished.
) else (
    echo The run stopped with an error, but completed chunks remain saved.
    echo Run the same recording again to resume.
)
echo.
pause
exit /b %CODE%

:install_failed
echo.
echo Package installation failed. Check your internet connection and run this file again.
pause
exit /b 1
