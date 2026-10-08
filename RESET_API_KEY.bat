@echo off
cd /d "%~dp0"
if exist ".env" (
    del ".env"
    echo Saved legacy API key file removed.
) else (
    echo No saved API key file was found.
)
pause
