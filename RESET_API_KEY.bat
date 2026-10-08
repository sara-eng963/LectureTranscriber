@echo off
cd /d "%~dp0"
if exist ".env" (
    del ".env"
    echo Saved Gemini API key removed.
) else (
    echo No saved API key was found.
)
pause
