@echo off
cd /d "%~dp0"
if not exist "transcripts" mkdir "transcripts"
start "" "%~dp0transcripts"
