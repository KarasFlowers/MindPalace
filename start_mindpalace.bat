@echo off
setlocal

cd /d "%~dp0"
title MindPalace

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run the initial setup first:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\Activate.ps1
    echo   pip install -e .
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m src %*
set "exit_code=%errorlevel%"

if not "%exit_code%"=="0" (
    echo.
    echo MindPalace exited with code %exit_code%.
    pause
)

exit /b %exit_code%
