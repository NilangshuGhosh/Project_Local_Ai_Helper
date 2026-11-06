@echo off
setlocal EnableDelayedExpansion

title Local RAG AI Knowledge Assistant

:: 1. Ensure working directory is always pinned to the script directory
cd /d "%~dp0"

echo ===================================================
echo     Local RAG AI Assistant - Starting Up...
echo ===================================================

:: 2. Pre-flight Check: Virtual Environment Python
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
        echo [!] Virtualenv 'venv' not found. Using system Python...
    ) else (
        echo [ERROR] Python environment not detected.
        echo Please ensure Python 3.10+ is installed or set up the virtualenv.
        echo.
        pause
        exit /b 1
    )
)

:: 3. Pre-flight Check: Ollama Service Status
curl.exe -s -m 2 http://localhost:11434/api/tags >nul 2>&1
if !errorlevel! neq 0 (
    echo [*] Ollama is not running on localhost:11434.
    echo [*] Attempting to start Ollama daemon in background...

    where ollama >nul 2>&1
    if !errorlevel! equ 0 (
        start "" /B ollama serve >nul 2>&1
        timeout /t 3 /nobreak >nul
    ) else (
        if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
            start "" /B "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve >nul 2>&1
            timeout /t 3 /nobreak >nul
        ) else (
            echo [!] Ollama binary was not found in PATH or standard install directory.
            echo [!] Please launch Ollama manually if the health check fails.
        )
    )
)

:: 4. Launch Application
"%PYTHON_EXE%" main.py

:: 5. Keep window open if crashed unexpectedly so user can read traceback
if %errorlevel% neq 0 (
    echo.
    echo ===================================================
    echo [!] Application exited with an error (Code: %errorlevel%).
    echo ===================================================
    pause
)

endlocal

