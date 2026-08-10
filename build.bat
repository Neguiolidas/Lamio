@echo off
REM Lamio build script - detects Git, CMake, compiler and CUDA.
REM Builds only the llama-server target (and llama-cli) with limited parallel jobs.
setlocal enabledelayedexpansion

cd /d "%~dp0"

set JOBS=2
set BUILD_DIR=build
set CUDA=auto

echo [Lamio] Build script

REM detect cmake
where cmake >nul 2>&1
if errorlevel 1 (
    echo [Lamio] ERROR: cmake not found. Install cmake ^>= 3.14.
    exit /b 1
)

REM detect CUDA
if "%CUDA%"=="auto" (
    where nvcc >nul 2>&1
    if errorlevel 1 (
        set CUDA=OFF
    ) else (
        set CUDA=ON
    )
)

if "%CUDA%"=="ON" (
    set CUDA_FLAG=-DGGML_CUDA=ON
    echo [Lamio] CUDA: ON
) else (
    set CUDA_FLAG=-DGGML_CUDA=OFF
    echo [Lamio] CUDA: OFF ^(CPU-only^)
)

REM configure
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo [Lamio] Configuring ^(%BUILD_DIR%^)...
    cmake -B %BUILD_DIR% -S . !CUDA_FLAG!
) else (
    echo [Lamio] Build dir exists, skipping configure
)

REM build (limited jobs)
echo [Lamio] Building llama-server + llama-cli ^(-j%JOBS%^)...
cmake --build %BUILD_DIR% --target llama-server llama-cli -j%JOBS%

echo [Lamio] Build complete: %BUILD_DIR%\bin\llama-server.exe
