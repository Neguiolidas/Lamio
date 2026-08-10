@echo off
REM Lamio start script - starts llama-server in router mode with sensible defaults.
REM Configurable via environment variables (see README).
setlocal enabledelayedexpansion

cd /d "%~dp0"

set SERVER=build\bin\llama-server.exe
if "%LAMIO_PORT%"=="" set LAMIO_PORT=8090
if "%LAMIO_CTX%"=="" set LAMIO_CTX=2048
if "%LAMIO_THREADS%"=="" set LAMIO_THREADS=8
if "%LAMIO_NGL%"=="" set LAMIO_NGL=40
if "%LAMIO_TIER_BUDGET%"=="" set LAMIO_TIER_BUDGET=4096
if "%LAMIO_EXPERT_K%"=="" set LAMIO_EXPERT_K=4

REM find model in models\
if "%LAMIO_MODEL%"=="" (
    for %%f in (models\*.gguf) do (
        if "%LAMIO_MODEL%"=="" set LAMIO_MODEL=%%f
    )
)

if "%LAMIO_MODEL%"=="" (
    echo [Lamio] ERROR: no model found. Place a .gguf in models\ or set LAMIO_MODEL.
    exit /b 1
)

if not exist "%SERVER%" (
    echo [Lamio] ERROR: %SERVER% not found. Run: build.bat
    exit /b 1
)

echo [Lamio] Model: %LAMIO_MODEL%
echo [Lamio] Port: %LAMIO_PORT% ^| Ctx: %LAMIO_CTX% ^| Threads: %LAMIO_THREADS% ^| NGL: %LAMIO_NGL%
echo [Lamio] Tier budget: %LAMIO_TIER_BUDGET%MiB ^| Expert-K: %LAMIO_EXPERT_K%
echo [Lamio] Starting server...

"%SERVER%" ^
    -m "%LAMIO_MODEL%" ^
    --host 0.0.0.0 ^
    --port %LAMIO_PORT% ^
    -c %LAMIO_CTX% ^
    -t %LAMIO_THREADS% ^
    -ngl %LAMIO_NGL% ^
    --lamio-tier-budget %LAMIO_TIER_BUDGET% ^
    --lamio-expert-k %LAMIO_EXPERT_K% ^
    --cache-type-k q4_0 ^
    --cache-type-v q4_0 ^
    -fa auto
