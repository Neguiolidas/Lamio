@echo off
REM Subir o servidor Lamio
REM Uso: start.bat

cd /d "%~dp0"
set "MODELS_DIR=%CD%\models"

REM Encontrar o binario
set "SERVER="
if exist "build\bin\Release\llama-server.exe" (
    set "SERVER=%CD%\build\bin\Release\llama-server.exe"
) else if exist "build\bin\Release\llama-server" (
    set "SERVER=%CD%\build\bin\Release\llama-server"
) else if exist "build\bin\llama-server.exe" (
    set "SERVER=%CD%\build\bin\llama-server.exe"
) else if exist "build\bin\llama-server" (
    set "SERVER=%CD%\build\bin\llama-server"
)

if "%SERVER%"=="" (
    echo  Erro: llama-server nao encontrado.
    echo  Rode primeiro: build.bat
    pause
    exit /b 1
)

REM Criar pasta de modelos se nao existir
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

echo.
echo  Servidor: %SERVER%
echo  Modelos:  %MODELS_DIR%
echo  Porta:    8090
echo  Acesse:   http://localhost:8090
echo  Pressione Ctrl+C para parar.
echo.

"%SERVER%" ^
    --models-dir "%MODELS_DIR%" ^
    --port 8090 ^
    --host 127.0.0.1 ^
    -c 2048 ^
    -t 8 ^
    -ngl 40 ^
    --lamio-tier-budget 4096 ^
    --lamio-expert-k 4

pause