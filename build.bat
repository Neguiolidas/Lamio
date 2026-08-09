@echo off
REM Build do Lamio para Windows
REM Uso: build.bat

echo.
echo  ==========================================
echo   LAMIO - Build
echo  ==========================================
echo.

set "LLAMA_DIR=%~dp0llama.cpp"

if not exist "%LLAMA_DIR%\CMakeLists.txt" (
    echo  Erro: llama.cpp nao encontrado em %LLAMA_DIR%
    echo  Rode: git clone --recurse-submodules https://github.com/Neguiolidas/Lamio.git
    pause
    exit /b 1
)

cd /d "%LLAMA_DIR%"

REM Detectar CUDA
where nvcc >nul 2>&1
if %errorlevel% equ 0 (
    echo  CUDA detectado
    set "CUDA_ARGS=-DGGML_CUDA=ON"
) else (
    echo  CUDA nao detectado. Compilando sem GPU.
    echo  Para compilar com CUDA, instale o CUDA Toolkit e rode novamente.
    set "CUDA_ARGS="
)

echo.
echo  Configurando build...
cmake -B build -S . %CUDA_ARGS%
if %errorlevel% neq 0 (
    echo  Erro na configuracao do CMake.
    pause
    exit /b 1
)

echo.
echo  Compilando (pode demorar varios minutos)...
cmake --build build --config Release -j
if %errorlevel% neq 0 (
    echo  Erro na compilacao.
    pause
    exit /b 1
)

echo.
echo  Build concluido com sucesso.
echo.
echo  Para subir o servidor, rode: start.bat
echo.
pause
