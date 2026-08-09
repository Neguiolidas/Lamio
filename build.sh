#!/usr/bin/env bash
# Build do Lamio. Roda em Linux, Git Bash e WSL.
# Uso: bash build.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Erro: pasta llama.cpp nao encontrada em $LLAMA_DIR"
    echo "Rode: git clone --recurse-submodules https://github.com/Neguiolidas/Lamio.git"
    exit 1
fi

cd "$LLAMA_DIR"

# Detectar CUDA
CUDA_ARGS=""
if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA detectado: $(nvcc --version 2>&1 | grep release | head -1)"
    CUDA_ARGS="-DGGML_CUDA=ON"
else
    echo "CUDA nao detectado. Compilando sem GPU (so CPU)."
    echo "Para compilar com CUDA, instale o CUDA Toolkit e rode novamente."
fi

echo ""
echo "Configurando build..."
cmake -B build -S . $CUDA_ARGS 2>&1 | tail -5

echo ""
echo "Compilando (isso pode demorar varios minutos)..."
cmake --build build -j 2>&1 | tail -10

# Verificar resultado
if [ -f "build/bin/llama-server" ]; then
    SERVER="build/bin/llama-server"
elif [ -f "build/bin/Release/llama-server" ]; then
    SERVER="build/bin/Release/llama-server"
elif [ -f "build/bin/Release/llama-server.exe" ]; then
    SERVER="build/bin/Release/llama-server.exe"
else
    echo ""
    echo "Erro: llama-server nao encontrado apos compilacao."
    echo "Verifique os erros acima."
    exit 1
fi

echo ""
echo "Build concluido com sucesso."
echo "  Servidor: $SERVER"
echo ""

# Verificar se Lamio foi incluido
if "$SERVER" --help 2>&1 | grep -q 'lamio-tier-budget'; then
    echo "  Lamio tiering: OK"
else
    echo "  Aviso: flag --lamio-tier-budget nao encontrada."
fi

echo ""
echo "Para subir o servidor, rode:"
echo "  bash start.sh"
