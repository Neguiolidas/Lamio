#!/usr/bin/env bash
# Build do Lamio. Roda em Linux, Git Bash e WSL.
# Compila apenas o llama-server (nao o projeto inteiro) com jobs limitados.
# Uso: bash build.sh  |  NOJOBS=2 bash build.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Limit jobs to avoid OOM on low-RAM boxes.
CORES=$(nproc 2>/dev/null || echo 4)
JOBS="${NOJOBS:-$(( CORES > 2 ? CORES / 2 : 1 ))}"

# Detect CUDA
CUDA_ARGS=""
if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA detectado: $(nvcc --version 2>&1 | grep release | head -1)"
    CUDA_ARGS="-DGGML_CUDA=ON"
else
    echo "CUDA nao detectado. Compilando sem GPU (so CPU)."
    echo "Para compilar com CUDA, instale o CUDA Toolkit e rode novamente."
fi

echo ""
echo "Configurando build (jobs=${JOBS})..."
cmake -B build -S . $CUDA_ARGS 2>&1 | tail -5

echo ""
echo "Compilando llama-server (jobs=${JOBS}, isso pode demorar)..."
cmake --build build -j"$JOBS" --target llama-server 2>&1 | tail -12

# Find the binary
SERVER=""
for candidate in \
    "build/bin/llama-server" \
    "build/bin/Release/llama-server" \
    "build/bin/Release/llama-server.exe" \
    "build/bin/llama-server.exe"; do
    if [ -f "$candidate" ]; then
        SERVER="$candidate"
        break
    fi
done

if [ -z "$SERVER" ]; then
    echo ""
    echo "Erro: llama-server nao encontrado apos compilacao."
    echo "Veja os erros acima."
    exit 1
fi

echo ""
echo "Build concluido com sucesso."
echo "  Servidor: $SERVER"
echo ""

# Verify Lamio flags were compiled in
if "$SERVER" --help 2>&1 | grep -q 'lamio-tier-budget'; then
    echo "  Lamio tiering: OK"
else
    echo "  Aviso: flag --lamio-tier-budget nao encontrada."
fi

echo ""
echo "Para subir o servidor: bash start.sh"