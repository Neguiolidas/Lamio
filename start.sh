#!/usr/bin/env bash
# Subir o servidor Lamio. Roda em Linux, Git Bash e WSL.
# Uso: bash start.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"

# Encontrar o binario
SERVER=""
for candidate in \
    "$LLAMA_DIR/build/bin/llama-server" \
    "$LLAMA_DIR/build/bin/Release/llama-server" \
    "$LLAMA_DIR/build/bin/Release/llama-server.exe" \
    "$LLAMA_DIR/build/bin/llama-server.exe"; do
    if [ -x "$candidate" ] || [ -f "$candidate" ]; then
        SERVER="$candidate"
        break
    fi
done

if [ -z "$SERVER" ]; then
    echo "Erro: llama-server nao encontrado."
    echo "Rode primeiro: bash build.sh"
    exit 1
fi

# Criar pasta de modelos se nao existir
mkdir -p "$LLAMA_DIR/models"

# Configuracao padrao
PORT="${LAMIO_PORT:-8090}"
CTX="${LAMIO_CTX:-2048}"
THREADS="${LAMIO_THREADS:-8}"
NGL="${LAMIO_NGL:-40}"
TIER_BUDGET="${LAMIO_TIER_BUDGET:-4096}"
EXPERT_K="${LAMIO_EXPERT_K:-4}"

echo "Servidor: $SERVER"
echo "Porta: $PORT"
echo "Contexto: $CTX tokens"
echo "Threads: $THREADS"
echo "GPU layers: $NGL"
echo "Tier budget: $TIER_BUDGET MiB"
echo "Expert top-K: $EXPERT_K"
echo ""
echo "Modelos em: $LLAMA_DIR/models/"
echo "Acesse: http://localhost:$PORT"
echo "Pressione Ctrl+C para parar."
echo ""

exec "$SERVER" \
    --models-dir "$LLAMA_DIR/models" \
    --port "$PORT" \
    --host 127.0.0.1 \
    -c "$CTX" \
    -t "$THREADS" \
    -ngl "$NGL" \
    --lamio-tier-budget "$TIER_BUDGET" \
    --lamio-expert-k "$EXPERT_K"
