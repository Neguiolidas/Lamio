#!/usr/bin/env bash
# Lamio start script - starts llama-server in router mode with sensible defaults.
# Configurable via environment variables (see README).
set -euo pipefail

cd "$(dirname "$0")"

SERVER="./build/bin/llama-server"
PORT="${LAMIO_PORT:-8090}"
CTX="${LAMIO_CTX:-2048}"
THREADS="${LAMIO_THREADS:-8}"
NGL="${LAMIO_NGL:-40}"
TIER_BUDGET="${LAMIO_TIER_BUDGET:-4096}"
EXPERT_K="${LAMIO_EXPERT_K:-4}"
MODEL="${LAMIO_MODEL:-}"

# find model in models/
if [ -z "$MODEL" ]; then
    MODEL=$(ls models/*.gguf 2>/dev/null | head -1)
fi
if [ -z "$MODEL" ]; then
    echo "[Lamio] ERROR: no model found. Place a .gguf in models/ or set LAMIO_MODEL."
    exit 1
fi

if [ ! -f "$SERVER" ]; then
    echo "[Lamio] ERROR: $SERVER not found. Run: bash build.sh"
    exit 1
fi

echo "[Lamio] Model: $MODEL"
echo "[Lamio] Port: $PORT | Ctx: $CTX | Threads: $THREADS | NGL: $NGL"
echo "[Lamio] Tier budget: ${TIER_BUDGET}MiB | Expert-K: $EXPERT_K"
echo "[Lamio] Starting server..."

exec "$SERVER" \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    -c "$CTX" \
    -t "$THREADS" \
    -ngl "$NGL" \
    --lamio-tier-budget "$TIER_BUDGET" \
    --lamio-expert-k "$EXPERT_K" \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    -fa auto
