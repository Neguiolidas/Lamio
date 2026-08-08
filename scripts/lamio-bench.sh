#!/usr/bin/env bash
# Lamio benchmark script - measures MoE inference performance.
# Works on Linux (bash) and Windows (Git Bash / WSL).
# Usage: ./lamio-bench.sh /path/to/model.gguf /path/to/llama-cli [label]
set -euo pipefail

MODEL="${1:?Usage: lamio-bench.sh <model.gguf> <llama-cli-path> [label]}"
CLI="${2:?Usage: lamio-bench.sh <model.gguf> <llama-cli-path> [label]}"
LABEL="${3:-lamio-bench}"
PROMPT="The quick brown fox jumps over the lazy dog."
N_TOKENS=20
CTX=512
THREADS="${LAMIO_THREADS:-4}"
NGL="${LAMIO_NGL:-0}"
TIMEOUT="${LAMIO_TIMEOUT:-120}"

run_bench() {
  local desc="$1"
  shift
  local out
  echo "=== $desc ==="
  out=$("$CLI" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n "$N_TOKENS" \
    -t "$THREADS" \
    -ngl "$NGL" \
    -c "$CTX" \
    --no-display-prompt \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    -fa auto \
    --simple-io \
    "$@" \
    2>&1) || true
  # Extract token generation stats from llama.cpp output
  local tps
  tps=$(echo "$out" | grep -oP 'tg\d+:\s+\K[\d.]+' | tail -1) || tps="N/A"
  local rss
  rss=$(echo "$out" | grep -i 'rss' | tail -1) || rss="N/A"
  # Lamio tier stats (from stderr)
  local tier
  tier=$(echo "$out" | grep 'lamio tier stats:' | tail -1) || tier="N/A"
  printf "  tokens/s: %s\n  tier: %s\n" "$tps" "$tier"
  echo "$out" | grep -E '^(llama_|lamio )' | head -10
  echo
}

echo "Lamio Benchmark: $LABEL"
echo "Model: $MODEL"
echo "Threads: $THREADS, GPU layers: $NGL, Context: $CTX"
echo "Timeout per test: ${TIMEOUT}s"
echo

# Test 1: baseline (no tiering, default top-k)
run_bench "Baseline (no tiering, top-k=8)" \
  --lamio-tier-budget 0

# Test 2: tiering on, default top-k
run_bench "Tiering on (4GB budget, top-k=8)" \
  --lamio-tier-budget 4096

# Test 3: tiering + top-k reduced
run_bench "Tiering on (4GB budget, top-k=4)" \
  --lamio-tier-budget 4096 \
  --lamio-expert-k 4

# Test 4: tiering + top-k=2 (aggressive)
run_bench "Tiering on (4GB budget, top-k=2)" \
  --lamio-tier-budget 4096 \
  --lamio-expert-k 2

# Test 5: tiering 8GB budget
run_bench "Tiering on (8GB budget, top-k=8)" \
  --lamio-tier-budget 8192

echo "=== Benchmark complete ==="
