#!/usr/bin/env bash
# Benchmark do Lamio para Linux, Git Bash e WSL.
# Uso: ./lamio-bench.sh modelo.gguf llama-cli [rotulo]
set -u

if [ "$#" -lt 2 ]; then
  echo "Uso: $0 <modelo.gguf> <llama-cli> [rotulo]" >&2
  exit 2
fi

MODEL="$1"
CLI="$2"
LABEL="${3:-lamio-bench}"
PROMPT="The quick brown fox jumps over the lazy dog."
N_TOKENS="${LAMIO_N_TOKENS:-20}"
CTX="${LAMIO_CTX:-512}"
THREADS="${LAMIO_THREADS:-4}"
NGL="${LAMIO_NGL:-0}"
TIMEOUT="${LAMIO_TIMEOUT:-120}"

run_bench() {
  desc="$1"
  shift
  out_file="${TMPDIR:-/tmp}/lamio-bench-$$.out"
  printf '=== %s ===\n' "$desc"

  if command -v timeout >/dev/null 2>&1; then
    timeout "$TIMEOUT" "$CLI" \
      -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" \
      -t "$THREADS" -ngl "$NGL" -c "$CTX" \
      --no-display-prompt --cache-type-k q4_0 --cache-type-v q4_0 \
      -fa auto --simple-io -st "$@" >"$out_file" 2>&1
    exit_code=$?
  else
    echo "Aviso: o comando timeout não está disponível; o teste não terá limite." >&2
    "$CLI" \
      -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" \
      -t "$THREADS" -ngl "$NGL" -c "$CTX" \
      --no-display-prompt --cache-type-k q4_0 --cache-type-v q4_0 \
      -fa auto --simple-io -st "$@" >"$out_file" 2>&1
    exit_code=$?
  fi

  total_line=$(grep -i 'eval time:' "$out_file" | tail -1 || true)
  prompt_line=$(grep -i 'prompt eval time:' "$out_file" | tail -1 || true)
  total_tokens=$(printf '%s\n' "$total_line" | sed -n 's/.*\/ *\([0-9][0-9]*\) tokens.*/\1/p')
  total_speed=$(printf '%s\n' "$total_line" | sed -n 's/.*( *\([0-9.][0-9.]*\) tokens per second.*/\1/p')
  prompt_speed=$(printf '%s\n' "$prompt_line" | sed -n 's/.*( *\([0-9.][0-9.]*\) tokens per second.*/\1/p')

  [ -n "$total_tokens" ] || total_tokens="N/A"
  [ -n "$total_speed" ] || total_speed="N/A"
  [ -n "$prompt_speed" ] || prompt_speed="N/A"

  # Detecta delimitadores de raciocínio no texto exibido. Sem delimitadores,
  # o número exato de tokens de raciocínio não pode ser recuperado do log.
  reasoning_text=$(sed -n '/<think>/,/<\/think>/p; /<|thinking|>/,/<|\/thinking|>/p' "$out_file" || true)
  reasoning_chars=$(printf '%s' "$reasoning_text" | wc -c | tr -d ' ')
  if [ "$reasoning_chars" -gt 0 ]; then
    reasoning_tokens="detectado-no-texto"
    content_tokens="não determinável"
  else
    reasoning_tokens="não determinável"
    content_tokens="não determinável"
  fi

  tier=$(grep 'lamio tier stats:' "$out_file" | tail -1 || true)
  [ -n "$tier" ] || tier="não disponível"

  printf '  exit: %s\n' "$exit_code"
  printf '  tokens gerados: %s\n' "$total_tokens"
  printf '  tokens de raciocínio: %s\n' "$reasoning_tokens"
  printf '  tokens de conteúdo: %s\n' "$content_tokens"
  printf '  velocidade de geração: %s tok/s\n' "$total_speed"
  printf '  velocidade do prompt: %s tok/s\n' "$prompt_speed"
  printf '  tier: %s\n' "$tier"
  rm -f "$out_file"
  printf '\n'
}

echo "Benchmark do Lamio: $LABEL"
echo "Modelo: $MODEL"
echo "Threads: $THREADS, camadas GPU: $NGL, contexto: $CTX"
echo "Limite por teste: ${TIMEOUT}s"
echo

run_bench "Baseline, sem tiering, top-k do modelo" \
  --lamio-tier-budget 0
run_bench "Tiering, 4 GB, top-k do modelo" \
  --lamio-tier-budget 4096
run_bench "Tiering, 4 GB, top-k 4" \
  --lamio-tier-budget 4096 --lamio-expert-k 4
run_bench "Tiering, 4 GB, top-k 2" \
  --lamio-tier-budget 4096 --lamio-expert-k 2
run_bench "Tiering, 8 GB, top-k do modelo" \
  --lamio-tier-budget 8192
run_bench "Reasoning ligado, tiering 4 GB, top-k 4" \
  --lamio-tier-budget 4096 --lamio-expert-k 4 \
  --reasoning-format deepseek --reasoning on
run_bench "Reasoning desligado, tiering 4 GB, top-k 4" \
  --lamio-tier-budget 4096 --lamio-expert-k 4 \
  --reasoning-format deepseek --reasoning off
run_bench "Conteúdo puro, reasoning off, top-k 2" \
  --lamio-tier-budget 4096 --lamio-expert-k 2 \
  --reasoning-format deepseek --reasoning off

echo "=== Benchmark concluído ==="
