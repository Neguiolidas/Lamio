# Lamio

MoE expert tiering for [llama.cpp](https://github.com/ggerganov/llama.cpp). Includes llama.cpp as a submodule so everything builds from one repo.

Started July 2026. MIT license.

## Quick start

```bash
# 1. Clone (with submodule)
git clone --recurse-submodules https://github.com/Neguiolidas/Lamio.git
cd Lamio

# 2. Build
bash build.sh

# 3. Place your .gguf model in llama.cpp/models/
cp /path/to/model.gguf llama.cpp/models/

# 4. Start
bash start.sh
```

Then open http://localhost:8090

## Windows

```cmd
REM 1. Clone (with submodule)
git clone --recurse-submodules https://github.com/Neguiolidas/Lamio.git
cd Lamio

REM 2. Build
build.bat

REM 3. Place your .gguf model in llama.cpp\models\

REM 4. Start
start.bat
```

Then open http://localhost:8090

## Configuration

Environment variables (start.sh only):

| Variable | Default | Description |
|---|---|---|
| LAMIO_PORT | 8090 | Server port |
| LAMIO_CTX | 2048 | Context window (tokens) |
| LAMIO_THREADS | 8 | CPU threads |
| LAMIO_NGL | 40 | GPU layers to offload |
| LAMIO_TIER_BUDGET | 4096 | Expert cache budget (MiB) |
| LAMIO_EXPERT_K | 4 | Active experts per token |

## What it does

MoE models (Mixtral, DeepSeek, Qwen MoE, etc.) have hundreds of expert tensors
but only activate a small subset per token. Lamio keeps inactive experts on
disk and loads them on demand, reducing RAM usage from the full model size to a
configurable budget.

## Flags

| Flag | Description |
|---|---|
| `--lamio-tier-budget N` | RAM budget for expert cache in MiB (0 = disabled) |
| `--lamio-expert-k N` | Override active experts per token (0 = model default) |

## Tier stats API

When tiering is enabled, `GET /lamio/tier-stats` returns:

```json
{
  "enabled": true,
  "hits": 1234,
  "misses": 56,
  "evictions": 12,
  "load_time_ms": 890.1,
  "bytes_loaded": 67108864,
  "used_mb": 3840.0,
  "capacity_mb": 4096.0,
  "n_layers": 40,
  "n_expert": 256,
  "n_expert_used": 4
}
```

## Benchmark

```bash
bash scripts/lamio-bench.sh model.gguf ./llama.cpp/build/bin/llama-cli "label"
```

8 tests covering tiering on/off, expert top-K, and reasoning overhead.

## Architecture

- `tier_manager.h/cpp` - LFRU cache with per-tensor-type keys, shallow-favoring budget
- `tier_bridge.h/cpp` - API between cache and llama.cpp model loader, next-layer prefetch
- `lamio_eval_callback.cpp` - Hook into ggml graph execution for expert selection and prefetch
- `expert_loader.h/cpp` - Disk-backed expert tensor loader using pread

## License

MIT. Based on [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT).
