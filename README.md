# Lamio

MoE expert tiering for [llama.cpp](https://github.com/ggerganov/llama.cpp). This repo is
a standalone fork of llama.cpp with Lamio integrated.

Started July 2026. MIT license.

## Build

```sh
cmake -B build -S . -DGGML_CUDA=ON   # use -DGGML_CUDA=OFF for CPU-only
cmake --build build --target llama-server llama-cli -j$(nproc)
```

## Quick start

### Linux

```bash
git clone https://github.com/Neguiolidas/Lamio.git
cd Lamio
bash build.sh
# place your .gguf model in models/
bash start.sh
```

Open http://localhost:8090

### Windows

```cmd
git clone https://github.com/Neguiolidas/Lamio.git
cd Lamio
build.bat
REM place your .gguf model in models\
start.bat
```

Open http://localhost:8090

`build.sh` / `build.bat` detects Git, CMake, compiler and CUDA and builds only what is missing
(only the `llama-server` target, with limited parallel jobs).
`start.sh` / `start.bat` starts the server in router mode with sensible defaults.

### Server configuration (start.sh)

| Variable | Default | Description |
|---|---|---|
| LAMIO_PORT | 8090 | Server port |
| LAMIO_CTX | 2048 | Context window (tokens) |
| LAMIO_THREADS | 8 | CPU threads |
| LAMIO_NGL | 40 | GPU layers to offload |
| LAMIO_TIER_BUDGET | 4096 | Expert cache budget (MiB) |
| LAMIO_EXPERT_K | 4 | Active experts per token |

### CLI usage

```sh
./build/bin/llama-cli \
  -m model.gguf \
  -p "Hello" \
  -n 128 \
  -t 8 \
  -ngl 40 \
  -c 2048 \
  --lamio-tier-budget 4096 \
  --lamio-expert-k 4 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -fa auto \
  --simple-io
```

## Lamio flags

| Flag | Description |
|---|---|
| `--lamio-tier-budget N` | RAM budget for expert cache in MiB (0 = disabled) |
| `--lamio-expert-k N` | Override number of active experts per token (0 = model default) |

## Telemetry

Tier cache stats are printed to stderr on shutdown:

```
lamio tier stats: hits=42 misses=8 evictions=3 hit_rate=84.0% bytes_loaded=4.50 MB load_time=120.3 ms used=3.20 MB capacity=4096.00 MB
```

The server also exposes `GET /lamio/tier-stats`:

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

```sh
chmod +x scripts/lamio-bench.sh
./scripts/lamio-bench.sh /path/to/model.gguf /path/to/llama-cli
```

Tests tiering on/off, expert top-K, and reasoning overhead.

## Architecture

- `src/lamio/tier_manager.cpp` - LFRU cache with per-tensor-type keys
- `src/lamio/tier_bridge.cpp` - Bridge between llama.cpp and tier_manager
- `src/lamio/lamio_eval_callback.cpp` - Intercepts MoE ops, triggers load + prefetch
- `src/lamio/expert_loader.cpp` - pread-based expert tensor loader

## License

MIT. Based on [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT).