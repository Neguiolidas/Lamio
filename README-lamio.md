# Lamio

MoE expert tiering for llama.cpp. Caches expert tensors in RAM with
per-tensor-type granularity (up/gate/down), LFRU eviction, shallow-favoring
budget distribution, and next-layer prefetch via posix_fadvise.

## Build

```sh
cmake -B build -DGGML_CUDA=ON   # use -DGGML_CUDA=OFF for CPU-only
cmake --build build -j$(nproc) --target llama-cli
```

## Usage

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

## Flags

| Flag | Description |
|------|-------------|
| `--lamio-tier-budget MiB` | RAM budget for MoE expert cache (0 = disabled) |
| `--lamio-expert-k N` | Override number of active experts per token (0 = model default) |

## Telemetry

Tier cache stats are printed to stderr on shutdown:

```
lamio tier stats: hits=42 misses=8 evictions=3 hit_rate=84.0% bytes_loaded=4.50 MB load_time=120.3 ms used=3.20 MB capacity=4096.00 MB
```

## Benchmark

```sh
chmod +x scripts/lamio-bench.sh
./scripts/lamio-bench.sh /path/to/model.gguf /path/to/llama-cli
```

Tests 5 configurations: baseline, tiering on (top-k 8), tiering + top-k 4,
tiering + top-k 2, and tiering 8GB budget.

## Web UI

The `ui/` directory contains a Vite + React dashboard for chatting with the
model and visualizing tier cache telemetry in real time.

```sh
cd ui && npm install && npm run dev
```

Requires a running `llama-server` instance on localhost:8080.

## Architecture

- `src/lamio/tier_manager.cpp` - LFRU cache with per-tensor-type keys
- `src/lamio/tier_bridge.cpp` - Bridge between llama.cpp and tier_manager
- `src/lamio/lamio_eval_callback.cpp` - Intercepts MoE ops, triggers load + prefetch
- `src/lamio/expert_loader.cpp` - pread-based expert tensor loader

## License

MIT
