# Changelog

## [0.3] - 2026-08-08

Standalone fork of llama.cpp with MoE expert tiering.

### Added
- `src/lamio/`: LFRU tier cache (per-tensor-type), expert loader via pread,
  eval callback for on-demand expert load and next-layer prefetch
- `--lamio-tier-budget` (MiB) and `--lamio-expert-k` (N) CLI flags
- `GET /lamio/tier-stats` server endpoint
- `build.sh` / `build.bat`: detect deps, build only `llama-server` with
  limited parallel jobs
- `start.sh` / `start.bat`: start server in router mode
- `scripts/lamio-bench.sh`: benchmark tiering on/off, expert top-K, reasoning

### Notes
- Based on llama.cpp (MIT). This repo is a standalone fork.