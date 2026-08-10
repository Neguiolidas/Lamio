#!/usr/bin/env bash
# Lamio build script - detects Git, CMake, compiler and CUDA.
# Builds only the llama-server target (and llama-cli) with limited parallel jobs.
set -euo pipefail

cd "$(dirname "$0")"

JOBS="${LAMIO_JOBS:-2}"
BUILD_DIR="${LAMIO_BUILD_DIR:-build}"
CUDA="${LAMIO_CUDA:-auto}"

echo "[Lamio] Build script"

# detect cmake
if ! command -v cmake &>/dev/null; then
    echo "[Lamio] ERROR: cmake not found. Install cmake >= 3.14."
    exit 1
fi

# detect compiler
if ! command -v cc &>/dev/null && ! command -v gcc &>/dev/null && ! command -v clang &>/dev/null; then
    echo "[Lamio] ERROR: no C compiler found."
    exit 1
fi

# detect CUDA
CUDA_FLAG=""
if [ "$CUDA" = "auto" ]; then
    if command -v nvcc &>/dev/null; then
        CUDA="ON"
    else
        CUDA="OFF"
    fi
fi
if [ "$CUDA" = "ON" ]; then
    CUDA_FLAG="-DGGML_CUDA=ON"
    echo "[Lamio] CUDA: ON"
else
    CUDA_FLAG="-DGGML_CUDA=OFF"
    echo "[Lamio] CUDA: OFF (CPU-only)"
fi

# configure
if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "[Lamio] Configuring ($BUILD_DIR)..."
    cmake -B "$BUILD_DIR" -S . $CUDA_FLAG
else
    echo "[Lamio] Build dir exists, skipping configure"
fi

# build (limited jobs to avoid disk/RAM pressure)
echo "[Lamio] Building llama-server + llama-cli (-j$JOBS)..."
cmake --build "$BUILD_DIR" --target llama-server llama-cli -j"$JOBS"

echo "[Lamio] Build complete: $BUILD_DIR/bin/llama-server"
