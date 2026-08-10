#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "model_config.h"
#include "block_tensors.h"
#include "weight_loader.h"

namespace lamio {

// Forward pass for one dense (non-MoE) transformer block.
// Returns the output hidden state tensor (owned by compute_ctx).
// Input x is the hidden state from the previous layer (or embedding).
//
// The compute graph is built, executed on the backend, and the result
// is read back. The caller is responsible for managing ctx lifetimes.
struct DenseForward {
    const ModelConfig & cfg;
    WeightLoader & weights;
    ggml_backend_t backend;
    ggml_context * compute_ctx;  // ctx for building the graph
    int n_threads;

    ggml_tensor * forward(const BlockTensors & blk, ggml_tensor * x);
};

} // namespace lamio