#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_reader.h"

namespace lamio {

// Metadata for a single tensor within a block, referencing the GgufReader
// tensor index so the loader can find the offset/size later.
struct TensorRef {
    std::string name;     // full name (e.g. "blk.0.attn_q.weight")
    int64_t tensor_idx = -1;  // index into GgufReader::tensors(); -1 = not found
    int n_dims = 0;
    std::vector<int64_t> ne;
    int64_t nbytes = 0;
    int ggml_type = 0;    // enum ggml_type (read from GGUF)
};

// Tensors belonging to one transformer block.
// Fields are filled by scanning tensor names for "blk.N.<suffix>".
// Pointers are optional: nullptr if the tensor doesn't exist in this block
// (e.g. MoE blocks have ffn_*_exps, dense blocks have ffn_*).
struct BlockTensors {
    int layer = 0;

    // attention
    TensorRef attn_q     = {};
    TensorRef attn_k     = {};
    TensorRef attn_v     = {};
    TensorRef attn_qkv   = {};  // fused QKV (hybrid/GQA models)
    TensorRef attn_output = {};
    TensorRef attn_norm   = {};  // pre-attention norm

    // FFN (dense)
    TensorRef ffn_gate   = {};
    TensorRef ffn_up     = {};
    TensorRef ffn_down   = {};
    TensorRef ffn_norm   = {};  // pre-ffn norm

    // FFN (MoE)
    TensorRef ffn_gate_exps = {};
    TensorRef ffn_up_exps   = {};
    TensorRef ffn_down_exps = {};
    TensorRef router        = {};  // "blk.N.ffn_gate_shrink.weight" or similar

    bool has_moe = false;
    bool has_dense_ffn = false;
};

// Non-block tensors (embedding, output norm, output weight, etc.)
struct GlobalTensors {
    TensorRef token_embd   = {};  // "token_embd.weight"
    TensorRef output_norm  = {};  // "output_norm.weight"
    TensorRef output_weight = {}; // "output.weight" (tied if absent)
    TensorRef output_norm_bias = {};
};

struct ModelTensors {
    GlobalTensors global;
    std::vector<BlockTensors> blocks;
};

// Scan the GgufReader tensor list and organize by block.
ModelTensors map_tensors(const GgufReader & r, int n_layers);

} // namespace lamio