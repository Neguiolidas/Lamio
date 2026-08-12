#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf_reader.h"

namespace lamio {

struct ModelConfig {
    std::string arch;          // "qwen35moe", "qwen35", "glm", etc.
    int n_layers       = 0;    // block_count
    int n_embd         = 0;    // embedding_length
    int n_heads        = 0;    // attention.head_count
    int n_kv_heads     = 0;    // attention.head_count_kv
    int n_experts      = 0;    // expert_count (0 = dense)
    int n_active       = 0;    // expert_used_count
    int n_ffn_expert   = 0;    // expert_feed_forward_length
    int n_ff_shexp     = 0;    // expert_shared_feed_forward_length
    int full_attn_interval = 4; // full_attention_interval (layers 4n-1)
    int vocab_size     = 0;
    int context_length = 0;
    int head_dim       = 0;    // attention.key_length (0 = n_embd/n_heads)
    float norm_eps     = 1e-6f;
    float rope_freq_base = 10000.0f;
    int rope_dim_count  = 0;

    bool is_moe() const { return n_experts > 0; }
    bool is_dense() const { return n_experts == 0; }
};

ModelConfig parse_model_config(const GgufReader & r);

} // namespace lamio