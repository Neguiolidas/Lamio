#pragma once

#include <vector>
#include <cstring>
#include "ggml.h"
#include "ggml-backend.h"
#include "model_config.h"
#include "block_tensors.h"
#include "weight_loader.h"

namespace lamio {

struct Qwen35HParams {
    // attention
    int n_head       = 0;
    int n_head_kv    = 0;
    int head_dim      = 0;   // key_length / value_length
    float norm_eps    = 1e-6f;
    float freq_base   = 10000.0f;
    int   rope_sections[4] = {0,0,0,0};

    // linear attention (delta net)
    int ssm_d_conv    = 0;   // conv_kernel
    int ssm_d_inner   = 0;   // inner_size
    int ssm_d_state   = 0;   // state_size (head_k_dim)
    int ssm_dt_rank   = 0;   // time_step_rank (= num_v_heads)
    int ssm_n_group   = 0;   // group_count (= num_k_heads)

    // derived
    int rot_dim = 0;            // RoPE applied dims (rope.dimension_count)

    int head_k_dim() const { return ssm_d_state; }
    int head_v_dim() const { return ssm_d_inner / ssm_dt_rank; }
    int num_k_heads() const { return ssm_n_group; }
    int num_v_heads() const { return ssm_dt_rank; }
    int key_dim()   const { return head_k_dim() * num_k_heads(); }
    int value_dim() const { return head_v_dim() * num_v_heads(); }
    int conv_dim()  const { return key_dim() * 2 + value_dim(); }
    int n_embd_r()  const { return (ssm_d_conv - 1) * conv_dim(); }
    int n_embd_s()  const { return head_k_dim() * head_v_dim() * num_v_heads(); }
};

// Extracts the qwen35-specific hparams from metadata + tensor shapes.
Qwen35HParams parse_qwen35_hparams(const ModelConfig & cfg, const GgufReader & r);

// Per-layer state passed between generation steps.
// For recurrent (GDN) layers: conv_state_in/out + ssm_state_in/out
// For attention layers: kv_k_in/out + kv_v_in/out
// All are tensors created by the caller in the INPUT context (set_input).
// The forward pass reads from *_in and writes to *_out (graph ops).
struct LayerState {
    // Set to true by caller to request state extraction (even on prefill)
    bool want_state = false;

    // GDN recurrent state (nullptr for attention layers)
    ggml_tensor * conv_state_in  = nullptr;  // [conv_kernel-1, conv_channels, 1]
    ggml_tensor * conv_state_out = nullptr;  // same shape, filled post-compute
    ggml_tensor * ssm_state_in   = nullptr;  // [head_v_dim, head_v_dim, num_v_heads, 1]
    ggml_tensor * ssm_state_out  = nullptr;  // same shape (extracted from gdn output)

    // Attention KV cache (nullptr for recurrent layers)
    ggml_tensor * kv_k_in  = nullptr;  // [head_dim, n_head_kv, kv_pos]
    ggml_tensor * kv_v_in  = nullptr;
    ggml_tensor * kv_k_out = nullptr;  // [head_dim, n_head_kv, kv_pos+1]
    ggml_tensor * kv_v_out = nullptr;
};

// Forward pass for one block. `cur` is the hidden state [n_embd, n_tokens].
// Returns the new hidden state. Computes everything into compute_ctx and
// expects the graph to be built+computed by the caller.
// pos_ids and mask are tensors created by the caller and filled post-alloc.
// state is per-layer persistent state (may be nullptr for prefill-from-zero).
struct Qwen35Forward {
    const ModelConfig & cfg;
    const Qwen35HParams & hp;
    WeightLoader & weights;
    ggml_context * ctx;     // compute ctx
    bool is_recr;           // true if this layer uses linear attention

    // tensors that must be zero-initialized after the gallocr allocs them
    std::vector<ggml_tensor *> zero_init;

    // Debug: pointers to intermediate tensors of the last build_layer call
    ggml_tensor * attn_norm_out = nullptr;
    ggml_tensor * gdn_output    = nullptr;
    ggml_tensor * final_output  = nullptr;

    // MoE: captures the top-k selected expert indices tensor after build_layer.
    // Used by the host orchestrator to drive expert eviction/prefetch. Null if
    // this layer is not MoE or the capture wasn't requested.
    ggml_tensor * selected_experts = nullptr;
    bool want_selected = false;

    // New state written by the forward pass (extracted from graph)
    ggml_tensor * new_conv_state = nullptr;
    ggml_tensor * new_ssm_state  = nullptr;

    ggml_tensor * build_layer(ggml_tensor * cur, const BlockTensors & blk,
                               ggml_tensor * pos_ids, ggml_tensor * mask,
                               LayerState * state = nullptr);
};

} // namespace lamio
