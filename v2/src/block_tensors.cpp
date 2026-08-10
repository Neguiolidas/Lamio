#include "block_tensors.h"

#include <cstring>
#include <cstdio>

namespace lamio {

static bool try_find(const GgufReader & r, const std::string & name, TensorRef & out) {
    const auto & tensors = r.tensors();
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].name == name) {
            out.name = tensors[i].name;
            out.tensor_idx = (int64_t)i;
            out.n_dims = tensors[i].n_dims;
            out.ne = tensors[i].ne;
            out.nbytes = tensors[i].nbytes;
            out.ggml_type = tensors[i].ggml_type;
            return true;
        }
    }
    return false;
}

static void scan_block(const GgufReader & r, BlockTensors & blk) {
    char prefix[32];
    snprintf(prefix, sizeof(prefix), "blk.%d.", blk.layer);

    auto find_in_block = [&](const char * suffix, TensorRef & out) -> bool {
        std::string name = std::string(prefix) + suffix;
        return try_find(r, name, out);
    };

    find_in_block("attn_q.weight", blk.attn_q);
    find_in_block("attn_k.weight", blk.attn_k);
    find_in_block("attn_v.weight", blk.attn_v);
    find_in_block("attn_qkv.weight", blk.attn_qkv);
    find_in_block("attn_output.weight", blk.attn_output);
    find_in_block("attn_norm.weight", blk.attn_norm);

    // Dense FFN: many architectures name the pre-FFN norm differently.
    find_in_block("ffn_gate.weight", blk.ffn_gate);
    find_in_block("ffn_up.weight", blk.ffn_up);
    find_in_block("ffn_down.weight", blk.ffn_down);
    find_in_block("ffn_norm.weight", blk.ffn_norm);
    if (blk.ffn_norm.tensor_idx < 0)
        find_in_block("post_attention_norm.weight", blk.ffn_norm);
    if (blk.ffn_norm.tensor_idx < 0)
        find_in_block("post_ffn_norm.weight", blk.ffn_norm);

    // MoE FFN
    find_in_block("ffn_gate_exps.weight", blk.ffn_gate_exps);
    find_in_block("ffn_up_exps.weight", blk.ffn_up_exps);
    find_in_block("ffn_down_exps.weight", blk.ffn_down_exps);

    // Router: try common names
    bool found_router = find_in_block("ffn_gate_shrink.weight", blk.router);
    if (!found_router) find_in_block("ffn_gate_inp.weight", blk.router);

    blk.has_moe = blk.ffn_gate_exps.tensor_idx >= 0;
    blk.has_dense_ffn = blk.ffn_gate.tensor_idx >= 0;
}

ModelTensors map_tensors(const GgufReader & r, int n_layers) {
    ModelTensors mt;
    mt.blocks.resize(n_layers);

    // Global tensors
    try_find(r, "token_embd.weight", mt.global.token_embd);
    try_find(r, "output_norm.weight", mt.global.output_norm);
    try_find(r, "output.weight", mt.global.output_weight);
    try_find(r, "output_norm.bias", mt.global.output_norm_bias);

    for (int l = 0; l < n_layers; ++l) {
        mt.blocks[l].layer = l;
        // Initialize all tensor_idx to -1 (not found)
        mt.blocks[l].attn_q.tensor_idx = -1;
        mt.blocks[l].attn_k.tensor_idx = -1;
        mt.blocks[l].attn_v.tensor_idx = -1;
        mt.blocks[l].attn_output.tensor_idx = -1;
        mt.blocks[l].attn_norm.tensor_idx = -1;
        mt.blocks[l].ffn_gate.tensor_idx = -1;
        mt.blocks[l].ffn_up.tensor_idx = -1;
        mt.blocks[l].ffn_down.tensor_idx = -1;
        mt.blocks[l].ffn_norm.tensor_idx = -1;
        mt.blocks[l].ffn_gate_exps.tensor_idx = -1;
        mt.blocks[l].ffn_up_exps.tensor_idx = -1;
        mt.blocks[l].ffn_down_exps.tensor_idx = -1;
        mt.blocks[l].router.tensor_idx = -1;
        scan_block(r, mt.blocks[l]);
    }

    return mt;
}

} // namespace lamio