#include "model_config.h"

#include <cstdlib>

namespace lamio {

static int get_int(const std::unordered_map<std::string, std::string> & m,
                   const std::string & key, int fallback = 0) {
    auto it = m.find(key);
    if (it == m.end()) return fallback;
    return atoi(it->second.c_str());
}

static float get_float(const std::unordered_map<std::string, std::string> & m,
                       const std::string & key, float fallback = 0.0f) {
    auto it = m.find(key);
    if (it == m.end()) return fallback;
    return strtof(it->second.c_str(), nullptr);
}

ModelConfig parse_model_config(const GgufReader & r) {
    ModelConfig cfg;
    const auto & md = r.metadata();

    cfg.arch = r.arch();

    // GGUF convention: <arch>.<field>
    const std::string & a = cfg.arch;
    cfg.n_layers       = get_int(md, a + ".block_count");
    cfg.n_embd         = get_int(md, a + ".embedding_length");
    cfg.n_heads        = get_int(md, a + ".attention.head_count");
    cfg.n_kv_heads     = get_int(md, a + ".attention.head_count_kv");
    cfg.n_experts      = get_int(md, a + ".expert_count");
    cfg.n_active        = get_int(md, a + ".expert_used_count");
    cfg.n_ffn_expert   = get_int(md, a + ".expert_feed_forward_length");
    cfg.n_ff_shexp     = get_int(md, a + ".expert_shared_feed_forward_length");
    cfg.full_attn_interval = get_int(md, a + ".full_attention_interval", 4);
    cfg.context_length = get_int(md, a + ".context_length");
    cfg.head_dim       = get_int(md, a + ".attention.key_length");
    cfg.norm_eps       = get_float(md, a + ".attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.rope_freq_base = get_float(md, a + ".rope.freq_base", 10000.0f);
    cfg.rope_dim_count = get_int(md, a + ".rope.dimension_count");

    // vocab: try tokenizer.ggml.tokens.size first (array length),
    // fall back to <arch>.vocab_size
    cfg.vocab_size = get_int(md, a + ".vocab_size");
    if (cfg.vocab_size == 0) {
        // tokenizer.ggml.tokens is an array; its length is in the metadata
        // as the count of elements. GgufReader stores arrays as "[array]".
        // We need the n_kv from gguf, but for now use a heuristic:
        // the eos_token_id + 1 is a lower bound.
        int eos = get_int(md, "tokenizer.ggml.eos_token_id");
        if (eos > 0) cfg.vocab_size = eos + 1;
    }

    if (cfg.head_dim == 0 && cfg.n_heads > 0) {
        cfg.head_dim = cfg.n_embd / cfg.n_heads;
    }

    return cfg;
}

} // namespace lamio