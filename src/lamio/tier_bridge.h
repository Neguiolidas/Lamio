#pragma once

#include "tier_manager.h"
#include "expert_loader.h"
#include "ggml.h"

#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <atomic>

namespace lamio {

// Global override for n_expert_used (0 = no override, use model default).
// Set via --lamio-expert-k CLI flag.
inline std::atomic<int> g_expert_k_override{0};

// Bridge between llama.cpp model loader and lamio tier system.
// During model load, expert tensors are registered but not allocated.
// During inference, on_select() loads the needed experts.

class tier_bridge {
public:
    // Initialize with GGUF path, RAM budget, model dimensions.
    static tier_bridge & instance();

    bool init(const char * gguf_path, size_t ram_budget,
              int n_layers, int n_expert, int n_expert_used = 0);

    // Called from llama_model_loader for each expert tensor.
    // tensor_name: e.g. "blk.0.ffn_up_exps.weight"
    // eid: expert index within the 3D tensor (0..n_expert-1)
    // tensor_stride: bytes between consecutive experts in the file
    // expert_bytes: size of one expert slice
    void register_expert_tensor(const char * tensor_name, int eid,
                                size_t tensor_stride, size_t expert_bytes);

    // Check if a tensor name is an expert tensor (ffn_*_exps).
    static bool is_expert_tensor(const char * name);

    // Called from build_moe_ffn after top-k selection.
    // Loads the k selected experts into cache and patches tensor data pointers.
    void on_expert_select(int layer, int type_idx, const int * selected_experts, int k);

    void on_select(int layer, int type_idx, const int * selected_experts, int k) {
        on_expert_select(layer, type_idx, selected_experts, k);
    }

    void on_select_async(int layer, int type_idx, const int * selected_experts, int k) {
        if (!enabled_ || !manager_) return;
        manager_->on_select_async(layer, type_idx, selected_experts, k);
    }
    void wait_async(int layer, int eid, int type_idx) {
        if (!enabled_ || !manager_) return;
        manager_->wait_async(layer, eid, type_idx);
    }

    void * get_expert_data(int layer, int eid, int type_idx) const;

    // Prefetch experts for the next layer into kernel page cache.
    void prefetch_layer(int layer, int type_idx,
                        const int * selected, int k, int next_layer);

    // Buffer management for expert weight tensors.
    // The eval callback needs to allocate a buffer for the full expert tensor
    // (all experts, not just selected ones) and fill in the selected slices.
    void * allocate_expert_buffer(int layer, const char * name, size_t total_size);
    void * get_expert_buffer(int layer, const char * name);

    bool is_enabled() const { return enabled_; }
    void set_enabled(bool on) { enabled_ = on; }

    tier_manager & manager() { return *manager_; }
    expert_loader & loader() { return loader_; }

private:
    tier_bridge() = default;
    bool enabled_ = false;
    std::unique_ptr<tier_manager> manager_;
    expert_loader loader_;

    // Per-layer, per-expert-type (gate/up/down) info for patching pointers
    struct expert_tensor_info {
        std::string name;      // e.g. "blk.0.ffn_up_exps.weight"
        size_t expert_bytes;   // size of one expert slice
        size_t tensor_stride;  // stride between experts in file
    };
    // Key: (layer * 3 + type_idx) where type_idx: 0=up, 1=gate, 2=down
    std::vector<expert_tensor_info> tensor_infos_;

    // Lookup tensor name by (layer, type_idx)
    // type_idx: 0=up, 1=gate, 2=down
    const expert_tensor_info * find_tensor_info(int layer, int type_idx) const;

    // Per-tensor buffers allocated on demand for the full expert tensor
    // (keyed by tensor name)
    std::map<std::string, void *> expert_buffers_;
};

// Check if a tensor name matches the expert tensor pattern.
// Matches: blk.%d.ffn_gate_exps.weight, blk.%d.ffn_up_exps.weight, blk.%d.ffn_down_exps.weight
// Only the actual weight tensors (ends in ".weight"). MXFP4 scales (ffn_*_exps_s, _in_s,
// _sb, _s2) are per-expert shape {n_expert} and must NOT be treated as expert weights.
inline bool tier_bridge::is_expert_tensor(const char * name) {
    if (!name) return false;
    // Must end in ".weight" to exclude scale/bias tensors.
    size_t len = strlen(name);
    if (len < 7 || strcmp(name + len - 7, ".weight") != 0) return false;
    // Quick check: must contain "ffn_" and "exps"
    const char * ffn = strstr(name, "ffn_");
    if (!ffn) return false;
    const char * exps = strstr(ffn, "exps");
    if (!exps) return false;
    // Must start with "blk."
    return strncmp(name, "blk.", 4) == 0;
}

} // namespace lamio