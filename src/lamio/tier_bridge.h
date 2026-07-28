#pragma once

#include "tier_manager.h"
#include "expert_loader.h"
#include "ggml.h"

#include <cstring>
#include <string>
#include <vector>
#include <memory>

namespace lamio {

// Bridge between llama.cpp model loader and lamio tier system.
// During model load, expert tensors are registered but not allocated.
// During inference, on_select() loads the needed experts.

class tier_bridge {
public:
    // Initialize with GGUF path, RAM budget, model dimensions.
    static tier_bridge & instance();

    bool init(const char * gguf_path, size_t ram_budget,
              int n_layers, int n_expert);

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
    void on_expert_select(int layer, const int * selected_experts, int k);

    // Get pointer to cached data for an expert.
    void * get_expert_data(int layer, int eid) const;

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
};

// Check if a tensor name matches the expert tensor pattern.
// Matches: blk.%d.ffn_gate_exps, blk.%d.ffn_up_exps, blk.%d.ffn_down_exps
inline bool tier_bridge::is_expert_tensor(const char * name) {
    if (!name) return false;
    // Quick check: must contain "ffn_" and "exps"
    const char * ffn = strstr(name, "ffn_");
    if (!ffn) return false;
    const char * exps = strstr(ffn, "exps");
    if (!exps) return false;
    // Must start with "blk."
    return strncmp(name, "blk.", 4) == 0;
}

} // namespace lamio