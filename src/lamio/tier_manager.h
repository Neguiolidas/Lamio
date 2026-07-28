#pragma once

#include "tier.h"
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <cstdint>

namespace lamio {

struct expert_key {
    int layer;
    int expert_id;
    bool operator==(const expert_key & o) const {
        return layer == o.layer && expert_id == o.expert_id;
    }
};

struct expert_key_hash {
    size_t operator()(const expert_key & k) const {
        return (size_t(k.layer) << 32) | size_t(k.expert_id);
    }
};

struct cache_slot {
    int layer     = -1;
    int expert_id = -1;
    void * data   = nullptr;
    size_t size   = 0;
    uint32_t heat = 0;
    uint32_t last_access = 0;
    bool valid    = false;
};

struct layer_cache {
    std::vector<cache_slot> slots;
    std::unordered_map<int, int> expert_to_slot; // eid -> slot index
    size_t capacity_bytes = 0;
    size_t used_bytes     = 0;
};

struct expert_record {
    int64_t file_offset;
    size_t  byte_size;
    int     type; // ggml_type (opaque here, just stored)
};

// Manages a pool of RAM slots per layer.
// Experts are staged into slots on first access and evicted via LFRU.
class tier_manager {
public:
    tier_manager(size_t ram_budget_bytes, int n_layers, int n_expert_per_layer);

    // Register an expert tensor location in the GGUF file.
    // Called during model load for each ffn_*_exps tensor.
    void register_expert(int layer, int eid,
                         int64_t file_offset, size_t byte_size, int ggml_type);

    // Ensure selected experts are resident. Evicts cold ones if cache is full.
    // Called after router selects top-k experts, before mul_mat_id.
    void on_select(int layer, const int * selected_experts, int k);

    // Bump heat for a resident expert.
    void bump_heat(int layer, int eid);

    // Check if an expert is cached.
    bool is_resident(int layer, int eid) const;

    // Get pointer to cached data for an expert (layer, eid).
    // Returns nullptr if not resident.
    void * get_data(int layer, int eid) const;

    // Decay all heat counters (called periodically, e.g. every N tokens).
    void decay_all();

    // Number of layers tracked.
    int n_layers() const { return (int)caches.size(); }

    // Stats.
    size_t total_used_bytes() const;
    size_t total_capacity_bytes() const;
    int    total_hits() const { return stats_hits; }
    int    total_misses() const { return stats_misses; }

    // Getters for cache data pointer (for bridge).
    void ** get_slot_data_ptr(int layer, int slot_idx);
    int get_slot_expert_id(int layer, int slot_idx) const;

    // Set external load callback: function that reads expert data from disk.
    // Called after eviction makes room, but before returning from on_select.
    // Signature: bool(int layer, int eid, void * dest_buf, size_t size).
    using load_fn = bool (*)(int layer, int eid, void * dest, size_t size);
    void set_load_callback(load_fn fn) { load_cb = fn; }

    ~tier_manager();

    // Toggle prefetch (route-ahead).
    void set_prefetch(bool on) { prefetch_enabled = on; }

private:
    std::vector<layer_cache> caches;
    std::unordered_map<expert_key, expert_record, expert_key_hash> registry;
    std::vector<uint32_t> heat;  // flattened: [layer][eid]
    std::vector<uint32_t> last;  // flattened: [layer][eid]
    uint32_t clock = 1;
    load_fn load_cb = nullptr;
    bool prefetch_enabled = false;

    int stats_hits   = 0;
    int stats_misses = 0;

    int n_expert;
    int idx(int layer, int eid) const { return layer * n_expert + eid; }
    void ensure_slot(int layer, int eid);
    void evict_one(int layer);
};

} // namespace lamio