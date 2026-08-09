#pragma once

#include "tier.h"
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <cstdint>
#include <future>
#include <mutex>

namespace lamio {

struct expert_key {
    int layer;
    int expert_id;
    int type_idx; // 0=up, 1=gate, 2=down
    bool operator==(const expert_key & o) const {
        return layer == o.layer && expert_id == o.expert_id && type_idx == o.type_idx;
    }
};

struct expert_key_hash {
    size_t operator()(const expert_key & k) const {
        return (size_t(k.layer) << 34) | (size_t(k.type_idx) << 32) | size_t(k.expert_id);
    }
};

struct cache_slot {
    int layer     = -1;
    int expert_id = -1;
    int type_idx  = -1; // 0=up, 1=gate, 2=down
    void * data   = nullptr;
    size_t size   = 0;
    uint32_t heat = 0;
    uint32_t last_access = 0;
    bool valid    = false;
};

struct slot_key {
    int expert_id;
    int type_idx;
    bool operator==(const slot_key & o) const {
        return expert_id == o.expert_id && type_idx == o.type_idx;
    }
};

struct slot_key_hash {
    size_t operator()(const slot_key & k) const {
        return (size_t(k.type_idx) << 16) | size_t(k.expert_id);
    }
};

struct layer_cache {
    std::vector<cache_slot> slots;
    std::unordered_map<slot_key, int, slot_key_hash> expert_to_slot;
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
    void register_expert(int layer, int eid, int type_idx,
                         int64_t file_offset, size_t byte_size, int ggml_type);

    // Ensure selected experts are resident for a given tensor type.
    void on_select(int layer, int type_idx, const int * selected_experts, int k);

    // Async version for a given tensor type.
    void on_select_async(int layer, int type_idx, const int * selected_experts, int k);
    void wait_async(int layer, int eid, int type_idx);

    // Bump heat for a resident expert.
    void bump_heat(int layer, int eid, int type_idx);

    // Check if an expert tensor is cached.
    bool is_resident(int layer, int eid, int type_idx) const;

    // Get pointer to cached data for an expert tensor.
    void * get_data(int layer, int eid, int type_idx) const;

    // Decay all heat counters (called periodically, e.g. every N tokens).
    void decay_all();

    // Number of layers tracked.
    int n_layers() const { return (int)caches.size(); }

    // Stats.
    size_t total_used_bytes() const;
    size_t total_capacity_bytes() const;
    int    total_hits() const { return stats_hits; }
    int    total_misses() const { return stats_misses; }
    int    total_evictions() const { return stats_evictions; }
    double total_load_time_ms() const { return stats_load_time_ms; }
    size_t total_bytes_loaded() const { return stats_bytes_loaded; }

    double used_mb() const {
        double total = 0;
        for (const auto & lc : caches) total += lc.used_bytes;
        return total / (1024.0 * 1024.0);
    }
    double capacity_mb() const { return ram_budget / (1024.0 * 1024.0); }
    int    get_n_layers() const { return (int)caches.size(); }
    int    get_n_expert() const { return n_expert; }
    int    get_n_expert_used() const { return n_expert_used; }
    void   set_n_expert_used(int k) { n_expert_used = k; }

    void print_stats() const;

    // Getters for cache data pointer (for bridge).
    void ** get_slot_data_ptr(int layer, int slot_idx);
    int get_slot_expert_id(int layer, int slot_idx) const;

    // Set external load callback.
    using load_fn = bool (*)(int layer, int eid, int type_idx, void * dest, size_t size);
    void set_load_callback(load_fn fn) { load_cb = fn; }

    ~tier_manager();

    // Toggle prefetch (route-ahead).
    void set_prefetch(bool on) { prefetch_enabled = on; }

private:
    std::vector<layer_cache> caches;
    std::unordered_map<expert_key, expert_record, expert_key_hash> registry;
    std::vector<uint32_t> heat;  // flattened: [layer][eid*3+type_idx]
    std::vector<uint32_t> last;  // flattened: [layer][eid*3+type_idx]
    uint32_t clock = 1;
    load_fn load_cb = nullptr;
    bool prefetch_enabled = false;

    int stats_hits   = 0;
    int stats_misses = 0;
    int stats_evictions = 0;
    double stats_load_time_ms = 0.0;
    size_t stats_bytes_loaded = 0;

    struct pending_load {
        std::future<bool> fut;
        int layer;
        int eid;
        int type_idx;
    };
    std::vector<pending_load> pending_loads_;
    std::mutex pending_mtx_;
    static constexpr int MAX_PENDING = 64;

    int n_expert;
    int n_expert_used = 0;
    size_t ram_budget = 0;
    int idx(int layer, int eid, int type_idx) const {
        return (layer * n_expert + eid) * 3 + type_idx;
    }
    void ensure_slot(int layer, int eid, int type_idx);
    void evict_one(int layer);
};

} // namespace lamio