#include "tier_manager.h"

namespace lamio {

tier_manager::~tier_manager() {
    for (auto & lc : caches) {
        for (auto & slot : lc.slots) {
            if (slot.data) {
                delete[] (uint8_t*)slot.data;
                slot.data = nullptr;
            }
        }
    }
}

tier_manager::tier_manager(size_t ram_budget_bytes, int n_layers, int n_expert_per_layer)
    : n_expert(n_expert_per_layer)
{
    caches.resize(n_layers);
    heat.resize(n_layers * n_expert, 0);
    last.resize(n_layers * n_expert, 0);

    // Distribute budget evenly across layers
    size_t per_layer = ram_budget_bytes / n_layers;
    // per-layer capacity: enough for a few expert slots
    // each slot is roughly the size of one expert tensor
    for (auto & lc : caches) {
        lc.capacity_bytes = per_layer;
        lc.used_bytes = 0;
        lc.slots.reserve(16);
    }
}

void tier_manager::register_expert(int layer, int eid,
                                    int64_t file_offset, size_t byte_size, int ggml_type) {
    expert_key k{layer, eid};
    registry[k] = {file_offset, byte_size, ggml_type};
}

bool tier_manager::is_resident(int layer, int eid) const {
    if (layer < 0 || layer >= (int)caches.size()) return false;
    auto it = caches[layer].expert_to_slot.find(eid);
    return it != caches[layer].expert_to_slot.end();
}

void tier_manager::bump_heat(int layer, int eid) {
    int i = idx(layer, eid);
    if (heat[i] < UINT32_MAX) heat[i]++;
    last[i] = clock++;
}

void * tier_manager::get_data(int layer, int eid) const {
    if (layer < 0 || layer >= (int)caches.size()) return nullptr;
    auto & lc = caches[layer];
    auto it = lc.expert_to_slot.find(eid);
    if (it == lc.expert_to_slot.end()) return nullptr;
    return lc.slots[it->second].data;
}

void tier_manager::ensure_slot(int layer, int eid) {
    auto & lc = caches[layer];
    // Already resident?
    if (lc.expert_to_slot.find(eid) != lc.expert_to_slot.end()) {
        bump_heat(layer, eid);
        stats_hits++;
        return;
    }

    // Find the record
    expert_key k{layer, eid};
    auto rit = registry.find(k);
    if (rit == registry.end()) return; // not in model (shouldn't happen)

    // Make room if needed. Max 64 evicts to prevent infinite loop on tiny budget
    size_t needed = rit->second.byte_size;
    int max_evict = 64;
    while (lc.used_bytes + needed > lc.capacity_bytes && !lc.slots.empty() && max_evict-- > 0) {
        evict_one(layer);
    }

    // Find an empty slot or reuse an evicted one
    int slot_idx = -1;
    for (int i = 0; i < (int)lc.slots.size(); i++) {
        if (!lc.slots[i].valid) { slot_idx = i; break; }
    }
    if (slot_idx < 0) {
        slot_idx = (int)lc.slots.size();
        lc.slots.push_back({});
    }

    auto & slot = lc.slots[slot_idx];
    // Allocate or reuse buffer
    if (slot.size < needed) {
        delete[] (uint8_t*)slot.data;
        slot.data = new uint8_t[needed];
        slot.size = needed;
    }

    slot.layer      = layer;
    slot.expert_id  = eid;
    slot.valid      = true;
    slot.heat       = 0;
    slot.last_access = 0;

    // Load data
    if (load_cb) {
        load_cb(layer, eid, slot.data, needed);
    }

    lc.expert_to_slot[eid] = slot_idx;
    lc.used_bytes += needed;

    bump_heat(layer, eid);
    stats_misses++;
}

void tier_manager::evict_one(int layer) {
    auto & lc = caches[layer];
    if (lc.slots.empty()) return;

    // Build pinned list (valid slots)
    std::vector<int> pinned;
    for (int i = 0; i < (int)lc.slots.size(); i++) {
        if (lc.slots[i].valid) pinned.push_back(i);
    }
    if (pinned.empty()) return;

    // Build heat/last vectors for slots
    std::vector<uint32_t> slot_heat(pinned.size());
    std::vector<uint32_t> slot_last(pinned.size());
    for (int i = 0; i < (int)pinned.size(); i++) {
        int si = pinned[i];
        int eid = lc.slots[si].expert_id;
        int hi = idx(layer, eid);
        slot_heat[i] = heat[hi];
        slot_last[i] = last[hi];
    }

    // Use LFRU to pick coldest slot
    int cold_slot_idx = 0;
    for (int i = 1; i < (int)pinned.size(); i++) {
        if (tier_lfru_score(slot_heat[i], slot_last[i], clock) <
            tier_lfru_score(slot_heat[cold_slot_idx], slot_last[cold_slot_idx], clock))
            cold_slot_idx = i;
    }

    int evict_slot = pinned[cold_slot_idx];
    auto & slot = lc.slots[evict_slot];
    if (!slot.valid) return;

    // Remove from map
    lc.expert_to_slot.erase(slot.expert_id);
    lc.used_bytes -= slot.size;
    slot.valid = false;
}

void tier_manager::on_select(int layer, const int * selected_experts, int k) {
    for (int i = 0; i < k; i++) {
        int eid = selected_experts[i];
        if (eid < 0) continue;
        ensure_slot(layer, eid);
    }
}

void tier_manager::decay_all() {
    tier_decay(heat);
}

size_t tier_manager::total_used_bytes() const {
    size_t total = 0;
    for (auto & lc : caches) total += lc.used_bytes;
    return total;
}

size_t tier_manager::total_capacity_bytes() const {
    size_t total = 0;
    for (auto & lc : caches) total += lc.capacity_bytes;
    return total;
}

void ** tier_manager::get_slot_data_ptr(int layer, int slot_idx) {
    if (layer < 0 || layer >= (int)caches.size()) return nullptr;
    auto & lc = caches[layer];
    if (slot_idx < 0 || slot_idx >= (int)lc.slots.size()) return nullptr;
    return &lc.slots[slot_idx].data;
}

int tier_manager::get_slot_expert_id(int layer, int slot_idx) const {
    if (layer < 0 || layer >= (int)caches.size()) return -1;
    auto & lc = caches[layer];
    if (slot_idx < 0 || slot_idx >= (int)lc.slots.size()) return -1;
    return lc.slots[slot_idx].expert_id;
}

} // namespace lamio