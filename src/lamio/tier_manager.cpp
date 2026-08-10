#include "tier_manager.h"
#include <chrono>
#include <cstdio>

namespace lamio {

tier_manager::~tier_manager() {
    print_stats();
    for (auto & lc : caches) {
        for (auto & slot : lc.slots) {
            if (slot.data) {
                delete[] (uint8_t*)slot.data;
                slot.data = nullptr;
            }
        }
        lc.slots.clear();
        lc.expert_to_slot.clear();
    }
    caches.clear();
    registry.clear();

    // Discard any pending async loads without waiting
    for (auto & pl : pending_loads_) {
        if (pl.fut.valid()) {
            try { pl.fut.wait(); } catch (...) {}
        }
    }
    pending_loads_.clear();
}

tier_manager::tier_manager(size_t ram_budget_bytes, int n_layers, int n_expert_per_layer)
    : n_expert(n_expert_per_layer)
    , ram_budget(ram_budget_bytes)
{
    caches.resize(n_layers);
    heat.resize(n_layers * n_expert * 3, 0);
    last.resize(n_layers * n_expert * 3, 0);

    // Distribute budget with shallow-favoring weights.
    // Layers 0-9 get 2x weight, 10-19 get 1.5x, rest get 1x.
    // This follows the observation that shallow layers are accessed first
    // and have higher temporal locality.
    std::vector<double> weights(n_layers, 1.0);
    double total_weight = 0.0;
    for (int i = 0; i < n_layers; i++) {
        if (i < 10) weights[i] = 2.0;
        else if (i < 20) weights[i] = 1.5;
        total_weight += weights[i];
    }
    for (int i = 0; i < n_layers; i++) {
        caches[i].capacity_bytes = (size_t)((double)ram_budget_bytes * weights[i] / total_weight);
        caches[i].used_bytes = 0;
        caches[i].slots.reserve(16);
    }
}

void tier_manager::register_expert(int layer, int eid, int type_idx,
                                    int64_t file_offset, size_t byte_size, int ggml_type) {
    expert_key k{layer, eid, type_idx};
    registry[k] = {file_offset, byte_size, ggml_type};
}

bool tier_manager::is_resident(int layer, int eid, int type_idx) const {
    if (layer < 0 || layer >= (int)caches.size()) return false;
    auto it = caches[layer].expert_to_slot.find({eid, type_idx});
    return it != caches[layer].expert_to_slot.end();
}

void tier_manager::bump_heat(int layer, int eid, int type_idx) {
    int i = idx(layer, eid, type_idx);
    if (heat[i] < UINT32_MAX) heat[i]++;
    last[i] = clock++;
}

void * tier_manager::get_data(int layer, int eid, int type_idx) const {
    if (layer < 0 || layer >= (int)caches.size()) return nullptr;
    auto & lc = caches[layer];
    auto it = lc.expert_to_slot.find({eid, type_idx});
    if (it == lc.expert_to_slot.end()) return nullptr;
    return lc.slots[it->second].data;
}

void tier_manager::ensure_slot(int layer, int eid, int type_idx) {
    auto & lc = caches[layer];
    slot_key sk{eid, type_idx};
    if (lc.expert_to_slot.find(sk) != lc.expert_to_slot.end()) {
        bump_heat(layer, eid, type_idx);
        stats_hits++;
        return;
    }

    expert_key k{layer, eid, type_idx};
    auto rit = registry.find(k);
    if (rit == registry.end()) return;

    size_t needed = rit->second.byte_size;
    int max_evict = 64;
    while (lc.used_bytes + needed > lc.capacity_bytes && !lc.slots.empty() && max_evict-- > 0) {
        evict_one(layer);
    }

    int slot_idx = -1;
    for (int i = 0; i < (int)lc.slots.size(); i++) {
        if (!lc.slots[i].valid) { slot_idx = i; break; }
    }
    if (slot_idx < 0) {
        slot_idx = (int)lc.slots.size();
        lc.slots.push_back({});
    }

    auto & slot = lc.slots[slot_idx];

    slot.layer      = layer;
    slot.expert_id  = eid;
    slot.type_idx   = type_idx;
    slot.valid      = true;
    slot.heat       = 0;
    slot.last_access = 0;

    // No dedicated RAM slot: EXPERTS are read directly from the mmap'd model,
    // which the kernel pages in. We only hint the kernel to keep the expert's
    // pages cached (via the prefetch callback, e.g. POSIX_FADV_WILLNEED).
    // This avoids duplicating model memory in user-space slots (which caused OOM
    // on the 23GB-RAM host) and avoids the page-cache eviction that corrupted
    // compute input. load_cb is optional and unused when prefetch_cb is set.
    if (prefetch_cb) {
        prefetch_cb(layer, eid, type_idx);
    } else if (load_cb) {
        auto t0 = std::chrono::steady_clock::now();
        load_cb(layer, eid, type_idx, slot.data, slot.size);
        auto t1 = std::chrono::steady_clock::now();
        stats_load_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_bytes_loaded += needed;
    }

    lc.expert_to_slot[sk] = slot_idx;
    lc.used_bytes += needed;

    bump_heat(layer, eid, type_idx);
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
        int tid = lc.slots[si].type_idx;
        int hi = idx(layer, eid, tid);
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
    lc.expert_to_slot.erase({slot.expert_id, slot.type_idx});
    lc.used_bytes -= slot.size;
    slot.valid = false;
    stats_evictions++;
}

void tier_manager::on_select(int layer, int type_idx, const int * selected_experts, int k) {
    for (int i = 0; i < k; i++) {
        int eid = selected_experts[i];
        if (eid < 0) continue;
        ensure_slot(layer, eid, type_idx);
    }
}

void tier_manager::on_select_async(int layer, int type_idx, const int * selected_experts, int k) {
    std::lock_guard<std::mutex> lk(pending_mtx_);
    for (int i = 0; i < k; i++) {
        int eid = selected_experts[i];
        if (eid < 0) continue;

        if (is_resident(layer, eid, type_idx)) {
            bump_heat(layer, eid, type_idx);
            stats_hits++;
            continue;
        }

        expert_key key{layer, eid, type_idx};
        auto rit = registry.find(key);
        if (rit == registry.end()) continue;

        auto & lc = caches[layer];
        size_t needed = rit->second.byte_size;
        int max_evict = 64;
        while (lc.used_bytes + needed > lc.capacity_bytes && !lc.slots.empty() && max_evict-- > 0) {
            evict_one(layer);
        }

        int slot_idx = -1;
        for (int j = 0; j < (int)lc.slots.size(); j++) {
            if (!lc.slots[j].valid) { slot_idx = j; break; }
        }
        if (slot_idx < 0) {
            slot_idx = (int)lc.slots.size();
            lc.slots.push_back({});
        }

        auto & slot = lc.slots[slot_idx];
        if (slot.size < needed) {
            delete[] (uint8_t*)slot.data;
            slot.data = new uint8_t[needed];
            slot.size = needed;
        }
        slot.layer = layer;
        slot.expert_id = eid;
        slot.type_idx = type_idx;
        slot.valid = true;
        slot.heat = 0;
        slot.last_access = 0;
        lc.expert_to_slot[{eid, type_idx}] = slot_idx;
        lc.used_bytes += needed;

        void * dest = slot.data;
        size_t sz = needed;
        load_fn cb = load_cb;
        pending_loads_.push_back({std::async(std::launch::async,
            [cb, layer, eid, type_idx, dest, sz]() -> bool {
                return cb(layer, eid, type_idx, dest, sz);
            }), layer, eid, type_idx});

        stats_misses++;
    }
}

void tier_manager::wait_async(int layer, int eid, int type_idx) {
    std::lock_guard<std::mutex> lk(pending_mtx_);
    for (auto it = pending_loads_.begin(); it != pending_loads_.end(); ++it) {
        if (it->layer == layer && it->eid == eid && it->type_idx == type_idx) {
            it->fut.wait();
            bump_heat(layer, eid, type_idx);
            pending_loads_.erase(it);
            return;
        }
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

void tier_manager::print_stats() const {
    int total = stats_hits + stats_misses;
    double hit_rate = total > 0 ? 100.0 * stats_hits / total : 0.0;
    fprintf(stderr, "lamio tier stats:"
           " hits=%d misses=%d evictions=%d"
           " hit_rate=%.1f%%"
           " bytes_loaded=%.2f MB"
           " load_time=%.1f ms"
           " used=%.2f MB capacity=%.2f MB\n",
           stats_hits, stats_misses, stats_evictions,
           hit_rate,
           stats_bytes_loaded / 1048576.0,
           stats_load_time_ms,
           total_used_bytes() / 1048576.0,
           total_capacity_bytes() / 1048576.0);
}

} // namespace lamio