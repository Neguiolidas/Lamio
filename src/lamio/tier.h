#pragma once

#include <cstdint>
#include <vector>

// Port of colibri c/tier.h — LFRU expert eviction policy for MoE tiering.
// Frequency is primary signal; recency breaks close calls.
// 25%+4 hysteresis prevents ping-pong.

namespace lamio {

// LFRU: frequency is primary; recency breaks close calls.
// A recent access is worth at most 255, frequency count is worth 256.
inline uint64_t tier_lfru_score(uint32_t heat, uint32_t last, uint32_t clock) {
    uint32_t age = clock - last;
    uint32_t recent = age < 255 ? 255 - age : 0;
    return (uint64_t(heat) << 8) | recent;
}

// Pick one hot-store slot to replace from routing heat (no recency).
// Returns true if swap is justified (hot > cold + 25%+4 hysteresis).
inline bool tier_pick_swap(const std::vector<uint32_t> & heat,
                           const std::vector<int>  & pinned,
                           int & slot, int & eid, long & gain) {
    if (heat.empty() || pinned.empty()) return false;
    int npin = (int)pinned.size();
    if (npin < 1) return false;

    int cold_idx = 0;
    for (int z = 1; z < npin; z++) {
        if (heat[pinned[z]] < heat[pinned[cold_idx]]) cold_idx = z;
    }

    int hot = -1;
    uint32_t fh = 0;
    for (int e = 0, nexpert = (int)heat.size(); e < nexpert; e++) {
        bool resident = false;
        for (int z = 0; z < npin; z++) {
            if (pinned[z] == e) { resident = true; break; }
        }
        if (!resident && heat[e] > fh) { fh = heat[e]; hot = e; }
    }
    if (hot < 0) return false;

    uint32_t fc = heat[pinned[cold_idx]];
    if (fh <= fc + (fc >> 2) + 4) return false;

    slot = cold_idx;
    eid  = hot;
    gain = (long)fh - (long)fc;
    return true;
}

// LFRU pick: frequency+recency score on both cold (pinned) and hot (candidate).
// Hysteresis in score units: hot_score > cold_score * 1.25 + 4*256.
inline bool tier_pick_lfru(const std::vector<uint32_t> & heat,
                           const std::vector<uint32_t> & last,
                           uint32_t clock,
                           const std::vector<int>  & pinned,
                           int & slot, int & eid, long & gain) {
    if (heat.empty() || last.empty() || pinned.empty()) return false;
    int npin = (int)pinned.size();
    if (npin < 1) return false;

    int cold_idx = 0;
    for (int z = 1; z < npin; z++) {
        if (tier_lfru_score(heat[pinned[z]], last[pinned[z]], clock) <
            tier_lfru_score(heat[pinned[cold_idx]], last[pinned[cold_idx]], clock))
            cold_idx = z;
    }

    int hot = -1;
    uint64_t hs = 0;
    for (int e = 0, nexpert = (int)heat.size(); e < nexpert; e++) {
        bool resident = false;
        for (int z = 0; z < npin; z++) {
            if (pinned[z] == e) { resident = true; break; }
        }
        uint64_t score = tier_lfru_score(heat[e], last[e], clock);
        if (!resident && (hot < 0 || score > hs)) { hot = e; hs = score; }
    }
    if (hot < 0) return false;

    uint64_t cs = tier_lfru_score(heat[pinned[cold_idx]], last[pinned[cold_idx]], clock);
    if (hs <= cs + (cs >> 2) + (4u << 8)) return false;

    slot = cold_idx;
    eid = hot;
    gain = (long)((hs - cs) >> 8);
    return true;
}

// Halve all heat counters (exponential decay).
inline void tier_decay(std::vector<uint32_t> & heat) {
    for (auto & h : heat) h >>= 1;
}

} // namespace lamio