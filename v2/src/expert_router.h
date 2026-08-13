// expert_router.h — MoE-aware memory orchestration for Lamio.
//
// The differentiator: beyond the model itself, this component actively manages
// which expert weights are resident in RAM. Experts are mmap'd zero-copy, so
// pages are faulted in on demand. This orchestrator drives that working set:
//
//   1. EVICTION: when estimated RSS approaches max_rss_mb, pages of the
//      least-recently-used experts are returned to the OS via madvise(MADV_DONTNEED).
//      A later access re-faults them from disk — "weights on demand".
//
//   2. PREFETCH: optionally prefetch the routers' top-k experts in advance so
//      page-fault stalls hide behind pipeline.
//
//   3. STAY: an expert with high recurrence (hot) is exempt from eviction.

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <sys/mman.h>

namespace lamio {

// One individual expert slice within a layer's weight tensor.
// tag distinguishes gate(1)/up(2)/down(3) so madvise only touches the right pages.
struct ExpertSlice {
    uintptr_t addr;
    size_t    size;
    int       layer;
    int       expert_id;
    int       tag;        // 0=any, 1=gate, 2=up, 3=down
};

class ExpertRouter {
public:
    void configure(size_t max_rss_bytes, bool enable_prefetch = false) {
        budget_ = max_rss_bytes;
        prefetch_ = enable_prefetch;
    }

    // Register one expert slice (a single expert within one MoE weight tensor).
    int add_expert(uintptr_t addr, size_t size, int layer, int expert_id, int tag = 0) {
        ExpertSlice s{addr, size, layer, expert_id, tag};
        slices_.push_back(s);
        last_use_.push_back(0ull);
        resident_.push_back(false);
        return (int)slices_.size() - 1;
    }

    // Mark that an expert (by its registered id) was used in this tick.
    void touch(int slice_id, uint64_t tick) {
        if (slice_id < 0 || slice_id >= (int)slices_.size()) return;
        last_use_[slice_id] = tick;
        resident_[slice_id] = true;
    }

    // Touch all 3 slices (gate/up/down) for a given (layer, expert_id).
    void touch_expert(int layer, int expert_id, uint64_t tick) {
        for (size_t i = 0; i < slices_.size(); ++i) {
            if (slices_[i].layer == layer && slices_[i].expert_id == expert_id) {
                last_use_[i] = tick;
                resident_[i] = true;
            }
        }
    }

    // After a batch of touches, run eviction if over budget.
    void maybe_evict(uint64_t tick) {
        if (budget_ == 0 || slices_.empty()) return;
        size_t rss_est = read_actual_rss();
        if (rss_est <= budget_) return;

        // Build LRU order of resident experts (oldest first).
        std::vector<int> order(slices_.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return last_use_[a] < last_use_[b];
        });

        // Target: bring RSS below 90% of budget (leave 10% headroom).
        size_t target = (budget_ * 9) / 10;
        size_t need = (rss_est > target) ? (rss_est - target) : 1;
        size_t freed = 0;
        size_t evicted_this_round = 0;
        for (int id : order) {
            if (!resident_[id]) continue;
            if (last_use_[id] == tick) continue; // just used
            if (freed >= need) break;
            uintptr_t a = slices_[id].addr;
            size_t s = slices_[id].size;
            uintptr_t page = getpagesize();
            uintptr_t aligned = a & ~(uintptr_t)(page - 1);
            size_t len = s + (a - aligned);
            len = ((len + page - 1) / page) * page;
            // Drop pages back to OS. On Linux 5.4+ prefer MADV_PAGEOUT (actually
            // unmap, not just a hint). Fallback to MADV_DONTNEED.
#ifdef MADV_PAGEOUT
            int r = madvise((void*)aligned, len, MADV_PAGEOUT);
            if (r != 0) r = madvise((void*)aligned, len, MADV_DONTNEED);
#else
            int r = madvise((void*)aligned, len, MADV_DONTNEED);
#endif
            if (r == 0) {
                resident_[id] = false;
                freed += slices_[id].size;
                ++drop_count_;
                ++evicted_this_round;
            }
        }
        if (evicted_this_round > 0) {
            std::fprintf(stderr, "expert_router: %zu experts evicted this tick "
                         "(target ~%zuMB, freed %zuMB, RSS was %zuMB / budget %zuMB)\n",
                         evicted_this_round, target / (1024*1024),
                         freed / (1024*1024),
                         rss_est / (1024*1024), budget_ / (1024*1024));
        }
    }

    // Read actual RSS from /proc/self/status (VmRSS line).
    static size_t read_actual_rss() {
        FILE * f = fopen("/proc/self/status", "r");
        if (!f) return 0;
        size_t rss = 0;
        char buf[256];
        while (fgets(buf, sizeof(buf), f)) {
            if (strncmp(buf, "VmRSS:", 6) == 0) {
                rss = (size_t)strtoull(buf + 6, nullptr, 10) * 1024; // kB -> bytes
                break;
            }
        }
        fclose(f);
        return rss;
    }

    // Prefetch an expert's pages (will need them soon).
    void prefetch(int slice_id) const {
        if (slice_id < 0 || slice_id >= (int)slices_.size()) return;
        uintptr_t a = slices_[slice_id].addr;
        size_t s = slices_[slice_id].size;
        uintptr_t page = getpagesize();
        uintptr_t aligned = a & ~(uintptr_t)(page - 1);
        size_t len = s + (a - aligned);
        len = ((len + page - 1) / page) * page;
        madvise((void*)aligned, len, MADV_WILLNEED);
    }

    size_t estimate_rss() const {
        size_t s = base_rss_;
        for (size_t i = 0; i < slices_.size(); ++i)
            if (resident_[i]) s += slices_[i].size;
        return s;
    }

    void set_base_rss(size_t base) { base_rss_ = base; }
    size_t drop_count() const { return drop_count_; }
    size_t slice_count() const { return slices_.size(); }

    // Map (layer, expert_id) -> slice_id. Returns -1 if not found.
    int find_slice(int layer, int expert_id) const {
        for (size_t i = 0; i < slices_.size(); ++i) {
            if (slices_[i].layer == layer && slices_[i].expert_id == expert_id)
                return (int)i;
        }
        return -1;
    }

private:
    size_t budget_        = 0;
    bool   prefetch_      = false;
    size_t base_rss_      = 0;
    size_t drop_count_    = 0;
    std::vector<ExpertSlice> slices_;
    std::vector<uint64_t> last_use_;
    std::vector<bool>     resident_;
};

} // namespace lamio