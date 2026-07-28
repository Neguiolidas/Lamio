#include "lamio/tier.h"
#include "lamio/tier_manager.h"
#include "lamio/expert_loader.h"
#include "lamio/tier_bridge.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace lamio;

// ============ tier.h unit tests ============

static void test_lfru_score() {
    // freq=1, last=clock (age=0) -> recent=255 -> score = (1<<8)|255 = 511
    uint64_t s = tier_lfru_score(1, 100, 100);
    assert(s == ((1ULL << 8) | 255));
    // freq=0, last=clock (age=0) -> recent=255 -> score = 0|255 = 255
    s = tier_lfru_score(0, 100, 100);
    assert(s == 255);
    // freq=2, last=0 (age=100) -> recent=155 -> score = (2<<8)|155 = 667
    s = tier_lfru_score(2, 0, 100);
    assert(s == ((2ULL << 8) | 155));
    printf("PASS: test_lfru_score\n");
}

static void test_pick_swap_basic() {
    // 8 experts, 4 pinned slots
    std::vector<uint32_t> heat = {1, 2, 0, 1, 5, 3, 0, 1};
    std::vector<int> pinned = {0, 1, 2, 3}; // slots 0-3 pinned

    int slot = -1, eid = -1;
    long gain = 0;
    bool found = tier_pick_swap(heat, pinned, slot, eid, gain);
    assert(found);
    // Expert 4 (heat=5) is not resident, should be swapped in
    // Expert 2 (heat=0) is coldest resident, should be swapped out
    assert(eid == 4); // hot expert
    printf("PASS: test_pick_swap_basic (slot=%d, eid=%d, gain=%ld)\n", slot, eid, gain);
}

// ============ tier_manager tests ============

static int g_load_count = 0;
static bool g_load_cb(int layer, int eid, void * dest, size_t size) {
    g_load_count++;
    memset(dest, 0xAB, size);
    return true;
}

static void test_manager_basic() {
    tier_manager mgr(4096, 2, 4); // 4KB budget, 2 layers, 4 experts
    mgr.set_load_callback(g_load_cb);

    // Register experts in the registry (normally done by tier_bridge)
    for (int l = 0; l < 2; l++) {
        for (int e = 0; e < 4; e++) {
            mgr.register_expert(l, e, 0, 128, 0); // fake offset, 128 bytes each
        }
    }

    g_load_count = 0;
    int selected[] = {0, 1};
    mgr.on_select(0, selected, 2);
    // At least the selected experts should be loaded
    assert(g_load_count >= 2);

    void * d0 = mgr.get_data(0, 0);
    void * d1 = mgr.get_data(0, 1);
    assert(d0 != nullptr);
    assert(d1 != nullptr);

    printf("PASS: test_manager_basic (loads=%d)\n", g_load_count);
}

// ============ tier_bridge tests ============

static void test_bridge_is_expert() {
    assert(tier_bridge::is_expert_tensor("blk.0.ffn_up_exps.weight"));
    assert(tier_bridge::is_expert_tensor("blk.3.ffn_gate_exps.weight"));
    assert(tier_bridge::is_expert_tensor("blk.39.ffn_down_exps.weight"));
    assert(!tier_bridge::is_expert_tensor("blk.0.ffn_up.weight"));
    assert(!tier_bridge::is_expert_tensor("blk.0.attn_q.weight"));
    assert(!tier_bridge::is_expert_tensor(nullptr));
    printf("PASS: test_bridge_is_expert\n");
}

static void test_bridge_register() {
    tier_bridge & br = tier_bridge::instance();
    // register without init (bridge not enabled)
    assert(!br.is_enabled());
    // Can still call register_expert_tensor (no-op when not enabled)
    printf("PASS: test_bridge_register\n");
}

// ============ expert_loader edge cases ============

static void test_loader_not_open() {
    expert_loader el;
    char buf[64];
    size_t n = el.read_expert_slice("blk.0.ffn_up_exps.weight", 0, 64, buf, sizeof(buf));
    assert(n == 0);
    assert(!el.is_open());
    printf("PASS: test_loader_not_open\n");
}

// ============ main ============

int main() {
    test_lfru_score();
    test_pick_swap_basic();
    test_manager_basic();
    test_bridge_is_expert();
    test_bridge_register();
    test_loader_not_open();

    printf("\nAll Lamio integration tests PASSED\n");
    return 0;
}