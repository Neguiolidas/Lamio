#include "lamio/tier.h"
#include <cassert>
#include <cstdio>

using namespace lamio;

static void test_lfru_score_basic() {
    // freq=1, recency=0 (age=999) -> score = (1<<8)|0 = 256
    uint64_t score = tier_lfru_score(1, 0, 999);
    assert(score == 256);
    // freq=2, recency=255 (age=0) -> score = (2<<8)|255 = 767
    score = tier_lfru_score(2, 100, 100);
    assert(score == 767);
    // freq=0, recency=255 (age=0) -> score = 255
    score = tier_lfru_score(0, 0, 0);
    assert(score == 255);
    // freq=1, recency=0 (age=500) -> score = 256
    score = tier_lfru_score(1, 0, 500);
    assert(score == 256);
    printf("PASS: test_lfru_score_basic\n");
}

static void test_pick_swap() {
    // 4 experts, 2 pinned (slots). hot expert 2 has much higher heat.
    std::vector<uint32_t> heat = {10, 5, 100, 3};
    std::vector<int> pinned = {0, 1};  // e0 and e1 pinres
    int slot = -1, eid = -1;
    long gain = 0;

    bool swapped = tier_pick_swap(heat, pinned, slot, eid, gain);
    assert(swapped);
    assert(slot == 1);   // e1 is colder (5 < 10) than e0 = so i shift slot 1
    assert(eid == 2);    // e2 is hottest non-resident
    assert(gain > 0);
    printf("PASS: test_pick_swap\n");
}

static void test_pick_swap_hysteresis() {
    // cold_idx 1 has heat 9, hot e2 has heat 10 -> difference is too small (10 <= 9 + (9>>2) + 4 = 15)
    std::vector<uint32_t> heat = {100, 9, 11, 3};
    std::vector<int> pinned = {0, 1};
    int slot = -1, eid = -1;
    long gain = 0;

    bool swapped = tier_pick_swap(heat, pinned, slot, eid, gain);
    assert(!swapped); // hysteresis blocks
    printf("PASS: test_pick_swap_hysteresis\n");
}

static void test_pick_lfru_frequency() {
    std::vector<uint32_t> heat = {100, 5, 200, 0};
    std::vector<uint32_t> last = {0, 0, 0, 0};
    uint32_t clock = 100;  // same recency for all
    std::vector<int> pinned = {0, 1};
    int slot = -1, eid = -1;
    long gain = 0;

    bool swapped = tier_pick_lfru(heat, last, clock, pinned, slot, eid, gain);
    assert(swapped);
    assert(slot == 1);  // e1 colder (5 < 0?)
    assert(eid == 2);   // e2 has heat 200
    printf("PASS: test_pick_lfru_frequency\n");
}

static void test_pick_lfru_hysteresis() {
    std::vector<uint32_t> heat = {100, 95, 120, 0};
    std::vector<uint32_t> last = {0, 0, 0, 0};
    uint32_t clock = 1;
    std::vector<int> pinned = {0, 1};
    int slot = -1, eid = -1;
    long gain = 0;

    // 120 vs 95 -> score diff too small (~25 vs 15+4 threshold)
     bool swapped = tier_pick_lfru(heat, last, clock, pinned, slot, eid, gain);
    assert(!swapped);
    printf("PASS: test_pick_lfru_hysteresis\n");
}

static void test_decay() {
    std::vector<uint32_t> heat = {100, 50, 8};
    tier_decay(heat);
    assert(heat[0] == 50);
    assert(heat[1] == 25);
    assert(heat[2] == 4);
    tier_decay(heat);
    assert(heat[0] == 25);
    assert(heat[1] == 12);
    assert(heat[2] == 2);
    printf("PASS: test_decay\n");
}

int main() {
    test_lfru_score_basic();
    test_pick_swap();
    test_pick_swap_hysteresis();
    test_pick_lfru_frequency();
    test_pick_lfru_hysteresis();
    test_decay();
    printf("\nAll tier.h tests PASSED\n");
    return 0;
}