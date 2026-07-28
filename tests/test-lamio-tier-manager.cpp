#include "lamio/tier_manager.h"
#include <cassert>
#include <cstdio>

using namespace lamio;

static int g_load_count = 0;
static bool test_load_cb(int layer, int eid, void * dest, size_t size) {
    g_load_count++;
    // Just fill with pattern to verify load happened
    for (size_t i = 0; i < size; i++) ((char*)dest)[i] = char(layer * 100 + eid);
    return true;
}

static void test_register_and_resident() {
    tier_manager tm(1024 * 1024, 4, 256); // 1MB budget, 4 layers, 256 experts
    tm.set_load_callback(test_load_cb);

    tm.register_expert(0, 0, 100, 64, 0); // layer 0, eid 0, 64 bytes

    assert(!tm.is_resident(0, 0)); // not loaded yet

    int selected[] = {0};
    g_load_count = 0;
    tm.on_select(0, selected, 1);

    assert(tm.is_resident(0, 0));
    assert(g_load_count == 1);
    void * data = tm.get_data(0, 0);
    assert(data != nullptr);
    assert(((char*)data)[0] == 0); // pattern: layer 0 * 100 + eid 0 = 0

    printf("PASS: test_register_and_resident\n");
}

static void test_cache_hit() {
    tier_manager tm(1024 * 1024, 4, 256);
    tm.set_load_callback(test_load_cb);

    tm.register_expert(0, 0, 100, 64, 0);
    tm.register_expert(0, 1, 200, 64, 0);

    int sel[] = {0, 1};
    g_load_count = 0;
    tm.on_select(0, sel, 2);
    assert(g_load_count == 2);

    // Select again - should be hits, no new loads
    tm.on_select(0, sel, 2);
    assert(g_load_count == 2); // still 2, hits dont trigger loads

    printf("PASS: test_cache_hit\n");
}

static void test_cache_evict() {
    // Tight budget: only fits 2 experts
    tier_manager tm(128, 1, 256); // 1 layer, 128 byte budget
    tm.set_load_callback(test_load_cb);

    tm.register_expert(0, 0, 100, 64, 0);
    tm.register_expert(0, 1, 200, 64, 0);
    tm.register_expert(0, 2, 300, 64, 0);

    g_load_count = 0;
    int sel0[] = {0};
    int sel1[] = {1};
    int sel2[] = {2};

    tm.on_select(0, sel0, 1);
    assert(g_load_count == 1);
    assert(tm.is_resident(0, 0));

    tm.on_select(0, sel1, 1);
    assert(g_load_count == 2);
    assert(tm.is_resident(0, 1));
    assert(tm.is_resident(0, 0)); // 2 experts fit in 128? 64+64=128, fits

    // Add third -> evict one
    tm.on_select(0, sel2, 1);
    assert(g_load_count == 3);
    assert(tm.is_resident(0, 2));

    // At least one of 0,1 was evicted (64+64+64 > 128)
    int resident_count = (tm.is_resident(0,0) ? 1 : 0) +
                         (tm.is_resident(0,1) ? 1 : 0) +
                         (tm.is_resident(0,2) ? 1 : 0);
    assert(resident_count == 2); // budget fits 2

    printf("PASS: test_cache_evict\n");
}

static void test_bump_heat() {
    tier_manager tm(1024 * 1024, 1, 256);
    tm.set_load_callback(test_load_cb);
    tm.register_expert(0, 0, 100, 64, 0);

    int sel[] = {0};
    tm.on_select(0, sel, 1);
    assert(tm.is_resident(0, 0));

    // Select again -> hits bump heat
    int prev_hits = tm.total_hits();
    tm.on_select(0, sel, 1);
    assert(tm.total_hits() == prev_hits + 1);

    printf("PASS: test_bump_heat\n");
}

static void test_multiple_layers() {
    tier_manager tm(1024 * 1024, 3, 256);
    tm.set_load_callback(test_load_cb);

    tm.register_expert(0, 5, 100, 64, 0);
    tm.register_expert(1, 10, 200, 64, 0);
    tm.register_expert(2, 15, 300, 64, 0);

    int sel0[] = {5};
    int sel1[] = {10};
    int sel2[] = {15};

    g_load_count = 0;
    tm.on_select(0, sel0, 1);
    tm.on_select(1, sel1, 1);
    tm.on_select(2, sel2, 1);

    assert(tm.is_resident(0, 5));
    assert(tm.is_resident(1, 10));
    assert(tm.is_resident(2, 15));
    assert(g_load_count == 3);

    printf("PASS: test_multiple_layers\n");
}

int main() {
    test_register_and_resident();
    test_cache_hit();
    test_cache_evict();
    test_bump_heat();
    test_multiple_layers();
    printf("\nAll tier_manager tests PASSED\n");
    return 0;
}