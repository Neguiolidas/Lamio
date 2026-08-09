#include "lamio/tier_manager.h"
#include <cassert>
#include <cstdio>

using namespace lamio;

static int g_load_count = 0;
static bool test_load_cb(int layer, int eid, int type_idx, void * dest, size_t size) {
    g_load_count++;
    for (size_t i = 0; i < size; i++) ((char*)dest)[i] = char(layer * 100 + eid * 10 + type_idx);
    return true;
}

static void test_register_and_resident() {
    tier_manager tm(1024 * 1024, 4, 256);
    tm.set_load_callback(test_load_cb);

    tm.register_expert(0, 0, 0, 100, 64, 0);

    assert(!tm.is_resident(0, 0, 0));

    int selected[] = {0};
    g_load_count = 0;
    tm.on_select(0, 0, selected, 1);

    assert(tm.is_resident(0, 0, 0));
    assert(g_load_count == 1);
    void * data = tm.get_data(0, 0, 0);
    assert(data != nullptr);
    assert(((char*)data)[0] == 0);

    printf("PASS: test_register_and_resident\n");
}

static void test_cache_hit() {
    tier_manager tm(1024 * 1024, 4, 256);
    tm.set_load_callback(test_load_cb);

    tm.register_expert(0, 0, 0, 100, 64, 0);
    tm.register_expert(0, 1, 0, 200, 64, 0);

    int sel[] = {0, 1};
    g_load_count = 0;
    tm.on_select(0, 0, sel, 2);
    assert(g_load_count == 2);

    tm.on_select(0, 0, sel, 2);
    assert(g_load_count == 2);

    printf("PASS: test_cache_hit\n");
}

static void test_cache_evict() {
    tier_manager tm(128, 1, 256);
    tm.set_load_callback(test_load_cb);

    tm.register_expert(0, 0, 0, 100, 64, 0);
    tm.register_expert(0, 1, 0, 200, 64, 0);
    tm.register_expert(0, 2, 0, 300, 64, 0);

    g_load_count = 0;
    int sel0[] = {0};
    int sel1[] = {1};
    int sel2[] = {2};

    tm.on_select(0, 0, sel0, 1);
    assert(g_load_count == 1);
    assert(tm.is_resident(0, 0, 0));

    tm.on_select(0, 0, sel1, 1);
    assert(g_load_count == 2);
    assert(tm.is_resident(0, 1, 0));
    assert(tm.is_resident(0, 0, 0));

    tm.on_select(0, 0, sel2, 1);
    assert(g_load_count == 3);
    assert(tm.is_resident(0, 2, 0));

    int resident_count = (tm.is_resident(0,0,0) ? 1 : 0) +
                         (tm.is_resident(0,1,0) ? 1 : 0) +
                         (tm.is_resident(0,2,0) ? 1 : 0);
    assert(resident_count == 2);

    printf("PASS: test_cache_evict\n");
}

static void test_bump_heat() {
    tier_manager tm(1024 * 1024, 1, 256);
    tm.set_load_callback(test_load_cb);
    tm.register_expert(0, 0, 0, 100, 64, 0);

    int sel[] = {0};
    tm.on_select(0, 0, sel, 1);
    assert(tm.is_resident(0, 0, 0));

    int prev_hits = tm.total_hits();
    tm.on_select(0, 0, sel, 1);
    assert(tm.total_hits() == prev_hits + 1);

    printf("PASS: test_bump_heat\n");
}

static void test_multiple_layers() {
    tier_manager tm(1024 * 1024, 3, 256);
    tm.set_load_callback(test_load_cb);

    tm.register_expert(0, 5, 0, 100, 64, 0);
    tm.register_expert(1, 10, 0, 200, 64, 0);
    tm.register_expert(2, 15, 0, 300, 64, 0);

    int sel0[] = {5};
    int sel1[] = {10};
    int sel2[] = {15};

    g_load_count = 0;
    tm.on_select(0, 0, sel0, 1);
    tm.on_select(1, 0, sel1, 1);
    tm.on_select(2, 0, sel2, 1);

    assert(tm.is_resident(0, 5, 0));
    assert(tm.is_resident(1, 10, 0));
    assert(tm.is_resident(2, 15, 0));
    assert(g_load_count == 3);

    printf("PASS: test_multiple_layers\n");
}

static void test_type_idx_independent() {
    tier_manager tm(1024 * 1024, 1, 256);
    tm.set_load_callback(test_load_cb);

    tm.register_expert(0, 0, 0, 100, 64, 0); // up
    tm.register_expert(0, 0, 1, 200, 64, 0); // gate
    tm.register_expert(0, 0, 2, 300, 64, 0); // down

    int sel[] = {0};
    g_load_count = 0;

    tm.on_select(0, 0, sel, 1);
    assert(tm.is_resident(0, 0, 0));
    assert(!tm.is_resident(0, 0, 1));
    assert(!tm.is_resident(0, 0, 2));
    assert(g_load_count == 1);

    tm.on_select(0, 1, sel, 1);
    assert(tm.is_resident(0, 0, 1));
    assert(!tm.is_resident(0, 0, 2));
    assert(g_load_count == 2);

    tm.on_select(0, 2, sel, 1);
    assert(tm.is_resident(0, 0, 2));
    assert(g_load_count == 3);

    void * up   = tm.get_data(0, 0, 0);
    void * gate = tm.get_data(0, 0, 1);
    void * down = tm.get_data(0, 0, 2);
    assert(((char*)up)[0]   == char(0 * 100 + 0 * 10 + 0));
    assert(((char*)gate)[0] == char(0 * 100 + 0 * 10 + 1));
    assert(((char*)down)[0] == char(0 * 100 + 0 * 10 + 2));

    printf("PASS: test_type_idx_independent\n");
}

int main() {
    test_register_and_resident();
    test_cache_hit();
    test_cache_evict();
    test_bump_heat();
    test_multiple_layers();
    test_type_idx_independent();
    printf("\nAll tier_manager tests PASSED\n");
    return 0;
}
