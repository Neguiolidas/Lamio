#include "lamio/expert_loader.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>

using namespace lamio;

static const char * TEST_GGUF = "test_expert_loader.gguf";

// Build a minimal GGUF with one fake expert tensor using gguf API
static void create_test_gguf() {
    // Use gguf API to write a minimal file with a tensor
    // We just create a binary blob that looks like a GGUF
    // For simplicity, write raw bytes directly since we test pread, not GGUF parsing
    FILE * f = fopen(TEST_GGUF, "wb");
    assert(f);

    // Write 1024 bytes of pattern data (simulates tensor data at offset 0)
    // Real GGUF would have header + kv + tensor info, but expert_loader uses
    // gguf_init_from_file to parse that. For test, we use a real small GGUF.
    // Instead, write a minimal valid GGUF using the C API is complex.
    // Alternative: test with a real GGUF if available, else skip.
    char data[1024];
    for (int i = 0; i < 1024; i++) data[i] = char(i % 256);
    fwrite(data, 1, 1024, f);
    fclose(f);
}

static void test_open_close() {
    expert_loader el;
    assert(!el.is_open());

    // Try opening a non-existent file
    bool ok = el.open("/nonexistent/path.gguf");
    assert(!ok);
    assert(!el.is_open());

    // Try opening a real GGUF (need one)
    // For now just test the open/close lifecycle on /dev/null (will fail on gguf parse)
    ok = el.open("/dev/null");
    assert(!ok);  // /dev/null is not valid GGUF
    assert(!el.is_open());

    printf("PASS: test_open_close\n");
}

static void test_read_nonexistent() {
    expert_loader el;
    // Reading without open should return 0
    char buf[64];
    size_t n = el.read_expert_slice("blk.0.ffn_up_exps.weight", 0, 64, buf, sizeof(buf));
    assert(n == 0);
    printf("PASS: test_read_nonexistent\n");
}

int main() {
    test_open_close();
    test_read_nonexistent();

    // Full integration test requires a real GGUF file.
    // That test is in test-lamio-integration.cpp (Ato 6).

    printf("\nAll expert_loader tests PASSED\n");

    // Cleanup
    remove(TEST_GGUF);

    return 0;
}