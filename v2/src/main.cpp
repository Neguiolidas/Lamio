#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_reader.h"

// Phase-1 smoke: open a .gguf, print arch + tensor count + MoE flag,
// and locate the router tensor. Proves the ggml-link + GGUF parser are live.
int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: lamio <model.gguf>\n");
        return 2;
    }

    lamio::GgufReader r(argv[1]);
    if (!r.ok()) {
        std::fprintf(stderr, "lamio: %s\n", r.error().c_str());
        return 1;
    }

    std::printf("arch      : %s\n", r.arch().c_str());
    std::printf("tensors   : %zu\n", r.tensors().size());
    std::printf("is_moe    : %s\n", r.is_moe() ? "yes" : "no");

    // Show MoE-relevant tensors (first 12) to confirm shape detection.
    int shown = 0;
    for (const auto & t : r.tensors()) {
        if (t.name.find("exps") != std::string::npos ||
            t.name.find("router") != std::string::npos) {
            std::printf("  %s  dims=%d ne=[%lld %lld %lld %lld] %lld bytes\n",
                        t.name.c_str(), t.n_dims,
                        (long long)t.ne[0], (long long)t.ne[1],
                        (long long)t.ne[2], (long long)t.ne[3],
                        (long long)t.nbytes);
            if (++shown >= 12) break;
        }
    }
    std::printf("MoE tensors shown: %d\n", shown);

    return 0;
}