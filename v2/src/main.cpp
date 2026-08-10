#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_reader.h"
#include "model_config.h"
#include "block_tensors.h"

static void print_info(const char * path) {
    lamio::GgufReader r(path);
    if (!r.ok()) {
        std::fprintf(stderr, "lamio: %s\n", r.error().c_str());
        std::exit(1);
    }

    lamio::ModelConfig cfg = lamio::parse_model_config(r);
    lamio::ModelTensors mt = lamio::map_tensors(r, cfg.n_layers);

    std::printf("file      : %s\n", path);
    std::printf("arch      : %s\n", cfg.arch.c_str());
    std::printf("layers    : %d\n", cfg.n_layers);
    std::printf("n_embd    : %d\n", cfg.n_embd);
    std::printf("n_heads   : %d (kv=%d)\n", cfg.n_heads, cfg.n_kv_heads);
    std::printf("head_dim  : %d\n", cfg.head_dim);
    std::printf("context   : %d\n", cfg.context_length);
    std::printf("vocab     : %d\n", cfg.vocab_size);
    std::printf("norm_eps  : %g\n", cfg.norm_eps);
    std::printf("rope_base : %g\n", cfg.rope_freq_base);
    std::printf("rope_dim  : %d\n", cfg.rope_dim_count);

    if (cfg.is_moe()) {
        std::printf("experts   : %d (active=%d, ffn_len=%d)\n",
                    cfg.n_experts, cfg.n_active, cfg.n_ffn_expert);
    } else {
        std::printf("experts   : 0 (dense)\n");
    }

    std::printf("tensors   : %zu\n", r.tensors().size());
    std::printf("is_moe    : %s\n", r.is_moe() ? "yes" : "no");

    // Block tensor mapping summary
    int moe_blocks = 0;
    int dense_blocks = 0;
    for (const auto & b : mt.blocks) {
        if (b.has_moe) ++moe_blocks;
        if (b.has_dense_ffn) ++dense_blocks;
    }
    std::printf("blocks    : %d (moe=%d, dense_ffn=%d)\n",
                (int)mt.blocks.size(), moe_blocks, dense_blocks);

    // Show first MoE block detail
    for (const auto & b : mt.blocks) {
        if (b.has_moe) {
            std::printf("blk.%d MoE detail:\n", b.layer);
            if (b.router.tensor_idx >= 0)
                std::printf("  router        : %s [%lld bytes]\n",
                            b.router.name.c_str(), (long long)b.router.nbytes);
            if (b.ffn_gate_exps.tensor_idx >= 0)
                std::printf("  gate_exps     : %s ne=[%lld %lld %lld] [%lld bytes]\n",
                            b.ffn_gate_exps.name.c_str(),
                            (long long)b.ffn_gate_exps.ne[0],
                            (long long)b.ffn_gate_exps.ne[1],
                            (long long)b.ffn_gate_exps.ne[2],
                            (long long)b.ffn_gate_exps.nbytes);
            if (b.ffn_up_exps.tensor_idx >= 0)
                std::printf("  up_exps       : %s ne=[%lld %lld %lld] [%lld bytes]\n",
                            b.ffn_up_exps.name.c_str(),
                            (long long)b.ffn_up_exps.ne[0],
                            (long long)b.ffn_up_exps.ne[1],
                            (long long)b.ffn_up_exps.ne[2],
                            (long long)b.ffn_up_exps.nbytes);
            if (b.ffn_down_exps.tensor_idx >= 0)
                std::printf("  down_exps     : %s ne=[%lld %lld %lld] [%lld bytes]\n",
                            b.ffn_down_exps.name.c_str(),
                            (long long)b.ffn_down_exps.ne[0],
                            (long long)b.ffn_down_exps.ne[1],
                            (long long)b.ffn_down_exps.ne[2],
                            (long long)b.ffn_down_exps.nbytes);
            break;
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: lamio [--info] <model.gguf>\n");
        return 2;
    }

    bool info_mode = false;
    const char * model_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--info") == 0) {
            info_mode = true;
        } else {
            model_path = argv[i];
        }
    }

    if (!model_path) {
        std::fprintf(stderr, "lamio: no model file specified\n");
        return 2;
    }

    if (info_mode) {
        print_info(model_path);

        // Task 2.3 probe: load the router tensor from the first MoE block
        // and verify data actually comes from disk (non-zero sample).
        lamio::GgufReader r(model_path);
        if (!r.ok()) { std::fprintf(stderr, "lamio: %s\n", r.error().c_str()); return 1; }
        lamio::ModelConfig cfg = lamio::parse_model_config(r);
        lamio::ModelTensors mt = lamio::map_tensors(r, cfg.n_layers);

        for (const auto & b : mt.blocks) {
            if (b.router.tensor_idx >= 0) {
                const size_t size = (size_t)(b.router.nbytes);
                std::vector<uint8_t> buf(size);
                size_t got = r.load_tensor_data(b.router.tensor_idx, buf.data(), buf.size());
                std::printf("router load: requested=%zu got=%zu\n", size, got);
                if (got == size && size > 0) {
                    // compute checksum + first bytes
                    uint32_t sum = 0;
                    for (size_t i = 0; i < got; ++i) sum += buf[i];
                    std::printf("router checksum=%u first8=%02x %02x %02x %02x %02x %02x %02x %02x\n",
                                sum,
                                buf[0], buf[1], buf[2], buf[3],
                                buf[4], buf[5], buf[6], buf[7]);
                } else {
                    std::printf("router load FAILED (got %zu != %zu)\n", got, size);
                }
                break;
            }
        }
        return 0;
    }

    // default: minimal GGUF dump (Fase 1 behavior)
    lamio::GgufReader r(model_path);
    if (!r.ok()) {
        std::fprintf(stderr, "lamio: %s\n", r.error().c_str());
        return 1;
    }

    std::printf("arch      : %s\n", r.arch().c_str());
    std::printf("tensors   : %zu\n", r.tensors().size());
    std::printf("is_moe    : %s\n", r.is_moe() ? "yes" : "no");

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