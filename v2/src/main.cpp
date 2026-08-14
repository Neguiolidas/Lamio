#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <thread>
#include <climits>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "gguf_reader.h"
#include "model_config.h"
#include "block_tensors.h"
#include "tokenizer.h"
#include "weight_loader.h"
#include "dense_forward.h"
#include "qwen35_forward.h"
#include "expert_router.h"
#include "sampling.h"
// server.h not needed - serve mode uses httplib inline

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

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
            if (b.ffn_gate_inp.tensor_idx >= 0)
                std::printf("  router        : %s [%lld bytes]\n",
                            b.ffn_gate_inp.name.c_str(), (long long)b.ffn_gate_inp.nbytes);
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
    bool tokenize_mode = false;
    bool list_mode = false;
    bool generate_mode = false;
    bool repl_mode = false;
    const char * model_path = nullptr;
    std::string prompt_text;
    lamio::SamplerConfig sampler_cfg;
    unsigned int rng_seed = 0;
    std::vector<int> stop_tokens;
    bool auto_stop = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--info") == 0) {
            info_mode = true;
        } else if (std::strcmp(argv[i], "--tokenize") == 0) {
            tokenize_mode = true;
            if (i + 1 < argc) prompt_text = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 < argc) prompt_text = argv[++i];
        } else if (std::strcmp(argv[i], "--list-tensors") == 0) {
            list_mode = true;
        } else if (std::strcmp(argv[i], "--generate") == 0) {
            generate_mode = true;
        } else if (std::strcmp(argv[i], "--repl") == 0) {
            repl_mode = true;
        } else if (std::strcmp(argv[i], "--n-gen") == 0) {
            if (i + 1 < argc) ++i;
        } else if (std::strcmp(argv[i], "--temperature") == 0 || std::strcmp(argv[i], "--temp") == 0) {
            if (i + 1 < argc) sampler_cfg.temperature = std::stof(argv[++i]);
        } else if (std::strcmp(argv[i], "--top-k") == 0) {
            if (i + 1 < argc) sampler_cfg.top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--top-p") == 0) {
            if (i + 1 < argc) sampler_cfg.top_p = std::stof(argv[++i]);
        } else if (std::strcmp(argv[i], "--repeat-penalty") == 0) {
            if (i + 1 < argc) sampler_cfg.repeat_penalty = std::stof(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) rng_seed = (unsigned)std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--stop-token") == 0) {
            if (i + 1 < argc) stop_tokens.push_back(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) ++i; // skip the value (parsed in second loop)
        } else if (std::strcmp(argv[i], "--max-rss-mb") == 0) {
            if (i + 1 < argc) ++i; // skip the value (parsed in second loop)
        } else if (std::strcmp(argv[i], "--auto-stop") == 0) {
            auto_stop = true;
        } else if (std::strcmp(argv[i], "--max-layers") == 0) {
            if (i + 1 < argc) ++i; // skip the value
        } else {
            model_path = argv[i];
        }
    }

    if (!model_path) {
        std::fprintf(stderr, "lamio: no model file specified\n");
        return 2;
    }

    if (tokenize_mode) {
        lamio::GgufReader r(model_path);
        if (!r.ok()) { std::fprintf(stderr, "lamio: %s\n", r.error().c_str()); return 1; }
        lamio::BpeTokenizer tok;
        if (!tok.load(r)) {
            std::fprintf(stderr, "lamio: tokenizer load failed\n");
            return 1;
        }
        auto ids = tok.encode(prompt_text);
        std::printf("tokens (%zu):", ids.size());
        for (int32_t id : ids) std::printf(" %d", id);
        std::printf("\ndecoded: %s\n", tok.decode(ids).c_str());
        return 0;
    }

    // --qwen35-generate: full forward pass with attention + delta net
    std::string gen_prompt = prompt_text;
    if (gen_prompt.empty()) gen_prompt = "Hello";
    int n_gen_tokens = 8;
    int max_layers = -1;  // -1 = all
    int n_cpu_threads = 1; // CPU threads for the matmul backend
    int max_rss_mb = 0;    // MoE expert cache budget (0 = unlimited); see ExpertRouter
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--max-layers") == 0 && i + 1 < argc) max_layers = atoi(argv[++i]);
        if (std::strcmp(argv[i], "--n-gen") == 0 && i + 1 < argc) n_gen_tokens = atoi(argv[++i]);
        if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) n_cpu_threads = atoi(argv[++i]);
        if (std::strcmp(argv[i], "--max-rss-mb") == 0 && i + 1 < argc) max_rss_mb = atoi(argv[++i]);
    }
    if (generate_mode || repl_mode) {
        std::mt19937 rng(rng_seed > 0 ? rng_seed : std::random_device{}());
        lamio::GgufReader r(model_path);
        if (!r.ok()) { std::fprintf(stderr, "lamio: %s\n", r.error().c_str()); return 1; }

        lamio::ModelConfig cfg = lamio::parse_model_config(r);
        lamio::ModelTensors mt = lamio::map_tensors(r, cfg.n_layers);
        lamio::BpeTokenizer tok;
        if (!tok.load(r)) { std::fprintf(stderr, "lamio: tokenizer load failed\n"); return 1; }

        std::vector<int32_t> all_ids;
        if (generate_mode) {
            all_ids = tok.encode(gen_prompt);
            if (all_ids.empty()) { std::fprintf(stderr, "lamio: empty prompt\n"); return 1; }
            std::fprintf(stderr, "prompt: %s\n", gen_prompt.c_str());
            std::fprintf(stderr, "tokens: %zu\n", all_ids.size());
        }

        // Parse qwen35 hparams
        lamio::Qwen35HParams hp = lamio::parse_qwen35_hparams(cfg, r);

        // ggml setup — auto-detect best backend (GPU of any kind, then CPU)
        ggml_backend_t backend = nullptr;
        ggml_backend_t gpu_backend = nullptr;
        ggml_backend_buffer_type_t gpu_buft = nullptr;
        size_t gpu_free_mem = 0, gpu_total_mem = 0;
        const char * gpu_name = nullptr;

        // Load all available backend plugins (CUDA, Metal, Vulkan, etc.)
        ggml_backend_load_all();

        // Try GPU first (any type: CUDA, Metal, Vulkan, ROCm)
        ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!gpu_dev) {
            // Try integrated GPU (shares host memory)
            gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
        if (gpu_dev) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(gpu_dev, &props);
            gpu_name = props.name;
            gpu_free_mem = props.memory_free;
            gpu_total_mem = props.memory_total;
            fprintf(stderr, "lamio: GPU detected: %s (%s) free=%zuMB total=%zuMB\n",
                    props.name ? props.name : "?",
                    props.description ? props.description : "?",
                    gpu_free_mem / (1024*1024), gpu_total_mem / (1024*1024));

            gpu_backend = ggml_backend_dev_init(gpu_dev, nullptr);
            if (gpu_backend) {
                gpu_buft = ggml_backend_dev_buffer_type(gpu_dev);
                backend = gpu_backend;
                fprintf(stderr, "lamio: using GPU backend: %s\n", gpu_name ? gpu_name : "unknown");
            } else {
                fprintf(stderr, "lamio: GPU init failed, falling back to CPU\n");
            }
        }

        if (!backend) {
            backend = ggml_backend_cpu_init();
            if (!backend) { std::fprintf(stderr, "lamio: backend init failed\n"); return 1; }
            fprintf(stderr, "lamio: using CPU backend\n");
        }
        // Use the configured CPU thread count. Default 1 preserves the original
        // single-thread behavior (optimal for small models where thread sync
        // overhead exceeds matmul gains); scale up via `--threads N` for large
        // models (e.g. Ornith-35B) where multi-core matmuls win big.
        ggml_backend_cpu_set_n_threads(backend, n_cpu_threads < 1 ? 1 : n_cpu_threads);

        // Weight context
        size_t wpool = ggml_tensor_overhead() * (size_t)(cfg.n_layers * 30 + 256) + 64*1024*1024;
        void * wbuf = malloc(wpool);
        struct ggml_init_params wparams = { wpool, wbuf, true };
        ggml_context * wctx = ggml_init(wparams);
        lamio::WeightLoader wl(backend, wctx, r);

        // Create ALL weight tensors (attention + FFN + delta net + norms)
        ggml_tensor * embd_w = wl.load(mt.global.token_embd);
        for (int l = 0; l < cfg.n_layers; ++l) {
            const auto & b = mt.blocks[l];
            bool is_recr = (l + 1) % 4 != 0;  // qwen35 hybrid: layer 3,7,11,15,17 = attention
            auto try_load = [&](const lamio::TensorRef & ref) {
                if (ref.tensor_idx >= 0) wl.load(ref);
            };
            if (is_recr) {
                try_load(b.attn_norm);
                try_load(b.attn_post_norm);
                try_load(b.attn_qkv);
                try_load(b.attn_gate);
                try_load(b.ssm_conv1d);
                try_load(b.ssm_dt);
                try_load(b.ssm_a);
                try_load(b.ssm_alpha);
                try_load(b.ssm_beta);
                try_load(b.ssm_norm);
                try_load(b.ssm_out);
            } else {
                try_load(b.attn_norm);
                try_load(b.attn_post_norm);
                try_load(b.attn_q);
                try_load(b.attn_k);
                try_load(b.attn_v);
                try_load(b.attn_output);
                try_load(b.attn_q_norm);
                try_load(b.attn_k_norm);
            }
            try_load(b.ffn_gate);
            try_load(b.ffn_up);
            try_load(b.ffn_down);
            try_load(b.ffn_norm);
            // MoE tensors
            try_load(b.ffn_gate_inp);
            try_load(b.ffn_gate_exps);
            try_load(b.ffn_up_exps);
            try_load(b.ffn_down_exps);
            try_load(b.ffn_gate_inp_shexp);
            try_load(b.ffn_gate_shexp);
            try_load(b.ffn_up_shexp);
            try_load(b.ffn_down_shexp);
        }
        if (mt.global.output_norm.tensor_idx >= 0) wl.load(mt.global.output_norm);
        if (mt.global.output_weight.tensor_idx >= 0) wl.load(mt.global.output_weight);

        // --- mmap-based zero-copy weight loading ---
        // mmap the GGUF file. Trunk tensors get copied into a small backend
        // buffer. Expert tensors (16GB) point directly into the mmap region
        // via ggml_backend_tensor_alloc on a buffer that wraps the mmap.
        // This gives ~1.4GB RSS (trunk) instead of 17GB.
        int gguf_fd = open(model_path, O_RDONLY);
        if (gguf_fd < 0) {
            std::fprintf(stderr, "lamio: cannot open %s for mmap\n", model_path);
            return 1;
        }
        off_t file_size = lseek(gguf_fd, 0, SEEK_END);
        void * mmap_addr = mmap(nullptr, (size_t)file_size, PROT_READ, MAP_PRIVATE, gguf_fd, 0);
        if (mmap_addr == MAP_FAILED) {
            std::fprintf(stderr, "lamio: mmap failed (%zu bytes)\n", (size_t)file_size);
            close(gguf_fd);
            return 1;
        }
        madvise(mmap_addr, (size_t)file_size, MADV_RANDOM);
        close(gguf_fd);

        // Phase 1: Allocate backend buffer ONLY for trunk tensors (non-expert).
        // We temporarily remove expert tensors from wctx so alloc_ctx_tensors
        // doesn't allocate space for them.
        // Phase 2: For expert tensors, set tensor->data directly to mmap+offset.
        size_t data_off = r.data_offset();
        size_t loaded = 0;

        // First: identify expert tensors and set their data to mmap (zero-copy)
        // We do this by calling ggml_backend_tensor_alloc on a buffer that wraps
        // the full mmap region.
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        // Create a buffer with size of the full file, but we won't actually
        // use allocated memory - we'll set tensor->data to mmap offsets.
        // The CPU backend buffer allows setting tensor data to arbitrary addrs
        // if the buffer is "meta" type. Since we can't create meta easily,
        // we'll use a different approach: manually set tensor->data.
        for (auto & kv : wl.refs_map()) {
            ggml_tensor * t = kv.first;
            const lamio::TensorRef & ref = kv.second;
            if (ref.tensor_idx < 0) continue;

            bool is_expert = (ref.name.find("ffn_gate_exps") != std::string::npos ||
                              ref.name.find("ffn_up_exps")   != std::string::npos ||
                              ref.name.find("ffn_down_exps") != std::string::npos);

            if (is_expert) {
                // Zero-copy: point tensor data directly into mmap region
                size_t toff = r.tensor_offset(ref.tensor_idx);
                size_t abs_off = data_off + toff;
                if (abs_off + (size_t)ref.nbytes > (size_t)file_size) continue;
                t->data = (uint8_t *)mmap_addr + abs_off;
                ++loaded;
            }
        }
        size_t n_expert = loaded;
        std::fprintf(stderr, "mmap: %zu expert tensors (zero-copy)\n", n_expert);

        // Now allocate backend buffer for remaining (trunk) tensors
        // Need to temporarily unlink expert tensors from wctx so alloc doesn't
        // try to allocate space for them. Since we already set t->data for
        // experts, alloc_ctx_tensors should skip them (it checks data == nullptr).
        ggml_backend_buffer_t wbackend_buf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        if (!wbackend_buf) { std::fprintf(stderr, "lamio: weight alloc failed (all tensors zero/allocated)\\n"); return 1; }
        std::fprintf(stderr, "mmap: backend buffer %zu MB, alloc OK\n", ggml_backend_buffer_get_size(wbackend_buf) / (1024*1024));

        // --- Expert Router: MoE-aware RSS orchestration ---
        // Register per-expert slices so eviction is granular (per expert, not
        // per layer). Each MoE weight tensor [d0, d1, n_experts] has stride
        // d0*d1*type per expert. The 3 tensors (gate/up/down) are ~142MB apart
        // in memory, so we register ONE slice per tensor per expert rather
        // than one big range (otherwise madvise would free the gap pages too).
        lamio::ExpertRouter expert_router;
        if (max_rss_mb > 0 && cfg.n_experts > 0) {
            expert_router.configure((size_t)max_rss_mb * 1024 * 1024);
            int slice_counter = 0;
            for (auto & kv : wl.refs_map()) {
                ggml_tensor * t = kv.first;
                const lamio::TensorRef & ref = kv.second;
                if (ref.tensor_idx < 0 || t->data == nullptr) continue;
                bool is_exp = (ref.name.find("ffn_gate_exps") != std::string::npos ||
                               ref.name.find("ffn_up_exps")   != std::string::npos ||
                               ref.name.find("ffn_down_exps") != std::string::npos);
                if (!is_exp) continue;
                int layer = -1;
                auto dot = ref.name.find('.');
                if (dot != std::string::npos) layer = std::atoi(ref.name.c_str() + dot + 1);
                if (layer < 0 || layer >= cfg.n_layers) continue;
                size_t per_exp = (size_t)t->nb[2];
                uintptr_t base = (uintptr_t)t->data;
                // Determine tensor type tag for find_slice lookup
                int tag = 0;
                if (ref.name.find("gate_exps") != std::string::npos) tag = 1;
                else if (ref.name.find("up_exps") != std::string::npos) tag = 2;
                else tag = 3;
                for (int e = 0; e < cfg.n_experts; ++e) {
                    uintptr_t a = base + (size_t)e * per_exp;
                    expert_router.add_expert(a, per_exp, layer, e, tag);
                    ++slice_counter;
                }
            }
            size_t trunk_rss = ggml_backend_buffer_get_size(wbackend_buf);
            expert_router.set_base_rss(trunk_rss + 256ULL*1024*1024 + 128ULL*1024*1024);
            std::fprintf(stderr, "expert_router: %zu expert slices (40L x 256E x 3T), budget=%dMB, base_rss=%zuMB\n",
                         expert_router.slice_count(), max_rss_mb,
                         expert_router.estimate_rss() / (1024*1024));
        }

        // Copy trunk tensor data from mmap into backend buffer
        size_t n_trunk = 0;
        for (auto & kv : wl.refs_map()) {
            ggml_tensor * t = kv.first;
            const lamio::TensorRef & ref = kv.second;
            if (ref.tensor_idx < 0 || t->data == nullptr) continue;
            // Skip experts (already set to mmap)
            bool is_expert = (ref.name.find("ffn_gate_exps") != std::string::npos ||
                              ref.name.find("ffn_up_exps")   != std::string::npos ||
                              ref.name.find("ffn_down_exps") != std::string::npos);
            if (is_expert) continue;

            size_t toff = r.tensor_offset(ref.tensor_idx);
            size_t abs_off = data_off + toff;
            if (abs_off + (size_t)ref.nbytes > (size_t)file_size) continue;

            // If alloc_ctx_tensors already allocated this tensor in the buffer,
            // t->data points to the buffer. We need to copy data there.
            void * src = (uint8_t *)mmap_addr + abs_off;
            ggml_backend_tensor_set(t, src, 0, (size_t)ref.nbytes);
            ++n_trunk;
        }
        loaded += n_trunk;
        std::fprintf(stderr, "loaded %zu weight tensors (%zu trunk + %zu expert mmap, file=%zuMB)\n",
                    loaded, n_trunk, n_expert, (size_t)file_size / (1024*1024));

        // Create the backend scheduler (reused across all gen steps)
        ggml_backend_sched_t sched = nullptr;
        if (gpu_backend && gpu_buft) {
            // GPU + CPU fallback scheduler
            ggml_backend_t backends[2] = { gpu_backend, ggml_backend_cpu_init() };
            ggml_backend_buffer_type_t bufts[2] = { gpu_buft, ggml_backend_cpu_buffer_type() };
            sched = ggml_backend_sched_new(backends, bufts, 2, 16384, false, false);
            fprintf(stderr, "lamio: scheduler: GPU+CPU (2 backends)\n");
        } else {
            // CPU-only scheduler
            ggml_backend_t sched_backends[1] = { backend };
            ggml_backend_buffer_type_t sched_bufts[1] = { ggml_backend_cpu_buffer_type() };
            sched = ggml_backend_sched_new(sched_backends, sched_bufts, 1, 16384, false, false);
            fprintf(stderr, "lamio: scheduler: CPU-only (1 backend)\n");
        }
        if (!sched) {
            std::fprintf(stderr, "lamio: failed to create backend scheduler\n");
            return 1;
        }

        // -------------------------------------------------------------------
        // Persistent state buffers (kept across all generation steps)
        // For GDN layers: conv_state [d_conv-1, conv_channels, 1] + ssm_state [S_v, S_v, H_v, 1]
        // For attn layers: kv_k [head_dim, n_head_kv, max_ctx] + kv_v [head_dim, n_head_kv, max_ctx]
        // -------------------------------------------------------------------
        const int conv_kernel   = hp.ssm_d_conv;
        const int conv_channels = hp.conv_dim();
        const int head_v_dim    = hp.head_v_dim();
        const int num_v_heads   = hp.num_v_heads();
        const int head_dim_attn = hp.head_dim;
        const int n_head_kv     = hp.n_head_kv;
        const int max_ctx       = 4096;

        // Allocate persistent backend buffers
        auto make_persistent = [&](ggml_type type, std::initializer_list<int64_t> ne) -> ggml_tensor * {
            size_t meta_sz = ggml_tensor_overhead() * 2 + 1024;
            void * mb = malloc(meta_sz);
            struct ggml_init_params mp = { meta_sz, mb, true };
            ggml_context * mc = ggml_init(mp);
            ggml_tensor * t;
            if (ne.size() == 3) t = ggml_new_tensor_3d(mc, type, ne.begin()[0], ne.begin()[1], ne.begin()[2]);
            else if (ne.size() == 4) t = ggml_new_tensor_4d(mc, type, ne.begin()[0], ne.begin()[1], ne.begin()[2], ne.begin()[3]);
            else t = ggml_new_tensor_2d(mc, type, ne.begin()[0], ne.begin()[1]);
            // Allocate buffer and bind tensor to it
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(t) + 256);
            ggml_backend_tensor_alloc(b, t, (char *)ggml_backend_buffer_get_base(b));
            ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            return t;
        };

        std::vector<ggml_tensor *> p_conv_state(cfg.n_layers, nullptr);
        std::vector<ggml_tensor *> p_ssm_state(cfg.n_layers, nullptr);
        std::vector<ggml_tensor *> p_kv_k(cfg.n_layers, nullptr);
        std::vector<ggml_tensor *> p_kv_v(cfg.n_layers, nullptr);
        int kv_pos = 0;

        for (int l = 0; l < cfg.n_layers; ++l) {
            bool is_recr = (l + 1) % 4 != 0;
            if (is_recr) {
                p_conv_state[l] = make_persistent(GGML_TYPE_F32, {conv_kernel - 1, conv_channels, 1});
                p_ssm_state[l]  = make_persistent(GGML_TYPE_F32, {head_v_dim, head_v_dim, num_v_heads, 1});
            } else {
                p_kv_k[l] = make_persistent(GGML_TYPE_F16, {head_dim_attn, n_head_kv, max_ctx});
                p_kv_v[l] = make_persistent(GGML_TYPE_F16, {head_dim_attn, n_head_kv, max_ctx});
            }
        }

        // -------------------------------------------------------------------
        // REPL loop: read prompts from stdin, generate for each one.
        // Model stays loaded across prompts. KV cache and GDN state persist.
        // -------------------------------------------------------------------
        if (repl_mode) {
            std::string line;
            std::fprintf(stderr, "repl: ready\n");
            std::fflush(stderr);
            while (std::getline(std::cin, line)) {
                if (line.empty()) continue;
                if (line == "quit" || line == "exit") break;

                // Tokenize the prompt
                auto ids = tok.encode(line);
                if (ids.empty()) {
                    std::printf("[]\n\n");
                    std::fflush(stdout);
                    continue;
                }

                // Append to all_ids for context continuity
                size_t prev_size = all_ids.size();
                all_ids.insert(all_ids.end(), ids.begin(), ids.end());

                int gen_step_start = 0;
                int n_gen = n_gen_tokens > 0 ? n_gen_tokens : 128;

                for (int gen_step = 0; gen_step < n_gen; ++gen_step) {
                    bool is_prefill = (gen_step == 0);
                    int n_tokens = is_prefill ? (int)ids.size() : 1;

                    size_t compute_ctx_size = 256 * 1024 * 1024;
                    void * cbuf = malloc(compute_ctx_size);
                    struct ggml_init_params cparams = { compute_ctx_size, cbuf, true };
                    ggml_context * cctx = ggml_init(cparams);

                    size_t ictx_size = ggml_tensor_overhead() * 64 + 128 * 1024 * 1024;
                    void * ibuf = malloc(ictx_size);
                    struct ggml_init_params iparams = { ictx_size, ibuf, true };
                    ggml_context * ictx = ggml_init(iparams);

                    int32_t pos_base = is_prefill ? (int)prev_size : (int)all_ids.size() - 1;
                    std::vector<int32_t> pos_data(n_tokens);
                    for (int i = 0; i < n_tokens; ++i) pos_data[i] = pos_base + i;

                    ggml_tensor * idx = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, n_tokens);
                    ggml_set_input(idx);
                    ggml_tensor * pos_ids = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, n_tokens);
                    ggml_set_input(pos_ids);
                    ggml_tensor * mask = nullptr;
                    if (is_prefill && n_tokens > 1) {
                        mask = ggml_new_tensor_2d(ictx, GGML_TYPE_F16, n_tokens, n_tokens);
                        ggml_set_input(mask);
                    }

                    std::vector<int32_t> cur_ids(all_ids.end() - n_tokens, all_ids.end());
                    ggml_tensor * x = ggml_get_rows(cctx, embd_w, idx);
                    ggml_set_output(x);
                    ggml_set_input(x);

                    int lim = cfg.n_layers;
                    std::vector<ggml_tensor *> all_zero_init;
                    std::vector<lamio::LayerState> layer_states(lim);
                    size_t graph_size = (size_t)lim * 256 + 2048;
                    ggml_cgraph * gf = ggml_new_graph_custom(cctx, graph_size, false);

                    for (int l = 0; l < lim; ++l) {
                        bool is_recr = (l + 1) % 4 != 0;
                        lamio::Qwen35Forward fwd{cfg, hp, wl, cctx, is_recr};
                        lamio::LayerState & ls = layer_states[l];
                        ls.want_state = (p_conv_state[l] != nullptr);
                        if (is_recr) {
                            if (!is_prefill && p_conv_state[l]) {
                                ggml_tensor * cs_in = ggml_new_tensor_3d(ictx, GGML_TYPE_F32,
                                    conv_kernel - 1, conv_channels, 1);
                                ggml_set_input(cs_in);
                                ls.conv_state_in = cs_in;
                            }
                            if (!is_prefill && p_ssm_state[l]) {
                                ggml_tensor * ss_in = ggml_new_tensor_4d(ictx, GGML_TYPE_F32,
                                    head_v_dim, head_v_dim, num_v_heads, 1);
                                ggml_set_input(ss_in);
                                ls.ssm_state_in = ss_in;
                            }
                        } else {
                            if (!is_prefill && p_kv_k[l] && kv_pos > 0) {
                                ggml_tensor * kk_in = ggml_view_3d(ictx, p_kv_k[l],
                                    head_dim_attn, n_head_kv, kv_pos,
                                    p_kv_k[l]->nb[1], p_kv_k[l]->nb[2], 0);
                                ggml_set_input(kk_in);
                                ls.kv_k_in = kk_in;
                                ggml_tensor * vv_in = ggml_view_3d(ictx, p_kv_v[l],
                                    head_dim_attn, n_head_kv, kv_pos,
                                    p_kv_v[l]->nb[1], p_kv_v[l]->nb[2], 0);
                                ggml_set_input(vv_in);
                                ls.kv_v_in = vv_in;
                            }
                            if (p_kv_k[l]) {
                                int out_pos = is_prefill ? (int)prev_size : kv_pos;
                                ggml_tensor * kk_out = ggml_view_3d(ictx, p_kv_k[l],
                                    head_dim_attn, n_head_kv, out_pos + n_tokens,
                                    p_kv_k[l]->nb[1], p_kv_k[l]->nb[2], 0);
                                ggml_set_output(kk_out);
                                ls.kv_k_out = kk_out;
                                ggml_tensor * vv_out = ggml_view_3d(ictx, p_kv_v[l],
                                    head_dim_attn, n_head_kv, out_pos + n_tokens,
                                    p_kv_v[l]->nb[1], p_kv_v[l]->nb[2], 0);
                                ggml_set_output(vv_out);
                                ls.kv_v_out = vv_out;
                            }
                        }
                        x = fwd.build_layer(x, mt.blocks[l], pos_ids, mask, &ls);
                        ggml_build_forward_expand(gf, x);
                        if (ls.conv_state_out) ggml_build_forward_expand(gf, ls.conv_state_out);
                        if (ls.ssm_state_out)  ggml_build_forward_expand(gf, ls.ssm_state_out);
                        if (ls.kv_k_out)       ggml_build_forward_expand(gf, ls.kv_k_out);
                        if (ls.kv_v_out)       ggml_build_forward_expand(gf, ls.kv_v_out);
                        if (!x) { std::fprintf(stderr, "lamio: forward block %d failed\n", l); break; }
                        for (auto * t : fwd.zero_init) all_zero_init.push_back(t);
                    }

                    ggml_tensor * out_norm_w = wl.load(mt.global.output_norm);
                    if (out_norm_w) {
                        x = ggml_rms_norm(cctx, x, cfg.norm_eps);
                        x = ggml_mul(cctx, x, out_norm_w);
                    }
                    ggml_tensor * output_w = wl.load(mt.global.output_weight);
                    if (!output_w) output_w = embd_w;
                    ggml_tensor * logits = ggml_mul_mat(cctx, output_w, x);
                    ggml_set_output(logits);
                    ggml_build_forward_expand(gf, logits);

                    ggml_backend_sched_reset(sched);
                    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                        std::fprintf(stderr, "lamio: sched alloc failed (step %d)\n", gen_step);
                        ggml_free(cctx); free(cbuf);
                        ggml_free(ictx); free(ibuf);
                        break;
                    }

                    ggml_backend_tensor_set(idx, cur_ids.data(), 0, n_tokens * sizeof(int32_t));
                    ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, n_tokens * sizeof(int32_t));
                    if (mask) {
                        std::vector<uint16_t> mask_data(n_tokens * n_tokens, 0);
                        for (int i = 0; i < n_tokens; ++i)
                            for (int j = 0; j < n_tokens; ++j)
                                mask_data[i * n_tokens + j] = (j <= i) ? 0x0000 : 0xFC00;
                        ggml_backend_tensor_set(mask, mask_data.data(), 0,
                            n_tokens * n_tokens * sizeof(uint16_t));
                    }

                    if (!is_prefill) {
                        for (int l = 0; l < lim; ++l) {
                            bool is_recr = (l + 1) % 4 != 0;
                            auto & ls = layer_states[l];
                            if (is_recr) {
                                if (ls.conv_state_in && p_conv_state[l])
                                    ggml_backend_tensor_copy(p_conv_state[l], ls.conv_state_in);
                                if (ls.ssm_state_in && p_ssm_state[l])
                                    ggml_backend_tensor_copy(p_ssm_state[l], ls.ssm_state_in);
                            }
                        }
                    }

                    for (auto * t : all_zero_init) {
                        if (t && t->buffer) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
                    }

                    ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
                    if (status != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "lamio: compute failed (step %d)\n", gen_step);
                        ggml_free(cctx); free(cbuf);
                        ggml_free(ictx); free(ibuf);
                        break;
                    }

                    {
                        for (int l = 0; l < lim; ++l) {
                            bool is_recr = (l + 1) % 4 != 0;
                            auto & ls = layer_states[l];
                            if (is_recr) {
                                if (ls.conv_state_out && p_conv_state[l])
                                    ggml_backend_tensor_copy(ls.conv_state_out, p_conv_state[l]);
                                if (ls.ssm_state_out && p_ssm_state[l])
                                    ggml_backend_tensor_copy(ls.ssm_state_out, p_ssm_state[l]);
                            }
                        }
                        if (is_prefill) {
                            kv_pos = (int)prev_size + n_tokens;
                        } else {
                            kv_pos++;
                        }
                    }

                    std::vector<float> logits_buf(cfg.vocab_size);
                    ggml_backend_tensor_get(logits, logits_buf.data(),
                        (n_tokens - 1) * cfg.vocab_size * sizeof(float),
                        cfg.vocab_size * sizeof(float));

                    int32_t best_id;
                    float best_val;
                    if (sampler_cfg.temperature == 1.0f && sampler_cfg.top_k == 0 &&
                        sampler_cfg.top_p == 1.0f && sampler_cfg.repeat_penalty == 1.0f) {
                        best_id = 0; best_val = -1e30f;
                        for (int i = 0; i < cfg.vocab_size; ++i) {
                            if (logits_buf[i] > best_val) { best_val = logits_buf[i]; best_id = i; }
                        }
                    } else {
                        best_val = logits_buf[0];
                        best_id = lamio::sample(logits_buf.data(), cfg.vocab_size,
                                               sampler_cfg, all_ids, rng);
                    }

                    std::string piece = tok.decode({best_id});
                    std::printf("%s", piece.c_str());
                    std::fflush(stdout);
                    all_ids.push_back(best_id);

                    ggml_free(ictx); free(ibuf);
                    ggml_free(cctx); free(cbuf);
                }

                std::printf("\n\n");
                std::fflush(stdout);
                std::fprintf(stderr, "repl: ready\n");
                std::fflush(stderr);
            }

            ggml_backend_sched_free(sched);
            ggml_backend_buffer_free(wbackend_buf);
            ggml_free(wctx); free(wbuf);
            ggml_backend_free(backend);
            return 0;
        }

        // --- One-shot generate mode ---
        // Pre-allocate the compute/input scratch buffers ONCE. Re-initializing
        // the ggml contexts on the same malloc'd buffer each step (instead of
        // malloc/free 384MB per token) removes page-table churn and heap
        // fragmentation from the decode hot path.
        void * cbuf = malloc(256 * 1024 * 1024);
        void * ibuf = malloc((size_t)ggml_tensor_overhead() * 64 + 128 * 1024 * 1024);
        ggml_cgraph * gf = nullptr;
        for (int gen_step = 0; gen_step < n_gen_tokens; ++gen_step) {
            bool is_prefill = (gen_step == 0);
            int n_tokens = is_prefill ? (int)all_ids.size() : 1;

            // Proactive eviction BEFORE compute: keep RSS budget tight so the
            // upcoming compute doesn't fault cold pages back in unnecessarily.
            if (!is_prefill && max_rss_mb > 0) {
                expert_router.maybe_evict(gen_step);
            }

            struct ggml_init_params cparams = { 256 * 1024 * 1024, cbuf, true };
            ggml_context * cctx = ggml_init(cparams);
            struct ggml_init_params iparams = { (size_t)ggml_tensor_overhead() * 64 + 128 * 1024 * 1024, ibuf, true };
            ggml_context * ictx = ggml_init(iparams);

            // Position IDs
            int32_t pos_base = is_prefill ? 0 : (int)all_ids.size() - 1;
            std::vector<int32_t> pos_data(n_tokens);
            for (int i = 0; i < n_tokens; ++i) pos_data[i] = pos_base + i;

            // Create input tensors in ictx
            ggml_tensor * idx = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, n_tokens);
            ggml_set_input(idx);
            ggml_tensor * pos_ids = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, n_tokens);
            ggml_set_input(pos_ids);
            // Mask: only needed for prefill (causal). For decode (1 token), no mask needed.
            ggml_tensor * mask = nullptr;
            if (is_prefill) {
                mask = ggml_new_tensor_2d(ictx, GGML_TYPE_F16, n_tokens, n_tokens);
                ggml_set_input(mask);
            }

            // Embedding lookup
            std::vector<int32_t> cur_ids(all_ids.end() - n_tokens, all_ids.end());
            ggml_tensor * x = ggml_get_rows(cctx, embd_w, idx);
            ggml_set_output(x);
            ggml_set_input(x);

            // Forward through all layers
            const int lim = (max_layers < 0 || max_layers > cfg.n_layers) ? cfg.n_layers : max_layers;
            std::vector<ggml_tensor *> all_zero_init;
            std::vector<lamio::LayerState> layer_states(lim);
            std::vector<ggml_tensor *> selected_tensors; // MoE router: top-k indices per layer

            // Create graph early and expand per-layer to avoid stack overflow
            // on deep models (40+ MoE layers cause deep recursion in build_forward_expand)
            // Use custom size: 40 layers x ~100 ops/layer + slack
            size_t graph_size = (size_t)lim * 256 + 2048;
            gf = ggml_new_graph_custom(cctx, graph_size, false);

            for (int l = 0; l < lim; ++l) {
                bool is_recr = (l + 1) % 4 != 0;
                lamio::Qwen35Forward fwd{cfg, hp, wl, cctx, is_recr};
                fwd.want_selected = (max_rss_mb > 0 && cfg.n_experts > 0 && !is_prefill);

                lamio::LayerState & ls = layer_states[l];
                ls.want_state = (p_conv_state[l] != nullptr);  // always extract state for GDN layers
                if (is_recr) {
                    // GDN: create input tensors for decode (prefill uses zeros)
                    if (!is_prefill && p_conv_state[l]) {
                        ggml_tensor * cs_in = ggml_new_tensor_3d(ictx, GGML_TYPE_F32,
                            conv_kernel - 1, conv_channels, 1);
                        ggml_set_input(cs_in);
                        ls.conv_state_in = cs_in;
                    }
                    if (!is_prefill && p_ssm_state[l]) {
                        ggml_tensor * ss_in = ggml_new_tensor_4d(ictx, GGML_TYPE_F32,
                            head_v_dim, head_v_dim, num_v_heads, 1);
                        ggml_set_input(ss_in);
                        ls.ssm_state_in = ss_in;
                    }
                    // Output tensors created as graph results (in cctx) by build_layer_attn_linear
                    // via ggml_cpy. We expand them into the graph and read them post-compute.
                } else {
                    // Attention: KV cache
                    if (!is_prefill && p_kv_k[l] && kv_pos > 0) {
                        ggml_tensor * kk_in = ggml_view_3d(ictx, p_kv_k[l],
                            head_dim_attn, n_head_kv, kv_pos,
                            p_kv_k[l]->nb[1], p_kv_k[l]->nb[2], 0);
                        ggml_set_input(kk_in);
                        ls.kv_k_in = kk_in;
                        ggml_tensor * vv_in = ggml_view_3d(ictx, p_kv_v[l],
                            head_dim_attn, n_head_kv, kv_pos,
                            p_kv_v[l]->nb[1], p_kv_v[l]->nb[2], 0);
                        ggml_set_input(vv_in);
                        ls.kv_v_in = vv_in;
                    }
                    // KV output: write directly into persistent buffer via view
                    if (p_kv_k[l]) {
                        int out_pos = is_prefill ? 0 : kv_pos;
                        ggml_tensor * kk_out = ggml_view_3d(ictx, p_kv_k[l],
                            head_dim_attn, n_head_kv, out_pos + n_tokens,
                            p_kv_k[l]->nb[1], p_kv_k[l]->nb[2], 0);
                        ggml_set_output(kk_out);
                        ls.kv_k_out = kk_out;
                        ggml_tensor * vv_out = ggml_view_3d(ictx, p_kv_v[l],
                            head_dim_attn, n_head_kv, out_pos + n_tokens,
                            p_kv_v[l]->nb[1], p_kv_v[l]->nb[2], 0);
                        ggml_set_output(vv_out);
                        ls.kv_v_out = vv_out;
                    }
                }

                x = fwd.build_layer(x, mt.blocks[l], pos_ids, mask, &ls);

                // Expand graph incrementally per-layer to avoid deep recursion
                ggml_build_forward_expand(gf, x);
                if (ls.conv_state_out) ggml_build_forward_expand(gf, ls.conv_state_out);
                if (ls.ssm_state_out)  ggml_build_forward_expand(gf, ls.ssm_state_out);
                if (ls.kv_k_out)       ggml_build_forward_expand(gf, ls.kv_k_out);
                if (ls.kv_v_out)       ggml_build_forward_expand(gf, ls.kv_v_out);
                // MoE router: expand the selected-experts output so it gets computed
                if (fwd.selected_experts) {
                    ggml_build_forward_expand(gf, fwd.selected_experts);
                    selected_tensors.push_back(fwd.selected_experts);
                } else {
                    selected_tensors.push_back(nullptr);
                }
                if (!x) { std::fprintf(stderr, "lamio: forward block %d failed\n", l); break; }
                for (auto * t : fwd.zero_init) all_zero_init.push_back(t);
            }

            // output norm + output weight -> logits
            ggml_tensor * out_norm_w = wl.load(mt.global.output_norm);
            if (out_norm_w) {
                x = ggml_rms_norm(cctx, x, cfg.norm_eps);
                x = ggml_mul(cctx, x, out_norm_w);
            }
            ggml_tensor * output_w = wl.load(mt.global.output_weight);
            if (!output_w) output_w = embd_w;
            ggml_tensor * logits = ggml_mul_mat(cctx, output_w, x);
            ggml_set_output(logits);

            // Graph was built incrementally per-layer above.
            // Just need to expand the final logits and any remaining state tensors.
            ggml_build_forward_expand(gf, logits);

            // State tensors already expanded per-layer in the loop above

            // Reset scheduler and allocate graph
            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                std::fprintf(stderr, "lamio: sched alloc failed (step %d)\n", gen_step);
                ggml_free(cctx);
                ggml_free(ictx);
                break;
            }

            // Set input data
            ggml_backend_tensor_set(idx, cur_ids.data(), 0, n_tokens * sizeof(int32_t));
            ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, n_tokens * sizeof(int32_t));
            if (mask) {
                std::vector<uint16_t> mask_data(n_tokens * n_tokens, 0);
                for (int i = 0; i < n_tokens; ++i)
                    for (int j = 0; j < n_tokens; ++j)
                        mask_data[i * n_tokens + j] = (j <= i) ? 0x0000 : 0xFC00;
                ggml_backend_tensor_set(mask, mask_data.data(), 0,
                    n_tokens * n_tokens * sizeof(uint16_t));
            }

            // Copy persistent state INTO graph input tensors (decode only)
            if (!is_prefill) {
                for (int l = 0; l < lim; ++l) {
                    bool is_recr = (l + 1) % 4 != 0;
                    auto & ls = layer_states[l];
                    if (is_recr) {
                        if (ls.conv_state_in && p_conv_state[l])
                            ggml_backend_tensor_copy(p_conv_state[l], ls.conv_state_in);
                        if (ls.ssm_state_in && p_ssm_state[l])
                            ggml_backend_tensor_copy(p_ssm_state[l], ls.ssm_state_in);
                    } else {
                        // KV cache views are views into p_kv_k/v, data is already there
                    }
                }
            }

            // Zero-init state/pad tensors (fresh buffers from sched)
            for (auto * t : all_zero_init) {
                if (t && t->buffer) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }

            // Compute
            ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
            if (status != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "lamio: compute failed (step %d)\n", gen_step);
                ggml_free(cctx);
                ggml_free(ictx);
                break;
            }

            // Copy persistent state FROM graph output tensors (for next step)
            {
                for (int l = 0; l < lim; ++l) {
                    bool is_recr = (l + 1) % 4 != 0;
                    auto & ls = layer_states[l];
                    if (is_recr) {
                        // conv_state_out and ssm_state_out are ggml_cpy results in cctx
                        if (ls.conv_state_out && p_conv_state[l])
                            ggml_backend_tensor_copy(ls.conv_state_out, p_conv_state[l]);
                        if (ls.ssm_state_out && p_ssm_state[l])
                            ggml_backend_tensor_copy(ls.ssm_state_out, p_ssm_state[l]);
                    }
                    // KV cache: output views are views into p_kv_k/v, data is already written
                }
                if (is_prefill) {
                    kv_pos = n_tokens;
                } else {
                    kv_pos++;
                }
            }

            // Read logits of last token
            std::vector<float> logits_buf(cfg.vocab_size);
            ggml_backend_tensor_get(logits, logits_buf.data(),
                (n_tokens - 1) * cfg.vocab_size * sizeof(float),
                cfg.vocab_size * sizeof(float));

            // MoE router: read selected experts and drive eviction
            if (max_rss_mb > 0 && cfg.n_experts > 0 && !is_prefill && !selected_tensors.empty()) {
                static uint64_t moe_tick = 0;
                moe_tick++;
                for (int l = 0; l < lim && l < (int)selected_tensors.size(); ++l) {
                    ggml_tensor * sel = selected_tensors[l];
                    if (!sel) continue;
                    int n_act = (int)sel->ne[0];   // n_expert_used
                    int n_tok = (int)sel->ne[1];   // should be 1 for decode
                    std::vector<int32_t> idxs(n_act * n_tok);
                    ggml_backend_tensor_get(sel, idxs.data(), 0, idxs.size() * sizeof(int32_t));
                    // Dedupe by expert_id (selected_experts may repeat)
                    std::vector<int> seen;
                    for (int k = 0; k < n_act * n_tok; ++k) {
                        int expert_id = idxs[k];
                        bool dup = false;
                        for (int s : seen) if (s == expert_id) { dup = true; break; }
                        if (dup) continue;
                        seen.push_back(expert_id);
                        expert_router.touch_expert(l, expert_id, moe_tick);
                    }
                }
                expert_router.maybe_evict(moe_tick);
            }

            // Sample next token
            int32_t best_id;
            float best_val;
            if (sampler_cfg.temperature == 1.0f && sampler_cfg.top_k == 0 &&
                sampler_cfg.top_p == 1.0f && sampler_cfg.repeat_penalty == 1.0f) {
                best_id = 0;
                best_val = -1e30f;
                for (int i = 0; i < cfg.vocab_size; ++i) {
                    if (logits_buf[i] > best_val) { best_val = logits_buf[i]; best_id = i; }
                }
            } else {
                best_val = logits_buf[0];
                best_id = lamio::sample(logits_buf.data(), cfg.vocab_size,
                                       sampler_cfg, all_ids, rng);
            }

            std::printf("[%d] token=%d logit=%.4f\n", gen_step, best_id, best_val);
            // Also print decoded token piece for streaming
            std::string piece = tok.decode({best_id});
            std::printf("piece:%s\n", piece.c_str());
            std::fflush(stdout);
            all_ids.push_back(best_id);

            // Check stop tokens
            bool should_stop = false;
            if (auto_stop && tok.eos_id() >= 0 && best_id == tok.eos_id()) {
                should_stop = true;
            }
            for (int st : stop_tokens) {
                if (best_id == st) { should_stop = true; break; }
            }
            // Also stop on <|im_end|> specifically (the EOS token from GGUF metadata)
            if (auto_stop && tok.eos_id() >= 0 && best_id == tok.eos_id()) {
                should_stop = true;
            }
            if (should_stop) {
                std::fprintf(stderr, "stop token %d reached\n", best_id);
                break;
            }

            ggml_free(ictx);
            ggml_free(cctx);
        }

        // Free the pre-allocated scratch buffers once
        free(ibuf);
        free(cbuf);
        ggml_backend_sched_free(sched);

        // Decode all generated tokens
        std::printf("\ndecoded: ");
        std::string result;
        for (size_t i = 0; i < all_ids.size(); ++i) {
            std::string piece = tok.decode({all_ids[i]});
            result += piece;
        }
        std::printf("%s\n", result.c_str());

        ggml_backend_buffer_free(wbackend_buf);
        ggml_free(wctx);
        free(wbuf);
        ggml_backend_free(backend);
        return 0;
    }

    if (list_mode) {
        lamio::GgufReader r(model_path);
        if (!r.ok()) { std::fprintf(stderr, "lamio: %s\n", r.error().c_str()); return 1; }
        const auto & tensors = r.tensors();
        for (size_t i = 0; i < tensors.size(); ++i) {
            std::printf("%d\t%s\tne=[", (int)i, tensors[i].name.c_str());
            for (int d = 0; d < tensors[i].n_dims; ++d)
                std::printf("%s%lld", d ? " " : "", (long long)tensors[i].ne[d]);
            std::printf("]\ttype=%d\n", tensors[i].ggml_type);
        }
        return 0;
    }

    bool dump_mode = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--dump-meta") == 0) dump_mode = true;
    if (dump_mode) {
        lamio::GgufReader r(model_path);
        if (!r.ok()) { std::fprintf(stderr, "lamio: %s\n", r.error().c_str()); return 1; }
        for (const auto & kv : r.metadata()) {
            std::printf("%s = %s\n", kv.first.c_str(), kv.second.c_str());
        }
        return 0;
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
            if (b.ffn_gate_inp.tensor_idx >= 0) {
                const size_t size = (size_t)(b.ffn_gate_inp.nbytes);
                std::vector<uint8_t> buf(size);
                size_t got = r.load_tensor_data(b.ffn_gate_inp.tensor_idx, buf.data(), buf.size());
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