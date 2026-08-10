#include "dense_forward.h"

#include <cstdio>
#include <cstring>

#include "ggml-alloc.h"
#include "ggml-cpu.h"

namespace lamio {

// Simplified forward: embedding -> rms_norm -> ffn (gate/up/silu/down) -> output.
// Full attention (rope, flash_attn, KV cache) is layered in after smoke validates
// that the matmul pipeline works end-to-end with real quantized weights.
ggml_tensor * DenseForward::forward(const BlockTensors & blk, ggml_tensor * x) {
    ggml_context * ctx = compute_ctx;

    // pre-FFN norm
    ggml_tensor * norm_w = weights.load(blk.ffn_norm);
    if (!norm_w) {
        std::fprintf(stderr, "dense_forward: ffn_norm weight missing\n");
        return nullptr;
    }
    ggml_tensor * h = ggml_rms_norm(ctx, x, cfg.norm_eps);
    h = ggml_mul(ctx, h, norm_w);

    // FFN: gate * silu(gate_w * h) + down_w * (up_w * h)
    // For SwiGLU: down = down_w * (silu(gate_w * h) * (up_w * h))
    ggml_tensor * gate_w = weights.load(blk.ffn_gate);
    ggml_tensor * up_w   = weights.load(blk.ffn_up);
    ggml_tensor * down_w = weights.load(blk.ffn_down);
    if (!gate_w || !up_w || !down_w) {
        std::fprintf(stderr, "dense_forward: ffn weights missing\n");
        return nullptr;
    }

    ggml_tensor * gate_out = ggml_mul_mat(ctx, gate_w, h);
    gate_out = ggml_silu(ctx, gate_out);

    ggml_tensor * up_out = ggml_mul_mat(ctx, up_w, h);

    ggml_tensor * gate_up = ggml_mul(ctx, gate_out, up_out);

    ggml_tensor * down_out = ggml_mul_mat(ctx, down_w, gate_up);

    // residual
    ggml_tensor * result = ggml_add(ctx, x, down_out);

    return result;
}

} // namespace lamio