#include "qwen35_forward.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace lamio {

// ---------------------------------------------------------------------------
// hparams extraction
// ---------------------------------------------------------------------------

Qwen35HParams parse_qwen35_hparams(const ModelConfig & cfg, const GgufReader & r) {
    Qwen35HParams hp;
    const auto & md = r.metadata();

    auto gi = [&](const std::string & key, int fb = 0) -> int {
        auto it = md.find(key);
        if (it == md.end()) return fb;
        return atoi(it->second.c_str());
    };
    auto gf = [&](const std::string & key, float fb = 0.0f) -> float {
        auto it = md.find(key);
        if (it == md.end()) return fb;
        return strtof(it->second.c_str(), nullptr);
    };

    const std::string & a = cfg.arch;

    hp.n_head    = cfg.n_heads;
    hp.n_head_kv = cfg.n_kv_heads;
    hp.head_dim  = cfg.head_dim;
    hp.norm_eps  = cfg.norm_eps;
    hp.freq_base = cfg.rope_freq_base;

    hp.ssm_d_conv  = gi(a + ".ssm.conv_kernel");
    hp.ssm_d_inner = gi(a + ".ssm.inner_size");
    hp.ssm_d_state = gi(a + ".ssm.state_size");
    hp.ssm_dt_rank = gi(a + ".ssm.time_step_rank");
    hp.ssm_n_group = gi(a + ".ssm.group_count");

    hp.rot_dim = gi(a + ".rope.dimension_count");

    // rope dimension sections (MRoPE) — read as int32 array
    std::vector<int32_t> secs;
    r.get_int32_array(a + ".rope.dimension_sections", secs);
    if (secs.size() >= 4) {
        for (int i = 0; i < 4; ++i) hp.rope_sections[i] = secs[i];
    } else {
        // fallback: all rope dims go into first section
        hp.rope_sections[0] = hp.head_dim;
        hp.rope_sections[1] = 0;
        hp.rope_sections[2] = 0;
        hp.rope_sections[3] = 0;
    }

    #ifdef LAMIO_TEST_ROPE
    hp.freq_base = 1.0f;  // DIAG
#endif
    fprintf(stderr, "[qwen35] n_head=%d kv=%d hd=%d conv=%d inner=%d state=%d dt_rank=%d group=%d\n",
            hp.n_head, hp.n_head_kv, hp.head_dim, hp.ssm_d_conv, hp.ssm_d_inner,
            hp.ssm_d_state, hp.ssm_dt_rank, hp.ssm_n_group);
    fprintf(stderr, "[qwen35] rope_sections=[%d %d %d %d] freq_base=%.0f\n",
            hp.rope_sections[0], hp.rope_sections[1], hp.rope_sections[2], hp.rope_sections[3],
            hp.freq_base);

    return hp;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_rms_norm(ctx, x, eps);
    // weight multiplication is done separately since ggml_rms_norm doesn't apply weight
}

static ggml_tensor * rms_norm_weighted(ggml_context * ctx, ggml_tensor * x,
                                        ggml_tensor * w, float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, w);
}

// SwiGLU FFN: gate * silu(up) -> down
static ggml_tensor * build_ffn(ggml_context * ctx, ggml_tensor * cur,
                                ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w) {
    ggml_tensor * gate = ggml_mul_mat(ctx, gate_w, cur);
    ggml_tensor * up   = ggml_mul_mat(ctx, up_w, cur);
    gate = ggml_silu(ctx, gate);
    ggml_tensor * prod = ggml_mul(ctx, gate, up);
    return ggml_mul_mat(ctx, down_w, prod);
}

// ---------------------------------------------------------------------------
// MoE FFN: router -> top-k -> mul_mat_id -> weighted sum + shared expert
// Follows qwen35moe.cpp build_moe_ffn + shared expert
// ---------------------------------------------------------------------------
static ggml_tensor * build_ffn_moe(
    ggml_context * ctx, ggml_tensor * cur,
    // routed experts
    ggml_tensor * gate_inp_w,  // [n_embd, n_expert]
    ggml_tensor * up_exps_w,   // [n_embd, n_ff_exp, n_expert] or [n_ff_exp, n_embd, n_expert]
    ggml_tensor * gate_exps_w, // same shape as up_exps
    ggml_tensor * down_exps_w, // [n_embd, n_ff_exp, n_expert] or inverse
    int n_expert, int n_expert_used,
    // shared expert (optional, can be nullptr)
    ggml_tensor * shexp_gate_w = nullptr,
    ggml_tensor * shexp_up_w   = nullptr,
    ggml_tensor * shexp_down_w = nullptr,
    ggml_tensor * shexp_gate_inp_w = nullptr)
{
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];

    // 1. Router logits: [n_expert, n_tokens]
    ggml_tensor * logits = ggml_mul_mat(ctx, gate_inp_w, cur);

    // 2. Softmax -> probs [n_expert, n_tokens]
    ggml_tensor * probs = ggml_soft_max(ctx, logits);

    // 3. Top-k selection: [n_expert_used, n_tokens] (i32 indices)
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx, probs, n_expert_used);

    // 4. Reshape probs for gather: [1, n_expert, n_tokens]
    probs = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);

    // 5. Gather weights: [1, n_expert_used, n_tokens]
    ggml_tensor * weights = ggml_get_rows(ctx, probs, selected_experts);

    // 5. Normalize weights
    weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
    ggml_tensor * w_sum = ggml_sum_rows(ctx, weights);
    w_sum = ggml_clamp(ctx, w_sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, w_sum);
    weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);

    // 6. cur -> [n_embd, 1, n_tokens] for mul_mat_id
    cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

    // 7. Expert FFN: up = mul_mat_id(up_exps, cur, selected_experts) -> [n_ff, k, tokens]
    ggml_tensor * up = ggml_mul_mat_id(ctx, up_exps_w, cur, selected_experts);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_exps_w, cur, selected_experts);

    // 8. SiLU: gate = silu(gate), prod = gate * up
    gate = ggml_silu(ctx, gate);
    ggml_tensor * prod = ggml_mul(ctx, gate, up);

    // 9. down = mul_mat_id(down_exps, prod, selected_experts) -> [n_embd, k, tokens]
    ggml_tensor * experts = ggml_mul_mat_id(ctx, down_exps_w, prod, selected_experts);

    // 10. Weight + sum experts
    experts = ggml_mul(ctx, experts, weights);

    // View each expert and sum as a balanced binary tree to minimize graph depth
    auto view_expert = [&](int i) -> ggml_tensor * {
        return ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], (size_t)i * experts->nb[1]);
    };
    std::vector<ggml_tensor *> nodes;
    for (int i = 0; i < n_expert_used; ++i)
        nodes.push_back(view_expert(i));
    while (nodes.size() > 1) {
        std::vector<ggml_tensor *> next;
        for (size_t i = 0; i < nodes.size(); i += 2) {
            if (i + 1 < nodes.size())
                next.push_back(ggml_add(ctx, nodes[i], nodes[i+1]));
            else
                next.push_back(nodes[i]);
        }
        nodes = std::move(next);
    }
    ggml_tensor * moe_out = nodes[0];
    if (n_expert_used == 1)
        moe_out = ggml_cont(ctx, moe_out);

    // 11. Shared expert (if present)
    if (shexp_gate_w && shexp_up_w && shexp_down_w) {
        // undo reshape_3d for shared expert input
        ggml_tensor * shexp_in = ggml_reshape_2d(ctx, cur, n_embd, n_tokens);
        ggml_tensor * s_gate = ggml_mul_mat(ctx, shexp_gate_w, shexp_in);
        ggml_tensor * s_up   = ggml_mul_mat(ctx, shexp_up_w, shexp_in);
        s_gate = ggml_silu(ctx, s_gate);
        ggml_tensor * s_prod = ggml_mul(ctx, s_gate, s_up);
        ggml_tensor * s_down = ggml_mul_mat(ctx, shexp_down_w, s_prod);

        // Shared gate: sigmoid(gate_inp_shexp @ cur) -> [1, n_tokens]
        if (shexp_gate_inp_w) {
            ggml_tensor * sgate = ggml_mul_mat(ctx, shexp_gate_inp_w, shexp_in);
            sgate = ggml_sigmoid(ctx, sgate);
            sgate = ggml_reshape_2d(ctx, sgate, 1, n_tokens);
            s_down = ggml_mul(ctx, s_down, sgate);
        }

        moe_out = ggml_add(ctx, moe_out, s_down);
    }

    return moe_out;
}

// ---------------------------------------------------------------------------
// Standard attention layer (full attention, non-recurrent)
// Qwen35 uses joint QG projection: wq outputs [2*n_embd_head*n_head, n_tokens]
// Q = first half, gate = second half
// ---------------------------------------------------------------------------

static ggml_tensor * build_layer_attn(
        ggml_context * ctx,
        ggml_tensor * cur,           // [n_embd, n_tokens]
        const BlockTensors & blk,
        const Qwen35HParams & hp,
        ggml_tensor * pos_ids,       // [n_tokens] I32
        ggml_tensor * mask,          // causal mask or null
        WeightLoader & wl,
        LayerState * state = nullptr) {

    const int n_embd     = hp.head_dim * hp.n_head;  // = n_embd
    const int n_embd_head = hp.head_dim;
    const int n_head      = hp.n_head;
    const int n_head_kv   = hp.n_head_kv;
    const int n_tokens    = (int)cur->ne[1];

    // Joint QG projection
    ggml_tensor * QG = ggml_mul_mat(ctx, wl.load(blk.attn_q), cur);
    // QG shape: [2*n_embd_head*n_head, n_tokens]

    // Q = view first half: [n_embd_head, n_head, n_tokens]
    // layout is interleaved [Q0|G0|Q1|G1|...], stride between heads = 2*head_dim
    const size_t es = ggml_element_size(QG);
    const size_t qg_stride = es * n_embd_head * 2;  // stride between heads (Q + gate interleaved)

    ggml_tensor * Qcur = ggml_view_3d(ctx, QG,
        n_embd_head, n_head, n_tokens,
        qg_stride,                  // nb1: skip Q+gate to next head
        qg_stride * n_head,          // nb2: stride between tokens
        0);                          // offset: start of Q
    // Q norm
    Qcur = rms_norm_weighted(ctx, Qcur, wl.load(blk.attn_q_norm), hp.norm_eps);

    // gate = view second half
    ggml_tensor * gate = ggml_view_3d(ctx, QG,
        n_embd_head, n_head, n_tokens,
        qg_stride,                   // nb1: same stride as Q
        qg_stride * n_head,          // nb2: same token stride
        es * n_embd_head);           // offset: start of gate (skip Q)
    gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);

    // K projection
    ggml_tensor * Kcur = ggml_mul_mat(ctx, wl.load(blk.attn_k), cur);
    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = rms_norm_weighted(ctx, Kcur, wl.load(blk.attn_k_norm), hp.norm_eps);

    // V projection
    ggml_tensor * Vcur = ggml_mul_mat(ctx, wl.load(blk.attn_v), cur);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE to Q and K
    const int n_rot = hp.rot_dim;  // rope.dimension_count, NOT head_dim
    const int rope_type = 2; // NEOX
    const int n_ctx_orig = 0;
    const float freq_scale = 1.0f;
    const float ext_factor = 0.0f;
    const float attn_factor = 1.0f;
    const float beta_fast = 32.0f;
    const float beta_slow = 1.0f;

    int sections[4];
    memcpy(sections, hp.rope_sections, sizeof(sections));

    Qcur = ggml_rope_multi(ctx, Qcur, pos_ids, nullptr,
        n_rot, sections, rope_type, n_ctx_orig, hp.freq_base,
        freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_multi(ctx, Kcur, pos_ids, nullptr,
        n_rot, sections, rope_type, n_ctx_orig, hp.freq_base,
        freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    // Attention via flash_attn_ext
    // flash_attn_ext expects q=[head_dim, n_tokens, n_head, n_seqs] — need permute(0,2,1,3)
    const float kq_scale = 1.0f / sqrtf((float)n_embd_head);
    const float max_alibi = 0.0f;

    // Cast K, V to F16 (flash_attn_ext requires F16 for k, v on CPU)
    Qcur = ggml_permute(ctx, Qcur, 0, 2, 1, 3);  // [head_dim, n_head, n_tokens, 1] -> [head_dim, n_tokens, n_head, 1]
    Kcur = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    Vcur = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

    Kcur = ggml_cast(ctx, Kcur, GGML_TYPE_F16);
    Vcur = ggml_cast(ctx, Vcur, GGML_TYPE_F16);

    // KV cache handling
    ggml_tensor * kq_k = Kcur;
    ggml_tensor * kq_v = Vcur;

    if (state) {
        if (state->kv_k_in && state->kv_v_in) {
            // Decode: concatenate cached K/V with current K/V
            Kcur = ggml_permute(ctx, Kcur, 0, 2, 1, 3);  // [head_dim, n_head_kv, n_tokens, 1]
            Vcur = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

            kq_k = ggml_concat(ctx, state->kv_k_in, Kcur, 2);
            kq_v = ggml_concat(ctx, state->kv_v_in, Vcur, 2);

            // Reshape back for flash_attn_ext
            kq_k = ggml_permute(ctx, kq_k, 0, 2, 1, 3);
            kq_v = ggml_permute(ctx, kq_v, 0, 2, 1, 3);
        }
        // Write K/V to persistent cache via cpy
        if (state->kv_k_out) {
            ggml_tensor * kq_k_3d = ggml_permute(ctx, kq_k, 0, 2, 1, 3);
            kq_k_3d = ggml_cont_3d(ctx, kq_k_3d, n_embd_head, n_head_kv, kq_k_3d->ne[2]);
            state->kv_k_out = ggml_cpy(ctx, kq_k_3d, state->kv_k_out);
            ggml_set_output(state->kv_k_out);
        }
        if (state->kv_v_out) {
            ggml_tensor * kq_v_3d = ggml_permute(ctx, kq_v, 0, 2, 1, 3);
            kq_v_3d = ggml_cont_3d(ctx, kq_v_3d, n_embd_head, n_head_kv, kq_v_3d->ne[2]);
            state->kv_v_out = ggml_cpy(ctx, kq_v_3d, state->kv_v_out);
            ggml_set_output(state->kv_v_out);
        }
    }

    cur = ggml_flash_attn_ext(ctx, Qcur, kq_k, kq_v, mask, kq_scale, max_alibi, false);
    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

    // flash_attn output: [head_dim, n_tokens, n_head, 1] -> reshape to [head_dim*n_head, n_tokens]
    cur = ggml_reshape_2d(ctx, cur, n_embd_head * n_head, n_tokens);
    ggml_tensor * gate_sig = ggml_sigmoid(ctx, gate);
    cur = ggml_mul(ctx, cur, gate_sig);

    // Output projection (Wo @ (attn * gate))
    cur = ggml_mul_mat(ctx, wl.load(blk.attn_output), cur);

    return cur;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GDN (recurrent) layer: conv1d + gated_delta_net + gated norm + output proj
// ---------------------------------------------------------------------------
static ggml_tensor * build_layer_attn_linear(
        ggml_context * ctx,
        ggml_tensor * cur,       // [n_embd, n_tokens] (after attn_norm)
        const BlockTensors & blk,
        const Qwen35HParams & hp,
        WeightLoader & wl,
        std::vector<ggml_tensor *> & zero_init,
        ggml_tensor ** out_gdn = nullptr,
        ggml_tensor ** out_final = nullptr,
        LayerState * state = nullptr) {

    const int64_t d_inner      = hp.ssm_d_inner;
    const int64_t head_k_dim   = hp.ssm_d_state;
    const int64_t num_k_heads  = hp.ssm_n_group;
    const int64_t num_v_heads  = hp.ssm_dt_rank;
    const int64_t head_v_dim   = hp.head_v_dim();
    const int64_t n_tokens     = cur->ne[1];
    const int64_t n_seqs       = 1;

    // --- Input projections ---
    // qkv_mixed = wqkv @ cur  -> [conv_dim, n_tokens]
    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, wl.load(blk.attn_qkv), cur);
    // z = wqkv_gate @ cur -> [value_dim, n_tokens]
    ggml_tensor * z = ggml_mul_mat(ctx, wl.load(blk.attn_gate), cur);

    // --- beta ---
    ggml_tensor * beta = ggml_mul_mat(ctx, wl.load(blk.ssm_beta), cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);

    // --- alpha (softplus) ---
    ggml_tensor * alpha = ggml_mul_mat(ctx, wl.load(blk.ssm_alpha), cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_tokens, n_seqs);
    ggml_tensor * alpha_biased = ggml_add(ctx, alpha, wl.load(blk.ssm_dt));
    ggml_tensor * alpha_sp     = ggml_softplus(ctx, alpha_biased);

    // gate = alpha_softplus * ssm_a
    ggml_tensor * gate = ggml_mul(ctx, alpha_sp, wl.load(blk.ssm_a));
    gate = ggml_reshape_4d(ctx, gate, 1, num_v_heads, n_tokens, n_seqs);

    // --- Conv1d ---
    // conv_dim = key_dim*2 + value_dim
    const int64_t conv_channels = hp.conv_dim();
    const int64_t conv_kernel_size = hp.ssm_d_conv;

    // For batch prefill (no persistent state), we process the full sequence.
    // conv_states are zeros for the first token; we use ggml_ssm_conv directly
    // which does the 1D causal convolution over the sequence dimension.

    // Reshape qkv_mixed for conv: [conv_channels, n_tokens] -> transpose -> [n_tokens, conv_channels]
    ggml_tensor * conv_input = ggml_transpose(ctx, qkv_mixed);

    // ssm_conv1d.weight shape in GGUF: [d_conv, d_inner] = [conv_kernel, conv_channels]
    // ggml_ssm_conv(ctx, conv_input, kernel) -> [conv_channels, n_tokens]
    // The conv_input for ggml_ssm_conv is [n_seq_tokens, n_seqs, conv_channels, 1]
    // But the simplest form: sx=[d_conv-1+n_tokens, conv_channels], c=[d_conv, conv_channels]
    // For full sequence conv without state, we need to prepend zeros.
    // ggml_ssm_conv actually handles this internally when given proper reshapes.

    // Reshape conv_input to 3D: [conv_channels, n_tokens, n_seqs]
    conv_input = ggml_reshape_3d(ctx, qkv_mixed, conv_channels, n_tokens, n_seqs);
    // Transpose to [n_tokens, conv_channels, n_seqs] — actually ggml_ssm_conv expects
    // sx: [d_conv + n_tokens - 1, conv_channels] and c: [d_conv, conv_channels]
    // For simplicity with no persistent state, prepend (d_conv-1) zeros

    // Build padded input: zeros [d_conv-1, conv_channels] concat qkv_mixed [conv_channels, n_tokens]
    // Actually ggml_ssm_conv signature: conv(ctx, sx, c) where sx is the input sequence
    // and c is the conv kernel. Output is {conv_channels, n_seq_tokens, n_seqs}.

    // ggml_ssm_conv expects sx: 3D [d_conv-1+n_tokens, d_inner, n_seqs]
    // c: 2D [d_conv, d_inner]
    // qkv_mixed is [conv_dim, n_tokens]. Need to transpose -> [n_tokens, conv_dim]
    // then pad with (d_conv-1) zeros at front -> [d_conv-1+n_tokens, conv_dim, 1]
    ggml_tensor * conv_kernel = wl.load(blk.ssm_conv1d);

    // Transpose: [conv_dim, n_tokens] -> [n_tokens, conv_dim]
    ggml_tensor * qkv_t = ggml_transpose(ctx, qkv_mixed);

    // Pad with conv_state (from previous step) or zeros (first step)
    ggml_tensor * pad;
    if (state && state->conv_state_in) {
        // Use persistent conv_state: copy into compute context
        pad = ggml_cont(ctx, state->conv_state_in);
        // Do NOT zero-init: this contains the previous step's state
    } else {
        pad = ggml_new_tensor_3d(ctx, qkv_t->type,
                                 conv_kernel_size - 1, conv_channels, n_seqs);
        zero_init.push_back(pad);
    }

    // sx = concat(pad, qkv_t, dim=0) -> [d_conv-1+n_tokens, conv_channels, n_seqs]
    ggml_tensor * sx = ggml_concat(ctx, pad, qkv_t, 0);
    sx = ggml_cont(ctx, sx);  // make contiguous for ssm_conv

    // Extract new conv_state (last conv_kernel-1 tokens of sx) for next step
    if (state && state->want_state) {
        int64_t s_idx = sx->ne[0] - (conv_kernel_size - 1);
        ggml_tensor * new_cs = ggml_view_3d(ctx, sx,
            conv_kernel_size - 1, conv_channels, n_seqs,
            sx->nb[1], sx->nb[2],
            ggml_row_size(sx->type, s_idx));
        ggml_tensor * cs_copy = ggml_cpy(ctx, ggml_cont(ctx, new_cs),
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, conv_kernel_size - 1, conv_channels, n_seqs));
        ggml_set_output(cs_copy);
        state->conv_state_out = cs_copy;
    }

    ggml_tensor * conv_out = ggml_ssm_conv(ctx, sx, conv_kernel);
    conv_out = ggml_silu(ctx, conv_out);

    // --- Split conv output into Q, K, V ---
    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const size_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

    // Q: [head_k_dim, num_k_heads, n_tokens, n_seqs]
    ggml_tensor * q_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv,
        nb1_qkv * n_tokens,
        0);

    // K: [head_k_dim, num_k_heads, n_tokens, n_seqs]
    ggml_tensor * k_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv,
        nb1_qkv * n_tokens,
        head_k_dim * num_k_heads * ggml_element_size(conv_out));

    // V: [head_v_dim, num_v_heads, n_tokens, n_seqs]
    ggml_tensor * v_conv = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_v_dim),
        nb1_qkv,
        nb1_qkv * n_tokens,
        ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

    // L2 normalize Q and K
    q_conv = ggml_l2_norm(ctx, q_conv, hp.norm_eps);
    k_conv = ggml_l2_norm(ctx, k_conv, hp.norm_eps);

    // Repeat Q,K if num_k_heads != num_v_heads (for matching shapes in GDN)
    if (num_k_heads != num_v_heads) {
        // q_conv/k_conv: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim, num_v_heads, ...]
        q_conv = ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
    }

    // --- Gated Delta Net (stateful recurrent op) ---
    // state: [head_v_dim, head_v_dim, num_v_heads, n_seqs] (S_v x S_v x H_v x n_seqs)
    // Use persistent ssm_state from previous step, or zero for prefill
    ggml_tensor * gdn_state;
    if (state && state->ssm_state_in) {
        gdn_state = ggml_cont(ctx, state->ssm_state_in);
        // Do NOT zero-init: contains previous step's state
    } else {
        gdn_state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
            head_v_dim, head_v_dim, num_v_heads, n_seqs);
        zero_init.push_back(gdn_state);
    }

    // ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K=1)
    // K=1: one snapshot slot (minimal). Output is the first n_tokens*n_seqs rows.
    // Result shape: [S_v*H, n_tokens*n_seqs + K*S_v*n_seqs, 1, 1]
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx, q_conv, k_conv, v_conv,
                                                  gate, beta, gdn_state, 1);
    if (out_gdn) *out_gdn = gdn_out;

    // Extract new ssm_state from gdn_out for next step
    if (state && state->want_state) {
        const int64_t state_offset = n_tokens * n_seqs;
        ggml_tensor * new_state = ggml_view_4d(ctx, gdn_out,
            head_v_dim, head_v_dim, num_v_heads, n_seqs,
            ggml_row_size(gdn_out->type, head_v_dim),
            ggml_row_size(gdn_out->type, head_v_dim * head_v_dim),
            ggml_row_size(gdn_out->type, head_v_dim * head_v_dim * num_v_heads),
            ggml_row_size(gdn_out->type, head_v_dim * num_v_heads * state_offset));
        ggml_tensor * ss_copy = ggml_cpy(ctx, ggml_cont(ctx, new_state),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_v_dim, head_v_dim, num_v_heads, n_seqs));
        ggml_set_output(ss_copy);
        state->ssm_state_out = ss_copy;
    }

    // Extract output: [head_v_dim, num_v_heads, n_tokens, n_seqs]
    ggml_tensor * output = ggml_view_4d(ctx, gdn_out,
        head_v_dim, num_v_heads, n_tokens, n_seqs,
        ggml_row_size(gdn_out->type, head_v_dim),
        ggml_row_size(gdn_out->type, head_v_dim * num_v_heads),
        ggml_row_size(gdn_out->type, head_v_dim * num_v_heads * n_tokens),
        0);

    // --- Gated normalization ---
    z = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_tokens, n_seqs);
    ggml_tensor * output_c = ggml_cont_4d(ctx, output, head_v_dim, num_v_heads, n_tokens, n_seqs);
    ggml_tensor * normed = ggml_rms_norm(ctx, output_c, hp.norm_eps);
    normed = ggml_mul(ctx, normed, wl.load(blk.ssm_norm));
    ggml_tensor * z_silu = ggml_silu(ctx, z);
    ggml_tensor * gated  = ggml_mul(ctx, normed, z_silu);

    // Reshape: [head_v_dim * num_v_heads, n_tokens] = [value_dim, n_tokens]
    ggml_tensor * final_output = ggml_reshape_2d(ctx, gated, head_v_dim * num_v_heads, n_tokens);
    if (out_final) *out_final = final_output;

    // Output projection: ssm_out @ final_output
    cur = ggml_mul_mat(ctx, wl.load(blk.ssm_out), final_output);

    // Reshape back: [n_embd, n_tokens]
    cur = ggml_reshape_2d(ctx, cur, cur->ne[0], n_tokens);

    return cur;
}

// ---------------------------------------------------------------------------
// Full layer: norm -> attention -> residual -> norm -> FFN -> residual
// ---------------------------------------------------------------------------

ggml_tensor * Qwen35Forward::build_layer(
        ggml_tensor * cur,
        const BlockTensors & blk,
        ggml_tensor * pos_ids,
        ggml_tensor * mask,
        LayerState * state) {

    const int n_tokens = (int)cur->ne[1];

    // Pre-attention residual
    ggml_tensor * inpSA = cur;

    // attn_norm
    cur = rms_norm_weighted(ctx, cur, weights.load(blk.attn_norm), hp.norm_eps);
    attn_norm_out = cur;

    // Dispatch attention type
    if (is_recr) {
        cur = build_layer_attn_linear(ctx, cur, blk, hp, weights, zero_init, &gdn_output, &final_output, state);
    } else {
        cur = build_layer_attn(ctx, cur, blk, hp, pos_ids, mask, weights, state);
    }

    // Attention residual
    cur = ggml_add(ctx, cur, inpSA);

    // Save for FFN residual
    ggml_tensor * ffn_residual = cur;

    // Post-attention norm
    ggml_tensor * post_norm_w = weights.load(blk.attn_post_norm);
    if (post_norm_w)
        cur = rms_norm_weighted(ctx, cur, post_norm_w, hp.norm_eps);
    else
        cur = rms_norm_weighted(ctx, cur, weights.load(blk.ffn_norm), hp.norm_eps);

    // FFN
    if (blk.has_moe && cfg.n_experts > 0) {
        // MoE FFN
        cur = build_ffn_moe(ctx, cur,
            weights.load(blk.ffn_gate_inp),
            weights.load(blk.ffn_up_exps),
            weights.load(blk.ffn_gate_exps),
            weights.load(blk.ffn_down_exps),
            cfg.n_experts, cfg.n_active,
            weights.load(blk.ffn_gate_shexp),
            weights.load(blk.ffn_up_shexp),
            weights.load(blk.ffn_down_shexp),
            weights.load(blk.ffn_gate_inp_shexp));
    } else {
        cur = build_ffn(ctx, cur,
            weights.load(blk.ffn_gate), weights.load(blk.ffn_up), weights.load(blk.ffn_down));
    }

    // FFN residual
    cur = ggml_add(ctx, cur, ffn_residual);

    return cur;
}

} // namespace lamio