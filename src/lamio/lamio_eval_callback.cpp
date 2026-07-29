#include "lamio_eval_callback.h"
#include "lamio/tier_bridge.h"
#include "lamio/tier_manager.h"

#include "ggml.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sys/mman.h>
#include <cstdint>

namespace lamio {

// Extract layer number from tensor name like "blk.5.ffn_up_exps.weight"
static int extract_layer(const char * name) {
    if (!name) return -1;
    int layer = -1;
    if (sscanf(name, "blk.%d.", &layer) == 1) {
        return layer;
    }
    return -1;
}

// The eval callback: called before (ask=true) and after (ask=false) each graph node.
// On mul_mat_id nodes with tier-managed expert tensors, loads the selected experts.
bool lamio_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    if (!ask) return true; // only act before compute
    if (!t || t->op != GGML_OP_MUL_MAT_ID) return true;

    auto & bridge = tier_bridge::instance();
    if (!bridge.is_enabled()) return true;

    // src[0] = expert weights tensor (up_exps, gate_exps, or down_exps)
    // src[1] = input activation
    // src[2] = selected expert IDs [n_expert_used, n_tokens]
    ggml_tensor * weights = t->src[0];
    ggml_tensor * ids     = t->src[2];

    if (!weights || !ids || !ids->data) return true;

    // Check if this tensor is tier-managed
    if (!bridge.is_expert_tensor(ggml_get_name(weights))) return true;

    int layer = extract_layer(ggml_get_name(weights));
    if (layer < 0) return true;

    // Read selected expert IDs from the ids tensor (I32 type)
    // Layout: [n_expert_used, n_tokens], column-major
    const int n_expert_used = (int)ids->ne[0];
    const int n_tokens      = (int)ids->ne[1];
    const int32_t * ids_data = (const int32_t *)ids->data;

    // Collect unique expert IDs across all tokens
    int selected[256];
    int n_selected = 0;
    for (int tt = 0; tt < n_tokens; tt++) {
        for (int eu = 0; eu < n_expert_used; eu++) {
            int eid = ids_data[tt * n_expert_used + eu];
            if (eid < 0) continue;
            // dedup
            bool found = false;
            for (int i = 0; i < n_selected; i++) {
                if (selected[i] == eid) { found = true; break; }
            }
            if (!found && n_selected < 256) {
                selected[n_selected++] = eid;
            }
        }
    }

    if (n_selected == 0) return true;

    // Determine which expert tensor type this is (up, gate, down)
    int type_idx = -1;
    const char * wname = ggml_get_name(weights);
    if (strstr(wname, "ffn_up_exps"))       type_idx = 0;
    else if (strstr(wname, "ffn_gate_exps")) type_idx = 1;
    else if (strstr(wname, "ffn_down_exps"))type_idx = 2;
    if (type_idx < 0) return true;

    // Async: kick off pread for all selected experts, then wait+memcpy.
    // This overlaps disk I/O with compute from prior layers.
    bridge.on_select_async(layer, selected, n_selected);

    const size_t expert_stride = weights->nb[2];
    const size_t expert_size   = ggml_nbytes(weights) / weights->ne[2];

    for (int i = 0; i < n_selected; i++) {
        int eid = selected[i];

        // Wait for async pread to complete
        bridge.wait_async(layer, eid);

        void * src = bridge.get_expert_data(layer, eid);
        if (src) {
            void * dst = (uint8_t *)weights->data + eid * expert_stride;
            memcpy(dst, src, expert_size);
        }
    }

    // DONTNEED selective: evict mmap pages for non-selected experts
    // to free page cache. Safe because selected experts are now in slots.
    {
        const int n_experts_total = (int)weights->ne[2];
        // Build a quick lookup of selected experts
        bool selected_mask[256] = {};
        for (int i = 0; i < n_selected; i++) {
            if (selected[i] >= 0 && selected[i] < 256) {
                selected_mask[selected[i]] = true;
            }
        }
        // Advise kernel to drop pages for non-selected experts
        for (int e = 0; e < n_experts_total && e < 256; e++) {
            if (!selected_mask[e]) {
                void * page = (uint8_t *)weights->data + e * expert_stride;
                size_t len = expert_size;
                // Align to page boundary
                size_t page_size = 4096;
                uintptr_t start = (uintptr_t)page;
                uintptr_t end = start + len;
                start &= ~(page_size - 1);
                end = (end + page_size - 1) & ~(page_size - 1);
                if (end > start) {
                    madvise((void *)start, end - start, MADV_DONTNEED);
                }
            }
        }
    }

    return true;
}

} // namespace lamio