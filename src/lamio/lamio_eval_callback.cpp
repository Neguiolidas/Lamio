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
#include <unistd.h>

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

    const size_t expert_stride = weights->nb[2];
    const size_t expert_size   = ggml_nbytes(weights) / weights->ne[2];

    // Phase 2+3: WILLNEED prefetch on selected expert pages.
    // DONTNEED is already done once at load time (phase 2).
    // The kernel readahead pulls in the selected expert pages async.
    for (int i = 0; i < n_selected; i++) {
        int eid = selected[i];
        if (eid < 0) continue;
        void * page = (uint8_t *)weights->data + eid * expert_stride;
        size_t page_size = 4096;
        uintptr_t start = (uintptr_t)page;
        uintptr_t end = start + expert_size;
        start &= ~(page_size - 1);
        end = (end + page_size - 1) & ~(page_size - 1);
        if (end > start) {
            madvise((void *)start, end - start, MADV_WILLNEED);
        }
    }

    return true;
}

} // namespace lamio