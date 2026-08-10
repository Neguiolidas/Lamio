#include "tier_bridge.h"

namespace lamio {

tier_bridge & tier_bridge::instance() {
    static tier_bridge inst;
    return inst;
}

bool tier_bridge::init(const char * gguf_path, size_t ram_budget,
                       int n_layers, int n_expert, int n_expert_used) {
    if (!loader_.open(gguf_path)) {
        return false;
    }

    manager_ = std::make_unique<tier_manager>(ram_budget, n_layers, n_expert);
    if (n_expert_used > 0) {
        manager_->set_n_expert_used(n_expert_used);
    }

    manager_->set_load_callback([](int layer, int eid, int type_idx, void * dest, size_t size) -> bool {
        tier_bridge & self = instance();
        const expert_tensor_info * info = self.find_tensor_info(layer, type_idx);
        if (!info) return false;
        size_t n = self.loader_.read_expert_slice(info->name.c_str(), eid,
                                                   info->expert_bytes,
                                                   dest, size);
        return n > 0;
    });

    // Prefer kernel page-cache prefetch over user-space slots: hints the kernel
    // to cache the expert's pages in the mmap'd model (POSIX_FADV_WILLNEED),
    // reading the correct data directly without duplicating model memory.
    manager_->set_prefetch_callback([](int layer, int eid, int type_idx) {
        tier_bridge & self = instance();
        const expert_tensor_info * info = self.find_tensor_info(layer, type_idx);
        if (!info) return;
        self.loader_.prefetch_expert(info->name.c_str(), eid, info->expert_bytes);
    });

    enabled_ = true;
    return true;
}

void tier_bridge::register_expert_tensor(const char * tensor_name, int eid,
                                          size_t tensor_stride, size_t expert_bytes) {
    // Parse layer number from tensor name "blk.N.ffn_*_exps.weight"
    int layer = -1;
    if (sscanf(tensor_name, "blk.%d.", &layer) != 1) return;

    // Determine type_idx: 0=up, 1=gate, 2=down
    int type_idx = -1;
    if (strstr(tensor_name, "ffn_up_exps"))       type_idx = 0;
    else if (strstr(tensor_name, "ffn_gate_exps")) type_idx = 1;
    else if (strstr(tensor_name, "ffn_down_exps"))type_idx = 2;
    if (type_idx < 0) return;

    int key = layer * 3 + type_idx;
    if (key >= (int)tensor_infos_.size()) {
        tensor_infos_.resize(key + 1);
    }

    tensor_infos_[key] = {tensor_name, expert_bytes, tensor_stride};

    // Lamio FIX: also populate the tier_manager registry, which ensure_slot() consults
    // to know an expert exists and how big it is. Without this, registry stays empty,
    // ensure_slot() always returns registry.end(), and no expert is ever loaded.
    // The file offset is resolved by name via expert_loader in the load callback, so 0.
    if (manager_ && layer >= 0) {
        for (int ex = 0; ex < (int)manager_->get_n_expert(); ex++) {
            manager_->register_expert(layer, ex, type_idx, 0, expert_bytes, 0);
        }
    }
}

void tier_bridge::on_expert_select(int layer, int type_idx, const int * selected_experts, int k) {
    if (!enabled_ || !manager_) return;
    manager_->on_select(layer, type_idx, selected_experts, k);
}

void * tier_bridge::get_expert_data(int layer, int eid, int type_idx) const {
    if (!enabled_ || !manager_) return nullptr;
    return manager_->get_data(layer, eid, type_idx);
}

void tier_bridge::prefetch_layer(int layer, int type_idx,
                                  const int * selected, int k, int next_layer) {
    if (!enabled_) return;
    for (int i = 0; i < k; i++) {
        int eid = selected[i];
        if (eid < 0) continue;
        const expert_tensor_info * info = find_tensor_info(next_layer, type_idx);
        if (!info) continue;
        loader_.prefetch_expert(info->name.c_str(), eid, info->expert_bytes);
    }
}

const tier_bridge::expert_tensor_info * tier_bridge::find_tensor_info(int layer, int type_idx) const {
    int key = layer * 3 + type_idx;
    if (key < 0 || key >= (int)tensor_infos_.size()) return nullptr;
    return &tensor_infos_[key];
}

void * tier_bridge::allocate_expert_buffer(int layer, const char * name, size_t total_size) {
    if (!name || total_size == 0) return nullptr;
    std::string key(name);
    auto it = expert_buffers_.find(key);
    if (it != expert_buffers_.end()) return it->second;

    void * buf = new uint8_t[total_size];
    if (buf) {
        memset(buf, 0, total_size);
        expert_buffers_[key] = buf;
    }
    return buf;
}

void * tier_bridge::get_expert_buffer(int layer, const char * name) {
    if (!name) return nullptr;
    auto it = expert_buffers_.find(std::string(name));
    if (it != expert_buffers_.end()) return it->second;
    return nullptr;
}

} // namespace lamio