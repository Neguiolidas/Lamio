import re

with open('src/llama-model-loader.cpp', 'r') as f:
    content = f.read()

old_block = """        // Lamio: skip expert tensors when tiering is enabled
        if (lamio::tier_bridge::is_expert_tensor(ggml_get_name(cur))) {
            auto & bridge = lamio::tier_bridge::instance();
            if (bridge.is_enabled()) {
                // Register the tensor info for later lazy loading
                size_t expert_bytes = ggml_nbytes(cur) / cur->ne[2]; // one expert slice
                size_t tensor_stride = expert_bytes;
                bridge.register_expert_tensor(ggml_get_name(cur), 0,
                                               tensor_stride, expert_bytes);
                LLAMA_LOG_DEBUG("%s: lamio tiering: skipped expert tensor %s (%zu bytes)\\n",
                                __func__, ggml_get_name(cur), ggml_nbytes(cur));

                // Allocate backend buffer for the tensor (data will be filled by eval callback)
                size_t n_size = ggml_nbytes(cur);
                if (use_mmap && cur->data == nullptr) {
                    const auto & mapping = mappings.at(weight->idx);
                    ggml_backend_buffer_t buf_mmap = nullptr;
                    if (bufs.count(weight->idx)) {
                        buf_mmap = bufs.at(weight->idx);
                    }
                    GGML_ASSERT(buf_mmap);
                    ggml_backend_tensor_alloc(buf_mmap, cur, (uint8_t *) mapping->addr() + weight->offs);
                }

                // Phase 2: DONTNEED on the entire expert tensor after allocation.
                // The mmap pages are loaded during file mapping but we don't need
                // them in RAM until the eval callback selects them. The kernel
                // will