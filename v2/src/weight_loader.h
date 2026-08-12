#pragma once

#include <string>
#include <unordered_map>

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf_reader.h"
#include "block_tensors.h"
#include "model_config.h"

namespace lamio {

// Loads tensor weights from the GGUF file into ggml tensors allocated on
// a ggml backend. Tensors are cached by name for repeated access.
class WeightLoader {
public:
    WeightLoader(ggml_backend_t backend, ggml_context * ctx, const GgufReader & r)
        : backend_(backend), ctx_(ctx), reader_(r) {}

    // Create a tensor in the context (metadata only, no data).
    // Returns nullptr on failure. Does NOT load data.
    ggml_tensor * create(const TensorRef & ref);

    // Load raw data into an already-allocated tensor.
    // Returns false on failure.
    bool load_data(ggml_tensor * t, const TensorRef & ref);

    // Load a tensor by its TensorRef. Creates tensor, caches it.
    // Data must be loaded separately after backend alloc.
    ggml_tensor * load(const TensorRef & ref);

    // Convenience: load by name (scans reader tensors).
    ggml_tensor * load_by_name(const std::string & name);

    // Load data for all cached tensors (call after backend alloc).
    // Returns the number of tensors successfully loaded.
    size_t load_all_data();

    // Access the tensor->TensorRef map for external loading (e.g. mmap).
    const std::unordered_map<ggml_tensor *, TensorRef> & refs_map() const { return refs_; }

private:
    ggml_backend_t backend_;
    ggml_context * ctx_;
    const GgufReader & reader_;
    std::unordered_map<std::string, ggml_tensor *> cache_;
    std::unordered_map<ggml_tensor *, TensorRef> refs_;  // for deferred data loading

    ggml_tensor * create_tensor(const TensorRef & ref);
};

} // namespace lamio