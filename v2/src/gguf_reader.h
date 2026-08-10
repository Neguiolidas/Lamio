#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace lamio {

// Thin, self-contained wrapper over ggml's GGUF parser enough to enumerate
// metadata and tensor metadata from a .gguf file. Deliberately minimal: full
// tensor-data loading is layered on later (expert loader etc.), this phase
// only validates that we can read a model's shape/architecture from disk.
class GgufReader {
public:
    explicit GgufReader(const std::string & path);
    ~GgufReader();

    bool ok() const { return ok_; }
    std::string error() const { return error_; }

    // Raw key -> value dump of all scalar metadata in the GGUF header.
    const std::unordered_map<std::string, std::string> & metadata() const {
        return metadata_;
    }

    // Human-readable architecture string if present (e.g. "qwen3moe").
    std::string arch() const;

    // Vector of [name, n_dims, ne[0..n_dims]] for every tensor. Names are
    // relative (no "blk.0." wrapper adjustments); used to detect MoE shape.
    struct TensorMeta {
        std::string name;
        int n_dims;
        std::vector<int64_t> ne;
        int64_t nbytes;
    };
    const std::vector<TensorMeta> & tensors() const { return tensors_; }

    // Heuristic MoE detection: true if any tensor name ends with
    // ffn_up_exps.weight / ffn_gate_exps.weight / ffn_down_exps.weight.
    bool is_moe() const;

    // Load raw tensor data from the GGUF file into dst.
    // Returns bytes read, or 0 on error.
    // Uses pread to seek to the tensor's offset without disturbing other reads.
    size_t load_tensor_data(int64_t tensor_idx, void * dst, size_t max_bytes) const;

private:
    bool ok_      = false;
    std::string   error_;
    std::string   path_;
    gguf_context * ctx_ = nullptr;

    std::unordered_map<std::string, std::string> metadata_;
    std::vector<TensorMeta> tensors_;
};

} // namespace lamio