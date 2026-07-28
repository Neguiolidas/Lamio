#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>

namespace lamio {

// Reads expert tensor slices directly from a GGUF file via pread.
// During open(), parses GGUF metadata and caches tensor name -> (offset, size).
class expert_loader {
public:
    expert_loader();
    ~expert_loader();

    bool open(const char * gguf_path);
    void close();
    bool is_open() const { return fd_ >= 0; }

    // Read a single expert slice from a 3D expert tensor.
    // tensor_name: e.g. "blk.0.ffn_up_exps.weight"
    // eid: expert index (0..n_expert-1)
    // expert_bytes: size of one expert slice in bytes
    // dest: buffer of at least expert_bytes
    // Returns bytes read, or 0 on failure.
    size_t read_expert_slice(const char * tensor_name, int eid,
                             size_t expert_bytes,
                             void * dest, size_t dest_size);

    // Prefetch (readahead) a single expert slice. Non-blocking.
    void prefetch_expert(const char * tensor_name, int eid,
                         size_t expert_bytes);

    size_t file_size() const { return file_size_; }

private:
    int fd_ = -1;
    size_t file_size_ = 0;
    std::string gguf_path_;
    int64_t data_offset_ = 0;  // start of tensor data section in file

    struct tensor_meta {
        int64_t data_offset;  // absolute offset in file (data_offset + gguf_offset)
        size_t  size;         // total tensor size in bytes
    };
    std::unordered_map<std::string, tensor_meta> tensor_cache_;

    bool parse_gguf_metadata();
};

} // namespace lamio