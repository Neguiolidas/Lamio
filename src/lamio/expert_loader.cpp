#include "expert_loader.h"
#include "gguf.h"
#include "ggml.h"

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdio>

namespace lamio {

expert_loader::expert_loader() = default;

expert_loader::~expert_loader() {
    close();
}

bool expert_loader::open(const char * gguf_path) {
    gguf_path_ = gguf_path;

    fd_ = ::open(gguf_path, O_RDONLY);
    if (fd_ < 0) {
        fprintf(stderr, "lamio: failed to open %s: %s\n", gguf_path, strerror(errno));
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) != 0) {
        fprintf(stderr, "lamio: fstat failed: %s\n", strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    file_size_ = (size_t)st.st_size;

    if (!parse_gguf_metadata()) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    return true;
}

bool expert_loader::parse_gguf_metadata() {
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(gguf_path_.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "lamio: gguf_init_from_file failed\n");
        return false;
    }

    data_offset_ = (int64_t)gguf_get_data_offset(ctx);
    int64_t n_tensors = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        size_t size = gguf_get_tensor_size(ctx, i);

        tensor_meta meta;
        meta.data_offset = data_offset_ + (int64_t)offset;
        meta.size = size;
        tensor_cache_[name] = meta;
    }

    fprintf(stderr, "lamio: cached %lld tensor offsets (data_offset=%lld)\n",
            (long long)n_tensors, (long long)data_offset_);

    gguf_free(ctx);
    return true;
}

void expert_loader::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    file_size_ = 0;
    data_offset_ = 0;
    tensor_cache_.clear();
    gguf_path_.clear();
}

size_t expert_loader::read_expert_slice(const char * tensor_name, int eid,
                                         size_t expert_bytes,
                                         void * dest, size_t dest_size) {
    if (fd_ < 0) return 0;
    if (dest_size < expert_bytes) return 0;

    auto it = tensor_cache_.find(tensor_name);
    if (it == tensor_cache_.end()) return 0;

    // Each expert is a contiguous slice of the 3D tensor.
    // Expert i starts at: tensor_data_offset + eid * expert_bytes
    // (experts are stored contiguously along the last dimension in GGUF)
    off_t abs_offset = (off_t)(it->second.data_offset + (int64_t)eid * (int64_t)expert_bytes);

    ssize_t n = pread(fd_, dest, expert_bytes, abs_offset);
    if (n != (ssize_t)expert_bytes) {
        fprintf(stderr, "lamio: pread failed for %s eid=%d: %s (got %zd want %zu)\n",
                tensor_name, eid, strerror(errno), (ssize_t)n, expert_bytes);
        return 0;
    }

    return (size_t)n;
}

void expert_loader::prefetch_expert(const char * tensor_name, int eid,
                                     size_t expert_bytes) {
    if (fd_ < 0) return;

    auto it = tensor_cache_.find(tensor_name);
    if (it == tensor_cache_.end()) return;

    off_t abs_offset = (off_t)(it->second.data_offset + (int64_t)eid * (int64_t)expert_bytes);

#ifdef __linux__
    posix_fadvise(fd_, abs_offset, (off_t)expert_bytes, POSIX_FADV_WILLNEED);
#else
    (void)abs_offset;
#endif
}

} // namespace lamio