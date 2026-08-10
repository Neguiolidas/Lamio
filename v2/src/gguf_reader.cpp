#include "gguf_reader.h"

#include <cstdio>
#include <cstring>
#include <sstream>

#include <fcntl.h>
#include <unistd.h>

#include "gguf.h"

namespace lamio {

GgufReader::GgufReader(const std::string & path) : path_(path) {
    static const struct gguf_init_params params = {
        /* no_alloc   = */ true,
        /* ctx        = */ nullptr,
    };
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        ok_ = false;
        error_ = "failed to init gguf from " + path;
        return;
    }

    // Dump all scalar metadata (string keys -> stringized values).
    const int64_t n_kv = gguf_get_n_kv(ctx_);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx_, i);
        if (!key) continue;
        const enum gguf_type t = gguf_get_kv_type(ctx_, i);
        std::string val;
        switch (t) {
            case GGUF_TYPE_STRING: val = gguf_get_val_str(ctx_, i); break;
            case GGUF_TYPE_UINT8:  val = std::to_string(gguf_get_val_u8(ctx_, i)); break;
            case GGUF_TYPE_INT8:   val = std::to_string((int)gguf_get_val_i8(ctx_, i)); break;
            case GGUF_TYPE_UINT16: val = std::to_string(gguf_get_val_u16(ctx_, i)); break;
            case GGUF_TYPE_INT16:  val = std::to_string(gguf_get_val_i16(ctx_, i)); break;
            case GGUF_TYPE_UINT32: val = std::to_string(gguf_get_val_u32(ctx_, i)); break;
            case GGUF_TYPE_INT32:  val = std::to_string(gguf_get_val_i32(ctx_, i)); break;
            case GGUF_TYPE_FLOAT32: val = std::to_string(gguf_get_val_f32(ctx_, i)); break;
            case GGUF_TYPE_BOOL:   val = gguf_get_val_bool(ctx_, i) ? "true" : "false"; break;
            case GGUF_TYPE_UINT64: val = std::to_string(gguf_get_val_u64(ctx_, i)); break;
            case GGUF_TYPE_INT64:  val = std::to_string(gguf_get_val_i64(ctx_, i)); break;
            case GGUF_TYPE_FLOAT64: val = std::to_string(gguf_get_val_f64(ctx_, i)); break;
            case GGUF_TYPE_ARRAY: {
                std::ostringstream os;
                os << "[array]";
                val = os.str();
                break;
            }
            default: val = "[unknown-type]"; break;
        }
        // Keep the original key; some keys are namespaced, but for phase 1 we
        // store the verbatim key so callers can look up "general.architecture".
        metadata_[key] = val;
    }

    // Enumerate tensor metadata seen in the header.
    const int64_t n_tensors = gguf_get_n_tensors(ctx_);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_, i);
        const int64_t * ne = gguf_get_tensor_ne(ctx_, i);
        TensorMeta tm;
        tm.name = name ? name : "";
        tm.nbytes = (int64_t)gguf_get_tensor_size(ctx_, i);
        // GGML_MAX_DIMS = 4 always for ggml. Determine n_dims from ne.
        int dims = 4;
        while (dims > 0 && ne[dims - 1] == 1) --dims;
        tm.n_dims = dims;
        tm.ne.assign(ne, ne + 4);
        tensors_.push_back(tm);
    }

    ok_ = true;
}

GgufReader::~GgufReader() {
    if (ctx_) gguf_free(ctx_);
}

std::string GgufReader::arch() const {
    auto it = metadata_.find("general.architecture");
    if (it != metadata_.end()) return it->second;
    return "";
}

static bool has_suffix(const std::string & n, const char * suffix) {
    const size_t sl = std::strlen(suffix);
    if (n.size() < sl) return false;
    return n.compare(n.size() - sl, sl, suffix) == 0;
}

bool GgufReader::is_moe() const {
    for (const auto & t : tensors_) {
        const std::string & n = t.name;
        if (has_suffix(n, "ffn_up_exps.weight") ||
            has_suffix(n, "ffn_gate_exps.weight") ||
            has_suffix(n, "ffn_down_exps.weight")) {
            return true;
        }
    }
    return false;
}

size_t GgufReader::load_tensor_data(int64_t tensor_idx, void * dst, size_t max_bytes) const {
    if (!ctx_ || dst == nullptr) return 0;
    if (tensor_idx < 0 || tensor_idx >= gguf_get_n_tensors(ctx_)) return 0;

    const size_t tsize = gguf_get_tensor_size(ctx_, tensor_idx);
    const size_t toffset = gguf_get_tensor_offset(ctx_, tensor_idx);
    const size_t data_off = gguf_get_data_offset(ctx_);
    if (tsize > max_bytes) return 0;

    int fd = open(path_.c_str(), O_RDONLY);
    if (fd < 0) return 0;

    const off_t abs_offset = (off_t)(data_off + toffset);
    ssize_t n = pread(fd, dst, tsize, abs_offset);
    close(fd);

    if (n != (ssize_t)tsize) return 0;
    return (size_t)n;
}

bool GgufReader::get_string_array(const std::string & key, std::vector<std::string> & out) const {
    if (!ctx_) return false;
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) return false;
    if (gguf_get_kv_type(ctx_, id) != GGUF_TYPE_ARRAY) return false;
    if (gguf_get_arr_type(ctx_, id) != GGUF_TYPE_STRING) return false;

    out.clear();
    const size_t n = gguf_get_arr_n(ctx_, id);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(gguf_get_arr_str(ctx_, id, i));
    }
    return true;
}

bool GgufReader::get_int32_array(const std::string & key, std::vector<int32_t> & out) const {
    if (!ctx_) return false;
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) return false;
    if (gguf_get_kv_type(ctx_, id) != GGUF_TYPE_ARRAY) return false;
    if (gguf_get_arr_type(ctx_, id) != GGUF_TYPE_INT32) return false;

    out.clear();
    const size_t n = gguf_get_arr_n(ctx_, id);
    out.reserve(n);
    const int32_t * data = static_cast<const int32_t *>(gguf_get_arr_data(ctx_, id));
    for (size_t i = 0; i < n; ++i) out.push_back(data[i]);
    return true;
}

} // namespace lamio