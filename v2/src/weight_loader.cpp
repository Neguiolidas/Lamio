#include "weight_loader.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml-backend.h"

namespace lamio {

static enum ggml_type to_ggml_type(int gguf_type) {
    return (enum ggml_type)gguf_type;
}

ggml_tensor * WeightLoader::create_tensor(const TensorRef & ref) {
    const int64_t * ne = ref.ne.data();
    if (ref.n_dims < 1 || ref.n_dims > 4) {
        std::fprintf(stderr, "weight_loader: bad n_dims=%d for %s ne=[%lld %lld %lld %lld]\n",
                     ref.n_dims, ref.name.c_str(),
                     (long long)ne[0], (long long)ne[1], (long long)ne[2], (long long)ne[3]);
        return nullptr;
    }
    return ggml_new_tensor(ctx_, to_ggml_type(ref.ggml_type), ref.n_dims, ne);
}

ggml_tensor * WeightLoader::create(const TensorRef & ref) {
    if (ref.tensor_idx < 0) return nullptr;
    auto it = cache_.find(ref.name);
    if (it != cache_.end()) return it->second;
    ggml_tensor * t = create_tensor(ref);
    if (t) { cache_[ref.name] = t; refs_[t] = ref; }
    return t;
}

bool WeightLoader::load_data(ggml_tensor * t, const TensorRef & ref) {
    if (!t || ref.tensor_idx < 0) return false;
    const size_t tsize = (size_t)ref.nbytes;
    std::vector<uint8_t> buf(tsize);
    size_t got = reader_.load_tensor_data(ref.tensor_idx, buf.data(), buf.size());
    if (got != tsize) {
        std::fprintf(stderr, "weight_loader: read %zu != %zu for %s\n",
                     got, tsize, ref.name.c_str());
        return false;
    }
    ggml_backend_tensor_set(t, buf.data(), 0, tsize);
    return true;
}

ggml_tensor * WeightLoader::load(const TensorRef & ref) {
    return create(ref);
}

ggml_tensor * WeightLoader::load_by_name(const std::string & name) {
    auto it = cache_.find(name);
    if (it != cache_.end()) return it->second;

    const auto & tensors = reader_.tensors();
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].name == name) {
            TensorRef ref;
            ref.name = tensors[i].name;
            ref.tensor_idx = (int64_t)i;
            ref.n_dims = tensors[i].n_dims;
            ref.ne = tensors[i].ne;
            ref.nbytes = tensors[i].nbytes;
            ref.ggml_type = tensors[i].ggml_type;
            return load(ref);
        }
    }
    return nullptr;
}

size_t WeightLoader::load_all_data() {
    size_t ok = 0;
    for (auto & kv : refs_) {
        if (load_data(kv.first, kv.second)) ++ok;
    }
    return ok;
}

} // namespace lamio