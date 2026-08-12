#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_reader.h"

namespace lamio {

// Byte-level BPE tokenizer read directly from GGUF vocab arrays.
// Self-contained: does not link llama-vocab. Supports GPT2-style pretokenizer
// regex (qwen, ornith, glm use this). decode() reverses the merge.
class BpeTokenizer {
public:
    // Build the vocabulary tables from the GgufReader metadata (tokenizer.ggml.*).
    // Returns false if required keys are missing.
    bool load(const GgufReader & r);

    // Encode text to token ids. add_bos prepends the bos token if configured.
    std::vector<int32_t> encode(const std::string & text, bool add_bos = true) const;

    // Decode token ids back to text (byte-level -> utf-8).
    std::string decode(const std::vector<int32_t> & ids) const;

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    size_t vocab_size() const { return tokens_.size(); }

    // Special token handling: maps special token strings to their IDs
    // so encode() recognises them as single tokens instead of splitting into BPE
    void add_special_token(const std::string & text, int32_t id) {
        special_tokens_[text] = id;
    }
    bool is_special_token(int32_t id) const {
        for (const auto & kv : special_tokens_)
            if (kv.second == id) return true;
        return false;
    }

private:
    std::vector<std::string>         tokens_;    // id -> token
    std::unordered_map<std::string, int32_t> token_map_; // token -> id
    std::unordered_map<std::string, int>  merges_rank_; // "a b" -> priority (0 = highest)
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    std::unordered_map<std::string, int32_t> special_tokens_; // special token -> id
};

} // namespace lamio