#include "tokenizer.h"

#include <algorithm>
#include <cstring>
#include <cstdio>

namespace lamio {

static bool is_ascii_word_char(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9');
}

// GPT-2 style pretokenizer: a leading space(s) run is fused to the following
// word token (the vocab stores space-prefixed tokens like "\xc4\xa0word").
// Trailing spaces at end of text become their own space token.
static std::vector<std::string> split_words(const std::string & text) {
    std::vector<std::string> out;
    std::string cur;
    unsigned cur_kind = 0; // 0=none, 1=word, 2=space, 3=punct

    auto flush = [&]() {
        if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    };

    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = (unsigned char)text[i];
        unsigned kind;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') kind = 2;
        else if (is_ascii_word_char(c) || c >= 0x80) kind = 1;
        else kind = 3;

        if (cur_kind == 2 && kind == 1) {
            // space run fuses into the following word token: replace ASCII space
            // with the byte-level prefix "\xc4\xa0" (" " in gpt2 vocab).
            cur.clear();
            cur.push_back((char)0xC4);
            cur.push_back((char)0xA0);
            cur_kind = 1;
            cur.push_back((char)c);
            continue;
        }
        if (cur_kind != 0 && cur_kind != kind) flush();
        cur_kind = kind;
        cur.push_back((char)c);
    }
    flush();
    return out;
}

bool BpeTokenizer::load(const GgufReader & r) {
    std::vector<std::string> merges_raw;
    if (!r.get_string_array("tokenizer.ggml.tokens", tokens_)) return false;
    if (!r.get_string_array("tokenizer.ggml.merges", merges_raw)) return false;

    for (size_t i = 0; i < tokens_.size(); ++i) {
        token_map_[tokens_[i]] = (int32_t)i;
    }

    // Build merge rank table: "a b" -> priority. Lower index = higher priority.
    std::unordered_map<std::string, int> rank_map;
    for (size_t i = 0; i < merges_raw.size(); ++i) {
        rank_map[merges_raw[i]] = (int)i;
    }

    // Build token -> byte representation for pretokenize-level lookup.
    // For byte-level BPE, tokens are already the actual byte strings/symbols.

    bos_id_ = -1;
    merges_rank_ = std::move(rank_map);

    // Auto-detect special tokens: any token containing <| or matching known
    // special tokens gets registered so encode() treats it as a single ID
    int32_t eos_id = -1;
    for (size_t i = 0; i < tokens_.size(); ++i) {
        const std::string & tok = tokens_[i];
        if (tok.find("<|") != std::string::npos) {
            add_special_token(tok, (int32_t)i);
        }
        // Also register known Qwen/Ornith special tokens
        if (tok == "" || tok == "" || tok == "" || tok == "" ||
            tok == "" || tok == "" || tok == "" || tok == "" ||
            tok == "" || tok == "" || tok == "" || tok == "" ||
            tok == "" || tok == "" || tok == "" || tok == "" ||
            tok == "" || tok == "" || tok == "" || tok == "" ||
            tok == "" || tok == "" || tok == "" || tok == "") {
            add_special_token(tok, (int32_t)i);
        }
    }

    // Read EOS token ID from metadata
    int32_t meta_eos = -1;
    if (r.get_int32("tokenizer.ggml.eos_token_id", meta_eos)) {
        eos_id_ = meta_eos;
    }

    return true;
}

// Encode: standard BPE with rank table (greedy, O(n^2) worst case, fine for smoke).
std::vector<int32_t> BpeTokenizer::encode(const std::string & text, bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos && bos_id_ >= 0) ids.push_back(bos_id_);

    // First pass: find all special tokens in the text and split around them.
    // Special tokens are encoded as single IDs, not BPE-split.
    size_t i = 0;
    while (i < text.size()) {
        // Check if any special token matches at position i
        bool found_special = false;
        // Try longest match first: sort by length descending
        // (simple linear scan since there are <30 special tokens)
        size_t best_len = 0;
        int32_t best_id = -1;
        for (const auto & kv : special_tokens_) {
            const std::string & st = kv.first;
            if (i + st.size() <= text.size() && text.compare(i, st.size(), st) == 0) {
                if (st.size() > best_len) {
                    best_len = st.size();
                    best_id = kv.second;
                }
            }
        }
        if (best_id >= 0) {
            ids.push_back(best_id);
            i += best_len;
            found_special = true;
        }
        if (found_special) continue;

        // Normal BPE: accumulate until next special token boundary
        // Find the next special token position
        size_t end = text.size();
        for (const auto & kv : special_tokens_) {
            size_t pos = text.find(kv.first, i + 1);
            if (pos != std::string::npos && pos < end) end = pos;
        }
        std::string segment = text.substr(i, end - i);

        // BPE encode the segment
        for (const auto & word : split_words(segment)) {
            std::vector<int32_t> w;
            auto try_sub = [&](const std::string & s) -> int32_t {
                auto it = token_map_.find(s);
                return it == token_map_.end() ? -1 : it->second;
            };

            size_t j = 0;
            while (j < word.size()) {
                int best_slen = -1; int32_t best_sid = -1;
                for (size_t len = 1; j + len <= word.size(); ++len) {
                    int32_t id = try_sub(word.substr(j, len));
                    if (id >= 0) { best_slen = (int)len; best_sid = id; }
                }
                if (best_slen > 0) {
                    w.push_back(best_sid);
                    j += best_slen;
                } else {
                    char b[2] = { word[j], 0 };
                    int32_t id = try_sub(b);
                    if (id >= 0) w.push_back(id);
                    ++j;
                }
            }

            bool changed = true;
            while (changed && w.size() > 1) {
                changed = false;
                int best_rank = INT32_MAX; size_t best_pos = SIZE_MAX;
                for (size_t k = 0; k + 1 < w.size(); ++k) {
                    if (w[k] < 0 || w[k+1] < 0) continue;
                    std::string pair = tokens_[w[k]] + " " + tokens_[w[k+1]];
                    auto it = merges_rank_.find(pair);
                    if (it != merges_rank_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_pos = k;
                    }
                }
                if (best_pos != SIZE_MAX) {
                    std::string merged = tokens_[w[best_pos]] + tokens_[w[best_pos+1]];
                    auto it = token_map_.find(merged);
                    if (it == token_map_.end()) {
                        best_pos = SIZE_MAX; break;
                    }
                    w[best_pos] = it->second;
                    w.erase(w.begin() + best_pos + 1);
                    changed = true;
                }
            }

            for (int32_t id : w) {
                if (id >= 0) ids.push_back(id);
            }
        }
        i = end;
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || id >= (int32_t)tokens_.size()) continue;
        const std::string & tok = tokens_[id];
        for (size_t i = 0; i < tok.size(); ++i) {
            // map "\xc4\xa0" (gpt2 byte-space) back to ASpace ' '
            if (tok[i] == (char)0xC4 && i + 1 < tok.size() && tok[i+1] == (char)0xA0) {
                out.push_back(' ');
                ++i;
            } else {
                out.push_back(tok[i]);
            }
        }
    }
    return out;
}

} // namespace lamio