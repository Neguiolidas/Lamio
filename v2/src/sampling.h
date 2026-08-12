#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace lamio {

// Sampling parameters
struct SamplerConfig {
    float temperature = 1.0f;   // 1.0 = no scaling, <1.0 = sharper, >1.0 = flatter
    int   top_k       = 0;       // 0 = disabled, >0 = keep only top-k tokens
    float top_p       = 1.0f;    // 1.0 = disabled, <1.0 = nucleus sampling
    float repeat_penalty = 1.0f; // 1.0 = disabled, >1.0 = penalize repeats
};

// Apply repeat penalty to recently used token ids
// logits: [n_vocab] float array (modified in-place)
// recent_tokens: token ids that appeared in the context so far
static inline void apply_repeat_penalty(float * logits, int n_vocab,
                                         const std::vector<int> & recent_tokens,
                                         float penalty) {
    if (penalty == 1.0f) return;
    for (int tok : recent_tokens) {
        if (tok >= 0 && tok < n_vocab) {
            if (logits[tok] > 0.0f) {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

// Apply temperature scaling
static inline void apply_temperature(float * logits, int n_vocab, float temp) {
    if (temp <= 0.0f || temp == 1.0f) return;
    for (int i = 0; i < n_vocab; ++i) {
        logits[i] /= temp;
    }
}

// Sort indices by logit descending. Returns indices.
static inline std::vector<int> argsort_desc(const float * logits, int n_vocab) {
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });
    return idx;
}

// Apply top-k filter: keep only top-k tokens, set rest to -INFINITY
static inline void apply_top_k(float * logits, int n_vocab, int k) {
    if (k <= 0 || k >= n_vocab) return;
    auto sorted = argsort_desc(logits, n_vocab);
    for (int i = k; i < n_vocab; ++i) {
        logits[sorted[i]] = -INFINITY;
    }
}

// Apply top-p (nucleus) filter: keep smallest set of tokens whose
// cumulative probability >= top_p, set rest to -INFINITY
static inline void apply_top_p(float * logits, int n_vocab, float p) {
    if (p >= 1.0f) return;
    auto sorted = argsort_desc(logits, n_vocab);

    // Softmax to get probabilities
    float max_logit = logits[sorted[0]];
    float sum = 0.0f;
    std::vector<float> probs(n_vocab, 0.0f);
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < n_vocab; ++i) probs[i] /= sum;

    // Accumulate until we reach top_p
    float cum = 0.0f;
    int cutoff = n_vocab;
    for (int i = 0; i < n_vocab; ++i) {
        cum += probs[sorted[i]];
        if (cum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out everything after the cutoff
    for (int i = cutoff; i < n_vocab; ++i) {
        logits[sorted[i]] = -INFINITY;
    }
}

// Sample from logits using the configured sampler.
// Returns a token id.
// rng: caller-provided RNG for reproducibility
static inline int sample(float * logits, int n_vocab,
                         const SamplerConfig & cfg,
                         const std::vector<int> & recent_tokens,
                         std::mt19937 & rng) {
    // Step 1: repeat penalty
    apply_repeat_penalty(logits, n_vocab, recent_tokens, cfg.repeat_penalty);

    // Step 2: temperature
    apply_temperature(logits, n_vocab, cfg.temperature);

    // Step 3: top-k
    apply_top_k(logits, n_vocab, cfg.top_k);

    // Step 4: top-p
    apply_top_p(logits, n_vocab, cfg.top_p);

    // Step 5: softmax + sample
    float max_logit = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    std::vector<float> probs(n_vocab, 0.0f);
    float sum = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        if (logits[i] > -INFINITY) {
            probs[i] = expf(logits[i] - max_logit);
            sum += probs[i];
        }
    }
    if (sum <= 0.0f) {
        // Fallback to argmax if everything got filtered
        int best = 0;
        float best_logit = logits[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best = i;
            }
        }
        return best;
    }
    for (int i = 0; i < n_vocab; ++i) probs[i] /= sum;

    // Greedy (temperature <= 0 or temperature == 0)
    if (cfg.temperature <= 0.0f) {
        int best = 0;
        float best_prob = probs[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (probs[i] > best_prob) {
                best_prob = probs[i];
                best = i;
            }
        }
        return best;
    }

    // Stochastic sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return n_vocab - 1; // last resort
}

} // namespace lamio
