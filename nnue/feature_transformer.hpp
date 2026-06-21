// nnue/feature_transformer.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <random>

namespace NNUE {

// --- Synchronized Architecture Constants ---
#ifndef NNUE_ACCUMULATOR_DIM_DEFINED
#define NNUE_ACCUMULATOR_DIM_DEFINED
constexpr int ACCUMULATOR_DIM = 512;
#endif             
constexpr int FEATURE_DIM = 64 * 10 * 64;           // 40,960 raw HalfKP coordinates

// --- Global Weight Registries ---
struct TransformerWeights {
    // Allocation includes row 0 exclusively for padding, matching your PyTorch configuration!
    // Shape maps to: [40960 + 1][512]
    int16_t embedding_weights[FEATURE_DIM + 1][ACCUMULATOR_DIM];
    int16_t bias[ACCUMULATOR_DIM];
};

inline TransformerWeights transformer_weights;

// --- Runtime Accumulator State Definition ---
struct Accumulator {
    int16_t v[ACCUMULATOR_DIM];

    // Clear and reset the accumulator to its structural baseline bias values
    void reset(const int16_t* bias_source) {
        std::memcpy(v, bias_source, sizeof(v));
    }
};

class FeatureTransformer {
public:
    FeatureTransformer() = default;

    static void initialize_weights_from_scratch(uint32_t seed = 20240620) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 4.0f);

        std::fill(std::begin(transformer_weights.bias), std::end(transformer_weights.bias), 0);
        std::fill(std::begin(transformer_weights.embedding_weights[0]),
                  std::end(transformer_weights.embedding_weights[0]), 0);

        for (int feature = 1; feature <= FEATURE_DIM; ++feature) {
            for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                float sample = std::clamp(dist(rng), -16.0f, 16.0f);
                transformer_weights.embedding_weights[feature][i] = static_cast<int16_t>(std::round(sample));
            }
        }
    }

    // ==========================================================
    // Full Reconstruction (Forward / Single Position Inference / Fallback)
    // ==========================================================
    static void forward(const std::vector<int>& active_features, Accumulator& out_accumulator) {
        // Initialize accumulator tracking layer using baseline bias vector parameters
        out_accumulator.reset(transformer_weights.bias);

        // Map and accumulate active feature embedding coordinates
        for (int raw_feature : active_features) {
            // 🎯 PyTorch Alignment Fix: Shift incoming indices by +1 to skip row 0 padding bounds
            int shifted_index = raw_feature + 1;

            // Boundary safety invariant check
            assert(shifted_index > 0 && shifted_index < (FEATURE_DIM + 1));

            // Vectorized summation block loops
            for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                out_accumulator.v[i] += transformer_weights.embedding_weights[shifted_index][i];
            }
        }
    }

    // ==========================================================
    // Incremental Modifiers (Preserved for future structural updates)
    // ==========================================================

    // Incrementally add features into a running accumulator
    static inline void add_features(Accumulator& accum, const std::vector<int>& added_features) {
        for (int raw_feature : added_features) {
            int shifted_index = raw_feature + 1; 
            assert(shifted_index > 0 && shifted_index < (FEATURE_DIM + 1));

            for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                accum.v[i] += transformer_weights.embedding_weights[shifted_index][i];
            }
        }
    }

    // Incrementally subtract features from a running accumulator
    static inline void remove_features(Accumulator& accum, const std::vector<int>& removed_features) {
        for (int raw_feature : removed_features) {
            int shifted_index = raw_feature + 1; 
            assert(shifted_index > 0 && shifted_index < (FEATURE_DIM + 1));;

            for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                accum.v[i] -= transformer_weights.embedding_weights[shifted_index][i];
            }
        }
    }

    // High-Velocity Compound Update Pass
    static inline void update_accumulator(Accumulator& accum,
                                          const std::vector<int>& removed_features,
                                          const std::vector<int>& added_features) {
        remove_features(accum, removed_features);
        add_features(accum, added_features);
    }
};

} // namespace NNUE
