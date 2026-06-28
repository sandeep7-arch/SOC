#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <limits>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace NNUE {

// --- Synchronized Architecture Dimensions ---
#ifndef NNUE_ACCUMULATOR_DIM_DEFINED
#define NNUE_ACCUMULATOR_DIM_DEFINED
constexpr int ACCUMULATOR_DIM = 512;
#endif
constexpr int INPUT_DIM = ACCUMULATOR_DIM * 2; // 1024
constexpr int HIDDEN1_DIM = 256;
constexpr int HIDDEN2_DIM = 32;
constexpr int OUTPUT_DIM = 1;
constexpr float MAX_CENTIPAWN_SCORE = 32000.0f;

// --- Flat Floating Point Weight Registry Maps ---
struct ModelParameters {
    // Hidden Layer 1 (1024 -> 256)
    float h1_weights[HIDDEN1_DIM][INPUT_DIM];
    float h1_bias[HIDDEN1_DIM];

    // Hidden Layer 2 (256 -> 32)
    float h2_weights[HIDDEN2_DIM][HIDDEN1_DIM];
    float h2_bias[HIDDEN2_DIM];

    // Output Layer (32 -> 1)
    float out_weights[OUTPUT_DIM][HIDDEN2_DIM];
    float out_bias[OUTPUT_DIM];
};

inline ModelParameters model_params;

struct QuantizedModelParameters {
    int16_t h1_weights[HIDDEN1_DIM][INPUT_DIM];
    float h1_scales[HIDDEN1_DIM];
    bool ready = false;
    bool enabled = true;
};

inline QuantizedModelParameters quantized_model_params;

class NNUEModel {
private:
    // --- Clipped ReLU Activation Primitives ---
    static inline float clipped_relu(float x) {
        return std::clamp<float>(x, 0.0f, 128.0f);
    }

    static inline float dot_i16_f32(const int16_t* input, const float* weights, int count) {
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= count; i += 8) {
            __m128i raw_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
            __m256 input_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(raw_i16));
            __m256 weight_ps = _mm256_loadu_ps(weights + i);
#if defined(__FMA__)
            acc = _mm256_fmadd_ps(input_ps, weight_ps, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(input_ps, weight_ps));
#endif
        }

        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, acc);
        float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3]
                  + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        for (; i < count; ++i) {
            sum += static_cast<float>(input[i]) * weights[i];
        }
        return sum;
#else
        float sum = 0.0f;
        for (int i = 0; i < count; ++i) {
            sum += static_cast<float>(input[i]) * weights[i];
        }
        return sum;
#endif
    }

    static inline int32_t dot_i16_i16(const int16_t* input, const int16_t* weights, int count) {
#if defined(__AVX2__)
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        for (; i + 16 <= count; i += 16) {
            __m256i input_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
            __m256i weight_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i));
            __m256i products = _mm256_madd_epi16(input_i16, weight_i16);
            acc = _mm256_add_epi32(acc, products);
        }

        alignas(32) int32_t lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
        int64_t sum = static_cast<int64_t>(lanes[0]) + lanes[1] + lanes[2] + lanes[3]
                    + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        for (; i < count; ++i) {
            sum += static_cast<int32_t>(input[i]) * static_cast<int32_t>(weights[i]);
        }
        return static_cast<int32_t>(std::clamp<int64_t>(
            sum,
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max()
        ));
#else
        int64_t sum = 0;
        for (int i = 0; i < count; ++i) {
            sum += static_cast<int32_t>(input[i]) * static_cast<int32_t>(weights[i]);
        }
        return static_cast<int32_t>(std::clamp<int64_t>(
            sum,
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max()
        ));
#endif
    }

public:
    NNUEModel() = default;

    // ==========================================================
    // Core Forward Inference Pass
    // ==========================================================
    static float forward(const float* oriented_input) {
        float layer1_out[HIDDEN1_DIM];
        float layer2_out[HIDDEN2_DIM];

        // 1. Process Hidden Layer 1 (1024 -> 256) + Clipped ReLU
        for (int i = 0; i < HIDDEN1_DIM; ++i) {
            float sum = model_params.h1_bias[i];
            for (int j = 0; j < INPUT_DIM; ++j) {
                sum += oriented_input[j] * model_params.h1_weights[i][j];
            }
            layer1_out[i] = clipped_relu(sum);
        }

        // 2. Process Hidden Layer 2 (256 -> 32) + Clipped ReLU
        for (int i = 0; i < HIDDEN2_DIM; ++i) {
            float sum = model_params.h2_bias[i];
            for (int j = 0; j < HIDDEN1_DIM; ++j) {
                sum += layer1_out[j] * model_params.h2_weights[i][j];
            }
            layer2_out[i] = clipped_relu(sum);
        }

        // 3. Process Output Layer (32 -> 1)
        float output_score = model_params.out_bias[0];
        for (int j = 0; j < HIDDEN2_DIM; ++j) {
            output_score += layer2_out[j] * model_params.out_weights[0][j];
        }

        return output_score;
    }

    // ==========================================================
    // Perspective Realignment Layer
    // ==========================================================
    static float evaluate_perspective(const int16_t* white_acc, const int16_t* black_acc, bool is_white_to_move) {
        const int16_t* stm_acc = is_white_to_move ? white_acc : black_acc;
        const int16_t* nstm_acc = is_white_to_move ? black_acc : white_acc;
        float layer1_out[HIDDEN1_DIM];
        float layer2_out[HIDDEN2_DIM];

        for (int i = 0; i < HIDDEN1_DIM; ++i) {
            float sum = model_params.h1_bias[i];
            if (quantized_model_params.ready && quantized_model_params.enabled) {
                const int16_t* q_row = quantized_model_params.h1_weights[i];
                int32_t q_sum = dot_i16_i16(stm_acc, q_row, ACCUMULATOR_DIM)
                              + dot_i16_i16(nstm_acc, q_row + ACCUMULATOR_DIM, ACCUMULATOR_DIM);
                sum += static_cast<float>(q_sum) * quantized_model_params.h1_scales[i];
            } else {
                const float* weights_row = model_params.h1_weights[i];
                sum += dot_i16_f32(stm_acc, weights_row, ACCUMULATOR_DIM)
                     + dot_i16_f32(nstm_acc, weights_row + ACCUMULATOR_DIM, ACCUMULATOR_DIM);
            }
            layer1_out[i] = clipped_relu(sum);
        }

        for (int i = 0; i < HIDDEN2_DIM; ++i) {
            float sum = model_params.h2_bias[i];
            const float* weights_row = model_params.h2_weights[i];
            for (int j = 0; j < HIDDEN1_DIM; ++j) {
                sum += layer1_out[j] * weights_row[j];
            }
            layer2_out[i] = clipped_relu(sum);
        }

        float output_score = model_params.out_bias[0];
        const float* output_weights = model_params.out_weights[0];
        for (int j = 0; j < HIDDEN2_DIM; ++j) {
            output_score += layer2_out[j] * output_weights[j];
        }

        return output_score;
    }

    // ==========================================================
    // Engine Evaluation Bridge
    // ==========================================================
    static int evaluate_cp(const int16_t* white_acc, const int16_t* black_acc, bool is_white_to_move, float output_scale = 1.0f) {
        float raw_score = evaluate_perspective(white_acc, black_acc, is_white_to_move);
        float scaled_score = raw_score * output_scale;
        scaled_score = std::max(-MAX_CENTIPAWN_SCORE, std::min(MAX_CENTIPAWN_SCORE, scaled_score));
        return static_cast<int>(std::round(scaled_score));
    }

    // ==========================================================
    // Weight Serializer Loader (Declared here, implemented below)
    // ==========================================================
    static bool load_weights(const std::string& filepath);

    static void build_quantized_inference_weights() {
        for (int row = 0; row < HIDDEN1_DIM; ++row) {
            float max_abs = 0.0f;
            for (int col = 0; col < INPUT_DIM; ++col) {
                max_abs = std::max(max_abs, std::abs(model_params.h1_weights[row][col]));
            }

            float scale = (max_abs > 0.0f) ? (max_abs / 32767.0f) : 1.0f;
            quantized_model_params.h1_scales[row] = scale;

            for (int col = 0; col < INPUT_DIM; ++col) {
                float normalized = model_params.h1_weights[row][col] / scale;
                int rounded = static_cast<int>(std::round(normalized));
                rounded = std::clamp(rounded, -32767, 32767);
                quantized_model_params.h1_weights[row][col] = static_cast<int16_t>(rounded);
            }
        }
        quantized_model_params.ready = true;
    }

    static void set_quantized_inference(bool enabled) {
        quantized_model_params.enabled = enabled;
    }

    // ==========================================================
    // Random Weight Seeding (Kaiming Initialization)
    // ==========================================================
    static void initialize_weights_from_scratch() {
        std::cout << " -> Seeding network layers with random Kaiming distributions...\n";
        
        std::mt19937 gen(42); 

        // 1. Layer 1 (1024 -> 256)
        float scale1 = std::sqrt(2.0f / static_cast<float>(INPUT_DIM));
        std::normal_distribution<float> dist1(0.0f, scale1);
        for (int i = 0; i < HIDDEN1_DIM; ++i) {
            model_params.h1_bias[i] = 0.0f;
            for (int j = 0; j < INPUT_DIM; ++j) {
                model_params.h1_weights[i][j] = dist1(gen);
            }
        }

        // 2. Layer 2 (256 -> 32)
        float scale2 = std::sqrt(2.0f / static_cast<float>(HIDDEN1_DIM));
        std::normal_distribution<float> dist2(0.0f, scale2);
        for (int i = 0; i < HIDDEN2_DIM; ++i) {
            model_params.h2_bias[i] = 0.0f;
            for (int j = 0; j < HIDDEN1_DIM; ++j) {
                model_params.h2_weights[i][j] = dist2(gen);
            }
        }

        // 3. Output Layer (32 -> 1)
        float scale_out = std::sqrt(2.0f / static_cast<float>(HIDDEN2_DIM));
        std::normal_distribution<float> dist_out(0.0f, scale_out);
        model_params.out_bias[0] = 0.0f;
        for (int j = 0; j < HIDDEN2_DIM; ++j) {
            model_params.out_weights[0][j] = dist_out(gen);
        }
    }
};

} // namespace NNUE

// ==========================================================
// 🎯 THE SAFETY VALVE BREAKING CIRCULAR DEPENDENCY
// ==========================================================
#include "nnue_loader.hpp"

namespace NNUE {
    // Implement the inline bridge function now that NNUELoader is fully known to the compiler
    inline bool NNUEModel::load_weights(const std::string& filepath) {
        return NNUELoader::load_model_file(filepath);
    }
}
