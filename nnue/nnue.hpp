// nnue/nnue.hpp
#pragma once
#include <cstdint>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "board.hpp"

namespace NNUE {

// --- Network Architecture Constants ---
// Synced 1:1 with your CONFIG settings (512 dimension layer instead of 256)
constexpr int TRANSFORMED_SIZE = 512;
constexpr int HALF_KP_FEATURES = 64 * 10 * 64; // 40,960

// --- Scaled Quantization Values ---
constexpr int OUTPUT_SCALE = 128;
constexpr int FV_SCALE = 127;

struct NetWeights {
    // 🎯 FIX: Allocate an extra index row to map directly to your padding index layout.
    // Index 0 represents your zero-padding fallback vector row.
    int16_t feature_weights[HALF_KP_FEATURES + 1][TRANSFORMED_SIZE];
    int16_t feature_bias[TRANSFORMED_SIZE];

    // Output Layer matches your unified dual perspective dimensions (512 * 2 = 1024)
    int16_t output_weights[TRANSFORMED_SIZE * 2];
    int16_t output_bias;
};

// Global network weight structures mapping memory allocations
inline NetWeights weights;

struct Accumulator {
    int16_t v[2][TRANSFORMED_SIZE]; // [0] = White perspective, [1] = Black perspective
};

// --- Perspective Index Feature Encoder ---
inline int get_feature_index(int king_sq, int piece_sq, int piece_type, int piece_color, int perspective) {
    if (perspective == BLACK) {
        king_sq ^= 56;
        piece_sq ^= 56;
        piece_color = !piece_color;
    }

    // Constant stride mapping blocks: 2 sides * 5 piece types * 64 squares = 640
    int token = piece_type + (piece_color == perspective ? 0 : 5);
    int raw_feature = king_sq * 640 + (token * 64 + piece_sq);

    // 🎯 FIX: Shift values up by +1 to guarantee clean lookup inside the embedding layout matrix
    return raw_feature + 1;
}

// Full evaluation calculation from raw bitboard inputs
inline void refresh_accumulator(const NativeBoard& board, Accumulator& accum) {
    for (int i = 0; i < TRANSFORMED_SIZE; ++i) {
        accum.v[WHITE][i] = weights.feature_bias[i];
        accum.v[BLACK][i] = weights.feature_bias[i];
    }

    int white_king = bit_scan_forward(board.get_piece_bb(WHITE, KING));
    int black_king = bit_scan_forward(board.get_piece_bb(BLACK, KING));

    for (int c = WHITE; c <= BLACK; ++c) {
        for (int p = PAWN; p <= QUEEN; ++p) {
            uint64_t pieces = board.get_piece_bb(static_cast<Color>(c), static_cast<PieceType>(p));
            while (pieces) {
                int sq = __builtin_ctzll(pieces);

                int idx_w = get_feature_index(white_king, sq, p, c, WHITE);
                int idx_b = get_feature_index(black_king, sq, p, c, BLACK);

                for (int i = 0; i < TRANSFORMED_SIZE; ++i) {
                    accum.v[WHITE][i] += weights.feature_weights[idx_w][i];
                    accum.v[BLACK][i] += weights.feature_weights[idx_b][i];
                }
                pieces &= (pieces - 1);
            }
        }
    }
}

// Forward evaluation pass utilizing Clipped ReLU activation layers
inline int evaluate(const Accumulator& accum, int turn) {
    int sum = weights.output_bias;

    int us = turn;
    int them = 1 - us;

    // Active perspective evaluation pass loops
    for (int i = 0; i < TRANSFORMED_SIZE; ++i) {
        int16_t val = accum.v[us][i];
        int16_t clipped = std::clamp<int16_t>(val, 0, FV_SCALE);
        sum += clipped * weights.output_weights[i];
    }

    // Passive perspective evaluation pass loops
    for (int i = 0; i < TRANSFORMED_SIZE; ++i) {
        int16_t val = accum.v[them][i];
        int16_t clipped = std::clamp<int16_t>(val, 0, FV_SCALE);
        sum += clipped * weights.output_weights[TRANSFORMED_SIZE + i];
    }

    return sum / (FV_SCALE * OUTPUT_SCALE);
}

// Read flat model weights exported by your binary tools
inline bool load_model(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) return false;

    f.read(reinterpret_cast<char*>(weights.feature_weights), sizeof(weights.feature_weights));
    f.read(reinterpret_cast<char*>(weights.feature_bias), sizeof(weights.feature_bias));
    f.read(reinterpret_cast<char*>(weights.output_weights), sizeof(weights.output_weights));
    f.read(reinterpret_cast<char*>(weights.output_bias), sizeof(weights.output_bias));

    return true;
}

} // namespace NNUE
