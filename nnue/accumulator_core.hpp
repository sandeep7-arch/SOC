// nnue/accumulator_core.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>
#include "board.hpp"

namespace NNUE {

// --- Network Constants Configuration ---
constexpr int TRANSFORMED_SIZE = 256;
constexpr int HALF_KP_FEATURES = 64 * 10 * 64; // 64 King Sq * 10 Piece Layers * 64 Target Sq

struct NetWeights {
    int16_t feature_weights[HALF_KP_FEATURES][TRANSFORMED_SIZE];
    int16_t feature_bias[TRANSFORMED_SIZE];
    int16_t output_weights[TRANSFORMED_SIZE * 2];
    int16_t output_bias;
};

// Global weight memory map matching your Python FeatureTransformer layers
inline NetWeights weights;

// --- Dual-Perspective Accumulator Structure ---
struct Accumulator {
    int16_t v[2][TRANSFORMED_SIZE]; // [0] = White Perspective, [1] = Black Perspective
};

// --- Perspective Index Feature Encoder ---
inline int get_feature_index(int king_sq, int piece_sq, int piece_type, int piece_color, int perspective) {
    if (perspective == BLACK) {
        king_sq ^= 56;
        piece_sq ^= 56;
        piece_color = !piece_color;
    }
    // Map piece identities to a 0-9 range (excluding Kings from the token index map)
    int token = piece_type + (piece_color == perspective ? 0 : 5);
    return king_sq * 640 + (token * 64 + piece_sq);
}

class AccumulatorManager {
public:
    AccumulatorManager() = default;

    // ==========================================================
    // Stateless Full Reconstruction (Optimized for Copy-on-Write)
    // ==========================================================
    static void compute_accumulator(const NativeBoard& board, Accumulator& out_accum) {
        // 1. Load structural bias weights directly into layers
        for (int i = 0; i < TRANSFORMED_SIZE; ++i) {
            out_accum.v[WHITE][i] = weights.feature_bias[i];
            out_accum.v[BLACK][i] = weights.feature_bias[i];
        }

        int white_king = bit_scan_forward(board.get_piece_bb(WHITE, KING));
        int black_king = bit_scan_forward(board.get_piece_bb(BLACK, KING));

        // 2. Scan full piece arrays to perform initial matrix additions from scratch
        for (int c = WHITE; c <= BLACK; ++c) {
            for (int p = PAWN; p <= QUEEN; ++p) {
                uint64_t pieces = board.get_piece_bb(static_cast<Color>(c), static_cast<PieceType>(p));
                while (pieces) {
                    int sq = __builtin_ctzll(pieces);

                    int idx_w = get_feature_index(white_king, sq, p, c, WHITE);
                    int idx_b = get_feature_index(black_king, sq, p, c, BLACK);

                    // Accumulate raw layers
                    for (int i = 0; i < TRANSFORMED_SIZE; ++i) {
                        out_accum.v[WHITE][i] += weights.feature_weights[idx_w][i];
                        out_accum.v[BLACK][i] += weights.feature_weights[idx_b][i];
                    }
                    pieces &= (pieces - 1); // Pop LSB
                }
            }
        }
    }
};

} // namespace NNUE
