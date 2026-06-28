// nnue/feature_encoder.hpp
#pragma once
#include <cstdint>
#include <vector>
#include "board.hpp"

namespace NNUE {

class FeatureEncoder {
public:
    // Maps standard piece types to continuous 0-4 offsets (excluding King)
    static constexpr int PIECE_TYPE_OFFSET[6] = {
        0, // PAWN = 0
        1, // KNIGHT = 1
        2, // BISHOP = 2
        3, // ROOK = 3
        4, // QUEEN = 4
        -1 // KING = 5 
    };

    static constexpr int PERSPECTIVE_STRIDE = 2 * 5 * 64; // 640
    static constexpr int COLOR_STRIDE = 5 * 64;           // 320
    static constexpr int PIECE_STRIDE = 64;               // 64

    // ==========================================================
    // CORE FEATURE INDEX GENERATOR (Matches HalfKP layout)
    // ==========================================================
    static inline int get_feature_index(int king_square, int square, int piece_type, int piece_color, int perspective) {
        int king_offset = king_square * PERSPECTIVE_STRIDE;

        int mapped_square = square;
        bool is_friendly = (piece_color == perspective);

        // From Black's perspective, mirror the square vertically (Rank 1 <-> Rank 8)
        if (perspective == BLACK) {
            mapped_square = square ^ 56;
        }

        int color_idx = is_friendly ? 0 : 1;
        int piece_idx = PIECE_TYPE_OFFSET[piece_type];

        return king_offset + (color_idx * COLOR_STRIDE) + (piece_idx * PIECE_STRIDE) + mapped_square;
    }

    // ==========================================================
    // FULL BOARD ENCODING (Zero-allocation feature gathering)
    // ==========================================================
    static void active_features(const NativeBoard& board, std::vector<int>& white_features, std::vector<int>& black_features) {
        // High Performance: Clear without deallocating internal vector capacity
        white_features.clear();
        black_features.clear();

        int white_king = bit_scan_forward(board.get_piece_bb(WHITE, KING));
        int black_king = bit_scan_forward(board.get_piece_bb(BLACK, KING));
        int black_king_flipped = black_king ^ 56;

        // Extract features using optimized bitboard pop routines
        for (int c = WHITE; c <= BLACK; ++c) {
            for (int p = PAWN; p <= QUEEN; ++p) {
                uint64_t pieces = board.get_piece_bb(static_cast<Color>(c), static_cast<PieceType>(p));
                while (pieces) {
                    int sq = __builtin_ctzll(pieces);

                    // Generate feature indices matching White's perspective
                    int idx_w = get_feature_index(white_king, sq, p, c, WHITE);
                    white_features.push_back(idx_w);

                    // Generate feature indices matching Black's perspective
                    int idx_b = get_feature_index(black_king_flipped, sq, p, c, BLACK);
                    black_features.push_back(idx_b);

                    pieces &= (pieces - 1); // Pop Least Significant Bit (LSB)
                }
            }
        }
    }
};

} // namespace NNUE
