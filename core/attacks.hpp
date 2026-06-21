#ifndef ATTACKS_HPP
#define ATTACKS_HPP

#include "types.hpp"
#include "bitboard.hpp"

namespace Attacks {

    // Precomputed global lookup tables
    inline Bitboard knight_attacks[64];
    inline Bitboard king_attacks[64];
    inline Bitboard pawn_attacks[COLOR_NB][64];

    // ============================================================================
    // Initialization Logic
    // ============================================================================

    inline void init() {
        for (int sq = 0; sq < 64; ++sq) {
            Bitboard b = 1ULL << sq;
            Square square = static_cast<Square>(sq);

            // 1. Precompute Knight Attacks
            Bitboard knight = 0ULL;
            // Up/Down shifts combined with Left/Right shifts
            knight |= (b << 17) & NOT_FILE_A;  // Up 2, Right 1
            knight |= (b << 10) & NOT_FILE_AB; // Up 1, Right 2
            knight |= (b >> 6)  & NOT_FILE_AB; // Down 1, Right 2
            knight |= (b >> 15) & NOT_FILE_A;  // Down 2, Right 1
            knight |= (b << 15) & NOT_FILE_H;  // Up 2, Left 1
            knight |= (b << 6)  & NOT_FILE_GH; // Up 1, Left 2
            knight |= (b >> 10) & NOT_FILE_GH; // Down 1, Left 2
            knight |= (b >> 17) & NOT_FILE_H;  // Down 2, Left 1
            knight_attacks[sq] = knight;

            // 2. Precompute King Attacks
            Bitboard king = 0ULL;
            king |= (b << 8);                  // Up
            king |= (b >> 8);                  // Down
            king |= (b << 1) & NOT_FILE_A;     // Right
            king |= (b >> 1) & NOT_FILE_H;     // Left
            king |= (b << 9) & NOT_FILE_A;     // Up-Right
            king |= (b << 7) & NOT_FILE_H;     // Up-Left
            king |= (b >> 7) & NOT_FILE_A;     // Down-Right
            king |= (b >> 9) & NOT_FILE_H;     // Down-Left
            king_attacks[sq] = king;

            // 3. Precompute Pawn Attacks (Directional)
            // White Pawn Attacks (Advance up ranks)
            Bitboard w_pawn = 0ULL;
            w_pawn |= (b << 9) & NOT_FILE_A;   // Up-Right capture
            w_pawn |= (b << 7) & NOT_FILE_H;   // Up-Left capture
            pawn_attacks[WHITE][sq] = w_pawn;

            // Black Pawn Attacks (Advance down ranks)
            Bitboard b_pawn = 0ULL;
            b_pawn |= (b >> 7) & NOT_FILE_H;   // Down-Left capture (from White's perspective, shifts right to file H)
            b_pawn |= (b >> 9) & NOT_FILE_A;   // Down-Right capture (shifts left to file A)
            pawn_attacks[BLACK][sq] = b_pawn;
        }
    }

    // ============================================================================
    // Accessor Getters
    // ============================================================================

    [[nodiscard]] inline Bitboard get_knight_attacks(Square sq) {
        return knight_attacks[sq];
    }

    [[nodiscard]] inline Bitboard get_king_attacks(Square sq) {
        return king_attacks[sq];
    }

    [[nodiscard]] inline Bitboard get_pawn_attacks(Color side, Square sq) {
        return pawn_attacks[side][sq];
    }
}

#endif // ATTACKS_HPP
