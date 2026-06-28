#ifndef ZOBRIST_HPP
#define ZOBRIST_HPP

#include <cstdint>
#include "types.hpp"

namespace Zobrist {

    // ============================================================================
    // Zobrist Keys Storage
    // ============================================================================
    inline uint64_t pieces[COLOR_NB][PIECE_TYPE_NB][64];
    inline uint64_t castling[16];
    inline uint64_t ep[65]; // 0-63 for squares, 64 for NO_SQ (no en passant available)
    inline uint64_t side_to_move;

    // ============================================================================
    // Deterministic PRNG (SplitMix64)
    // ============================================================================
    // Standard splitmix64 generator ensuring identical hash tables on all platforms.
    struct SplitMix64 {
        uint64_t state;

        constexpr explicit SplitMix64(uint64_t seed) : state(seed) {}

        constexpr uint64_t next() {
            state += 0x9e3779b97f4a7c15ULL;
            uint64_t z = state;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
        }
    };

    // ============================================================================
    // Table Initialization
    // ============================================================================
    inline void init_keys() {
        // Use a fixed arbitrary seed for strict determinism
        SplitMix64 rng(0x123456789ABCDEF0ULL);

        // 1. Initialize piece keys
        for (int color = 0; color < COLOR_NB; ++color) {
            for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
                for (int sq = 0; sq < 64; ++sq) {
                    pieces[color][pt][sq] = rng.next();
                }
            }
        }

        // 2. Initialize castling rights keys (16 possible combinations)
        for (int i = 0; i < 16; ++i) {
            castling[i] = rng.next();
        }

        // 3. Initialize en passant keys (0-63 represent target files/squares, 64 represents none)
        for (int i = 0; i < 65; ++i) {
            ep[i] = rng.next();
        }

        // 4. Initialize side to move key
        side_to_move = rng.next();
    }
}

#endif // ZOBRIST_HPP
