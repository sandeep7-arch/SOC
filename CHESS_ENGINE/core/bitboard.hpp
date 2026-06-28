#ifndef BITBOARD_HPP
#define BITBOARD_HPP

#include <cstdint>
#include <iostream>
#include "types.hpp"

// Type alias for clarity
using Bitboard = uint64_t;

// ============================================================================
// Bitboard Constants
// ============================================================================
constexpr Bitboard EMPTY_BB = 0ULL;
constexpr Bitboard ALL_BB   = ~0ULL;

// File Masks
constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_B = FILE_A << 1;
constexpr Bitboard FILE_C = FILE_A << 2;
constexpr Bitboard FILE_D = FILE_A << 3;
constexpr Bitboard FILE_E = FILE_A << 4;
constexpr Bitboard FILE_F = FILE_A << 5;
constexpr Bitboard FILE_G = FILE_A << 6;
constexpr Bitboard FILE_H = FILE_A << 7;

// Rank Masks
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = RANK_1 << 8;
constexpr Bitboard RANK_3 = RANK_1 << 16;
constexpr Bitboard RANK_4 = RANK_1 << 24;
constexpr Bitboard RANK_5 = RANK_1 << 32;
constexpr Bitboard RANK_6 = RANK_1 << 40;
constexpr Bitboard RANK_7 = RANK_1 << 48;
constexpr Bitboard RANK_8 = RANK_1 << 56;

// Intercept Mask Helpers (Avoids wrap-around on edges)
constexpr Bitboard NOT_FILE_A = ~FILE_A;
constexpr Bitboard NOT_FILE_H = ~FILE_H;
constexpr Bitboard NOT_FILE_AB = ~(FILE_A | FILE_B);
constexpr Bitboard NOT_FILE_GH = ~(FILE_G | FILE_H);

// ============================================================================
// Core Inline Bit Manipulation
// ============================================================================

[[nodiscard]] constexpr Bitboard square_bb(Square sq) {
    return 1ULL << static_cast<int>(sq);
}

[[nodiscard]] constexpr bool test_bit(Bitboard bb, Square sq) {
    return (bb & square_bb(sq)) != 0;
}

constexpr void set_bit(Bitboard& bb, Square sq) {
    bb |= square_bb(sq);
}

constexpr void clear_bit(Bitboard& bb, Square sq) {
    bb &= ~square_bb(sq);
}

constexpr void toggle_bit(Bitboard& bb, Square sq) {
    bb ^= square_bb(sq);
}

[[nodiscard]] constexpr int pop_count(Bitboard bb) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bb);
#elif defined(_MSC_VER)
    return static_cast<int>(__popcnt64(bb));
#else
    // Fallback software implementation if modern compilers are absent
    bb -= (bb >> 1) & 0x5555555555555555ULL;
    bb = (bb & 0x3333333333333333ULL) + ((bb >> 2) & 0x3333333333333333ULL);
    return static_cast<int>((((bb + (bb >> 4)) & 0xF0F0F0F0F0F0F0Full) * 0x101010101010101ULL) >> 56);
#endif
}

[[nodiscard]] inline Square bit_scan_forward(Bitboard bb) {
    // Behavior is undefined for bb = 0. Caller must guarantee bb != EMPTY_BB
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<Square>(__builtin_ctzll(bb));
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, bb);
    return static_cast<Square>(index);
#endif
}

inline Square pop_lsb(Bitboard& bb) {
    Square lsb = bit_scan_forward(bb);
    bb &= bb - 1; // Clears the lowest set bit
    return lsb;
}

// ============================================================================
// Debug Utilities
// ============================================================================

inline void print_bitboard(Bitboard bb) {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int rank = 7; rank >= 0; --rank) {
        std::cout << (rank + 1) << " | ";
        for (int file = 0; file < 8; ++file) {
            int square = rank * 8 + file;
            std::cout << (test_bit(bb, static_cast<Square>(square)) ? "1" : ".") << " | ";
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n";
    std::cout << "Bitboard Value: 0x" << std::hex << bb << std::dec << "\n\n";
}

#endif // BITBOARD_HPP
