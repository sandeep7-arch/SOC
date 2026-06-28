#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>

// ============================================================================
// Core Types & Enums
// ============================================================================

enum Color : uint8_t {
    WHITE = 0,
    BLACK = 1,
    BOTH = 2,
    COLOR_NB = 2
};

enum PieceType : uint8_t {
    PAWN   = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK   = 3,
    QUEEN  = 4,
    KING   = 5,
    NONE   = 6,
    PIECE_TYPE_NB = 6
};

enum Square : uint8_t {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    NO_SQ = 64
};

// Castling rights represented as a 4-bit mask
enum CastlingRights : uint8_t {
    NO_CASTLING = 0,
    WHITE_OO    = 1,
    WHITE_OOO   = 2,
    BLACK_OO    = 4,
    BLACK_OOO   = 8,
    WHITE_ANY   = WHITE_OO | WHITE_OOO,
    BLACK_ANY   = BLACK_OO | BLACK_OOO,
    ALL_CASTLING = WHITE_ANY | BLACK_ANY
};

// ============================================================================
// Move MoveFlags Encoding
// ============================================================================
// Bits 12-15 represent the type of move performed.
namespace MoveFlags {
    enum Flags : uint16_t {
        QUIET           = 0,     // 0000
        DOUBLE_PUSH     = 1,     // 0001
        OO              = 2,     // 0010
        OOO             = 3,     // 0011
        CAPTURE         = 4,     // 0100
        EN_PASSANT      = 5,     // 0101
        PR_KNIGHT       = 8,     // 1000
        PR_BISHOP       = 9,     // 1001
        PR_ROOK         = 10,    // 1010
        PR_QUEEN        = 11,    // 1011
        PC_KNIGHT       = 12,    // 1100
        PC_BISHOP       = 13,    // 1101
        PC_ROOK         = 14,    // 1110
        PC_QUEEN        = 15     // 1111
    };
}

// ============================================================================
// 16-bit Packed Move Representation
// ============================================================================
// Format:
// Bits 0-5   : From Square (0-63)
// Bits 6-11  : To Square (0-63)
// Bits 12-15 : Move Flags
struct Move {
    uint16_t data;

    // Default constructor generates a null/invalid move
    constexpr Move() : data(0) {}
    constexpr explicit Move(uint16_t raw) : data(raw) {}
    constexpr Move(Square from, Square to, MoveFlags::Flags flags)
        : data(static_cast<uint16_t>(from) |
               (static_cast<uint16_t>(to) << 6) |
               (static_cast<uint16_t>(flags) << 12)) {}

    constexpr bool operator==(const Move& other) const { return data == other.data; }
    constexpr bool operator!=(const Move& other) const { return data != other.data; }
    constexpr bool is_none() const { return data == 0; }
};

// ============================================================================
// Inline Bit Extraction Utilities
// ============================================================================

[[nodiscard]] constexpr Square move_from(Move m) {
    return static_cast<Square>(m.data & 0x3F);
}

[[nodiscard]] constexpr Square move_to(Move m) {
    return static_cast<Square>((m.data >> 6) & 0x3F);
}

[[nodiscard]] constexpr MoveFlags::Flags move_flags(Move m) {
    return static_cast<MoveFlags::Flags>((m.data >> 12) & 0x0F);
}

[[nodiscard]] constexpr bool move_is_capture(Move m) {
    // Captures have bit 14 set (Flags 4..7 and 12..15)
    return (m.data & 0x4000) != 0;
}

[[nodiscard]] constexpr bool move_is_promotion(Move m) {
    // Promotions have bit 15 set (Flags 8..15)
    return (m.data & 0x8000) != 0;
}

[[nodiscard]] constexpr bool move_is_ep(Move m) {
    return move_flags(m) == MoveFlags::EN_PASSANT;
}

[[nodiscard]] constexpr PieceType move_promotion_piece(Move m) {
    // Extracts the promotion target from promotion/promotion-capture flags
    // Mapping: KNIGHT(8/12)->KNIGHT(1), BISHOP(9/13)->BISHOP(2), ROOK(10/14)->ROOK(3), QUEEN(11/15)->QUEEN(4)
    return static_cast<PieceType>((move_flags(m) & 0x3) + 1);
}

#endif // TYPES_HPP
