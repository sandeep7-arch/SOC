#ifndef BOARD_HPP
#define BOARD_HPP

#include <array>
#include <vector>
#include <cassert>
#include "types.hpp"
#include "bitboard.hpp"
#include "Zobrist.hpp"

// Structure to preserve history for perfect unmaking
struct UndoState {
    Move move;
    uint8_t castling_rights;
    Square ep_square;
    uint8_t halfmove_clock;
    PieceType captured_piece;
    uint64_t hash;
};

class NativeBoard {
private:
    std::array<std::array<Bitboard, PIECE_TYPE_NB>, COLOR_NB> bitboards;
    std::array<Bitboard, 3> occupancy; // WHITE, BLACK, BOTH

    // Fast piece lookup array to identify captured pieces instantly
    std::array<PieceType, 64> piece_map;

    Color side_to_move;
    uint8_t castling_rights;
    Square ep_square;
    uint8_t halfmove_clock;
    uint16_t fullmove_number;
    uint64_t hash_key;

    std::vector<UndoState> undo_stack;

    // Castling rights update lookup table (indexed by square)
    // Ensures efficient, branchless castling rights updating when pieces move or are captured
    static constexpr std::array<uint8_t, 64> castling_rights_mask = []() {
        std::array<uint8_t, 64> masks{};
        for (auto& mask : masks) {
            mask = ALL_CASTLING;
        }
        masks[SQ_A1] = ~WHITE_OOO;
        masks[SQ_E1] = ~(WHITE_OO | WHITE_OOO);
        masks[SQ_H1] = ~WHITE_OO;
        masks[SQ_A8] = ~BLACK_OOO;
        masks[SQ_E8] = ~(BLACK_OO | BLACK_OOO);
        masks[SQ_H8] = ~BLACK_OO;
        return masks;
    }();

public:
    NativeBoard() { clear(); }

    void clear() {
        for (int c = 0; c < COLOR_NB; ++c) bitboards[c].fill(EMPTY_BB);
        occupancy.fill(EMPTY_BB);
        piece_map.fill(NONE);
        side_to_move = WHITE;
        castling_rights = NO_CASTLING;
        ep_square = NO_SQ;
        halfmove_clock = 0;
        fullmove_number = 1;
        hash_key = 0ULL;
        undo_stack.clear();
    }

    // ============================================================================
    // Core Utility Syncs
    // ============================================================================

    void set_state(Color side, uint8_t rights, Square ep) {
        side_to_move = side;
        castling_rights = rights;
        ep_square = ep;
    }

    void update_occupancy() {
        occupancy[WHITE] = EMPTY_BB;
        occupancy[BLACK] = EMPTY_BB;
        for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
            occupancy[WHITE] |= bitboards[WHITE][pt];
            occupancy[BLACK] |= bitboards[BLACK][pt];
        }
        occupancy[BOTH] = occupancy[WHITE] | occupancy[BLACK];
    }

    void generate_hash_key() {
        uint64_t h = 0ULL;
        for (int c = 0; c < COLOR_NB; ++c) {
            for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
                Bitboard bb = bitboards[c][pt];
                while (bb) {
                    Square sq = pop_lsb(bb);
                    h ^= Zobrist::pieces[c][pt][sq];
                }
            }
        }
        h ^= Zobrist::castling[castling_rights];
        h ^= Zobrist::ep[ep_square];
        if (side_to_move == BLACK) h ^= Zobrist::side_to_move;
        hash_key = h;
    }

    // Manual piece placement utility (useful for FEN parsing/testing)
    void put_piece(Color c, PieceType pt, Square sq) {
        set_bit(bitboards[c][pt], sq);
        piece_map[sq] = pt;
    }

    // ============================================================================
    // Make Move
    // ============================================================================

    void make_move(Move move) {
        const Square from = move_from(move);
        const Square to = move_to(move);
        const MoveFlags::Flags flags = move_flags(move);
        const Color us = side_to_move;
        const Color them = static_cast<Color>(us ^ 1);
        const PieceType moving_piece = piece_map[from];

        // 1. Identify and Track Captures
        PieceType captured = NONE;
        if (flags == MoveFlags::EN_PASSANT) {
            captured = PAWN;
        } else {
            captured = piece_map[to];
        }

        // 2. Push Current State onto Undo Stack
        undo_stack.push_back({move, castling_rights, ep_square, halfmove_clock, captured, hash_key});

        // 3. Clear Current State Transient Hashes
        hash_key ^= Zobrist::castling[castling_rights];
        hash_key ^= Zobrist::ep[ep_square];

        // 4. Handle En Passant Capture
        if (flags == MoveFlags::EN_PASSANT) {
            Square ep_cap_sq = static_cast<Square>(to + (us == WHITE ? -8 : 8));
            clear_bit(bitboards[them][PAWN], ep_cap_sq);
            piece_map[ep_cap_sq] = NONE;
            hash_key ^= Zobrist::pieces[them][PAWN][ep_cap_sq];
        }
        // Handle Standard Capture
        else if (captured != NONE) {
            clear_bit(bitboards[them][captured], to);
            hash_key ^= Zobrist::pieces[them][captured][to];
        }

        // 5. Move the actual piece on the bitboards
        clear_bit(bitboards[us][moving_piece], from);
        hash_key ^= Zobrist::pieces[us][moving_piece][from];
        piece_map[from] = NONE;

        if (!move_is_promotion(move)) {
            set_bit(bitboards[us][moving_piece], to);
            hash_key ^= Zobrist::pieces[us][moving_piece][to];
            piece_map[to] = moving_piece;
        } else {
            // Handle Promotion Transformation
            PieceType promo_piece = move_promotion_piece(move);
            set_bit(bitboards[us][promo_piece], to);
            hash_key ^= Zobrist::pieces[us][promo_piece][to];
            piece_map[to] = promo_piece;
        }

        // 6. Handle Castling (Move Rook)
        if (flags == MoveFlags::OO) {
            Square r_from = (us == WHITE) ? SQ_H1 : SQ_H8;
            Square r_to   = (us == WHITE) ? SQ_F1 : SQ_F8;
            clear_bit(bitboards[us][ROOK], r_from);
            set_bit(bitboards[us][ROOK], r_to);
            piece_map[r_from] = NONE;
            piece_map[r_to] = ROOK;
            hash_key ^= Zobrist::pieces[us][ROOK][r_from] ^ Zobrist::pieces[us][ROOK][r_to];
        } else if (flags == MoveFlags::OOO) {
            Square r_from = (us == WHITE) ? SQ_A1 : SQ_A8;
            Square r_to   = (us == WHITE) ? SQ_D1 : SQ_D8;
            clear_bit(bitboards[us][ROOK], r_from);
            set_bit(bitboards[us][ROOK], r_to);
            piece_map[r_from] = NONE;
            piece_map[r_to] = ROOK;
            hash_key ^= Zobrist::pieces[us][ROOK][r_from] ^ Zobrist::pieces[us][ROOK][r_to];
        }

        // 7. Update State Metrics (Castling, En Passant Status, and Clocks)
        castling_rights &= castling_rights_mask[from];
        castling_rights &= castling_rights_mask[to];

        if (flags == MoveFlags::DOUBLE_PUSH) {
            ep_square = static_cast<Square>(to + (us == WHITE ? -8 : 8));
        } else {
            ep_square = NO_SQ;
        }

        if (moving_piece == PAWN || captured != NONE) {
            halfmove_clock = 0;
        } else {
            halfmove_clock++;
        }

        if (us == BLACK) {
            fullmove_number++;
        }

        // 8. Re-apply Hashing Metrics & Toggle Side
        hash_key ^= Zobrist::castling[castling_rights];
        hash_key ^= Zobrist::ep[ep_square];
        hash_key ^= Zobrist::side_to_move;

        side_to_move = them;

        // 9. Synchronize Occupancies
        update_occupancy();
    }

    // ============================================================================
    // Unmake Move
    // ============================================================================

    void unmake_move(Move move) {
        assert(!undo_stack.empty());
        const UndoState state = undo_stack.back();
        undo_stack.pop_back();

        const Square from = move_from(move);
        const Square to = move_to(move);
        const MoveFlags::Flags flags = move_flags(move);

        side_to_move = static_cast<Color>(side_to_move ^ 1);
        const Color us = side_to_move;
        const Color them = static_cast<Color>(us ^ 1);

        // 1. Revert Moved or Promoted Pieces
        PieceType put_back_piece = move_is_promotion(move) ? PAWN : piece_map[to];

        if (move_is_promotion(move)) {
            PieceType promo_piece = move_promotion_piece(move);
            clear_bit(bitboards[us][promo_piece], to);
        } else {
            clear_bit(bitboards[us][put_back_piece], to);
        }

        set_bit(bitboards[us][put_back_piece], from);
        piece_map[from] = put_back_piece;
        piece_map[to] = NONE;

        // 2. Revert Captures (Standard or En Passant)
        if (flags == MoveFlags::EN_PASSANT) {
            Square ep_cap_sq = static_cast<Square>(to + (us == WHITE ? -8 : 8));
            set_bit(bitboards[them][PAWN], ep_cap_sq);
            piece_map[ep_cap_sq] = PAWN;
        } else if (state.captured_piece != NONE) {
            set_bit(bitboards[them][state.captured_piece], to);
            piece_map[to] = state.captured_piece;
        }

        // 3. Revert Castling Rooks
        if (flags == MoveFlags::OO) {
            Square r_from = (us == WHITE) ? SQ_H1 : SQ_H8;
            Square r_to   = (us == WHITE) ? SQ_F1 : SQ_F8;
            clear_bit(bitboards[us][ROOK], r_to);
            set_bit(bitboards[us][ROOK], r_from);
            piece_map[r_to] = NONE;
            piece_map[r_from] = ROOK;
        } else if (flags == MoveFlags::OOO) {
            Square r_from = (us == WHITE) ? SQ_A1 : SQ_A8;
            Square r_to   = (us == WHITE) ? SQ_D1 : SQ_D8;
            clear_bit(bitboards[us][ROOK], r_to);
            set_bit(bitboards[us][ROOK], r_from);
            piece_map[r_to] = NONE;
            piece_map[r_from] = ROOK;
        }

        // 4. Copy Exact Scalar and Hash History State Back
        castling_rights = state.castling_rights;
        ep_square = state.ep_square;
        halfmove_clock = state.halfmove_clock;
        hash_key = state.hash;

        if (us == BLACK) {
            fullmove_number--;
        }

        // 5. Synchronize Occupancies
        update_occupancy();
    }

    // ============================================================================
    // Structural Inspection Getters
    // ============================================================================
    [[nodiscard]] Bitboard get_occupancy(Color c) const { return occupancy[c]; }
    [[nodiscard]] Bitboard get_combined_occupancy() const { return occupancy[BOTH]; }
    [[nodiscard]] Bitboard get_piece_bb(Color c, PieceType pt) const { return bitboards[c][pt]; }
    [[nodiscard]] PieceType get_piece_on(Square sq) const { return piece_map[sq]; }
    [[nodiscard]] Color get_side_to_move() const { return side_to_move; }
    [[nodiscard]] Square get_ep_square() const { return ep_square; }
    [[nodiscard]] uint8_t get_castling_rights() const { return castling_rights; }
    [[nodiscard]] uint8_t get_halfmove_clock() const { return halfmove_clock; }
    [[nodiscard]] uint64_t get_hash() const { return hash_key; }

    [[nodiscard]] int count_repetitions(uint64_t hash) const {
        int count = 0;
        for (auto it = undo_stack.rbegin(); it != undo_stack.rend(); ++it) {
            if (it->hash == hash) {
                ++count;
            }
        }
        return count;
    }
};

#endif // BOARD_HPP
