#ifndef MOVEGEN_HPP
#define MOVEGEN_HPP

#include <array>
#include "types.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"
#include "board.hpp"

// ============================================================================
// Move List Structure
// ============================================================================
struct MoveList {
    std::array<Move, 256> moves;
    int count = 0;

    void add(Move m) {
        moves[count++] = m;
    }
};

namespace MoveGen {

    // Ray offsets for sliding pieces
    constexpr int bishop_offsets[4] = { 9, 7, -9, -7 };
    constexpr int rook_offsets[4]   = { 8, 1, -8, -1 };

    // ============================================================================
    // Sliding Piece Attack Helpers
    // ============================================================================
    inline Bitboard generate_sliding_attacks(Square sq, Bitboard occ, const int offsets[4]) {
        Bitboard attacks = EMPTY_BB;
        for (int i = 0; i < 4; ++i) {
            int offset = offsets[i];
            int current_sq = static_cast<int>(sq);

            while (true) {
                int r = current_sq / 8;
                int f = current_sq % 8;

                // Track directional borders manually to prevent wrapping bugs
                if (offset == 1  && f == 7) break; // Right boundary
                if (offset == -1 && f == 0) break; // Left boundary
                if (offset == 8  && r == 7) break; // Top boundary
                if (offset == -8 && r == 0) break; // Bottom boundary
                if (offset == 9  && (f == 7 || r == 7)) break; // Up-Right boundary
                if (offset == 7  && (f == 0 || r == 7)) break; // Up-Left boundary
                if (offset == -7 && (f == 7 || r == 0)) break; // Down-Right boundary
                if (offset == -9 && (f == 0 || r == 0)) break; // Down-Left boundary

                current_sq += offset;
                Square target = static_cast<Square>(current_sq);
                set_bit(attacks, target);

                if (test_bit(occ, target)) break; // Blocked by piece
            }
        }
        return attacks;
    }

    // ============================================================================
    // Core Move Generator
    // ============================================================================
    inline void generate_moves(const NativeBoard& board, MoveList& list) {
        const Color us = board.get_side_to_move();
        const Color them = static_cast<Color>(us ^ 1);
        const Bitboard us_occ = board.get_occupancy(us);
        const Bitboard them_occ = board.get_occupancy(them);
        const Bitboard empty_occ = ~board.get_combined_occupancy();

        // ------------------------------------------------------------------------
        // 1. Pawn Moves
        // ------------------------------------------------------------------------
        Bitboard pawns = board.get_piece_bb(us, PAWN);
        const int push_offset = (us == WHITE) ? 8 : -8;
        const Bitboard promo_rank = (us == WHITE) ? RANK_8 : RANK_1;
        const Bitboard start_rank = (us == WHITE) ? RANK_2 : RANK_7;

        while (pawns) {
            Square from = pop_lsb(pawns);
            Square to = static_cast<Square>(static_cast<int>(from) + push_offset);

            // Single Push
            if (test_bit(empty_occ, to)) {
                if (square_bb(to) & promo_rank) {
                    list.add(Move(from, to, MoveFlags::PR_QUEEN));
                    list.add(Move(from, to, MoveFlags::PR_ROOK));
                    list.add(Move(from, to, MoveFlags::PR_BISHOP));
                    list.add(Move(from, to, MoveFlags::PR_KNIGHT));
                } else {
                    list.add(Move(from, to, MoveFlags::QUIET));

                    // Double Push (Only valid if the single push was also empty)
                    Square double_to = static_cast<Square>(static_cast<int>(to) + push_offset);
                    if ((square_bb(from) & start_rank) && test_bit(empty_occ, double_to)) {
                        list.add(Move(from, double_to, MoveFlags::DOUBLE_PUSH));
                    }
                }
            }

            // Standard Pawn Captures
            Bitboard p_attacks = Attacks::get_pawn_attacks(us, from) & them_occ;
            while (p_attacks) {
                Square cap_to = pop_lsb(p_attacks);
                if (square_bb(cap_to) & promo_rank) {
                    list.add(Move(from, cap_to, MoveFlags::PC_QUEEN));
                    list.add(Move(from, cap_to, MoveFlags::PC_ROOK));
                    list.add(Move(from, cap_to, MoveFlags::PC_BISHOP));
                    list.add(Move(from, cap_to, MoveFlags::PC_KNIGHT));
                } else {
                    list.add(Move(from, cap_to, MoveFlags::CAPTURE));
                }
            }

            // En Passant Capture
            Square ep_sq = board.get_ep_square();
            if (ep_sq != NO_SQ) {
                Bitboard ep_attack = Attacks::get_pawn_attacks(us, from) & square_bb(ep_sq);
                if (ep_attack) {
                    list.add(Move(from, ep_sq, MoveFlags::EN_PASSANT));
                }
            }
        }

        // ------------------------------------------------------------------------
        // 2. Knight Moves
        // ------------------------------------------------------------------------
        Bitboard knights = board.get_piece_bb(us, KNIGHT);
        while (knights) {
            Square from = pop_lsb(knights);
            Bitboard targets = Attacks::get_knight_attacks(from) & ~us_occ;
            while (targets) {
                Square to = pop_lsb(targets);
                if (test_bit(them_occ, to)) {
                    list.add(Move(from, to, MoveFlags::CAPTURE));
                } else {
                    list.add(Move(from, to, MoveFlags::QUIET));
                }
            }
        }

        // ------------------------------------------------------------------------
        // 3. Sliding Moves (Bishops, Rooks, Queens)
        // ------------------------------------------------------------------------
        Bitboard combined_occ = board.get_combined_occupancy();

        // Bishops
        Bitboard bishops = board.get_piece_bb(us, BISHOP);
        while (bishops) {
            Square from = pop_lsb(bishops);
            Bitboard targets = generate_sliding_attacks(from, combined_occ, bishop_offsets) & ~us_occ;
            while (targets) {
                Square to = pop_lsb(targets);
                list.add(Move(from, to, test_bit(them_occ, to) ? MoveFlags::CAPTURE : MoveFlags::QUIET));
            }
        }

        // Rooks
        Bitboard rooks = board.get_piece_bb(us, ROOK);
        while (rooks) {
            Square from = pop_lsb(rooks);
            Bitboard targets = generate_sliding_attacks(from, combined_occ, rook_offsets) & ~us_occ;
            while (targets) {
                Square to = pop_lsb(targets);
                list.add(Move(from, to, test_bit(them_occ, to) ? MoveFlags::CAPTURE : MoveFlags::QUIET));
            }
        }

        // Queens
        Bitboard queens = board.get_piece_bb(us, QUEEN);
        while (queens) {
            Square from = pop_lsb(queens);
            Bitboard targets = (generate_sliding_attacks(from, combined_occ, bishop_offsets) |
                                generate_sliding_attacks(from, combined_occ, rook_offsets)) & ~us_occ;
            while (targets) {
                Square to = pop_lsb(targets);
                list.add(Move(from, to, test_bit(them_occ, to) ? MoveFlags::CAPTURE : MoveFlags::QUIET));
            }
        }

        // ------------------------------------------------------------------------
        // 4. King Moves & Castling (Pseudo-Legal Context)
        // ------------------------------------------------------------------------
        Bitboard king = board.get_piece_bb(us, KING);
        if (king) {
            Square from = bit_scan_forward(king);
            Bitboard targets = Attacks::get_king_attacks(from) & ~us_occ;
            while (targets) {
                Square to = pop_lsb(targets);
                list.add(Move(from, to, test_bit(them_occ, to) ? MoveFlags::CAPTURE : MoveFlags::QUIET));
            }

            // Pseudo-legal castling (Checks ONLY path occupancy and rights flags)
            uint8_t rights = board.get_castling_rights();
            if (us == WHITE) {
                if ((rights & WHITE_OO) && !test_bit(combined_occ, SQ_F1) && !test_bit(combined_occ, SQ_G1)) {
                    list.add(Move(SQ_E1, SQ_G1, MoveFlags::OO));
                }
                if ((rights & WHITE_OOO) && !test_bit(combined_occ, SQ_D1) && !test_bit(combined_occ, SQ_C1) && !test_bit(combined_occ, SQ_B1)) {
                    list.add(Move(SQ_E1, SQ_C1, MoveFlags::OOO));
                }
            } else {
                if ((rights & BLACK_OO) && !test_bit(combined_occ, SQ_F8) && !test_bit(combined_occ, SQ_G8)) {
                    list.add(Move(SQ_E8, SQ_G8, MoveFlags::OO));
                }
                if ((rights & BLACK_OOO) && !test_bit(combined_occ, SQ_D8) && !test_bit(combined_occ, SQ_C8) && !test_bit(combined_occ, SQ_B8)) {
                    list.add(Move(SQ_E8, SQ_C8, MoveFlags::OOO));
                }
            }
        }
    }
}

#endif // MOVEGEN_HPP
