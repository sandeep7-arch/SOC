#ifndef LEGAL_HPP
#define LEGAL_HPP

#include "types.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"
#include "board.hpp"
#include "movegen.hpp"

namespace Legal {

    // ============================================================================
    // Attack Determination Helper (Uses Attack Symmetry)
    // ============================================================================
    [[nodiscard]] inline bool is_square_attacked(const NativeBoard& board, Square sq, Color attacker) {
        const Bitboard combined_occ = board.get_combined_occupancy();

        // 1. Attacked by Pawns
        // Use the opposite color's pawn attack mask from the target square
        Color defender = static_cast<Color>(attacker ^ 1);
        if (Attacks::get_pawn_attacks(defender, sq) & board.get_piece_bb(attacker, PAWN)) return true;

        // 2. Attacked by Knights
        if (Attacks::get_knight_attacks(sq) & board.get_piece_bb(attacker, KNIGHT)) return true;

        // 3. Attacked by Kings
        if (Attacks::get_king_attacks(sq) & board.get_piece_bb(attacker, KING)) return true;

        // 4. Attacked by Bishops / Queens (Sliding Diagonal)
        Bitboard sliders_diagonal = board.get_piece_bb(attacker, BISHOP) | board.get_piece_bb(attacker, QUEEN);
        if (sliders_diagonal) {
            Bitboard bishop_attacks = MoveGen::generate_sliding_attacks(sq, combined_occ, MoveGen::bishop_offsets);
            if (bishop_attacks & sliders_diagonal) return true;
        }

        // 5. Attacked by Rooks / Queens (Sliding Orthogonal)
        Bitboard sliders_orthogonal = board.get_piece_bb(attacker, ROOK) | board.get_piece_bb(attacker, QUEEN);
        if (sliders_orthogonal) {
            Bitboard rook_attacks = MoveGen::generate_sliding_attacks(sq, combined_occ, MoveGen::rook_offsets);
            if (rook_attacks & sliders_orthogonal) return true;
        }

        return false;
    }

    // ============================================================================
    // The Core Legality Filter
    // ============================================================================
    [[nodiscard]] inline bool is_move_legal(NativeBoard& board, Move move) {
        const Color us = board.get_side_to_move();
        const Color them = static_cast<Color>(us ^ 1);
        const MoveFlags::Flags flags = move_flags(move);

        // --- Castling Pre-Check Safety Rules ---
        // Castling is pseudo-legalized, but the King cannot escape, pass through, or land in check.
        if (flags == MoveFlags::OO || flags == MoveFlags::OOO) {
            Square king_start = (us == WHITE) ? SQ_E1 : SQ_E8;

            // Cannot castle out of check
            if (is_square_attacked(board, king_start, them)) return false;

            if (flags == MoveFlags::OO) {
                Square transit = (us == WHITE) ? SQ_F1 : SQ_F8;
                // Cannot pass through check
                if (is_square_attacked(board, transit, them)) return false;
            } else {
                Square transit = (us == WHITE) ? SQ_D1 : SQ_D8;
                // Cannot pass through check
                if (is_square_attacked(board, transit, them)) return false;
            }
        }

        // --- Make / Unmake Verification Pass ---
        board.make_move(move);

        // Find where our king is after the move
        Bitboard king_bb = board.get_piece_bb(us, KING);

        // If the king was captured or missing (should not happen in correct movegen), it's illegal
        if (!king_bb) {
            board.unmake_move(move);
            return false;
        }

        Square king_sq = bit_scan_forward(king_bb);

        // Check if our king is exposed to an attack from the opponent
        bool is_legal = !is_square_attacked(board, king_sq, them);

        board.unmake_move(move);

        return is_legal;
    }
}

#endif // LEGAL_HPP
