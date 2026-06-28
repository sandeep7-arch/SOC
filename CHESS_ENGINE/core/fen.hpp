#ifndef FEN_HPP
#define FEN_HPP

#include <cctype>
#include <string>
#include "board.hpp"

inline bool parse_fen_checked(NativeBoard& board, const std::string& fen) {
    board.clear();
    size_t i = 0;

    int row = 7;
    int col = 0;
    for (; i < fen.length() && fen[i] != ' '; ++i) {
        const unsigned char raw = static_cast<unsigned char>(fen[i]);
        const char c = static_cast<char>(raw);
        if (c == '/') {
            if (col != 8) return false;
            row--;
            col = 0;
            if (row < 0) return false;
        } else if (std::isdigit(raw)) {
            col += c - '0';
            if (col > 8) return false;
        } else {
            if (col >= 8 || row < 0) return false;
            const Color color = std::isupper(raw) ? WHITE : BLACK;
            PieceType pt = NONE;
            switch (std::toupper(raw)) {
                case 'P': pt = PAWN; break;
                case 'N': pt = KNIGHT; break;
                case 'B': pt = BISHOP; break;
                case 'R': pt = ROOK; break;
                case 'Q': pt = QUEEN; break;
                case 'K': pt = KING; break;
                default: return false;
            }

            board.put_piece(color, pt, static_cast<Square>(row * 8 + col));
            col++;
        }
    }

    if (row != 0 || col != 8) return false;
    if (i >= fen.length() || fen[i] != ' ') return false;
    while (i < fen.length() && fen[i] == ' ') i++;

    if (i >= fen.length()) return false;
    Color side = WHITE;
    if (fen[i] == 'w') side = WHITE;
    else if (fen[i] == 'b') side = BLACK;
    else return false;
    i++;

    while (i < fen.length() && fen[i] == ' ') i++;

    uint8_t rights = NO_CASTLING;
    if (i >= fen.length()) return false;
    if (fen[i] == '-') {
        i++;
    } else {
        while (i < fen.length() && fen[i] != ' ') {
            switch (fen[i]) {
                case 'K': rights |= WHITE_OO; break;
                case 'Q': rights |= WHITE_OOO; break;
                case 'k': rights |= BLACK_OO; break;
                case 'q': rights |= BLACK_OOO; break;
                default: return false;
            }
            i++;
        }
    }

    while (i < fen.length() && fen[i] == ' ') i++;

    Square ep = NO_SQ;
    if (i >= fen.length()) return false;
    if (fen[i] == '-') {
        i++;
    } else {
        if (i + 1 >= fen.length()) return false;
        const int file = fen[i] - 'a';
        const int rank = fen[i + 1] - '1';
        if (file < 0 || file > 7 || rank < 0 || rank > 7) return false;
        ep = static_cast<Square>(rank * 8 + file);
        i += 2;
    }

    board.set_state(side, rights, ep);
    board.update_occupancy();
    board.generate_hash_key();
    return true;
}

inline void parse_fen(NativeBoard& board, const std::string& fen) {
    (void)parse_fen_checked(board, fen);
}

namespace FenParser {
    inline bool parse_fen(const std::string& fen, NativeBoard& board) {
        return parse_fen_checked(board, fen);
    }
}

#endif // FEN_HPP
