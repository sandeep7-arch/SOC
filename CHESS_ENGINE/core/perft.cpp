#include <iostream>
#include <chrono>
#include <string>
#include "types.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"
#include "board.hpp"
#include "movegen.hpp"
#include "legal.hpp"

// ============================================================================
// Coordinate String Helper for Divide Output
// ============================================================================
inline std::string square_to_uci(Square sq) {
    if (sq == NO_SQ) return "-";
    std::string s = "";
    s += static_cast<char>('a' + (sq % 8));
    s += static_cast<char>('1' + (sq / 8));
    return s;
}

inline std::string move_to_uci(Move m) {
    if (m.is_none()) return "0000";
    std::string s = square_to_uci(move_from(m)) + square_to_uci(move_to(m));

    // Append promotion piece identifier if applicable
    if (move_is_promotion(m)) {
        MoveFlags::Flags flags = move_flags(m);
        if (flags == MoveFlags::PR_KNIGHT || flags == MoveFlags::PC_KNIGHT) s += "n";
        else if (flags == MoveFlags::PR_BISHOP || flags == MoveFlags::PC_BISHOP) s += "b";
        else if (flags == MoveFlags::PR_ROOK   || flags == MoveFlags::PC_ROOK)   s += "r";
        else if (flags == MoveFlags::PR_QUEEN  || flags == MoveFlags::PC_QUEEN)  s += "q";
    }
    return s;
}

// ============================================================================
// Core Perft Counters
// ============================================================================

// Pure recursive perft node counter
uint64_t perft(int depth, NativeBoard& board) {
    if (depth <= 0) return 1ULL;

    uint64_t nodes = 0;
    MoveList list;
    MoveGen::generate_moves(board, list);

    for (int i = 0; i < list.count; ++i) {
        Move move = list.moves[i];

        // Rely exclusively on our legal move filter pass
        if (!Legal::is_move_legal(board, move)) {
            continue;
        }

        board.make_move(move);
        nodes += perft(depth - 1, board);
        board.unmake_move(move);
    }

    return nodes;
}

// Perft with Divide: Prints the move allocation breakdown at root depth
void perft_divide(int depth, NativeBoard& board) {
    if (depth <= 0) return;

    std::cout << "\n--- Perft Divide (Depth " << depth << ") ---" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t total_nodes = 0;

    MoveList list;
    MoveGen::generate_moves(board, list);

    for (int i = 0; i < list.count; ++i) {
        Move move = list.moves[i];

        if (!Legal::is_move_legal(board, move)) {
            continue;
        }

        board.make_move(move);
        uint64_t branch_nodes = perft(depth - 1, board);
        total_nodes += branch_nodes;
        board.unmake_move(move);

        std::cout << move_to_uci(move) << ": " << branch_nodes << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "\nTotal Nodes : " << total_nodes << std::endl;
    std::cout << "Time Taken  : " << duration << " ms" << std::endl;
    if (duration > 0) {
        std::cout << "NPS         : " << (total_nodes * 1000) / duration << std::endl;
    }
    std::cout << "-----------------------------\n" << std::endl;
}

void parse_fen(NativeBoard& board, const std::string& fen) {
    board.clear();
    size_t i = 0;

    // 1. Parse Pieces
    int row = 7;
    int col = 0;
    for (; i < fen.length() && fen[i] != ' '; ++i) {
        char c = fen[i];
        if (c == '/') {
            row--;
            col = 0;
        } else if (isdigit(c)) {
            col += (c - '0');
        } else {
            Color color = isupper(c) ? WHITE : BLACK;
            PieceType pt = NONE;
            char upper = toupper(c);
            if (upper == 'P') pt = PAWN;
            else if (upper == 'N') pt = KNIGHT;
            else if (upper == 'B') pt = BISHOP;
            else if (upper == 'R') pt = ROOK;
            else if (upper == 'Q') pt = QUEEN;
            else if (upper == 'K') pt = KING;

            Square sq = static_cast<Square>(row * 8 + col);
            board.put_piece(color, pt, sq);
            col++;
        }
    }

    // Skip the trailing space
    if (i < fen.length() && fen[i] == ' ') i++;

    // 2. Parse Side to Move
    Color side = WHITE;
    if (i < fen.length()) {
        side = (fen[i] == 'w') ? WHITE : BLACK;
        i++;
    }

    // Direct state injection requires access helpers or an init function in NativeBoard.
    // To keep our design clean without hacking board.hpp internals, we can use a quick
    // pointer/reference layout or add an explicit state setter inside NativeBoard.

    // Skip spaces
    while (i < fen.length() && fen[i] == ' ') i++;

    // 3. Parse Castling Rights
    uint8_t rights = NO_CASTLING;
    while (i < fen.length() && fen[i] != ' ') {
        char c = fen[i];
        if (c == 'K') rights |= WHITE_OO;
        if (c == 'Q') rights |= WHITE_OOO;
        if (c == 'k') rights |= BLACK_OO;
        if (c == 'q') rights |= BLACK_OOO;
        i++;
    }

    // Skip spaces
    while (i < fen.length() && fen[i] == ' ') i++;

    // 4. Parse En Passant Square
    Square ep = NO_SQ;
    if (i < fen.length() && fen[i] != '-') {
        int f = fen[i] - 'a';
        int r = fen[i+1] - '1';
        ep = static_cast<Square>(r * 8 + f);
    }

    // --- State Injection Context ---
    // Since NativeBoard encapsulates these variables privately, we can use a small backdoor method
    // or add a public layout sync method to your NativeBoard class in board.hpp:
    //
    // void set_state(Color side, uint8_t rights, Square ep) {
    //     side_to_move = side;
    //     castling_rights = rights;
    //     ep_square = ep;
    // }

    board.set_state(side, rights, ep);
    board.update_occupancy();
    board.generate_hash_key();
}

// ============================================================================
// Execution Framework Wrapper (For Quick Verification Setup)
// ============================================================================
int main() {
    Attacks::init();
    Zobrist::init_keys();
    NativeBoard board;

    // KiwiPete position (Tests castling permissions, promotions, and intensive EP tracking)
    std::string kiwi_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    parse_fen(board, kiwi_fen);

    // Ground Truth Targets for KiwiPete (White to move):
    // Depth 1: 48
    // Depth 2: 2,039
    // Depth 3: 97,862
    perft_divide(3, board);
    return 0;
}
