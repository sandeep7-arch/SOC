// search/native_search.cpp
#include "board.hpp"       // Synchronized with our exact, production-grade layout
#include "bitboard.hpp"
#include "attacks.hpp"
#include "movegen.hpp"
#include "legal.hpp"
#include "transposition.hpp"
#include "move_ordering.hpp"
#include "time_manager.hpp"
#include "Zobrist.hpp"
#include "fen.hpp"

// NNUE INTEGRATION HEADERS
#include "feature_transformer.hpp"
#include "nnue_model.hpp"
#include "feature_encoder.hpp"
#include "nnue_loader.hpp"

#include <chrono>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

// Performance globals and diagnostic counters
long long native_nodes = 0;
auto search_start_time = std::chrono::steady_clock::now();
double native_time_limit = 0.0;
int native_last_root_score = 0;

constexpr int INF = 1000000;
constexpr int MATE_SCORE = 30000;
constexpr int TIMEOUT_SIGNAL = -999999;
constexpr int MAX_QPLY = 32;
constexpr int DELTA_MARGIN = 200;
constexpr int FUTILITY_MARGIN = 120;
constexpr int REVERSE_FUTILITY_MARGIN = 95;
constexpr int NATIVE_PIECE_VALUES[] = { 100, 320, 330, 500, 900, 20000 };

// Instantiate unified subsystem managers
TranspositionTable global_tt(2000000);
MoveOrderingManager order_manager;
TimeManager time_manager;

struct NativePVLine {
    Move moves[64];
    int count = 0;
};

struct NativeEvalState {
    NNUE::Accumulator white_acc;
    NNUE::Accumulator black_acc;
    int white_king = 0;
    int black_king_flipped = 0;
};

struct SearchSettings {
    bool enable_lmr = true;
    bool enable_futility = true;
    bool enable_reverse_futility = true;
    int lmr_depth = 3;
    int lmr_move_number = 4;
};

SearchSettings search_settings;

inline bool is_time_up() {
    return time_manager.should_stop_search();
}

inline char get_promo_char(MoveFlags::Flags flags) {
    if (flags == MoveFlags::PR_KNIGHT || flags == MoveFlags::PC_KNIGHT) return 'n';
    if (flags == MoveFlags::PR_BISHOP || flags == MoveFlags::PC_BISHOP) return 'b';
    if (flags == MoveFlags::PR_ROOK   || flags == MoveFlags::PC_ROOK)   return 'r';
    if (flags == MoveFlags::PR_QUEEN  || flags == MoveFlags::PC_QUEEN)  return 'q';
    return '\0';
}

inline PieceType piece_on(const NativeBoard& board, Color color, Square sq) {
    PieceType pt = board.get_piece_on(sq);
    if (pt == NONE) return NONE;
    return (board.get_piece_bb(color, pt) & square_bb(sq)) ? pt : NONE;
}

inline PieceType moving_piece_for_move(const NativeBoard& board, Move move) {
    return board.get_piece_on(move_from(move));
}

inline PieceType captured_piece_for_move(const NativeBoard& board, Move move) {
    if (move_is_ep(move)) return PAWN;
    return board.get_piece_on(move_to(move));
}

inline bool castling_path_is_safe(const NativeBoard& board, Move move) {
    const MoveFlags::Flags flags = move_flags(move);
    if (flags != MoveFlags::OO && flags != MoveFlags::OOO) return true;

    const Color us = board.get_side_to_move();
    const Color them = static_cast<Color>(us ^ 1);
    const Square king_start = (us == WHITE) ? SQ_E1 : SQ_E8;
    if (Legal::is_square_attacked(board, king_start, them)) return false;

    const Square transit = (flags == MoveFlags::OO)
        ? ((us == WHITE) ? SQ_F1 : SQ_F8)
        : ((us == WHITE) ? SQ_D1 : SQ_D8);
    return !Legal::is_square_attacked(board, transit, them);
}

inline bool side_king_is_safe_after_make(const NativeBoard& board, Color moved_side) {
    Bitboard king_bb = board.get_piece_bb(moved_side, KING);
    if (!king_bb) return false;
    Square king_sq = bit_scan_forward(king_bb);
    Color attacker = static_cast<Color>(moved_side ^ 1);
    return !Legal::is_square_attacked(board, king_sq, attacker);
}

inline std::string move_to_uci(Move move) {
    Square from = move_from(move);
    Square to = move_to(move);

    std::string uci;
    uci += static_cast<char>('a' + (from % 8));
    uci += static_cast<char>('1' + (from / 8));
    uci += static_cast<char>('a' + (to % 8));
    uci += static_cast<char>('1' + (to / 8));

    char promo = get_promo_char(move_flags(move));
    if (promo) uci += promo;
    return uci;
}

inline bool find_legal_root_move(const NativeBoard& root_board, Move candidate, Move& selected) {
    NativeBoard board = root_board;
    MoveList move_list;
    MoveGen::generate_moves(board, move_list);

    bool have_fallback = false;
    Move fallback;
    for (int i = 0; i < move_list.count; ++i) {
        Move move = move_list.moves[i];
        if (!Legal::is_move_legal(board, move)) continue;

        if (!have_fallback) {
            fallback = move;
            have_fallback = true;
        }

        if (move == candidate) {
            selected = move;
            return true;
        }
    }

    if (have_fallback) {
        selected = fallback;
        return true;
    }
    return false;
}

inline int state_feature_index(const NativeEvalState& state, Square sq, PieceType pt, Color color, Color perspective) {
    int king_sq = (perspective == WHITE) ? state.white_king : state.black_king_flipped;
    return NNUE::FeatureEncoder::get_feature_index(king_sq, sq, pt, color, perspective);
}

inline void add_feature(NNUE::Accumulator& accum, int raw_feature) {
    const int shifted_index = raw_feature + 1;
#if defined(__AVX2__)
    for (int i = 0; i < NNUE::ACCUMULATOR_DIM; i += 16) {
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accum.v + i));
        __m256i emb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(NNUE::transformer_weights.embedding_weights[shifted_index] + i));
        acc = _mm256_add_epi16(acc, emb);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accum.v + i), acc);
    }
#else
    for (int i = 0; i < NNUE::ACCUMULATOR_DIM; ++i) {
        accum.v[i] += NNUE::transformer_weights.embedding_weights[shifted_index][i];
    }
#endif
}

inline void remove_feature(NNUE::Accumulator& accum, int raw_feature) {
    const int shifted_index = raw_feature + 1;
#if defined(__AVX2__)
    for (int i = 0; i < NNUE::ACCUMULATOR_DIM; i += 16) {
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accum.v + i));
        __m256i emb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(NNUE::transformer_weights.embedding_weights[shifted_index] + i));
        acc = _mm256_sub_epi16(acc, emb);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accum.v + i), acc);
    }
#else
    for (int i = 0; i < NNUE::ACCUMULATOR_DIM; ++i) {
        accum.v[i] -= NNUE::transformer_weights.embedding_weights[shifted_index][i];
    }
#endif
}

inline void add_piece_features(NativeEvalState& state, Square sq, PieceType pt, Color color) {
    if (pt == KING || pt == NONE) return;
    add_feature(state.white_acc, state_feature_index(state, sq, pt, color, WHITE));
    add_feature(state.black_acc, state_feature_index(state, sq, pt, color, BLACK));
}

inline void remove_piece_features(NativeEvalState& state, Square sq, PieceType pt, Color color) {
    if (pt == KING || pt == NONE) return;
    remove_feature(state.white_acc, state_feature_index(state, sq, pt, color, WHITE));
    remove_feature(state.black_acc, state_feature_index(state, sq, pt, color, BLACK));
}

inline void refresh_eval_state(const NativeBoard& board, NativeEvalState& state) {
    static thread_local std::vector<int> w_features;
    static thread_local std::vector<int> b_features;

    NNUE::FeatureEncoder::active_features(board, w_features, b_features);
    NNUE::FeatureTransformer::forward(w_features, state.white_acc);
    NNUE::FeatureTransformer::forward(b_features, state.black_acc);

    state.white_king = bit_scan_forward(board.get_piece_bb(WHITE, KING));
    state.black_king_flipped = bit_scan_forward(board.get_piece_bb(BLACK, KING)) ^ 56;
}

inline void update_eval_state_for_move(
    const NativeBoard& before,
    Move move,
    const NativeEvalState& parent,
    NativeEvalState& child
) {
    const Color us = before.get_side_to_move();
    const Color them = static_cast<Color>(us ^ 1);
    const Square from = move_from(move);
    const Square to = move_to(move);
    const MoveFlags::Flags flags = move_flags(move);
    const PieceType moving_piece = moving_piece_for_move(before, move);

    if (moving_piece == KING || moving_piece == NONE) {
        child = parent;
        return;
    }

    child = parent;

    remove_piece_features(child, from, moving_piece, us);

    if (flags == MoveFlags::EN_PASSANT) {
        Square ep_cap_sq = static_cast<Square>(to + (us == WHITE ? -8 : 8));
        remove_piece_features(child, ep_cap_sq, PAWN, them);
    } else {
        PieceType captured = captured_piece_for_move(before, move);
        remove_piece_features(child, to, captured, them);
    }

    PieceType placed_piece = move_is_promotion(move) ? move_promotion_piece(move) : moving_piece;
    add_piece_features(child, to, placed_piece, us);
}

inline void apply_eval_state_for_move(const NativeBoard& before, Move move, NativeEvalState& state) {
    const Color us = before.get_side_to_move();
    const Color them = static_cast<Color>(us ^ 1);
    const Square from = move_from(move);
    const Square to = move_to(move);
    const MoveFlags::Flags flags = move_flags(move);
    const PieceType moving_piece = moving_piece_for_move(before, move);

    if (moving_piece == KING || moving_piece == NONE) return;

    remove_piece_features(state, from, moving_piece, us);

    if (flags == MoveFlags::EN_PASSANT) {
        Square ep_cap_sq = static_cast<Square>(to + (us == WHITE ? -8 : 8));
        remove_piece_features(state, ep_cap_sq, PAWN, them);
    } else {
        remove_piece_features(state, to, captured_piece_for_move(before, move), them);
    }

    PieceType placed_piece = move_is_promotion(move) ? move_promotion_piece(move) : moving_piece;
    add_piece_features(state, to, placed_piece, us);
}

inline void undo_eval_state_for_move(const NativeBoard& restored_parent, Move move, NativeEvalState& state) {
    const Color us = restored_parent.get_side_to_move();
    const Color them = static_cast<Color>(us ^ 1);
    const Square from = move_from(move);
    const Square to = move_to(move);
    const MoveFlags::Flags flags = move_flags(move);
    const PieceType moving_piece = moving_piece_for_move(restored_parent, move);

    if (moving_piece == KING || moving_piece == NONE) return;

    PieceType placed_piece = move_is_promotion(move) ? move_promotion_piece(move) : moving_piece;
    remove_piece_features(state, to, placed_piece, us);

    if (flags == MoveFlags::EN_PASSANT) {
        Square ep_cap_sq = static_cast<Square>(to + (us == WHITE ? -8 : 8));
        add_piece_features(state, ep_cap_sq, PAWN, them);
    } else {
        add_piece_features(state, to, captured_piece_for_move(restored_parent, move), them);
    }

    add_piece_features(state, from, moving_piece, us);
}

extern "C" {
    void initialize_native_evaluator(size_t cache_size);
    int probe_native_eval_cache(uint64_t hash, bool turn, int* out_found);
    void store_native_eval_cache(uint64_t hash, bool turn, int score);
    void clear_native_eval_cache();
}

inline bool is_repetition_or_draw(const NativeBoard& board) {
    if (board.get_halfmove_clock() >= 100) return true;

    const uint64_t current_hash = board.get_hash();
    if (current_hash == 0) return false;
    return board.count_repetitions(current_hash) >= 2;
}

// 🎯 HIGH-PERFORMANCE STATIC ALLOCATION EVALUATOR
int native_evaluate(const NativeBoard& board, const NativeEvalState& eval_state) {
    uint64_t current_hash = board.get_hash();

    int found_in_cache = 0;
    bool white_to_move = (board.get_side_to_move() == WHITE);
    int cached_score = probe_native_eval_cache(current_hash, white_to_move, &found_in_cache);

    if (found_in_cache == 1) {
        return cached_score;
    }

    int centipawn_score = NNUE::NNUEModel::evaluate_cp(
        eval_state.white_acc.v,
        eval_state.black_acc.v,
        white_to_move,
        1.0f
    );

    store_native_eval_cache(current_hash, white_to_move, centipawn_score);
    return centipawn_score;
}

inline bool check_detection(const NativeBoard& board) {
    Color us = board.get_side_to_move();
    Bitboard king_bb = board.get_piece_bb(us, KING);
    if (!king_bb) return false;
    Square king_sq = bit_scan_forward(king_bb);
    return Legal::is_square_attacked(board, king_sq, static_cast<Color>(us ^ 1));
}

int native_quiescence(NativeBoard& board, NativeEvalState& eval_state, int alpha, int beta, int ply) {
    if (ply >= MAX_QPLY) return native_evaluate(board, eval_state);
    native_nodes++;

    if ((native_nodes & 2047) == 0 && is_time_up()) return TIMEOUT_SIGNAL;

    bool in_check = check_detection(board);
    int stand_pat = -INF;
    if (!in_check) {
        stand_pat = native_evaluate(board, eval_state);
        if (stand_pat >= beta) return stand_pat;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    MoveList move_list;
    MoveGen::generate_moves(board, move_list);

    int move_scores[256] = {0};
    Color us = board.get_side_to_move();

    // Fast inline capture evaluator using piece arrays
    for (int i = 0; i < move_list.count; ++i) {
        Move m = move_list.moves[i];
        if (!move_is_capture(m)) continue;

        PieceType attacker_type = moving_piece_for_move(board, m);
        PieceType victim_type = captured_piece_for_move(board, m);
        if (attacker_type == NONE || victim_type == NONE) continue;

        move_scores[i] = (NATIVE_PIECE_VALUES[victim_type] * 10) - NATIVE_PIECE_VALUES[attacker_type];
    }

    // Sort moves inline
    for (int i = 0; i < move_list.count - 1; ++i) {
        for (int j = i + 1; j < move_list.count; ++j) {
            if (move_scores[j] > move_scores[i]) {
                std::swap(move_scores[i], move_scores[j]);
                std::swap(move_list.moves[i], move_list.moves[j]);
            }
        }
    }

    for (int i = 0; i < move_list.count; ++i) {
        Move move = move_list.moves[i];

        if (!move_is_capture(move) && !in_check) continue;

        if (!in_check && move_is_capture(move)) {
            // Basic Delta Pruning Check
            PieceType victim_type = captured_piece_for_move(board, move);
            int gain = (victim_type == NONE) ? NATIVE_PIECE_VALUES[PAWN] : NATIVE_PIECE_VALUES[victim_type];
            if (stand_pat + gain + DELTA_MARGIN < alpha) continue;
        }

        const Color moved_side = board.get_side_to_move();
        const PieceType moving_piece = moving_piece_for_move(board, move);
        const bool king_move = moving_piece == KING;
        if (!castling_path_is_safe(board, move)) continue;
        NativeEvalState saved_eval;
        if (king_move) {
            saved_eval = eval_state;
        } else {
            apply_eval_state_for_move(board, move, eval_state);
        }
        board.make_move(move);
        if (!side_king_is_safe_after_make(board, moved_side)) {
            board.unmake_move(move);
            if (!king_move) {
                undo_eval_state_for_move(board, move, eval_state);
            }
            continue;
        }
        if (king_move) {
            refresh_eval_state(board, eval_state);
        }
        int score = -native_quiescence(board, eval_state, -beta, -alpha, ply + 1);
        board.unmake_move(move);
        if (king_move) {
            eval_state = saved_eval;
        } else {
            undo_eval_state_for_move(board, move, eval_state);
        }

        if (score == TIMEOUT_SIGNAL) return TIMEOUT_SIGNAL;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

int native_alpha_beta(
    NativeBoard& board,
    NativeEvalState& eval_state,
    int alpha,
    int beta,
    int depth,
    int ply,
    NativePVLine& pv_line,
    bool pv_node = false
) {
    if ((native_nodes & 2047) == 0 && is_time_up()) return TIMEOUT_SIGNAL;

    native_nodes++;
    pv_line.count = 0;

    if (ply > 0 && is_repetition_or_draw(board)) {
        return 0;
    }

    if (depth <= 0) {
        return native_quiescence(board, eval_state, alpha, beta, ply);
    }

    uint64_t current_hash = board.get_hash();
    int tt_score = -INF;
    uint16_t tt_move_raw = 0;
    
    if (
        global_tt.probe(current_hash, depth, alpha, beta, ply, board.get_side_to_move() == WHITE, tt_score, tt_move_raw)
        && ply > 0
        && !pv_node
    ) {
        return tt_score;
    }

    MoveList move_list;
    MoveGen::generate_moves(board, move_list);

    int move_scores[256] = {0};
    Color us = board.get_side_to_move();
    bool in_check = check_detection(board);

    for (int i = 0; i < move_list.count; ++i) {
        Move m = move_list.moves[i];
        PieceType attacker_type = moving_piece_for_move(board, m);
        PieceType victim_type = move_is_capture(m) ? captured_piece_for_move(board, m) : NONE;
        if (attacker_type == NONE) attacker_type = PAWN;
        if (victim_type == NONE) victim_type = PAWN;
        move_scores[i] = order_manager.score_move(
            m.data, tt_move_raw, ply, us,
            move_from(m), move_to(m), attacker_type, victim_type, move_flags(m), move_is_capture(m), move_is_ep(m)
        );
    }

    for (int i = 0; i < move_list.count - 1; ++i) {
        for (int j = i + 1; j < move_list.count; ++j) {
            if (move_scores[j] > move_scores[i]) {
                std::swap(move_scores[i], move_scores[j]);
                std::swap(move_list.moves[i], move_list.moves[j]);
            }
        }
    }

    int legal_moves_counted = 0;
    int best_score = -INF;
    Move best_move_found;
    uint8_t tt_entry_flag = TT_ALPHA;
    NativePVLine child_pv;
    const bool can_prune_quiets = search_settings.enable_futility && !pv_node && !in_check && depth <= 2;
    int static_eval = 0;
    if ((can_prune_quiets || (search_settings.enable_reverse_futility && !pv_node && !in_check && depth <= 3))
        && std::abs(alpha) < MATE_SCORE - 1000
        && std::abs(beta) < MATE_SCORE - 1000) {
        static_eval = native_evaluate(board, eval_state);
        if (search_settings.enable_reverse_futility
            && depth <= 3
            && static_eval - (REVERSE_FUTILITY_MARGIN * depth) >= beta) {
            return static_eval;
        }
    }

    for (int i = 0; i < move_list.count; ++i) {
        Move move = move_list.moves[i];
        const bool tactical_move = move_is_capture(move) || move_is_promotion(move);

        if (can_prune_quiets
            && !tactical_move
            && legal_moves_counted > 0
            && static_eval + (FUTILITY_MARGIN * depth) <= alpha) {
            continue;
        }

        const Color moved_side = board.get_side_to_move();
        const PieceType moving_piece = moving_piece_for_move(board, move);
        const bool king_move = moving_piece == KING;
        if (!castling_path_is_safe(board, move)) continue;
        NativeEvalState saved_eval;
        if (king_move) {
            saved_eval = eval_state;
        } else {
            apply_eval_state_for_move(board, move, eval_state);
        }
        board.make_move(move);
        if (!side_king_is_safe_after_make(board, moved_side)) {
            board.unmake_move(move);
            if (!king_move) {
                undo_eval_state_for_move(board, move, eval_state);
            }
            continue;
        }
        legal_moves_counted++;
        child_pv.count = 0;

        if (king_move) {
            refresh_eval_state(board, eval_state);
        }
        const bool first_legal_move = legal_moves_counted == 1;
        const bool child_pv_node = pv_node && first_legal_move;
        int score;
        if (first_legal_move) {
            score = -native_alpha_beta(
                board,
                eval_state,
                -beta,
                -alpha,
                depth - 1,
                ply + 1,
                child_pv,
                child_pv_node
            );
        } else {
            NativePVLine scout_pv;
            int reduction = 0;
            if (search_settings.enable_lmr
                && depth >= search_settings.lmr_depth
                && legal_moves_counted >= search_settings.lmr_move_number
                && !tactical_move
                && !in_check
                && !pv_node) {
                reduction = 1 + (depth >= 6 && legal_moves_counted >= 8 ? 1 : 0);
            }
            score = -native_alpha_beta(
                board,
                eval_state,
                -alpha - 1,
                -alpha,
                std::max(0, depth - 1 - reduction),
                ply + 1,
                scout_pv,
                false
            );
            if (reduction > 0 && score > alpha) {
                score = -native_alpha_beta(
                    board,
                    eval_state,
                    -alpha - 1,
                    -alpha,
                    depth - 1,
                    ply + 1,
                    scout_pv,
                    false
                );
            }
            if (score > alpha && score < beta) {
                score = -native_alpha_beta(
                    board,
                    eval_state,
                    -beta,
                    -alpha,
                    depth - 1,
                    ply + 1,
                    child_pv,
                    child_pv_node
                );
            } else {
                child_pv.count = 0;
            }
        }
        board.unmake_move(move);
        if (king_move) {
            eval_state = saved_eval;
        } else {
            undo_eval_state_for_move(board, move, eval_state);
        }

        if (score == TIMEOUT_SIGNAL) return TIMEOUT_SIGNAL;

        if (score > best_score) {
            best_score = score;
            best_move_found = move;

            pv_line.moves[0] = move;
            std::memcpy(pv_line.moves + 1, child_pv.moves, child_pv.count * sizeof(Move));
            pv_line.count = child_pv.count + 1;
        }

        if (score > alpha) {
            tt_entry_flag = TT_EXACT;
            alpha = score;
        }

        if (alpha >= beta) {
            tt_entry_flag = TT_BETA;
            if (!move_is_capture(move)) {
                order_manager.add_killer(ply, move.data);
                order_manager.add_history(us, move_from(move), move_to(move), depth);
            }
            break;
        }
    }

    if (legal_moves_counted == 0) {
        return check_detection(board) ? (-MATE_SCORE + ply) : 0;
    }

    global_tt.store(current_hash, depth, best_score, tt_entry_flag, best_move_found.data, ply, board.get_side_to_move() == WHITE);
    return best_score;
}

extern "C" {

    EXPORT_API bool init_engine_native(const char* model_path, size_t tt_size, size_t cache_size) {
        if (!model_path) return false;

        bool nnue_success = NNUE::NNUELoader::load_model_file(std::string(model_path));
        if (!nnue_success) return false;

        global_tt.resize(tt_size);
        initialize_native_evaluator(cache_size);
        return true;
    }

    EXPORT_API bool search_position_native(
        const char* fen_str, int max_depth,
        int wtime, int btime, int winc, int binc, int movestogo,
        char* out_best_move_uci
    ) {
        if (!fen_str) return false;

        static bool initialized = false;
        if (!initialized) {
            Attacks::init();
            Zobrist::init_keys();
            initialized = true;
        }

        NativeBoard board;
        if (!parse_fen_checked(board, std::string(fen_str))) return false;

        native_nodes = 0;
        native_last_root_score = 0;
        order_manager.clear();
        clear_native_eval_cache();
        time_manager.reset();

        bool is_white = (board.get_side_to_move() == WHITE);
        time_manager.max_depth = max_depth;
        time_manager.allocate_time(wtime, btime, is_white ? winc : binc, movestogo, is_white);
        time_manager.start_timer();

        NativePVLine main_pv;
        NativeEvalState root_eval;
        refresh_eval_state(board, root_eval);
        int last_score = 0;
        constexpr int ASPIRATION_WINDOW = 40;

        for (int depth = 1; depth <= time_manager.max_depth; ++depth) {
            int score = 0;
            NativePVLine temp_pv;

            if (depth > 2) {
                int alpha = last_score - ASPIRATION_WINDOW;
                int beta = last_score + ASPIRATION_WINDOW;

                score = native_alpha_beta(board, root_eval, alpha, beta, depth, 0, temp_pv, true);
                if (score <= alpha || score >= beta) {
                    temp_pv.count = 0;
                    score = native_alpha_beta(board, root_eval, -INF, INF, depth, 0, temp_pv, true);
                }
            } else {
                score = native_alpha_beta(board, root_eval, -INF, INF, depth, 0, temp_pv, true);
            }

            if (score == TIMEOUT_SIGNAL) break;

            main_pv = temp_pv;
            last_score = score;
            native_last_root_score = score;

            std::string uci_pv_string = "";
            for (int i = 0; i < main_pv.count; ++i) {
                uci_pv_string += move_to_uci(main_pv.moves[i]) + " ";
            }

            int ms = static_cast<int>(time_manager.get_elapsed() * 1000);
            long long nps = (native_nodes * 1000LL) / static_cast<long long>(ms > 0 ? ms : 1);
            std::cout << "info depth " << depth
                      << " score cp " << score
                      << " nodes " << native_nodes
                      << " time " << (ms > 0 ? ms : 1)
                      << " nps " << nps
                      << " pv " << uci_pv_string << std::endl;

            if (time_manager.check_soft_bound()) break;
        }

        time_manager.stop();

        Move selected_move;
        if (main_pv.count > 0 && find_legal_root_move(board, main_pv.moves[0], selected_move)) {
            std::string uci = move_to_uci(selected_move);
            std::strcpy(out_best_move_uci, uci.c_str());
        } else if (find_legal_root_move(board, Move(), selected_move)) {
            std::string uci = move_to_uci(selected_move);
            std::strcpy(out_best_move_uci, uci.c_str());
        } else {
            std::strcpy(out_best_move_uci, "0000");
        }

        return true;
    }

    EXPORT_API bool search_position_native_movetime(
        const char* fen_str, int max_depth, int movetime_ms, char* out_best_move_uci
    ) {
        if (!fen_str) return false;

        static bool initialized = false;
        if (!initialized) {
            Attacks::init();
            Zobrist::init_keys();
            initialized = true;
        }

        NativeBoard board;
        if (!parse_fen_checked(board, std::string(fen_str))) return false;

        native_nodes = 0;
        native_last_root_score = 0;
        order_manager.clear();
        clear_native_eval_cache();
        time_manager.reset();
        time_manager.max_depth = std::max(1, max_depth);
        time_manager.set_fixed_move_time(std::max(1, movetime_ms));
        time_manager.start_timer();

        NativePVLine main_pv;
        NativeEvalState root_eval;
        refresh_eval_state(board, root_eval);
        int last_score = 0;
        constexpr int ASPIRATION_WINDOW = 40;

        for (int depth = 1; depth <= time_manager.max_depth; ++depth) {
            int score = 0;
            NativePVLine temp_pv;

            if (depth > 2) {
                int alpha = last_score - ASPIRATION_WINDOW;
                int beta = last_score + ASPIRATION_WINDOW;

                score = native_alpha_beta(board, root_eval, alpha, beta, depth, 0, temp_pv, true);
                if (score <= alpha || score >= beta) {
                    temp_pv.count = 0;
                    score = native_alpha_beta(board, root_eval, -INF, INF, depth, 0, temp_pv, true);
                }
            } else {
                score = native_alpha_beta(board, root_eval, -INF, INF, depth, 0, temp_pv, true);
            }

            if (score == TIMEOUT_SIGNAL) break;

            main_pv = temp_pv;
            last_score = score;
            native_last_root_score = score;

            std::string uci_pv_string = "";
            for (int i = 0; i < main_pv.count; ++i) {
                uci_pv_string += move_to_uci(main_pv.moves[i]) + " ";
            }

            int ms = static_cast<int>(time_manager.get_elapsed() * 1000);
            long long nps = (native_nodes * 1000LL) / static_cast<long long>(ms > 0 ? ms : 1);
            std::cout << "info depth " << depth
                      << " score cp " << score
                      << " nodes " << native_nodes
                      << " time " << (ms > 0 ? ms : 1)
                      << " nps " << nps
                      << " pv " << uci_pv_string << std::endl;

            if (time_manager.check_soft_bound()) break;
        }

        time_manager.stop();

        Move selected_move;
        if (main_pv.count > 0 && find_legal_root_move(board, main_pv.moves[0], selected_move)) {
            std::string uci = move_to_uci(selected_move);
            std::strcpy(out_best_move_uci, uci.c_str());
        } else if (find_legal_root_move(board, Move(), selected_move)) {
            std::string uci = move_to_uci(selected_move);
            std::strcpy(out_best_move_uci, uci.c_str());
        } else {
            std::strcpy(out_best_move_uci, "0000");
        }

        return true;
    }

    EXPORT_API int get_last_search_score_native() {
        return native_last_root_score;
    }

    EXPORT_API void set_quantized_inference_native(bool enabled) {
        NNUE::NNUEModel::set_quantized_inference(enabled);
    }
} // extern "C"
