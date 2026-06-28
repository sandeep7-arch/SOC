// search/move_ordering.hpp
#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>

const int MAX_PLY = 128;
const int TT_MOVE_BONUS       = 10000000;
const int PROMOTION_BONUS     = 9000000;
const int WINNING_CAPTURE_BONUS = 8000000;
const int KILLER_BONUS        = 7000000;
const int LOSING_CAPTURE_BONUS  = 6000000;

class MoveOrderingManager {
private:
    uint16_t killer_moves[MAX_PLY][2];
    int history[2][64][64];
    const int NATIVE_PIECE_VALUES[6] = { 100, 320, 330, 500, 900, 20000 };

public:
    MoveOrderingManager() {
        clear_all();
    }

    void clear() {
        std::memset(killer_moves, 0, sizeof(killer_moves));
    }

    void clear_all() {
        clear();
        std::memset(history, 0, sizeof(history));
    }

    void add_killer(int ply, uint16_t move_raw) {
        if (ply >= MAX_PLY || move_raw == 0) return;
        if (killer_moves[ply][0] == move_raw) return;

        killer_moves[ply][1] = killer_moves[ply][0];
        killer_moves[ply][0] = move_raw;
    }

    void add_history(int side, int from, int to, int depth) {
        if (from >= 0 && from < 64 && to >= 0 && to < 64) {
            int bonus = depth * depth;
            if (history[side][from][to] < 2000000) {
                history[side][from][to] += bonus;
            }
        }
    }

    void decay_history() {
        for (int c = 0; c < 2; ++c) {
            for (int f = 0; f < 64; ++f) {
                for (int t = 0; t < 64; ++t) {
                    history[c][f][t] >>= 1;
                }
            }
        }
    }

    int score_move(uint16_t move_raw, uint16_t tt_move_raw, int ply, int side,
                   int from, int to, int piece_type, int victim_type, int promo_type, bool is_cap, bool is_ep) {

        if (tt_move_raw != 0 && move_raw == tt_move_raw) {
            return TT_MOVE_BONUS;
        }

        if (promo_type >= 8) {
            int promoted_piece = (promo_type & 0x3) + 1;
            int score = PROMOTION_BONUS + NATIVE_PIECE_VALUES[promoted_piece];
            if (is_cap) {
                int victim_val = is_ep ? 100 : NATIVE_PIECE_VALUES[victim_type];
                score += (victim_val * 10) - NATIVE_PIECE_VALUES[piece_type] + 500000;
            }
            return score;
        }

        if (is_cap) {
            int victim_val = is_ep ? 100 : NATIVE_PIECE_VALUES[victim_type];
            int mvv_lva = (victim_val * 10) - NATIVE_PIECE_VALUES[piece_type];

            if (NATIVE_PIECE_VALUES[piece_type] <= victim_val) {
                return WINNING_CAPTURE_BONUS + mvv_lva;
            } else {
                return LOSING_CAPTURE_BONUS + mvv_lva;
            }
        }

        if (ply < MAX_PLY) {
            if (move_raw == killer_moves[ply][0]) return KILLER_BONUS + 1000;
            if (move_raw == killer_moves[ply][1]) return KILLER_BONUS;
        }

        return history[side][from][to];
    }
};
