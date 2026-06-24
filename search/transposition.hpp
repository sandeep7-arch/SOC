// search/transposition.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <algorithm>

enum TTFlag : uint8_t {
    TT_EXACT = 0,
    TT_BETA  = 1, // LOWERBOUND
    TT_ALPHA = 2  // UPPERBOUND
};

struct TTEntry {
    uint64_t zobrist_key = 0;
    int16_t  score        = 0;
    uint16_t move_raw     = 0;
    uint8_t  depth        = 0;
    uint8_t  flag         = 0;
    bool     occupied     = false;
};

class TranspositionTable {
private:
    std::vector<TTEntry> table;
    size_t max_entries;
    size_t index_mask;

    // Preserving your legacy side-to-move separation constant
    const uint64_t BLACK_TURN_HASH = 0xABCDEF1234567890ULL;
    const int MATE_THRESHOLD = 29000;

    inline uint64_t get_perspective_key(uint64_t zobrist, bool board_turn) const {
        return !board_turn ? (zobrist ^ BLACK_TURN_HASH) : zobrist;
    }

public:
    long long hits = 0;
    long long misses = 0;
    long long collisions = 0;
    long long stores = 0;

    TranspositionTable(size_t entries = 2000000) {
        max_entries = rounded_power_of_two(std::max<size_t>(1, entries));
        index_mask = max_entries - 1;
        table.resize(max_entries);
    }

    void resize(size_t entries) {
        max_entries = rounded_power_of_two(std::max<size_t>(1, entries));
        index_mask = max_entries - 1;
        table.clear();
        table.resize(max_entries);
        clear();
    }

    void clear() {
        std::fill(table.begin(), table.end(), TTEntry());
        hits = 0;
        misses = 0;
        collisions = 0;
        stores = 0;
    }

    static size_t rounded_power_of_two(size_t value) {
        size_t power = 1;
        while (power < value) power <<= 1;
        return power;
    }

    bool probe(uint64_t key, int depth, int alpha, int beta, int ply, bool board_turn, int& tt_score, uint16_t& tt_move) {
        key = get_perspective_key(key, board_turn);
        size_t index = key & index_mask;
        const TTEntry& entry = table[index];

        if (!entry.occupied || entry.zobrist_key != key) {
            misses++;
            return false;
        }

        hits++;
        tt_move = entry.move_raw;

        if (entry.depth >= depth) {
            // Uncompress/Restore distance-to-mate score mapping
            int return_score = entry.score;
            if (return_score >= MATE_THRESHOLD) return_score -= ply;
            else if (return_score <= -MATE_THRESHOLD) return_score += ply;

            if (entry.flag == TT_EXACT) {
                tt_score = return_score;
                return true;
            }
            if (entry.flag == TT_BETA && return_score >= beta) {
                tt_score = beta;
                return true;
            }
            if (entry.flag == TT_ALPHA && return_score <= alpha) {
                tt_score = alpha;
                return true;
            }
        }
        return false;
    }

    void store(uint64_t key, int depth, int score, uint8_t flag, uint16_t move_raw, int ply, bool board_turn) {
        key = get_perspective_key(key, board_turn);
        size_t index = key & index_mask;
        TTEntry& existing = table[index];

        if (existing.occupied) {
            if (existing.zobrist_key == key) {
                if (depth < existing.depth) return; // Guard against overwriting deeper research
            } else {
                collisions++;
                if (depth < existing.depth) return; // Depth-preferred eviction policy
            }
        }

        // Compress/Normalize distance-to-mate score mapping
        if (score >= MATE_THRESHOLD) score += ply;
        else if (score <= -MATE_THRESHOLD) score -= ply;

        existing.zobrist_key = key;
        existing.score       = static_cast<int16_t>(score);
        existing.move_raw    = move_raw;
        existing.depth       = static_cast<uint8_t>(depth);
        existing.flag        = flag;
        existing.occupied    = true;
        stores++;
    }

    size_t get_size() const {
        size_t count = 0;
        for (const auto& entry : table) {
            if (entry.occupied) count++;
        }
        return count;
    }
};
