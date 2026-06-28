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
    uint8_t  generation   = 0;
    bool     occupied     = false;
};

class TranspositionTable {
private:
    static constexpr size_t CLUSTER_SIZE = 4;

    struct TTBucket {
        TTEntry entries[CLUSTER_SIZE];
    };

    std::vector<TTBucket> table;
    size_t max_entries;
    size_t bucket_count;
    size_t index_mask;
    uint8_t current_generation = 1;

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
        resize(entries);
    }

    void resize(size_t entries) {
        max_entries = rounded_power_of_two(std::max<size_t>(1, entries));
        bucket_count = rounded_power_of_two(std::max<size_t>(1, max_entries / CLUSTER_SIZE));
        index_mask = bucket_count - 1;
        table.clear();
        table.resize(bucket_count);
        clear();
    }

    void clear() {
        for (auto& bucket : table) {
            for (auto& entry : bucket.entries) {
                entry = TTEntry();
            }
        }
        current_generation = 1;
        hits = 0;
        misses = 0;
        collisions = 0;
        stores = 0;
    }

    void start_new_search() {
        current_generation++;
        if (current_generation == 0) current_generation = 1;
    }

    static size_t rounded_power_of_two(size_t value) {
        size_t power = 1;
        while (power < value) power <<= 1;
        return power;
    }

    bool probe(uint64_t key, int depth, int alpha, int beta, int ply, bool board_turn, int& tt_score, uint16_t& tt_move) {
        key = get_perspective_key(key, board_turn);
        size_t index = key & index_mask;
        const TTBucket& bucket = table[index];

        const TTEntry* matched = nullptr;
        for (const TTEntry& entry : bucket.entries) {
            if (entry.occupied && entry.zobrist_key == key) {
                matched = &entry;
                break;
            }
        }

        if (matched == nullptr) {
            misses++;
            tt_move = 0;
            return false;
        }

        const TTEntry& entry = *matched;
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
        TTBucket& bucket = table[index];
        TTEntry* replace = &bucket.entries[0];

        for (TTEntry& entry : bucket.entries) {
            if (!entry.occupied) {
                replace = &entry;
                break;
            }
            if (entry.zobrist_key == key) {
                replace = &entry;
                break;
            }
            if (replacement_score(entry) < replacement_score(*replace)) {
                replace = &entry;
            }
        }

        if (replace->occupied && replace->zobrist_key != key) {
            collisions++;
        } else if (replace->occupied && depth < replace->depth && flag != TT_EXACT) {
            return;
        }

        // Compress/Normalize distance-to-mate score mapping
        if (score >= MATE_THRESHOLD) score += ply;
        else if (score <= -MATE_THRESHOLD) score -= ply;

        replace->zobrist_key = key;
        replace->score       = static_cast<int16_t>(score);
        replace->move_raw    = move_raw;
        replace->depth       = static_cast<uint8_t>(std::min(depth, 255));
        replace->flag        = flag;
        replace->generation  = current_generation;
        replace->occupied    = true;
        stores++;
    }

    size_t get_size() const {
        size_t count = 0;
        for (const auto& bucket : table) {
            for (const auto& entry : bucket.entries) {
                if (entry.occupied) count++;
            }
        }
        return count;
    }

private:
    int replacement_score(const TTEntry& entry) const {
        int age_bonus = entry.generation == current_generation ? 8 : 0;
        int exact_bonus = entry.flag == TT_EXACT ? 4 : 0;
        return static_cast<int>(entry.depth) * 2 + exact_bonus + age_bonus;
    }
};
