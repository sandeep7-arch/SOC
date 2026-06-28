// search/evaluator_core.cpp
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

#if defined(_WIN32)
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

const int MATE_THRESHOLD = 29000;

struct EvalEntry {
    uint64_t hash;
    int score;
    uint32_t generation;
    bool turn;
};

class NativeEvaluator {
private:
    std::vector<EvalEntry> cache_table;
    size_t table_size;
    size_t table_mask; // Bitwise mask for instantaneous indexing
    size_t active_entries;
    uint32_t current_generation = 1;

public:
    long long cache_hits = 0;
    long long cache_misses = 0;

    NativeEvaluator(size_t cache_size) : active_entries(0) {
        // Enforce power-of-two rounding for high-performance bitwise masking
        size_t power_of_two = 1;
        while (power_of_two < cache_size) power_of_two <<= 1;

        table_size = power_of_two;
        table_mask = table_size - 1;
        cache_table.resize(table_size, {0, 0, 0, false});
    }

    // Fixed out_found parameter lane to use standard int* to match Python's ctypes.c_int
    int probe(uint64_t hash, bool turn, int* out_found) {
        size_t index = hash & table_mask; // Blazing fast 1-cycle bitwise lookup
        const EvalEntry& entry = cache_table[index];

        if (entry.generation == current_generation && entry.hash == hash && entry.turn == turn) {
            cache_hits++;
            *out_found = 1; // Explicit integer assignment flags
            return entry.score;
        }

        cache_misses++;
        *out_found = 0;
        return 0;
    }

    void store(uint64_t hash, bool turn, int score) {
        size_t index = hash & table_mask;

        if (cache_table[index].generation != current_generation) {
            active_entries++;
        }
        cache_table[index] = {hash, score, current_generation, turn};
    }

    void clear() {
        current_generation++;
        if (current_generation == 0) {
            std::fill(cache_table.begin(), cache_table.end(), EvalEntry{0, 0, 0, false});
            current_generation = 1;
        }
        active_entries = 0;
        cache_hits = 0;
        cache_misses = 0;
    }

    size_t size() const {
        return active_entries;
    }
};

static NativeEvaluator* global_evaluator = nullptr;

extern "C" {
    EXPORT_API void initialize_native_evaluator(size_t cache_size) {
        if (global_evaluator != nullptr) {
            delete global_evaluator;
        }
        global_evaluator = new NativeEvaluator(cache_size);
    }

    EXPORT_API int probe_native_eval_cache(uint64_t hash, bool turn, int* out_found) {
        if (!global_evaluator) {
            *out_found = 0;
            return 0;
        }
        return global_evaluator->probe(hash, turn, out_found);
    }

    EXPORT_API void store_native_eval_cache(uint64_t hash, bool turn, int score) {
        if (global_evaluator) {
            global_evaluator->store(hash, turn, score);
        }
    }

    EXPORT_API void clear_native_eval_cache() {
        if (global_evaluator) {
            global_evaluator->clear();
        }
    }

    EXPORT_API size_t get_native_eval_cache_size() {
        return global_evaluator ? global_evaluator->size() : 0;
    }

    EXPORT_API void get_native_eval_counters(long long* out_hits, long long* out_misses) {
        if (global_evaluator) {
            *out_hits = global_evaluator->cache_hits;
            *out_misses = global_evaluator->cache_misses;
        }
    }

    EXPORT_API int normalize_score_native(int score) {
        if (score >= MATE_THRESHOLD) return MATE_THRESHOLD - 1;
        if (score <= -MATE_THRESHOLD) return -MATE_THRESHOLD + 1;
        return score;
    }
}
