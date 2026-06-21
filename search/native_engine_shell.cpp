// search/native_engine_shell.cpp
#include <cstddef>
#include <algorithm>
#include <cstring>

// 🌟 FIX: Force visibility on Linux/macOS explicitly so ctypes can bind seamlessly
#if defined(_WIN32)
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

// Forward declare the deep search engine functions from native_search.cpp
extern "C" {
    bool init_engine_native(const char* model_path, size_t tt_size, size_t cache_size);
    bool search_position_native(const char* fen_str, int max_depth,
                                int wtime, int btime, int winc, int binc, int movestogo,
                                char* out_best_move_uci);
}

extern "C" {

    // 🎯 ONE-TIME HANDSHAKE: Boots NNUE weights and handles cache memory layout allocation
    EXPORT_API bool load_nnue_model_and_caches(const char* model_path, size_t tt_size, size_t eval_cache_size) {
        return init_engine_native(model_path, tt_size, eval_cache_size);
    }

    // 🎯 SEARCH GATEWAY: Translates basic incoming wrappers into the iterative-deepening engine execution
    EXPORT_API bool load_fen_to_native_search(const char* fen_str, int depth, double allocated_time_ms, char* out_best_move_uci) {
        if (!fen_str) return false;

        // Map generic parameters into standard structural side-to-move constraints
        int requested_ms = std::max(1, static_cast<int>(allocated_time_ms));
        int pseudo_wtime = requested_ms * 24;
        int pseudo_btime = requested_ms * 24;

        // Execute the full Alpha-Beta/NNUE iterative zero-allocation search pipeline
        bool search_success = search_position_native(
            fen_str,
            depth,
            pseudo_wtime,
            pseudo_btime,
            0,    // winc
            0,    // binc
            0,    // movestogo
            out_best_move_uci
        );

        return search_success;
    }
}