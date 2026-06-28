// search/time_manager.hpp
#pragma once
#include <chrono>
#include <algorithm>

class TimeManager {
private:
    std::chrono::steady_clock::time_point start_time;
    double allocated_time = 0.0; // Soft bound target budget
    double hard_limit     = 0.0; // Hard bound emergency abort
    bool fixed_depth      = false;
    bool fixed_time       = false;
    bool timer_active     = false;

public:
    int max_depth = 64;

    void reset() {
        allocated_time = 0.0;
        hard_limit = 0.0;
        fixed_depth = false;
        fixed_time = false;
        timer_active = false;
        max_depth = 64;
    }

    void set_fixed_depth(int depth) {
        fixed_depth = true;
        fixed_time = false;
        max_depth = depth;
    }

    void set_fixed_move_time(int milliseconds) {
        fixed_time = true;
        fixed_depth = false;
        allocated_time = milliseconds / 1000.0;
        hard_limit = allocated_time;
    }

    void allocate_time(int wtime_ms, int btime_ms, int inc_ms, int movestogo, bool is_white) {
        if (fixed_time) return;

        if (wtime_ms < 0 || btime_ms < 0) {
            allocated_time = 3.0;
            hard_limit = 15.0;
            return;
        }

        double remaining = is_white ? (wtime_ms / 1000.0) : (btime_ms / 1000.0);
        double increment = inc_ms / 1000.0;
        double moves_to_map = (movestogo > 0) ? (movestogo + 1.0) : 24.0;

        double base_allocation = remaining / moves_to_map;
        double allocation = base_allocation + (increment * 0.75);

        double soft_ceiling = remaining * 0.40;
        if (allocation > soft_ceiling) allocation = soft_ceiling;
        if (allocation < 0.015) allocation = std::max(0.015, std::min(increment, 0.500));

        allocated_time = allocation;

        if (movestogo > 0) {
            hard_limit = std::min(remaining * 0.80, allocation * 3.0);
        } else {
            hard_limit = std::min(remaining * 0.60, allocation * 2.5);
        }

        if (hard_limit < allocated_time) hard_limit = allocated_time;
    }

    void start_timer() {
        start_time = std::chrono::steady_clock::now();
        timer_active = true;
    }

    void stop() {
        timer_active = false;
    }

    inline double get_elapsed() const {
        if (!timer_active) return 0.0;
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = now - start_time;
        return diff.count();
    }

    inline bool should_stop_search() const {
        if (fixed_depth || !timer_active) return false;
        return get_elapsed() >= hard_limit;
    }

    inline bool check_soft_bound() const {
        if (fixed_depth || !timer_active) return false;
        return get_elapsed() >= allocated_time;
    }
};
