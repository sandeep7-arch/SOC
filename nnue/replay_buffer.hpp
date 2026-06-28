// nnue/replay_buffer.hpp
#pragma once
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include "board.hpp"
#include "fen.hpp"
#include "feature_encoder.hpp"  

namespace NNUE {

    enum class EvalPerspective {
        White,
        SideToMove
    };

    struct Experience {
        std::vector<int> white_features; 
        std::vector<int> black_features; 
        float target_value;
        bool is_white_to_move;
        float weight;
        std::string game_id;
        int32_t ply;
    };

    class ReplayBuffer {
    private:
        size_t capacity;
        size_t head = 0;
        size_t current_size = 0;
        std::vector<Experience> buffer;
        mutable std::mt19937 rng{std::random_device{}()};

    public:
        // Constructor correctly reserves total capacity bounds
        explicit ReplayBuffer(size_t capacity = 1000000) : capacity(capacity) {
            buffer.resize(capacity);
        }

        void add(const std::vector<int>& white_features,
                 const std::vector<int>& black_features,
                 float target_value,
                 bool is_white_to_move,
                 float weight = 1.0f,
                 const std::string& game_id = "",
                 int32_t ply = 0) {

            buffer[head] = Experience{
                white_features,
                black_features,
                target_value,
                is_white_to_move,
                weight,
                game_id,
                ply
            };

            head = (head + 1) % capacity;
            if (current_size < capacity) {
                current_size++;
            }
        }

        void add_raw(Experience&& exp) {
            buffer[head] = std::move(exp);
            head = (head + 1) % capacity;
            if (current_size < capacity) {
                current_size++;
            }
        }

        void add_raw(const Experience& exp) {
            buffer[head] = exp;
            head = (head + 1) % capacity;
            if (current_size < capacity) {
                current_size++;
            }
        }

        void add_game(const std::vector<std::tuple<std::vector<int>, std::vector<int>, bool>>& game_positions,
                       float result,
                       const std::string& game_id = "") {
             size_t total_positions = game_positions.size();
             if (total_positions == 0) return;

             for (size_t ply = 0; ply < total_positions; ++ply) {
                 const auto& [w_feats, b_feats, is_w_move] = game_positions[ply];
                 float progress = static_cast<float>(ply + 1) / static_cast<float>(total_positions);
                 float weight = 0.25f + 0.75f * progress;
                 add(w_feats, b_feats, result, is_w_move, weight, game_id, static_cast<int32_t>(ply));
             }
        }

        // 🎯 FIX: Converted limit from 'int' to 'size_t' to support 12.5M+ rows cleanly without integer truncation issues
        void load_fen_file(const std::string& filepath,
                           size_t limit = 13000000,
                           EvalPerspective eval_perspective = EvalPerspective::White) {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                std::cerr << " -> [Error] Failed to open FEN file: " << filepath << "\n";
                return;
            }

            std::cout << " -> Streaming paired evaluations from: " << filepath << "\n";
            std::cout << " -> Eval perspective: "
                      << (eval_perspective == EvalPerspective::White ? "white" : "side-to-move")
                      << "\n";
            std::string line;
            size_t count = 0;
            size_t malformed_count = 0;
            size_t parse_fail_count = 0;
            size_t eval_fail_count = 0;
            size_t clamped_count = 0;
            size_t weighted_count = 0;
            size_t white_to_move_count = 0;
            size_t black_to_move_count = 0;
            double target_sum = 0.0;
            double abs_target_sum = 0.0;
            float min_target = std::numeric_limits<float>::infinity();
            float max_target = -std::numeric_limits<float>::infinity();

            std::vector<int> w_feats;
            std::vector<int> b_feats;

            while (std::getline(file, line)) {
                if (count >= limit) {
                    std::cout << " -> [Info] Reached loading cap of " << limit << " entries.\n";
                    break;
                }
                if (line.empty()) continue;

                size_t last_comma_pos = line.find_last_of(',');
                if (last_comma_pos == std::string::npos) {
                    malformed_count++;
                    continue;
                }

                size_t prev_comma_pos = last_comma_pos == 0
                    ? std::string::npos
                    : line.find_last_of(',', last_comma_pos - 1);
                std::string fen_str;
                std::string eval_str;
                std::string weight_str;
                if (prev_comma_pos == std::string::npos) {
                    fen_str = line.substr(0, last_comma_pos);
                    eval_str = line.substr(last_comma_pos + 1);
                } else {
                    fen_str = line.substr(0, prev_comma_pos);
                    eval_str = line.substr(prev_comma_pos + 1, last_comma_pos - prev_comma_pos - 1);
                    weight_str = line.substr(last_comma_pos + 1);
                }

                try {
                    NativeBoard board;
                    if (!FenParser::parse_fen(fen_str, board)) {
                        parse_fail_count++;
                        continue;
                    }

                    float cp = std::stof(eval_str);
                    float target_cp = std::clamp(cp, -4000.0f, 4000.0f);
                    float sample_weight = 1.0f;
                    if (!weight_str.empty()) {
                        sample_weight = std::clamp(std::stof(weight_str), 0.05f, 4.0f);
                        weighted_count++;
                    }
                    if (target_cp != cp) {
                        clamped_count++;
                    }

                    if (eval_perspective == EvalPerspective::White &&
                        board.get_side_to_move() == BLACK) {
                        target_cp = -target_cp;
                    }
                    if (board.get_side_to_move() == WHITE) {
                        white_to_move_count++;
                    } else {
                        black_to_move_count++;
                    }
                    target_sum += target_cp;
                    abs_target_sum += std::abs(target_cp);
                    min_target = std::min(min_target, target_cp);
                    max_target = std::max(max_target, target_cp);

                    w_feats.clear();
                    b_feats.clear();
                    FeatureEncoder::active_features(board, w_feats, b_feats);

                    add(w_feats, b_feats, target_cp, (board.get_side_to_move() == WHITE), sample_weight, "csv_import", static_cast<int32_t>(count));
                    count++;
                } catch (const std::invalid_argument&) {
                    eval_fail_count++;
                    continue;
                } catch (const std::out_of_range&) {
                    eval_fail_count++;
                    continue;
                }
            }
            std::cout << " -> Ingestion complete. Total buffer capacity used: " << current_size << "\n";
            if (count > 0) {
                std::cout << " -> Label stats: min=" << min_target
                          << " max=" << max_target
                          << " mean=" << (target_sum / static_cast<double>(count))
                          << " mean_abs=" << (abs_target_sum / static_cast<double>(count))
                          << " weighted_rows=" << weighted_count
                          << " | stm white=" << white_to_move_count
                          << " black=" << black_to_move_count << "\n";
            }
            if (malformed_count || parse_fail_count || eval_fail_count || clamped_count) {
                std::cout << " -> Skipped/clamped rows: malformed=" << malformed_count
                          << " fen_parse=" << parse_fail_count
                          << " eval_parse=" << eval_fail_count
                          << " clamped=" << clamped_count << "\n";
            }
        }

        std::vector<Experience> sample(size_t batch_size) const {
            size_t actual_batch = std::min(batch_size, current_size);
            std::vector<Experience> results;
            results.reserve(actual_batch);

            if (actual_batch == 0) return results;
            std::uniform_int_distribution<size_t> dist(0, current_size - 1);

            for (size_t i = 0; i < actual_batch; ++i) {
                results.push_back(buffer[dist(rng)]);
            }
            return results;
        }

        std::vector<Experience> shuffled_snapshot() const {
            std::vector<Experience> results;
            results.reserve(current_size);
            for (size_t i = 0; i < current_size; ++i) {
                results.push_back(buffer[i]);
            }
            std::shuffle(results.begin(), results.end(), rng);
            return results;
        }

        void clear() { head = 0; current_size = 0; }
        size_t size() const { return current_size; }
        bool is_empty() const { return current_size == 0; }

        bool save(const std::string& filepath) const {
            std::ofstream f(filepath, std::ios::binary);
            if (!f.is_open()) return false;
            f.write(reinterpret_cast<const char*>(&capacity), sizeof(capacity));
            f.write(reinterpret_cast<const char*>(&current_size), sizeof(current_size));
            f.write(reinterpret_cast<const char*>(&head), sizeof(head));

            for (size_t i = 0; i < current_size; ++i) {
                const auto& exp = buffer[i];
                size_t w_len = exp.white_features.size();
                f.write(reinterpret_cast<const char*>(&w_len), sizeof(w_len));
                if (w_len > 0) f.write(reinterpret_cast<const char*>(exp.white_features.data()), w_len * sizeof(int));

                size_t b_len = exp.black_features.size();
                f.write(reinterpret_cast<const char*>(&b_len), sizeof(b_len));
                if (b_len > 0) f.write(reinterpret_cast<const char*>(exp.black_features.data()), b_len * sizeof(int));

                f.write(reinterpret_cast<const char*>(&exp.target_value), sizeof(exp.target_value));
                f.write(reinterpret_cast<const char*>(&exp.is_white_to_move), sizeof(exp.is_white_to_move));
                f.write(reinterpret_cast<const char*>(&exp.weight), sizeof(exp.weight));
                f.write(reinterpret_cast<const char*>(&exp.ply), sizeof(exp.ply));

                size_t str_len = exp.game_id.size();
                f.write(reinterpret_cast<const char*>(&str_len), sizeof(str_len));
                if (str_len > 0) f.write(exp.game_id.data(), str_len);
            }
            return f.good();
        }

        bool load(const std::string& filepath) {
            std::ifstream f(filepath, std::ios::binary);
            if (!f.is_open()) return false;
            f.read(reinterpret_cast<char*>(&capacity), sizeof(capacity));
            f.read(reinterpret_cast<char*>(&current_size), sizeof(current_size));
            f.read(reinterpret_cast<char*>(&head), sizeof(head));

            buffer.resize(capacity);
            for (size_t i = 0; i < current_size; ++i) {
                Experience exp;
                size_t w_len = 0;
                f.read(reinterpret_cast<char*>(&w_len), sizeof(w_len));
                exp.white_features.resize(w_len);
                if (w_len > 0) f.read(reinterpret_cast<char*>(exp.white_features.data()), w_len * sizeof(int));

                size_t b_len = 0;
                f.read(reinterpret_cast<char*>(&b_len), sizeof(b_len));
                exp.black_features.resize(b_len);
                if (b_len > 0) f.read(reinterpret_cast<char*>(exp.black_features.data()), b_len * sizeof(int));

                f.read(reinterpret_cast<char*>(&exp.target_value), sizeof(exp.target_value));
                f.read(reinterpret_cast<char*>(&exp.is_white_to_move), sizeof(exp.is_white_to_move));
                f.read(reinterpret_cast<char*>(&exp.weight), sizeof(exp.weight));
                f.read(reinterpret_cast<char*>(&exp.ply), sizeof(exp.ply));

                size_t str_len = 0;
                f.read(reinterpret_cast<char*>(&str_len), sizeof(str_len));
                if (str_len > 0) {
                    exp.game_id.resize(str_len);
                    f.read(&exp.game_id[0], str_len);
                }
                buffer[i] = std::move(exp);
            }
            return !f.bad();
        }
    };

} // namespace NNUE
