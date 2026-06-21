// nnue/nnue_evaluator.hpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cassert>

#include "board.hpp"
#include "feature_encoder.hpp"
#include "feature_transformer.hpp"
#include "nnue_model.hpp"
#include "nnue_loader.hpp"

namespace NNUE {

    class NNUEEvaluator {
    private:
        bool use_cache;
        std::unordered_map<uint64_t, int> evaluation_cache;

        inline uint64_t get_board_hash(const NativeBoard& board) const {
            return board.get_hash();
        }

    public:
        NNUEEvaluator(const std::string& model_path = "", bool use_cache = false)
        : use_cache(use_cache) {
            if (!model_path.empty()) {
                load_model(model_path);
            }
        }

        void load_model(const std::string& filepath) {
            if (!NNUELoader::load_model_file(filepath)) {
                std::cerr << " -> [Fatal] Incomplete weight mapping during evaluator construction.\n";
            }
        }

        // ==========================================================
        // STATELESS FULL RECONSTRUCTION EVAL (Optimized for tight loop search)
        // ==========================================================
        int evaluate(const NativeBoard& board) {
            uint64_t zobrist = get_board_hash(board);
            
            if (use_cache) {
                auto it = evaluation_cache.find(zobrist);
                if (it != evaluation_cache.end()) {
                    return it->second;
                }
            }

            // Thread-local variables to completely bypass heap allocation overhead in tight search loops
            static thread_local std::vector<int> w_features;
            static thread_local std::vector<int> b_features;

            // 1. Generate active coordinate indices matching the network's internal HalfKP input dimensions
            FeatureEncoder::active_features(board, w_features, b_features);

            // 2. Build temporary continuous accumulators from scratch
            Accumulator temp_w, temp_b;
            FeatureTransformer::forward(w_features, temp_w);
            FeatureTransformer::forward(b_features, temp_b);

            // 3. Feed the raw underlying int16_t vectors into the network layers
            bool is_white_to_move = (board.get_side_to_move() == WHITE);
            int score = NNUEModel::evaluate_cp(temp_w.v, temp_b.v, is_white_to_move, 1.0f);

            if (use_cache) {
                evaluation_cache[zobrist] = score;
            }

            return score;
        }

        // ==========================================================
        // POSITION BATCHING
        // ==========================================================
        std::vector<int> batch_evaluate(const std::vector<NativeBoard>& boards) {
            std::vector<int> results;
            results.reserve(boards.size());

            for (const auto& board : boards) {
                results.push_back(evaluate(board));
            }

            return results;
        }

        void clear_cache() { evaluation_cache.clear(); }
        size_t cache_size() const { return evaluation_cache.size(); }
    };

} // namespace NNUE
