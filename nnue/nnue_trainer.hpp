// nnue/nnue_trainer.hpp
#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <cassert>
#include <limits>
#include <cstring>
#include <atomic>
#include <unordered_map>
#include <filesystem>

#include "loss.hpp"
#include "checkpoint.hpp"
#include "nnue_model.hpp"
#include "feature_transformer.hpp"
#include "replay_buffer.hpp"

namespace NNUE {

    struct TrainerConfig {
        float learning_rate = 1e-3f;
        float weight_decay = 1e-4f;
        float grad_clip_norm = 1.0f;
        int num_epochs = 30;
        size_t batch_size = 8192;
    };

    inline TrainerConfig GLOBAL_CONFIG;

    class AdamWOptimizer {
    public:
        float lr;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float weight_decay;
        uint32_t step_count = 0;

        std::vector<float> m_h1_w, v_h1_w;
        std::vector<float> m_ft_w, v_ft_w;
        std::vector<float> m_ft_b, v_ft_b;
        std::vector<float> m_h1_b, v_h1_b;
        std::vector<float> m_h2_w, v_h2_w;
        std::vector<float> m_h2_b, v_h2_b;
        std::vector<float> m_out_w, v_out_w;
        std::vector<float> m_out_b, v_out_b;

        AdamWOptimizer(float lr, float wd) : lr(lr), weight_decay(wd) {
            reset_momentum_states();
        }

        void reset_momentum_states() {
            m_ft_w.assign((FEATURE_DIM + 1) * ACCUMULATOR_DIM, 0.0f);
            v_ft_w.assign((FEATURE_DIM + 1) * ACCUMULATOR_DIM, 0.0f);
            m_ft_b.assign(ACCUMULATOR_DIM, 0.0f);
            v_ft_b.assign(ACCUMULATOR_DIM, 0.0f);

            m_h1_w.assign(HIDDEN1_DIM * INPUT_DIM, 0.0f);  v_h1_w.assign(HIDDEN1_DIM * INPUT_DIM, 0.0f);
            m_h1_b.assign(HIDDEN1_DIM, 0.0f);              v_h1_b.assign(HIDDEN1_DIM, 0.0f);

            m_h2_w.assign(HIDDEN2_DIM * HIDDEN1_DIM, 0.0f); v_h2_w.assign(HIDDEN2_DIM * HIDDEN1_DIM, 0.0f);
            m_h2_b.assign(HIDDEN2_DIM, 0.0f);              v_h2_b.assign(HIDDEN2_DIM, 0.0f);

            m_out_w.assign(OUTPUT_DIM * HIDDEN2_DIM, 0.0f); v_out_w.assign(OUTPUT_DIM * HIDDEN2_DIM, 0.0f);
            m_out_b.assign(OUTPUT_DIM, 0.0f);              v_out_b.assign(OUTPUT_DIM, 0.0f);
            
            step_count = 0;
        }

        void update_learning_rate(float new_lr) { lr = new_lr; }

        void step_core(float* weights, const float* gradients, size_t size, std::vector<float>& m, std::vector<float>& v) {
            if (size == 0) return;
            float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step_count + 1));
            float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step_count + 1));

            for (size_t i = 0; i < size; ++i) {
                weights[i] -= lr * weight_decay * weights[i];

                m[i] = beta1 * m[i] + (1.0f - beta1) * gradients[i];
                v[i] = beta2 * v[i] + (1.0f - beta2) * (gradients[i] * gradients[i]);

                float hat_m = m[i] / bias_correction1;
                float hat_v = v[i] / bias_correction2;
                weights[i] -= lr * hat_m / (std::sqrt(hat_v) + eps);
            }
        }

        void step_sparse_rows(float* weights,
                              const std::unordered_map<int, std::vector<float>>& gradients,
                              std::vector<float>& m,
                              std::vector<float>& v) {
            float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step_count + 1));
            float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step_count + 1));

            for (const auto& [row, row_grad] : gradients) {
                if (row <= 0 || row > FEATURE_DIM) continue;
                size_t base = static_cast<size_t>(row) * ACCUMULATOR_DIM;
                for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                    size_t idx = base + static_cast<size_t>(i);
                    weights[idx] -= lr * weight_decay * weights[idx];

                    float grad = row_grad[i];
                    m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
                    v[idx] = beta2 * v[idx] + (1.0f - beta2) * (grad * grad);

                    float hat_m = m[idx] / bias_correction1;
                    float hat_v = v[idx] / bias_correction2;
                    weights[idx] -= lr * hat_m / (std::sqrt(hat_v) + eps);
                }
            }
        }

        void update_all_layers(float* ft_w,
                               const std::unordered_map<int, std::vector<float>>& ft_w_g,
                               float* ft_b,
                               const float* ft_b_g,
                               const float* h1_w_g, const float* h1_b_g,
                               const float* h2_w_g, const float* h2_b_g,
                               const float* out_w_g, const float* out_b_g) {

            step_sparse_rows(ft_w, ft_w_g, m_ft_w, v_ft_w);
            step_core(ft_b, ft_b_g, ACCUMULATOR_DIM, m_ft_b, v_ft_b);

            step_core(reinterpret_cast<float*>(model_params.h1_weights), h1_w_g, HIDDEN1_DIM * INPUT_DIM, m_h1_w, v_h1_w);
            step_core(model_params.h1_bias, h1_b_g, HIDDEN1_DIM, m_h1_b, v_h1_b);

            step_core(reinterpret_cast<float*>(model_params.h2_weights), h2_w_g, HIDDEN2_DIM * HIDDEN1_DIM, m_h2_w, v_h2_w);
            step_core(model_params.h2_bias, h2_b_g, HIDDEN2_DIM, m_h2_b, v_h2_b);

            step_core(reinterpret_cast<float*>(model_params.out_weights), out_w_g, OUTPUT_DIM * HIDDEN2_DIM, m_out_w, v_out_w);
            step_core(model_params.out_bias, out_b_g, OUTPUT_DIM, m_out_b, v_out_b);

            step_count++;
        }
    };

    class NNUETrainer {
    private:
        ReplayBuffer& train_buffer;
        ReplayBuffer* val_buffer;
        NNUEWDLLoss loss_fn;
        std::unique_ptr<AdamWOptimizer> optimizer;
        std::vector<float> ft_embedding_params;
        std::vector<float> ft_bias_params;

        int start_epoch = 0;
        int current_epoch = 0;
        uint32_t global_step = 0;
        float best_val_loss = std::numeric_limits<float>::infinity();

        static size_t ft_row_offset(int shifted_feature) {
            return static_cast<size_t>(shifted_feature) * ACCUMULATOR_DIM;
        }

        static int shifted_feature_index(int raw_feature) {
            return raw_feature + 1;
        }

        void initialize_feature_shadow_from_globals() {
            ft_embedding_params.resize((FEATURE_DIM + 1) * ACCUMULATOR_DIM);
            ft_bias_params.resize(ACCUMULATOR_DIM);

            for (int row = 0; row <= FEATURE_DIM; ++row) {
                size_t base = ft_row_offset(row);
                for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                    ft_embedding_params[base + static_cast<size_t>(i)] =
                        static_cast<float>(transformer_weights.embedding_weights[row][i]);
                }
            }
            for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                ft_bias_params[i] = static_cast<float>(transformer_weights.bias[i]);
            }
        }

        void sync_feature_shadow_to_globals() const {
            for (int row = 0; row <= FEATURE_DIM; ++row) {
                size_t base = ft_row_offset(row);
                for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                    int rounded = static_cast<int>(std::round(ft_embedding_params[base + static_cast<size_t>(i)]));
                    rounded = std::clamp(rounded, -32768, 32767);
                    transformer_weights.embedding_weights[row][i] = static_cast<int16_t>(rounded);
                }
            }
            for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                int rounded = static_cast<int>(std::round(ft_bias_params[i]));
                rounded = std::clamp(rounded, -32768, 32767);
                transformer_weights.bias[i] = static_cast<int16_t>(rounded);
            }
        }

        void build_float_accumulator(const std::vector<int>& active_features, float* out) const {
            std::copy(ft_bias_params.begin(), ft_bias_params.end(), out);
            for (int raw_feature : active_features) {
                int shifted = shifted_feature_index(raw_feature);
                if (shifted <= 0 || shifted > FEATURE_DIM) continue;
                const float* row = ft_embedding_params.data() + ft_row_offset(shifted);
                for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                    out[i] += row[i];
                }
            }
        }

        void add_feature_gradients(std::unordered_map<int, std::vector<float>>& ft_grads,
                                   const std::vector<int>& active_features,
                                   const float* accumulator_grad) const {
            for (int raw_feature : active_features) {
                int shifted = shifted_feature_index(raw_feature);
                if (shifted <= 0 || shifted > FEATURE_DIM) continue;
                auto [it, inserted] = ft_grads.try_emplace(shifted, ACCUMULATOR_DIM, 0.0f);
                std::vector<float>& row_grad = it->second;
                for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                    row_grad[i] += accumulator_grad[i];
                }
            }
        }

        void clip_gradients(std::vector<float*>& blocks,
                            const std::vector<size_t>& sizes,
                            std::unordered_map<int, std::vector<float>>& ft_w_grads,
                            std::vector<float>& ft_b_grads) const {
            if (GLOBAL_CONFIG.grad_clip_norm <= 0.0f) return;

            double norm_sq = 0.0;
            for (size_t block = 0; block < blocks.size(); ++block) {
                const float* data = blocks[block];
                for (size_t i = 0; i < sizes[block]; ++i) {
                    norm_sq += static_cast<double>(data[i]) * static_cast<double>(data[i]);
                }
            }
            for (float grad : ft_b_grads) {
                norm_sq += static_cast<double>(grad) * static_cast<double>(grad);
            }
            for (const auto& [_, row_grad] : ft_w_grads) {
                for (float grad : row_grad) {
                    norm_sq += static_cast<double>(grad) * static_cast<double>(grad);
                }
            }

            const double norm = std::sqrt(norm_sq);
            if (norm <= static_cast<double>(GLOBAL_CONFIG.grad_clip_norm) || norm == 0.0) return;

            const float scale = static_cast<float>(static_cast<double>(GLOBAL_CONFIG.grad_clip_norm) / norm);
            for (size_t block = 0; block < blocks.size(); ++block) {
                float* data = blocks[block];
                for (size_t i = 0; i < sizes[block]; ++i) {
                    data[i] *= scale;
                }
            }
            for (float& grad : ft_b_grads) {
                grad *= scale;
            }
            for (auto& [_, row_grad] : ft_w_grads) {
                for (float& grad : row_grad) {
                    grad *= scale;
                }
            }
        }

        void capture_checkpoint_snapshots(ModelWeights& w_out, OptimizerState& opt_out) const {
            w_out.ft_embedding_weights = ft_embedding_params;
            w_out.ft_bias = ft_bias_params;
            w_out.h1_weights.assign(reinterpret_cast<float*>(model_params.h1_weights), reinterpret_cast<float*>(model_params.h1_weights) + (HIDDEN1_DIM * INPUT_DIM));
            w_out.h1_bias.assign(model_params.h1_bias, model_params.h1_bias + HIDDEN1_DIM);
            w_out.h2_weights.assign(reinterpret_cast<float*>(model_params.h2_weights), reinterpret_cast<float*>(model_params.h2_weights) + (HIDDEN2_DIM * HIDDEN1_DIM));
            w_out.h2_bias.assign(model_params.h2_bias, model_params.h2_bias + HIDDEN2_DIM);
            w_out.out_weights.assign(reinterpret_cast<float*>(model_params.out_weights), reinterpret_cast<float*>(model_params.out_weights) + (OUTPUT_DIM * HIDDEN2_DIM));
            w_out.out_bias.assign(model_params.out_bias, model_params.out_bias + OUTPUT_DIM);

            opt_out.step_count = optimizer->step_count;
            opt_out.m_ft_w = optimizer->m_ft_w; opt_out.v_ft_w = optimizer->v_ft_w;
            opt_out.m_ft_b = optimizer->m_ft_b; opt_out.v_ft_b = optimizer->v_ft_b;
            opt_out.m_h1_w = optimizer->m_h1_w; opt_out.v_h1_w = optimizer->v_h1_w;
            opt_out.m_h1_b = optimizer->m_h1_b; opt_out.v_h1_b = optimizer->v_h1_b;
            opt_out.m_h2_w = optimizer->m_h2_w; opt_out.v_h2_w = optimizer->v_h2_w;
            opt_out.m_h2_b = optimizer->m_h2_b; opt_out.v_h2_b = optimizer->v_h2_b;
            opt_out.m_out_w = optimizer->m_out_w; opt_out.v_out_w = optimizer->v_out_w;
            opt_out.m_out_b = optimizer->m_out_b; opt_out.v_out_b = optimizer->v_out_b;
        }

        void apply_checkpoint_snapshots(const ModelWeights& w_in, const OptimizerState& opt_in) {
            if (w_in.ft_embedding_weights.size() == (FEATURE_DIM + 1) * ACCUMULATOR_DIM &&
                w_in.ft_bias.size() == ACCUMULATOR_DIM) {
                ft_embedding_params = w_in.ft_embedding_weights;
                ft_bias_params = w_in.ft_bias;
                sync_feature_shadow_to_globals();
            }

            std::memcpy(model_params.h1_weights, w_in.h1_weights.data(), HIDDEN1_DIM * INPUT_DIM * sizeof(float));
            std::memcpy(model_params.h1_bias, w_in.h1_bias.data(), HIDDEN1_DIM * sizeof(float));
            std::memcpy(model_params.h2_weights, w_in.h2_weights.data(), HIDDEN2_DIM * HIDDEN1_DIM * sizeof(float));
            std::memcpy(model_params.h2_bias, w_in.h2_bias.data(), HIDDEN2_DIM * sizeof(float));
            std::memcpy(model_params.out_weights, w_in.out_weights.data(), OUTPUT_DIM * HIDDEN2_DIM * sizeof(float));
            std::memcpy(model_params.out_bias, w_in.out_bias.data(), OUTPUT_DIM * sizeof(float));

            optimizer->step_count = opt_in.step_count;
            if (opt_in.m_ft_w.size() == (FEATURE_DIM + 1) * ACCUMULATOR_DIM &&
                opt_in.v_ft_w.size() == (FEATURE_DIM + 1) * ACCUMULATOR_DIM &&
                opt_in.m_ft_b.size() == ACCUMULATOR_DIM &&
                opt_in.v_ft_b.size() == ACCUMULATOR_DIM) {
                optimizer->m_ft_w = opt_in.m_ft_w; optimizer->v_ft_w = opt_in.v_ft_w;
                optimizer->m_ft_b = opt_in.m_ft_b; optimizer->v_ft_b = opt_in.v_ft_b;
            }
            optimizer->m_h1_w = opt_in.m_h1_w; optimizer->v_h1_w = opt_in.v_h1_w;
            optimizer->m_h1_b = opt_in.m_h1_b; optimizer->v_h1_b = opt_in.v_h1_b;
            optimizer->m_h2_w = opt_in.m_h2_w; optimizer->v_h2_w = opt_in.v_h2_w;
            optimizer->m_h2_b = opt_in.m_h2_b; optimizer->v_h2_b = opt_in.v_h2_b;
            optimizer->m_out_w = opt_in.m_out_w; optimizer->v_out_w = opt_in.v_out_w;
            optimizer->m_out_b = opt_in.m_out_b; optimizer->v_out_b = opt_in.v_out_b;
        }

    public:
        explicit NNUETrainer(ReplayBuffer& train_buf, const NNUEWDLLoss& loss, ReplayBuffer* val_buf = nullptr)
        : train_buffer(train_buf), val_buffer(val_buf), loss_fn(loss) {
            initialize_feature_shadow_from_globals();
            optimizer = std::make_unique<AdamWOptimizer>(GLOBAL_CONFIG.learning_rate, GLOBAL_CONFIG.weight_decay);
        }

        void refresh_feature_transformer_from_globals() {
            initialize_feature_shadow_from_globals();
            optimizer->m_ft_w.assign((FEATURE_DIM + 1) * ACCUMULATOR_DIM, 0.0f);
            optimizer->v_ft_w.assign((FEATURE_DIM + 1) * ACCUMULATOR_DIM, 0.0f);
            optimizer->m_ft_b.assign(ACCUMULATOR_DIM, 0.0f);
            optimizer->v_ft_b.assign(ACCUMULATOR_DIM, 0.0f);
        }

        float get_scheduled_lr() const {
            float progress = static_cast<float>(current_epoch) / static_cast<float>(GLOBAL_CONFIG.num_epochs);
            
            // Establish a minimum learning rate floor (1% of your base learning rate)
            float min_lr = GLOBAL_CONFIG.learning_rate * 0.01f; 
            float base_lr = GLOBAL_CONFIG.learning_rate - min_lr;

            // Smoothly decay from base_lr down to min_lr using a cosine curve
            return min_lr + 0.5f * base_lr * (1.0f + std::cos(progress * 3.14159265f));
        }

        float train_epoch() {
            size_t total_samples = train_buffer.size();
            if (total_samples == 0) return 0.0f;

            size_t batch_size = GLOBAL_CONFIG.batch_size;
            auto samples = train_buffer.shuffled_snapshot();

            float epoch_loss_sum = 0.0f;
            size_t iterations = 0;

            std::vector<float> predictions(batch_size);
            std::vector<float> targets(batch_size);
            std::vector<float> weights(batch_size);
            std::vector<float> loss_grads(batch_size);

            // Global accumulator arrays for the optimizer
            std::vector<float> h1_w_grads(HIDDEN1_DIM * INPUT_DIM, 0.0f);
            std::vector<float> h1_b_grads(HIDDEN1_DIM, 0.0f);
            std::vector<float> h2_w_grads(HIDDEN2_DIM * HIDDEN1_DIM, 0.0f);
            std::vector<float> h2_b_grads(HIDDEN2_DIM, 0.0f);
            std::vector<float> out_w_grads(OUTPUT_DIM * HIDDEN2_DIM, 0.0f);
            std::vector<float> out_b_grads(OUTPUT_DIM, 0.0f);
            std::vector<float> ft_b_grads(ACCUMULATOR_DIM, 0.0f);
            std::unordered_map<int, std::vector<float>> ft_w_grads;

            std::vector<std::vector<float>> cache_oriented_input(batch_size, std::vector<float>(INPUT_DIM));
            std::vector<std::vector<float>> cache_layer1_out(batch_size, std::vector<float>(HIDDEN1_DIM));
            std::vector<std::vector<float>> cache_layer2_out(batch_size, std::vector<float>(HIDDEN2_DIM));

            for (size_t offset = 0; offset < total_samples; offset += batch_size) {
                size_t actual_size = std::min(batch_size, total_samples - offset);
                if (actual_size < 2) break;

                std::fill(h1_w_grads.begin(), h1_w_grads.end(), 0.0f); std::fill(h1_b_grads.begin(), h1_b_grads.end(), 0.0f);
                std::fill(h2_w_grads.begin(), h2_w_grads.end(), 0.0f); std::fill(h2_b_grads.begin(), h2_b_grads.end(), 0.0f);
                std::fill(out_w_grads.begin(), out_w_grads.end(), 0.0f); std::fill(out_b_grads.begin(), out_b_grads.end(), 0.0f);
                std::fill(ft_b_grads.begin(), ft_b_grads.end(), 0.0f);
                ft_w_grads.clear();

                // --- 1. FORWARD PASS ---
                #pragma omp parallel for schedule(static)
                for (size_t b = 0; b < actual_size; ++b) {
                    const auto& exp = samples[offset + b];
                    targets[b] = exp.target_value;
                    weights[b] = exp.weight;

                    float white_acc[ACCUMULATOR_DIM];
                    float black_acc[ACCUMULATOR_DIM];
                    build_float_accumulator(exp.white_features, white_acc);
                    build_float_accumulator(exp.black_features, black_acc);
                    float* oriented = cache_oriented_input[b].data();
                    if (exp.is_white_to_move) {
                        for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                            oriented[i] = white_acc[i];
                            oriented[i + ACCUMULATOR_DIM] = black_acc[i];
                        }
                    } else {
                        for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                            oriented[i] = black_acc[i];
                            oriented[i + ACCUMULATOR_DIM] = white_acc[i];
                        }
                    }

                    // Dense Layer 1 + Clip ReLU 
                    for (int i = 0; i < HIDDEN1_DIM; ++i) {
                        float sum = model_params.h1_bias[i];
                        for (int j = 0; j < INPUT_DIM; ++j) {
                            sum += oriented[j] * model_params.h1_weights[i][j];
                        }
                        cache_layer1_out[b][i] = std::clamp<float>(sum, 0.0f, 128.0f);
                    }

                    // Dense Layer 2 + Clip ReLU
                    for (int i = 0; i < HIDDEN2_DIM; ++i) {
                        float sum = model_params.h2_bias[i];
                        for (int j = 0; j < HIDDEN1_DIM; ++j) {
                            sum += cache_layer1_out[b][j] * model_params.h2_weights[i][j];
                        }
                        cache_layer2_out[b][i] = std::clamp<float>(sum, 0.0f, 128.0f);
                    }

                    // Output Layer
                    float output_score = model_params.out_bias[0];
                    for (int j = 0; j < HIDDEN2_DIM; ++j) {
                        output_score += cache_layer2_out[b][j] * model_params.out_weights[0][j];
                    }
                    predictions[b] = output_score;
                }

                predictions.resize(actual_size);
                targets.resize(actual_size);
                weights.resize(actual_size);

                float batch_loss = loss_fn.forward(predictions, targets, weights);
                loss_fn.backward(predictions, targets, weights, loss_grads);

                predictions.resize(batch_size);
                targets.resize(batch_size);
                weights.resize(batch_size);

                epoch_loss_sum += batch_loss;
                iterations++;

                // --- 2. BACKPROPAGATION PASS (THREAD-LOCAL REDUCTION) ---
                // 🎯 FIX: Removed extra 'inv_batch' scale factor to prevent vanishing underflow issues
                #pragma omp parallel
                {
                    // Every running hardware worker thread gets a zeroed private buffer workspace
                    std::vector<float> local_h1_w(HIDDEN1_DIM * INPUT_DIM, 0.0f);
                    std::vector<float> local_h1_b(HIDDEN1_DIM, 0.0f);
                    std::vector<float> local_h2_w(HIDDEN2_DIM * HIDDEN1_DIM, 0.0f);
                    std::vector<float> local_h2_b(HIDDEN2_DIM, 0.0f);
                    std::vector<float> local_out_w(OUTPUT_DIM * HIDDEN2_DIM, 0.0f);
                    std::vector<float> local_out_b(OUTPUT_DIM, 0.0f);
                    std::vector<float> local_ft_b(ACCUMULATOR_DIM, 0.0f);
                    std::unordered_map<int, std::vector<float>> local_ft_w;

                    #pragma omp for schedule(static)
                    for (size_t b = 0; b < actual_size; ++b) {
                        const auto& exp = samples[offset + b];
                        float dL_dout = loss_grads[b]; // Pure un-mutilated single position gradient component

                        local_out_b[0] += dL_dout;
                        float dL_dlayer2[HIDDEN2_DIM];
                        for (int j = 0; j < HIDDEN2_DIM; ++j) {
                            local_out_w[0 * HIDDEN2_DIM + j] += dL_dout * cache_layer2_out[b][j];
                            dL_dlayer2[j] = dL_dout * model_params.out_weights[0][j];
                        }

                        // Hidden Layer 2 Backpass
                        float dL_dlayer1[HIDDEN1_DIM] = {0.0f};
                        for (int i = 0; i < HIDDEN2_DIM; ++i) {
                            float act_out = cache_layer2_out[b][i];
                            if (act_out <= 0.0f || act_out >= 128.0f) continue;
                            
                            local_h2_b[i] += dL_dlayer2[i];
                            for (int j = 0; j < HIDDEN1_DIM; ++j) {
                                local_h2_w[i * HIDDEN1_DIM + j] += dL_dlayer2[i] * cache_layer1_out[b][j];
                                dL_dlayer1[j] += dL_dlayer2[i] * model_params.h2_weights[i][j];
                            }
                        }

                        // Hidden Layer 1 Backpass
                        const float* oriented = cache_oriented_input[b].data();
                        float dL_doriented[INPUT_DIM] = {0.0f};
                        for (int i = 0; i < HIDDEN1_DIM; ++i) {
                            float act_out = cache_layer1_out[b][i];
                            if (act_out <= 0.0f || act_out >= 128.0f) continue;

                            local_h1_b[i] += dL_dlayer1[i];
                            for (int j = 0; j < INPUT_DIM; ++j) {
                                local_h1_w[i * INPUT_DIM + j] += dL_dlayer1[i] * oriented[j];
                                dL_doriented[j] += dL_dlayer1[i] * model_params.h1_weights[i][j];
                            }
                        }

                        float white_acc_grad[ACCUMULATOR_DIM];
                        float black_acc_grad[ACCUMULATOR_DIM];
                        if (exp.is_white_to_move) {
                            std::copy(dL_doriented, dL_doriented + ACCUMULATOR_DIM, white_acc_grad);
                            std::copy(dL_doriented + ACCUMULATOR_DIM, dL_doriented + INPUT_DIM, black_acc_grad);
                        } else {
                            std::copy(dL_doriented, dL_doriented + ACCUMULATOR_DIM, black_acc_grad);
                            std::copy(dL_doriented + ACCUMULATOR_DIM, dL_doriented + INPUT_DIM, white_acc_grad);
                        }

                        for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                            local_ft_b[i] += white_acc_grad[i] + black_acc_grad[i];
                        }
                        add_feature_gradients(local_ft_w, exp.white_features, white_acc_grad);
                        add_feature_gradients(local_ft_w, exp.black_features, black_acc_grad);
                    }

                    // Critical section cleanly aggregates private thread workspaces back into the central block
                    #pragma omp critical
                    {
                        for (size_t i = 0; i < h1_w_grads.size(); ++i) h1_w_grads[i] += local_h1_w[i];
                        for (size_t i = 0; i < h1_b_grads.size(); ++i) h1_b_grads[i] += local_h1_b[i];
                        for (size_t i = 0; i < h2_w_grads.size(); ++i) h2_w_grads[i] += local_h2_w[i];
                        for (size_t i = 0; i < h2_b_grads.size(); ++i) h2_b_grads[i] += local_h2_b[i];
                        for (size_t i = 0; i < out_w_grads.size(); ++i) out_w_grads[i] += local_out_w[i];
                        for (size_t i = 0; i < out_b_grads.size(); ++i) out_b_grads[i] += local_out_b[i];
                        for (size_t i = 0; i < ft_b_grads.size(); ++i) ft_b_grads[i] += local_ft_b[i];
                        for (auto& [row, row_grad] : local_ft_w) {
                            auto [it, inserted] = ft_w_grads.try_emplace(row, ACCUMULATOR_DIM, 0.0f);
                            std::vector<float>& merged = it->second;
                            for (int i = 0; i < ACCUMULATOR_DIM; ++i) {
                                merged[i] += row_grad[i];
                            }
                        }
                    }
                }

                std::vector<float*> grad_blocks = {
                    h1_w_grads.data(), h1_b_grads.data(),
                    h2_w_grads.data(), h2_b_grads.data(),
                    out_w_grads.data(), out_b_grads.data()
                };
                std::vector<size_t> grad_sizes = {
                    h1_w_grads.size(), h1_b_grads.size(),
                    h2_w_grads.size(), h2_b_grads.size(),
                    out_w_grads.size(), out_b_grads.size()
                };
                clip_gradients(grad_blocks, grad_sizes, ft_w_grads, ft_b_grads);

                optimizer->update_all_layers(ft_embedding_params.data(), ft_w_grads,
                                             ft_bias_params.data(), ft_b_grads.data(),
                                             h1_w_grads.data(), h1_b_grads.data(),
                                             h2_w_grads.data(), h2_b_grads.data(),
                                             out_w_grads.data(), out_b_grads.data());
                global_step++;
            }

            return epoch_loss_sum / static_cast<float>(std::max(size_t(1), iterations));
        }

        float validate() {
            if (!val_buffer || val_buffer->size() == 0) return 0.0f;

            size_t total_samples = val_buffer->size();
            size_t batch_size = GLOBAL_CONFIG.batch_size;
            auto samples = val_buffer->shuffled_snapshot();

            float total_loss = 0.0f;
            size_t iterations = 0;

            std::vector<float> predictions(batch_size);
            std::vector<float> targets(batch_size);
            std::vector<float> weights(batch_size);

            for (size_t offset = 0; offset < total_samples; offset += batch_size) {
                size_t actual_size = std::min(batch_size, total_samples - offset);
                if (actual_size < 2) break;

                #pragma omp parallel for schedule(static)
                for (size_t b = 0; b < actual_size; ++b) {
                    const auto& exp = samples[offset + b];
                    targets[b] = exp.target_value;
                    weights[b] = exp.weight;

                    float white_acc[ACCUMULATOR_DIM];
                    float black_acc[ACCUMULATOR_DIM];
                    float oriented[INPUT_DIM];
                    build_float_accumulator(exp.white_features, white_acc);
                    build_float_accumulator(exp.black_features, black_acc);
                    const float* stm = exp.is_white_to_move ? white_acc : black_acc;
                    const float* nstm = exp.is_white_to_move ? black_acc : white_acc;
                    std::copy(stm, stm + ACCUMULATOR_DIM, oriented);
                    std::copy(nstm, nstm + ACCUMULATOR_DIM, oriented + ACCUMULATOR_DIM);
                    predictions[b] = NNUEModel::forward(oriented);
                }

                predictions.resize(actual_size);
                targets.resize(actual_size);
                weights.resize(actual_size);
                total_loss += loss_fn.forward(predictions, targets, weights);
                predictions.resize(batch_size);
                targets.resize(batch_size);
                weights.resize(batch_size);
                iterations++;
            }

            return total_loss / static_cast<float>(std::max(size_t(1), iterations));
        }

        void train(int epochs = GLOBAL_CONFIG.num_epochs, const std::string& checkpoint_dir = "checkpoints") {
            std::cout << " -> Training dense network and feature-transformer embeddings...\n";

            extern std::atomic<bool> g_interrupted;
            const std::filesystem::path checkpoint_root(checkpoint_dir);
            const std::string resume_checkpoint = (checkpoint_root / "resume.ckpt").string();
            const std::string best_checkpoint = (checkpoint_root / "best.ckpt").string();

            for (int epoch = start_epoch; epoch < epochs; ++epoch) {
                if (g_interrupted.load()) {
                    std::cout << "\n -> Training interrupted. Saving resume checkpoint...\n";
                    save_checkpoint(resume_checkpoint);
                    break;
                }

                current_epoch = epoch;
                float active_lr = get_scheduled_lr();
                optimizer->update_learning_rate(active_lr);

                float train_loss = train_epoch();
                float val_loss = validate();

                std::cout << "[Epoch " << (epoch + 1) << "/" << epochs << "] "
                << "train_loss=" << train_loss << " | val_loss=" << val_loss << "\n";

                if (val_buffer && val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    save_checkpoint(best_checkpoint);
                    std::cout << " -> New best checkpoint: " << best_checkpoint
                              << " | best_val_loss=" << best_val_loss << "\n";
                }

                const std::string epoch_checkpoint =
                    (checkpoint_root / ("epoch_" + std::to_string(epoch + 1) + ".ckpt")).string();
                save_checkpoint(resume_checkpoint);
            }
        }

        bool export_to_engine_binary(const std::string& path) const {
            sync_feature_shadow_to_globals();
            std::ofstream f(path, std::ios::binary);
            if (!f.is_open()) return false;

            size_t ft_w_bytes = sizeof(transformer_weights.embedding_weights);
            size_t ft_b_bytes = sizeof(transformer_weights.bias);
            f.write(reinterpret_cast<const char*>(transformer_weights.embedding_weights), ft_w_bytes);
            f.write(reinterpret_cast<const char*>(transformer_weights.bias), ft_b_bytes);

            size_t h1_w_bytes = sizeof(model_params.h1_weights);
            size_t h1_b_bytes = sizeof(model_params.h1_bias);
            f.write(reinterpret_cast<const char*>(model_params.h1_weights), h1_w_bytes);
            f.write(reinterpret_cast<const char*>(model_params.h1_bias), h1_b_bytes);

            size_t h2_w_bytes = sizeof(model_params.h2_weights);
            size_t h2_b_bytes = sizeof(model_params.h2_bias);
            f.write(reinterpret_cast<const char*>(model_params.h2_weights), h2_w_bytes);
            f.write(reinterpret_cast<const char*>(model_params.h2_bias), h2_b_bytes);

            size_t out_w_bytes = sizeof(model_params.out_weights);
            size_t out_b_bytes = sizeof(model_params.out_bias);
            f.write(reinterpret_cast<const char*>(model_params.out_weights), out_w_bytes);
            f.write(reinterpret_cast<const char*>(model_params.out_bias), out_b_bytes);

            return !f.bad();
        }

        bool save_checkpoint(const std::string& path) const {
            ModelWeights model_snapshot; OptimizerState optimizer_snapshot;
            capture_checkpoint_snapshots(model_snapshot, optimizer_snapshot);
            std::unordered_map<std::string, float> metadata = { {"best_val_loss", best_val_loss} };
            return CheckpointManager::save_checkpoint(path, current_epoch, global_step, model_snapshot, optimizer_snapshot, metadata);
        }

        bool resume(const std::string& path, bool fine_tuning = false) {
            CheckpointState loaded_state;
            if (!CheckpointManager::load_checkpoint(path, loaded_state)) return false;
            current_epoch = loaded_state.epoch;
            global_step = loaded_state.step;
            if (loaded_state.metadata.find("best_val_loss") != loaded_state.metadata.end()) {
                best_val_loss = loaded_state.metadata.at("best_val_loss");
            }
            
            if (fine_tuning) {
                if (loaded_state.model_state.ft_embedding_weights.size() == (FEATURE_DIM + 1) * ACCUMULATOR_DIM &&
                    loaded_state.model_state.ft_bias.size() == ACCUMULATOR_DIM) {
                    ft_embedding_params = loaded_state.model_state.ft_embedding_weights;
                    ft_bias_params = loaded_state.model_state.ft_bias;
                    sync_feature_shadow_to_globals();
                }
                std::memcpy(model_params.h1_weights, loaded_state.model_state.h1_weights.data(), HIDDEN1_DIM * INPUT_DIM * sizeof(float));
                std::memcpy(model_params.h1_bias, loaded_state.model_state.h1_bias.data(), HIDDEN1_DIM * sizeof(float));
                std::memcpy(model_params.h2_weights, loaded_state.model_state.h2_weights.data(), HIDDEN2_DIM * HIDDEN1_DIM * sizeof(float));
                std::memcpy(model_params.h2_bias, loaded_state.model_state.h2_bias.data(), HIDDEN2_DIM * sizeof(float));
                std::memcpy(model_params.out_weights, loaded_state.model_state.out_weights.data(), OUTPUT_DIM * HIDDEN2_DIM * sizeof(float));
                std::memcpy(model_params.out_bias, loaded_state.model_state.out_bias.data(), OUTPUT_DIM * sizeof(float));

                optimizer->reset_momentum_states();
                start_epoch = 0;
                global_step = 0;
                best_val_loss = std::numeric_limits<float>::infinity();
            } else {
                apply_checkpoint_snapshots(loaded_state.model_state, loaded_state.optimizer_state);
                start_epoch = current_epoch + 1;
            }
            return true;
        }

        bool restore_for_export(const std::string& path) {
            CheckpointState loaded_state;
            if (!CheckpointManager::load_checkpoint(path, loaded_state)) return false;
            apply_checkpoint_snapshots(loaded_state.model_state, loaded_state.optimizer_state);
            current_epoch = loaded_state.epoch;
            global_step = loaded_state.step;
            auto best_it = loaded_state.metadata.find("best_val_loss");
            if (best_it != loaded_state.metadata.end()) {
                best_val_loss = best_it->second;
            }
            return true;
        }

        void save_final(const std::string& path) const {
            save_checkpoint(path);
        }
    };

} // namespace NNUE
