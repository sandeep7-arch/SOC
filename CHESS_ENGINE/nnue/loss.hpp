// nnue/loss.hpp
#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <numeric>
#include <cassert>
#include <algorithm>

namespace NNUE {

    class NNUEWDLLoss {
    public:
        enum class LossType { MSE, BCE };
        enum class Reduction { MEAN, SUM };

    private:
        float K;
        LossType loss_type;
        Reduction reduction;

        inline float sigmoid_wdl(float raw_score) const {
            return 1.0f / (1.0f + std::exp(-raw_score / K));
        }

        inline float binary_cross_entropy(float pred, float target) const {
            pred = std::max(1e-7f, std::min(pred, 1.0f - 1e-7f));
            return -(target * std::log(pred) + (1.0f - target) * std::log(1.0f - pred));
        }

    public:
        NNUEWDLLoss(float K = 400.0f, const std::string& type_str = "mse", const std::string& reduction_str = "mean")
        : K(K) {
            if (type_str == "mse") loss_type = LossType::MSE;
            else if (type_str == "bce") loss_type = LossType::BCE;
            else throw std::invalid_argument("Unsupported loss_type specified.");

            if (reduction_str == "sum") reduction = Reduction::SUM;
            else reduction = Reduction::MEAN;
        }

        // --- FORWARD PASS ---
        float forward(const std::vector<float>& predictions,
                      const std::vector<float>& targets,
                      const std::vector<float>& weights = {}) const {
                          assert(predictions.size() == targets.size());
                          size_t n = predictions.size();
                          if (n == 0) return 0.0f;

                          float total_loss = 0.0f;
                          for (size_t i = 0; i < n; ++i) {
                              float wdl_pred = sigmoid_wdl(predictions[i]);
                              float wdl_target = sigmoid_wdl(targets[i]);

                              float element_loss = (loss_type == LossType::MSE)
                              ? (wdl_pred - wdl_target) * (wdl_pred - wdl_target)
                              : binary_cross_entropy(wdl_pred, wdl_target);

                              if (!weights.empty()) element_loss *= weights[i];
                              total_loss += element_loss;
                          }

                          return (reduction == Reduction::SUM) ? total_loss : (total_loss / static_cast<float>(n));
                      }

        // --- OPTIMIZED BACKWARD PASS ---
        // 🎯 FIX: Added default initialization `= {}` to weights to prevent compilation mismatches
        // --- OPTIMIZED BACKWARD PASS INSIDE nnue/loss.hpp ---
void backward(const std::vector<float>& predictions,
              const std::vector<float>& targets,
              const std::vector<float>& weights,
              std::vector<float>& out_gradients) const {
                  size_t n = predictions.size();
                  out_gradients.resize(n);

                  float scale = (reduction == Reduction::MEAN) ? (1.0f / static_cast<float>(n)) : 1.0f;
                  bool has_weights = !weights.empty();

                  for (size_t i = 0; i < n; ++i) {
                      // Pred and target are both pushed to WDL space [0.0, 1.0]
                      float pred_wdl = sigmoid_wdl(predictions[i]);
                      float target_wdl = sigmoid_wdl(targets[i]); 

                      float grad = 0.0f;

                      if (loss_type == LossType::MSE) {
                          // Standard Derivative Chain Rule: 2 * (pred - target) * derivation
                          float sigmoid_deriv = pred_wdl * (1.0f - pred_wdl) / K;
                          grad = 2.0f * (pred_wdl - target_wdl) * sigmoid_deriv * scale;
                      }
                      else {
                        float epsilon = 1e-7f;
                        float safe_pred = std::clamp(pred_wdl, epsilon, 1.0f - epsilon);
                        
                        // Exact derivative calculation accounting for continuous probabilistic targets
                        float dL_dpred = (safe_pred - target_wdl) / (safe_pred * (1.0f - safe_pred));
                        float dpred_dx = (pred_wdl * (1.0f - pred_wdl)) / K;
                        
                        grad = dL_dpred * dpred_dx * scale;
                    }

                      if (has_weights) {
                          grad *= weights[i];
                      }
                      out_gradients[i] = grad;
                  }
              }
    };

} // namespace NNUE