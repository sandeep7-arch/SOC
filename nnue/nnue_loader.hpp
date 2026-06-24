#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include "feature_transformer.hpp"

namespace NNUE {

class NNUELoader {
public:
    // ==========================================================
    // Unified Binary Weight Loader
    // ==========================================================
    static bool load_model_file(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << " -> [Error] Failed to locate engine binary file: " << filepath << "\n";
            return false;
        }

        std::cerr << " -> Parsing NNUE binary weights: " << filepath << "\n";

        // 1. Read Feature Transformer parameters into globally accessible slots
        file.read(reinterpret_cast<char*>(transformer_weights.embedding_weights),
                  sizeof(transformer_weights.embedding_weights));
        file.read(reinterpret_cast<char*>(transformer_weights.bias),
                  sizeof(transformer_weights.bias));

        // 2. Read Dense Layer parameters matching NNUEModel expectations
        file.read(reinterpret_cast<char*>(model_params.h1_weights),
                  sizeof(model_params.h1_weights));
        file.read(reinterpret_cast<char*>(model_params.h1_bias),
                  sizeof(model_params.h1_bias));

        file.read(reinterpret_cast<char*>(model_params.h2_weights),
                  sizeof(model_params.h2_weights));
        file.read(reinterpret_cast<char*>(model_params.h2_bias),
                  sizeof(model_params.h2_bias));

        file.read(reinterpret_cast<char*>(model_params.out_weights),
                  sizeof(model_params.out_weights));
        file.read(reinterpret_cast<char*>(model_params.out_bias),
                  sizeof(model_params.out_bias));

        // 🎯 FIX: 'fail()' catches structural truncation, premature EOF, and stream breakdown issues
        if (!file) {
            std::cerr << " -> [Error] Binary file corrupted or missing parameter blocks (Premature EOF).\n";
            return false;
        }

        // Integrity verification: Check if there are unread trailing bytes
        char leftover;
        file.read(&leftover, 1);
        if (!file.eof()) {
            std::cerr << " -> [Warning] Extra trailing bytes detected inside binary file structure.\n";
        }

        NNUEModel::build_quantized_inference_weights();
        std::cerr << " -> Weights integrated successfully into active engine layers.\n";
        return true;
    }
};

} // namespace NNUE
