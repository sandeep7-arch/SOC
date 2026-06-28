// search/checkpoint.hpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <memory>
#include <algorithm>
#include <filesystem>
#include <cstdint>
#include <cstdio>

namespace NNUE {

// Constants for cross-module validation
inline const std::string CHECKPOINT_VERSION = "3.0"; // Includes trainable feature-transformer state.

// Fully allocated container representing all trained network memory blocks.
struct ModelWeights {
    std::vector<float> ft_embedding_weights;
    std::vector<float> ft_bias;

    // Layer 1 (1024 -> 256)
    std::vector<float> h1_weights;
    std::vector<float> h1_bias;

    // Layer 2 (256 -> 32)
    std::vector<float> h2_weights;
    std::vector<float> h2_bias;

    // Output Layer (32 -> 1)
    std::vector<float> out_weights;
    std::vector<float> out_bias;
};

// Tracks AdamW momentum states synchronized across all deep network layers
struct OptimizerState {
    uint32_t step_count = 0;

    std::vector<float> m_ft_w;
    std::vector<float> v_ft_w;
    std::vector<float> m_ft_b;
    std::vector<float> v_ft_b;
    
    // Layer 1
    std::vector<float> m_h1_w;
    std::vector<float> v_h1_w;
    std::vector<float> m_h1_b;
    std::vector<float> v_h1_b;

    // Layer 2
    std::vector<float> m_h2_w;
    std::vector<float> v_h2_w;
    std::vector<float> m_h2_b;
    std::vector<float> v_h2_b;

    // Output Layer
    std::vector<float> m_out_w;
    std::vector<float> v_out_w;
    std::vector<float> m_out_b;
    std::vector<float> v_out_b;
};

struct CheckpointState {
    std::string version = CHECKPOINT_VERSION;
    int32_t epoch = 0;
    int32_t step = 0;
    ModelWeights model_state;
    OptimizerState optimizer_state;
    std::unordered_map<std::string, float> metadata;
};

class CheckpointManager {
private:
    // Helper routine to streamline vector binary stream serialization passes
    static void write_vector(std::ofstream& f, const std::vector<float>& vec) {
        size_t sz = vec.size();
        f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        if (sz > 0) {
            f.write(reinterpret_cast<const char*>(vec.data()), sz * sizeof(float));
        }
    }

    // Helper routine to streamline vector binary stream recovery deserialization passes
    static void read_vector(std::ifstream& f, std::vector<float>& vec) {
        size_t sz = 0;
        f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        vec.resize(sz);
        if (sz > 0) {
            f.read(reinterpret_cast<char*>(vec.data()), sz * sizeof(float));
        }
    }

public:
    // =========================================================================
    // SAVE CHECKPOINT
    // =========================================================================
    static bool save_checkpoint(const std::string& filepath,
                                 int32_t epoch,
                                 int32_t step,
                                 const ModelWeights& model,
                                 const OptimizerState& optimizer,
                                 const std::unordered_map<std::string, float>& metadata = {}) {

        namespace fs = std::filesystem;
        try {
            fs::path p(filepath);
            if (p.has_parent_path()) {
                fs::create_directories(p.parent_path());
            }
        } catch (...) {
            return false;
        }

        const std::string tmp_filepath = filepath + ".tmp";
        std::ofstream f(tmp_filepath, std::ios::binary);
        if (!f.is_open()) return false;

        // 1. Version validation header serialization
        size_t ver_len = CHECKPOINT_VERSION.size();
        f.write(reinterpret_cast<const char*>(&ver_len), sizeof(ver_len));
        f.write(CHECKPOINT_VERSION.data(), ver_len);

        // 2. Structural metadata scalars
        f.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
        f.write(reinterpret_cast<const char*>(&step), sizeof(step));

        // 3. Complete network model weights and biases.
        write_vector(f, model.ft_embedding_weights);
        write_vector(f, model.ft_bias);
        write_vector(f, model.h1_weights);
        write_vector(f, model.h1_bias);
        write_vector(f, model.h2_weights);
        write_vector(f, model.h2_bias);
        write_vector(f, model.out_weights);
        write_vector(f, model.out_bias);

        // 4. Complete Optimizer Track History Vectors Serializations
        f.write(reinterpret_cast<const char*>(&optimizer.step_count), sizeof(optimizer.step_count));
        write_vector(f, optimizer.m_ft_w);   write_vector(f, optimizer.v_ft_w);
        write_vector(f, optimizer.m_ft_b);   write_vector(f, optimizer.v_ft_b);
        write_vector(f, optimizer.m_h1_w);  write_vector(f, optimizer.v_h1_w);
        write_vector(f, optimizer.m_h1_b);  write_vector(f, optimizer.v_h1_b);
        write_vector(f, optimizer.m_h2_w);  write_vector(f, optimizer.v_h2_w);
        write_vector(f, optimizer.m_h2_b);  write_vector(f, optimizer.v_h2_b);
        write_vector(f, optimizer.m_out_w); write_vector(f, optimizer.v_out_w);
        write_vector(f, optimizer.m_out_b); write_vector(f, optimizer.v_out_b);

        // 5. Metric Map Serializations
        size_t meta_size = metadata.size();
        f.write(reinterpret_cast<const char*>(&meta_size), sizeof(meta_size));
        for (const auto& [key, value] : metadata) {
            size_t k_len = key.size();
            f.write(reinterpret_cast<const char*>(&k_len), sizeof(k_len));
            f.write(key.data(), k_len);
            f.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }

        const bool ok = f.good();
        f.close();
        if (!ok) {
            std::remove(tmp_filepath.c_str());
            return false;
        }
        try {
            std::filesystem::rename(tmp_filepath, filepath);
        } catch (...) {
            std::remove(tmp_filepath.c_str());
            return false;
        }
        return true;
    }

    // =========================================================================
    // LOAD CHECKPOINT
    // =========================================================================
    static bool load_checkpoint(const std::string& filepath, CheckpointState& out_state) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open()) return false;

        // 1. Unpack validation header versions
        size_t ver_len = 0;
        f.read(reinterpret_cast<char*>(&ver_len), sizeof(ver_len));
        out_state.version.resize(ver_len);
        f.read(&out_state.version[0], ver_len);
        if (!f || out_state.version != CHECKPOINT_VERSION) {
            std::cerr << " -> [Error] Unsupported checkpoint version: " << out_state.version
                      << " (expected " << CHECKPOINT_VERSION << ")\n";
            return false;
        }

        // 2. Unpack core structural scalars
        f.read(reinterpret_cast<char*>(&out_state.epoch), sizeof(out_state.epoch));
        f.read(reinterpret_cast<char*>(&out_state.step), sizeof(out_state.step));

        // 3. Unpack complete network parameter structures.
        read_vector(f, out_state.model_state.ft_embedding_weights);
        read_vector(f, out_state.model_state.ft_bias);
        read_vector(f, out_state.model_state.h1_weights);
        read_vector(f, out_state.model_state.h1_bias);
        read_vector(f, out_state.model_state.h2_weights);
        read_vector(f, out_state.model_state.h2_bias);
        read_vector(f, out_state.model_state.out_weights);
        read_vector(f, out_state.model_state.out_bias);

        // 4. Unpack optimization track running history states
        f.read(reinterpret_cast<char*>(&out_state.optimizer_state.step_count), sizeof(out_state.optimizer_state.step_count));
        read_vector(f, out_state.optimizer_state.m_ft_w);   read_vector(f, out_state.optimizer_state.v_ft_w);
        read_vector(f, out_state.optimizer_state.m_ft_b);   read_vector(f, out_state.optimizer_state.v_ft_b);
        read_vector(f, out_state.optimizer_state.m_h1_w);  read_vector(f, out_state.optimizer_state.v_h1_w);
        read_vector(f, out_state.optimizer_state.m_h1_b);  read_vector(f, out_state.optimizer_state.v_h1_b);
        read_vector(f, out_state.optimizer_state.m_h2_w);  read_vector(f, out_state.optimizer_state.v_h2_w);
        read_vector(f, out_state.optimizer_state.m_h2_b);  read_vector(f, out_state.optimizer_state.v_h2_b);
        read_vector(f, out_state.optimizer_state.m_out_w); read_vector(f, out_state.optimizer_state.v_out_w);
        read_vector(f, out_state.optimizer_state.m_out_b); read_vector(f, out_state.optimizer_state.v_out_b);

        // 5. Unpack floating-point training metadata analytics maps
        size_t meta_size = 0;
        f.read(reinterpret_cast<char*>(&meta_size), sizeof(meta_size));
        out_state.metadata.clear();
        for (size_t i = 0; i < meta_size; ++i) {
            size_t k_len = 0;
            f.read(reinterpret_cast<char*>(&k_len), sizeof(k_len));
            std::string key(k_len, ' ');
            f.read(&key[0], k_len);
            float value = 0.0f;
            f.read(reinterpret_cast<char*>(&value), sizeof(value));
            out_state.metadata[key] = value;
        }

        return !f.bad();
    }

    // =========================================================================
    // CHECKPOINT INFO INSPECTOR
    // =========================================================================
    static bool checkpoint_info(const std::string& filepath,
                                 std::string& out_version,
                                 int32_t& out_epoch,
                                 int32_t& out_step) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open()) return false;

        size_t ver_len = 0;
        f.read(reinterpret_cast<char*>(&ver_len), sizeof(ver_len));
        out_version.resize(ver_len);
        f.read(&out_version[0], ver_len);

        f.read(reinterpret_cast<char*>(&out_epoch), sizeof(out_epoch));
        f.read(reinterpret_cast<char*>(&out_step), sizeof(out_step));

        return !f.bad();
    }

    // =========================================================================
    // AUTOMATIC FILE CONTEXT FINDER
    // =========================================================================
    static std::string latest_checkpoint(const std::string& directory) {
        namespace fs = std::filesystem;
        if (!fs::exists(directory) || !fs::is_directory(directory)) {
            return "";
        }

        fs::path resume_file = fs::path(directory) / "resume.ckpt";
        if (fs::exists(resume_file) && fs::is_regular_file(resume_file)) {
            return resume_file.string();
        }

        return "";
    }
};

} // namespace NNUE
