// nnue/main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <csignal>
#include <atomic>

namespace fs = std::filesystem;

namespace NNUE {
    std::atomic<bool> g_interrupted{false};
}

void signal_handler(int signal) {
    if (signal == SIGINT) {
        NNUE::g_interrupted.store(true);
    }
}

#include "loss.hpp"
#include "checkpoint.hpp"
#include "nnue_trainer.hpp"
#include "replay_buffer.hpp"

int main() {
    std::signal(SIGINT, signal_handler);

    std::cout << "====================================================================\n";
    std::cout << "    🚀 INITIALIZING PRODUCTION-GRADE C++ NNUE TRAINING PIPELINE     \n";
    std::cout << "====================================================================\n\n";

    fs::path data_dir("data");
    fs::path pgn_dir = data_dir / "PGN_files";
    fs::path fen_path = data_dir / "fen_files" / "chessData.fen";
    std::string checkpoint_dir = "checkpoints";
    std::string export_dir = "exports";

    if (!fs::exists(checkpoint_dir)) fs::create_directories(checkpoint_dir);
    if (!fs::exists(export_dir)) fs::create_directories(export_dir);

    // 🎯 FIX: Balanced hyperparameters to ensure steady, clean gradient drops
    NNUE::GLOBAL_CONFIG.learning_rate = 1e-4f;   
    NNUE::GLOBAL_CONFIG.weight_decay = 1e-5f;    
    NNUE::GLOBAL_CONFIG.batch_size = 8192;      
    NNUE::GLOBAL_CONFIG.num_epochs = 15;

    std::cout << "[1/5] Ingesting source files from drive space...\n";
    
    // 🎯 FIX: Allocate space for your true dataset volume!
    NNUE::ReplayBuffer full_dataset(13000000); 
    const size_t game_limit = 12500000;

    if (fs::exists(fen_path)) {
        std::cout << " -> Appending raw evaluation snapshot records from: " << fen_path.string() << "\n";
        full_dataset.load_fen_file(fen_path.string(), game_limit);
    }

    if (full_dataset.size() == 0) {
        std::cout << "⚠️ Warning: No training data found on drive. Generating synthetic testing samples.\n";
        NNUE::Experience mock_exp;
        mock_exp.target_value = 0.5f;
        mock_exp.weight = 1.0f;
        mock_exp.is_white_to_move = true;
        mock_exp.white_features = {14, 55}; 
        mock_exp.black_features = {12, 43};
        mock_exp.game_id = "synthetic_mock";
        mock_exp.ply = 1;
        
        full_dataset.add_raw(std::move(mock_exp));
    }

    std::cout << " -> Success! Compiled " << full_dataset.size() << " unique training positions.\n";

    size_t total_items = full_dataset.size();
    size_t train_size = static_cast<size_t>(0.9f * total_items);
    size_t val_size = total_items - train_size;

    NNUE::ReplayBuffer train_buffer(train_size + 10);
    NNUE::ReplayBuffer val_buffer(val_size + 10);

    auto all_samples = full_dataset.shuffled_snapshot();
    for (size_t i = 0; i < all_samples.size(); ++i) {
        if (i < train_size) train_buffer.add_raw(std::move(all_samples[i]));
        else val_buffer.add_raw(std::move(all_samples[i]));
    }
    std::cout << " -> Data allocation: " << train_size << " training items | " << val_size << " validation items.\n";

    NNUE::NNUEWDLLoss loss_function(400.0f, "mse", "mean");

    std::cout << "\n[2/5] Building dual-perspective network layers...\n";
    NNUE::FeatureTransformer::initialize_weights_from_scratch();

    std::cout << "\n[3/5] Initializing execution trainer matrix...\n";
    NNUE::NNUETrainer trainer(train_buffer, loss_function, &val_buffer);

    std::cout << "\n[4/5] Searching for existing checkpoint intervals...\n";
    std::string newest_checkpoint = NNUE::CheckpointManager::latest_checkpoint(checkpoint_dir);

    if (!newest_checkpoint.empty()) {
        std::cout << " -> Active backup located! Restoring variables from: " << newest_checkpoint << "\n";
        trainer.resume(newest_checkpoint, false);
    } else {
        std::cout << " -> No active checkpoints located. Commencing baseline training.\n";
        NNUE::NNUEModel::initialize_weights_from_scratch();
    }

    std::cout << "\n[5/5] Igniting backpropagation loop. Tuning parameters active...\n";
    std::cout << "--------------------------------------------------------------------\n";

    trainer.train(NNUE::GLOBAL_CONFIG.num_epochs, checkpoint_dir);

    std::cout << "--------------------------------------------------------------------\n";

    fs::path final_output_path = fs::path(checkpoint_dir) / "nnue_brain.ckpt";
    trainer.save_final(final_output_path.string());

    std::cout << "\n[Post-Processing] Running model export routines...\n";
    std::string engine_ready_file = (fs::path(export_dir) / "nnue_inference.bin").string();

    if (trainer.export_to_engine_binary(engine_ready_file)) {
        std::cout << " -> Production engine file exported directly to: " << engine_ready_file << "\n";
    } else {
        std::cout << " ⚠️ Warning: Native engine export failed!\n";
    }

    return 0;
}
