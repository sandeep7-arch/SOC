// nnue/main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <csignal>
#include <atomic>
#include <cstdlib>

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

enum class TrainingMode {
    ContinueCheckpoint,
    FineTuneInference,
    TrainScratch
};

struct TrainOptions {
    fs::path fen_path = fs::path("data") / "fen_files" / "chessData.fen";
    std::string checkpoint_dir = (fs::path("checkpoints") / "resume").string();
    std::string export_dir = "exports";
    fs::path base_model_path = fs::path("exports") / "nnue_inference_m.bin";
    fs::path output_model_path;
    fs::path val_fen_path;
    TrainingMode mode = TrainingMode::ContinueCheckpoint;
    NNUE::EvalPerspective eval_perspective = NNUE::EvalPerspective::White;
    bool base_model_path_set = false;
    bool output_model_path_set = false;
    bool val_fen_path_set = false;
    bool allow_synthetic = true;
    size_t game_limit = 12500000;
    size_t val_limit = 1250000;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --mode continue|finetune|scratch\n"
        << "  --fen PATH\n"
        << "  --val-fen PATH\n"
        << "  --base-model PATH\n"
        << "  --output-model PATH\n"
        << "  --checkpoint-dir PATH\n"
        << "  --export-dir PATH\n"
        << "  --eval-perspective white|stm\n"
        << "  --lr VALUE\n"
        << "  --epochs N\n"
        << "  --batch-size N\n"
        << "  --limit N\n"
        << "  --val-limit N\n"
        << "  --no-synthetic\n";
}

bool parse_options(int argc, char** argv, TrainOptions& options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--mode") {
            const char* value = require_value(arg);
            if (!value) return false;
            std::string mode = value;
            if (mode == "continue") options.mode = TrainingMode::ContinueCheckpoint;
            else if (mode == "finetune") options.mode = TrainingMode::FineTuneInference;
            else if (mode == "scratch") options.mode = TrainingMode::TrainScratch;
            else {
                std::cerr << "Unsupported mode: " << mode << "\n";
                return false;
            }
        } else if (arg == "--fen") {
            const char* value = require_value(arg);
            if (!value) return false;
            options.fen_path = value;
        } else if (arg == "--val-fen") {
            const char* value = require_value(arg);
            if (!value) return false;
            options.val_fen_path = value;
            options.val_fen_path_set = true;
        } else if (arg == "--base-model") {
            const char* value = require_value(arg);
            if (!value) return false;
            options.base_model_path = value;
            options.base_model_path_set = true;
        } else if (arg == "--output-model") {
            const char* value = require_value(arg);
            if (!value) return false;
            options.output_model_path = value;
            options.output_model_path_set = true;
        } else if (arg == "--checkpoint-dir") {
            const char* value = require_value(arg);
            if (!value) return false;
            options.checkpoint_dir = value;
        } else if (arg == "--export-dir") {
            const char* value = require_value(arg);
            if (!value) return false;
            options.export_dir = value;
        } else if (arg == "--eval-perspective") {
            const char* value = require_value(arg);
            if (!value) return false;
            std::string perspective = value;
            if (perspective == "white") {
                options.eval_perspective = NNUE::EvalPerspective::White;
            } else if (perspective == "stm" || perspective == "side-to-move") {
                options.eval_perspective = NNUE::EvalPerspective::SideToMove;
            } else {
                std::cerr << "Unsupported eval perspective: " << perspective << "\n";
                return false;
            }
        } else if (arg == "--lr") {
            const char* value = require_value(arg);
            if (!value) return false;
            NNUE::GLOBAL_CONFIG.learning_rate = std::stof(value);
        } else if (arg == "--epochs") {
            const char* value = require_value(arg);
            if (!value) return false;
            NNUE::GLOBAL_CONFIG.num_epochs = std::stoi(value);
        } else if (arg == "--batch-size") {
            const char* value = require_value(arg);
            if (!value) return false;
            NNUE::GLOBAL_CONFIG.batch_size = static_cast<size_t>(std::stoull(value));
        } else if (arg == "--limit") {
            const char* value = require_value(arg);
            if (!value) return false;
            options.game_limit = static_cast<size_t>(std::stoull(value));
        } else if (arg == "--val-limit") {
            const char* value = require_value(arg);
            if (!value) return false;
            options.val_limit = static_cast<size_t>(std::stoull(value));
        } else if (arg == "--no-synthetic") {
            options.allow_synthetic = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }

    if (!options.base_model_path_set) {
        options.base_model_path = fs::path(options.export_dir) / "nnue_inference.bin";
    }
    if (!options.output_model_path_set) {
        options.output_model_path = fs::path(options.export_dir) / "nnue_inference.bin";
    }
    return true;
}

const char* mode_name(TrainingMode mode) {
    switch (mode) {
        case TrainingMode::ContinueCheckpoint: return "continue";
        case TrainingMode::FineTuneInference: return "finetune";
        case TrainingMode::TrainScratch: return "scratch";
    }
    return "unknown";
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);

    std::cout << "====================================================================\n";
    std::cout << "             INITIALIZING NATIVE C++ NNUE TRAINING PIPELINE        \n";
    std::cout << "====================================================================\n\n";

    NNUE::GLOBAL_CONFIG.learning_rate = 1e-3f;   
    NNUE::GLOBAL_CONFIG.weight_decay = 1e-5f;    
    NNUE::GLOBAL_CONFIG.batch_size = 8192;      
    NNUE::GLOBAL_CONFIG.num_epochs = 24;

    TrainOptions options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }

    if (!fs::exists(options.checkpoint_dir)) fs::create_directories(options.checkpoint_dir);
    if (!fs::exists(options.export_dir)) fs::create_directories(options.export_dir);
    if (options.output_model_path.has_parent_path()) fs::create_directories(options.output_model_path.parent_path());

    std::cout << " -> Training mode: " << mode_name(options.mode) << "\n";
    std::cout << " -> Learning rate: " << NNUE::GLOBAL_CONFIG.learning_rate
              << " | epochs: " << NNUE::GLOBAL_CONFIG.num_epochs
              << " | batch_size: " << NNUE::GLOBAL_CONFIG.batch_size << "\n";
    std::cout << " -> Dataset eval perspective: "
              << (options.eval_perspective == NNUE::EvalPerspective::White ? "white" : "side-to-move")
              << "\n";
    if (options.val_fen_path_set) {
        std::cout << " -> Validation FEN file: " << options.val_fen_path.string()
                  << " | val_limit: " << options.val_limit << "\n";
    }
    if (options.mode == TrainingMode::FineTuneInference &&
        fs::absolute(options.base_model_path).lexically_normal() ==
            fs::absolute(options.output_model_path).lexically_normal()) {
        std::cout << " -> Warning: fine-tune output path matches base model; export will overwrite the input model.\n";
    }
    if (options.mode != TrainingMode::ContinueCheckpoint) {
        fs::path stale_resume = fs::path(options.checkpoint_dir) / "resume.ckpt";
        fs::path stale_best = fs::path(options.checkpoint_dir) / "best.ckpt";
        if (fs::exists(stale_resume) || fs::exists(stale_best)) {
            std::cout << " -> Note: scratch/finetune ignores existing checkpoints in this directory, "
                      << "but new checkpoints will overwrite resume.ckpt/best.ckpt.\n";
        }
    }

    std::cout << "[1/5] Ingesting source files from drive space...\n";
    
    NNUE::ReplayBuffer full_dataset(30000000); 

    if (fs::exists(options.fen_path)) {
        std::cout << " -> Appending raw evaluation snapshot records from: " << options.fen_path.string() << "\n";
        full_dataset.load_fen_file(options.fen_path.string(), options.game_limit, options.eval_perspective);
    }

    if (full_dataset.size() == 0 && !options.allow_synthetic) {
        std::cerr << " -> [Error] No training data loaded and --no-synthetic was set.\n";
        return 1;
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
    size_t train_size = options.val_fen_path_set
        ? total_items
        : static_cast<size_t>(0.9f * total_items);
    size_t val_size = options.val_fen_path_set
        ? options.val_limit
        : (total_items - train_size);

    NNUE::ReplayBuffer train_buffer(train_size + 10);
    NNUE::ReplayBuffer val_buffer(val_size + 10);

    if (options.val_fen_path_set) {
        auto train_samples = full_dataset.shuffled_snapshot();
        for (auto& sample : train_samples) {
            train_buffer.add_raw(std::move(sample));
        }
        if (fs::exists(options.val_fen_path)) {
            std::cout << " -> Loading held-out validation records from: "
                      << options.val_fen_path.string() << "\n";
            val_buffer.load_fen_file(options.val_fen_path.string(), options.val_limit, options.eval_perspective);
            val_size = val_buffer.size();
        } else {
            std::cerr << " -> [Error] Validation FEN file not found: "
                      << options.val_fen_path.string() << "\n";
            return 1;
        }
    } else {
        auto all_samples = full_dataset.shuffled_snapshot();
        for (size_t i = 0; i < all_samples.size(); ++i) {
            if (i < train_size) train_buffer.add_raw(std::move(all_samples[i]));
            else val_buffer.add_raw(std::move(all_samples[i]));
        }
    }
    std::cout << " -> Data allocation: " << train_buffer.size()
              << " training items | " << val_buffer.size()
              << " validation items.\n";

    NNUE::NNUEWDLLoss loss_function(400.0f, "mse", "mean");

    std::cout << "\n[2/5] Building dual-perspective network layers...\n";
    if (options.mode == TrainingMode::FineTuneInference) {
        std::cout << " -> Loading engine inference weights from: " << options.base_model_path.string() << "\n";
        if (!NNUE::NNUEModel::load_weights(options.base_model_path.string())) {
            std::cerr << " -> [Error] Failed to load base inference model for fine-tuning.\n";
            return 1;
        }
        NNUE::NNUEModel::set_quantized_inference(false);
    } else {
        NNUE::FeatureTransformer::initialize_weights_from_scratch();
    }

    std::cout << "\n[3/5] Initializing execution trainer matrix...\n";
    NNUE::NNUETrainer trainer(train_buffer, loss_function, &val_buffer);

    std::cout << "\n[4/5] Preparing initial training state...\n";

    if (options.mode == TrainingMode::ContinueCheckpoint) {
        std::string newest_checkpoint = NNUE::CheckpointManager::latest_checkpoint(options.checkpoint_dir);
        if (!newest_checkpoint.empty()) {
            std::cout << " -> Continuing interrupted run from checkpoint: " << newest_checkpoint << "\n";
            if (!trainer.resume(newest_checkpoint, false)) {
                std::cerr << " -> [Error] Checkpoint resume failed. Use --mode scratch with a fresh checkpoint-dir to restart.\n";
                return 1;
            }
        } else {
            std::cout << " -> No checkpoint found. Commencing baseline training.\n";
            NNUE::NNUEModel::initialize_weights_from_scratch();
        }
    } else if (options.mode == TrainingMode::TrainScratch) {
        std::cout << " -> Ignoring checkpoints. Commencing baseline training.\n";
        NNUE::NNUEModel::initialize_weights_from_scratch();
    } else {
        std::cout << " -> Fine-tuning loaded inference weights with fresh optimizer state.\n";
    }

    std::cout << "\n[5/5] Running native backpropagation loop...\n";
    std::cout << "--------------------------------------------------------------------\n";

    trainer.train(NNUE::GLOBAL_CONFIG.num_epochs, options.checkpoint_dir);

    std::cout << "--------------------------------------------------------------------\n";

    if (NNUE::g_interrupted.load()) {
        std::cout << " -> Training interrupted. Resume checkpoint saved; skipping model export.\n";
        return 130;
    }

    fs::path resume_checkpoint = fs::path(options.checkpoint_dir) / "resume.ckpt";
    fs::path best_checkpoint = fs::path(options.checkpoint_dir) / "best.ckpt";
    trainer.save_final(resume_checkpoint.string());

    std::cout << "\n[Post-Processing] Exporting engine model...\n";
    if (fs::exists(best_checkpoint)) {
        std::cout << " -> Restoring best validation checkpoint for export: "
                  << best_checkpoint.string() << "\n";
        if (!trainer.restore_for_export(best_checkpoint.string())) {
            std::cout << " ⚠️ Warning: Failed to restore best checkpoint. Exporting latest weights instead.\n";
        }
    } else {
        std::cout << " -> No best checkpoint found. Exporting latest weights.\n";
    }

    std::string engine_ready_file = options.output_model_path.string();

    if (trainer.export_to_engine_binary(engine_ready_file)) {
        std::cout << " -> Engine NNUE file exported to: " << engine_ready_file << "\n";
    } else {
        std::cout << " ⚠️ Warning: Native engine export failed!\n";
    }

    return 0;
}
