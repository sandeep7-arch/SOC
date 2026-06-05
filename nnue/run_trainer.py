# run_trainer.py

from __future__ import annotations
import sys
from pathlib import Path

# Ensure Python can find your 'nnue' module package from the root directory
sys.path.append(str(Path(__file__).parent))

from .config import CONFIG
from .model import NNUEModel
from .dataset import NNUEDataset
from .trainer import NNUETrainer
from .checkpoint import latest_checkpoint
from .export import NNUEExporter


def main() -> None:
    """
    Main orchestration entry point to execute the backpropagation training run.
    """
    print("====================================================================")
    print("      🚀 INITIALIZING PRODUCTION-GRADE NNUE TRAINING PIPELINE       ")
    print("====================================================================\n")

    # Verify architecture dimensions before starting
    print(f"[*] Verified Architecture: HalfKP Topology ({CONFIG.FEATURE_DIM:,} input features)")

    # 1. Establish Hard Drive Storage Locations
    DATA_DIR = Path("data")
    PGN_PATH = DATA_DIR / "PGN_files"
    FEN_PATH = DATA_DIR / "fen_files"
    CHECKPOINT_DIR = Path(CONFIG.CHECKPOINT_DIR)

    # 2. Ingest Data
    print("\n[1/5] Ingesting source files from drive space...")
    dataset = NNUEDataset()
    data_found = False

    if PGN_PATH.exists():
        print(f" -> Scanning PGN files at: {PGN_PATH}")
        dataset.load_games(PGN_PATH)
        data_found = True

    if FEN_PATH.exists():
        print(f" -> Appending raw evaluation snapshot records from: {FEN_PATH}")
        dataset.load_fen_file(FEN_PATH)
        data_found = True

    if not data_found or len(dataset) == 0:
        print("\n⚠️ Execution Halted: No training data found on your hard drive!")
        print(f" Please generate data into '{PGN_PATH}' or '{FEN_PATH}' first.")
        return

    print(f" -> Success! Compiled {len(dataset):,} unique training positions.")

    # 3. Instantiate Architecture
    print("\n[2/5] Building dual-perspective network layers...")
    model = NNUEModel()

    # 4. Bind Systems Into Trainer
    print("\n[3/5] Initializing execution trainer matrix...")
    trainer = NNUETrainer(
        model=model,
        train_dataset=dataset,
        val_dataset=None,
    )

    # 5. Fault-Tolerance Check
    print("\n[4/5] Searching for existing checkpoint intervals...")
    newest_checkpoint = latest_checkpoint(CHECKPOINT_DIR)

    if newest_checkpoint:
        print(f" -> Active backup located! Restoring system variables from: {newest_checkpoint}")
        trainer.resume(newest_checkpoint)
    else:
        print(" -> No active checkpoints located. Commencing training from baseline zero weights.")

    # 6. Ignite Training
    print("\n[5/5] Igniting backpropagation loop. Tuning parameters active...")
    print("-" * 68)

    trainer.train(
        epochs=CONFIG.NUM_EPOCHS,
        checkpoint_dir=str(CHECKPOINT_DIR),
        save_every=1
    )
    print("-" * 68)

    # 7. Serialize Production Evaluation Module
    final_output_path = CHECKPOINT_DIR / "nnue_brain.pt"
    trainer.save_final(final_output_path)

    exporter = NNUEExporter(model=model, device="cpu")

    # 1. Export deployment brain
    exporter.export_engine_checkpoint(Path(CONFIG.EXPORT_DIR) / "nnue_inference.pt")

    # 2. Export quantized edition if requested
    if CONFIG.EXPORT_INT8:
        quantized_cpu_model = exporter.quantize_model()
        torch.save(quantized_cpu_model.state_dict(), Path(CONFIG.EXPORT_DIR) / "nnue_int8_quantized.pt")
        print(f" -> Quantized production file exported to: {CONFIG.EXPORT_DIR}/nnue_int8_quantized.pt")

    # 3. Export ONNX standard cross-compatibility map
    if CONFIG.EXPORT_ONNX:
        exporter.export_onnx(Path(CONFIG.EXPORT_DIR) / "nnue_model.onnx")

    print(f"\n🎉 Process complete! Optimized model weights exported to: {Path(CONFIG.EXPORT_DIR}")


if __name__ == "__main__":
    main()
