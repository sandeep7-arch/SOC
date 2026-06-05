# nnue/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NNUEConfig:
    """
    Central configuration for the NNUE system.

    All other NNUE modules should import this config instance instead of
    hardcoding dimension parameters.
    """

    # ============================================================
    # Feature Encoding (HalfKP-style indexing)
    # ============================================================

    NUM_KING_SQUARES: int = 64
    NUM_COLORS: int = 2
    NUM_PIECE_TYPES: int = 5  # Excludes king (P, N, B, R, Q)
    NUM_SQUARES: int = 64

    # HalfKP Topology:
    # 64 King Squares × (2 Colors * 5 Piece Types * 64 Squares)
    # Total input vector size = 64 * 10 * 64 = 40,960 dimensions.
    FEATURE_DIM: int = (
        NUM_KING_SQUARES
        * (NUM_COLORS * NUM_PIECE_TYPES)
        * NUM_SQUARES
    )

    # ============================================================
    # Network Architecture
    # ============================================================

    ACCUMULATOR_DIM: int = 512  # Transformed input layer side (L1)

    HIDDEN1_DIM: int = 256      # L2 hidden nodes
    HIDDEN2_DIM: int = 32       # L3 hidden nodes

    OUTPUT_DIM: int = 1         # Raw evaluation evaluation score output

    # ============================================================
    # Quantization
    # ============================================================

    QUANTIZATION_SCALE: int = 127

    WEIGHT_CLAMP_MIN: float = -1.0
    WEIGHT_CLAMP_MAX: float = 1.0

    # ============================================================
    # Training Hyperparameters
    # ============================================================

    LEARNING_RATE: float = 1e-3

    WEIGHT_DECAY: float = 1e-5

    BATCH_SIZE: int = 4096

    NUM_EPOCHS: int = 50

    GRAD_CLIP_NORM: float = 1.0

    VALIDATION_SPLIT: float = 0.1

    # ============================================================
    # Evaluation Scaling
    # ============================================================

    MAX_CENTIPAWN_SCORE: int = 10000

    OUTPUT_SCALE: float = 600.0

    # ============================================================
    # Export Format Controls
    # ============================================================

    EXPORT_INT8: bool = True

    EXPORT_ONNX: bool = True

    # ============================================================
    # System Operational Paths
    # ============================================================

    CHECKPOINT_DIR: Path = Path("checkpoints")

    EXPORT_DIR: Path = Path("exports")

    LOG_DIR: Path = Path("logs")

    def __post_init__(self) -> None:
        """Create storage system directories if missing on initialization."""
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


CONFIG = NNUEConfig()
