"""Bridge from Python RL orchestration to the native NNUE trainer."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativeTrainerConfig:
    root: Path = Path(__file__).resolve().parents[1]
    trainer_bin: Path = Path(__file__).resolve().parents[1] / "nnue_trainer"
    dataset_path: Path = Path(__file__).resolve().parents[1] / "data" / "fen_files" / "chessData.fen"


class RLTrainer:
    """Runs the repo's C++ trainer after Python has generated self-play data."""

    def __init__(self, config: NativeTrainerConfig | None = None) -> None:
        self.config = config or NativeTrainerConfig()

    def train(self) -> None:
        if not self.config.dataset_path.exists():
            raise FileNotFoundError(f"missing FEN dataset: {self.config.dataset_path}")
        if not self.config.trainer_bin.exists():
            raise FileNotFoundError(f"missing native trainer binary: {self.config.trainer_bin}")
        subprocess.run([str(self.config.trainer_bin)], cwd=str(self.config.root), check=True)
