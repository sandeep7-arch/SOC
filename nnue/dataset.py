# nnue/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset, DataLoader

from engine.board import ChessBoard  # <-- Wired your custom state machine wrapper
from .feature_encoder import ENCODER, FeatureEncoder
from .replay_buffer import ReplayBuffer, Experience


class NNUEDataset(Dataset):
    """
    Production-grade NNUE training dataset interface.
    Bridges raw training files/buffers and structural PyTorch training loops.

    Updated to leverage the custom ChessBoard state storage architecture.
    """

    def __init__(self, encoder: Optional[FeatureEncoder] = None) -> None:
        self.encoder = encoder or ENCODER
        self.samples: List[Experience] = []

    # =====================================================
    # Replay Buffer Import (Fast RL Lane)
    # =====================================================

    def load_replay_buffer(self, replay_buffer: ReplayBuffer) -> None:
        """Loads pre-encoded structural experience fragments instantly from RAM."""
        self.samples.extend(replay_buffer.buffer)

    # =====================================================
    # FEN Raw File Import (Using ChessBoard Vault)
    # =====================================================

    def load_fen_file(self, path: str | Path, target_value: float = 0.0) -> None:
        """Reads standalone FEN lines, tracking states through ChessBoard."""
        path = Path(path)

        with open(path, "r") as f:
            for line in f:
                fen_str = line.strip()
                if not fen_str:
                    continue

                # Use your static validation check to avoid unnecessary object allocation
                if not ChessBoard.is_valid_fen(fen_str):
                    continue

                try:
                    # Initialize using your custom data vault
                    custom_board = ChessBoard(fen_str)

                    # Pass the underlying chess.Board core to the encoder
                    w_feats, b_feats = self.encoder.encode(custom_board.board)

                    self.samples.append(
                        Experience(
                            white_features=w_feats,
                            black_features=b_feats,
                            target_value=target_value,
                            is_white_to_move=(custom_board.board.turn == chess.WHITE),
                            weight=1.0,
                        )
                    )
                except Exception:
                    continue  # Skip corrupt configurations safely

    # =====================================================
    # PGN Directory Import (Using ChessBoard Moves)
    # =====================================================

    def load_games(self, directory: str | Path) -> None:
        """Recursively parses a directory layout for legacy PGN game records."""
        directory = Path(directory)
        for pgn_file in directory.rglob("*.pgn"):
            self.extract_positions_from_games(pgn_file)

    def extract_positions_from_games(self, pgn_path: str | Path) -> None:
        """Extracts and pre-encodes all valid match nodes from a target PGN log."""
        pgn_path = Path(pgn_path)

        with open(pgn_path, encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                self._extract_game(game)

    def _extract_game(self, game: chess.pgn.Game) -> None:
        """Extracts chess positions from a game using custom ChessBoard mutations."""
        result_map = {
            "1-0": 1.0,
            "0-1": -1.0,
            "1/2-1/2": 0.0,
        }

        result = result_map.get(game.headers.get("Result", "*"), 0.0)

        # Instantiate your clean, unanalyzed state storage machine
        custom_board = ChessBoard()

        # Capture raw game paths dynamically
        game_nodes: List[Tuple[List[int], List[int], bool]] = []

        for move in game.mainline_moves():
            w_feats, b_feats = self.encoder.encode(custom_board.board)
            game_nodes.append((w_feats, b_feats, custom_board.board.turn == chess.WHITE))

            # Step your custom vault forward using your exact verification method
            custom_board.make_move(move)

        total_positions = max(1, len(game_nodes))

        for ply, (w_feats, b_feats, is_w_move) in enumerate(game_nodes):
            progress = (ply + 1) / total_positions
            weight = 0.25 + 0.75 * progress

            self.samples.append(
                Experience(
                    white_features=w_feats,
                    black_features=b_feats,
                    target_value=result,
                    is_white_to_move=is_w_move,
                    weight=weight,
                    ply=ply,
                )
            )

    # =====================================================
    # Dataset Core API
    # =====================================================

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetches a pre-encoded training sample instantly, bypassing on-the-fly parsing."""
        sample = self.samples[idx]

        w_tensor = torch.tensor(sample.white_features, dtype=torch.long)
        b_tensor = torch.tensor(sample.black_features, dtype=torch.long)
        is_w_move = torch.tensor(sample.is_white_to_move, dtype=torch.bool)
        target = torch.tensor(sample.target_value, dtype=torch.float32)
        weight = torch.tensor(sample.weight, dtype=torch.float32)

        return w_tensor, b_tensor, is_w_move, target, weight


# =====================================================
# High Performance Batching Tools
# =====================================================

def collate_nnue_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate processor handles varying feature list dimensions seamlessly."""
    w_feats, b_feats, turns, targets, weights = zip(*batch)

    return (
        list(w_feats),
        list(b_feats),
        torch.stack(turns),
        torch.stack(targets),
        torch.stack(weights),
    )


def create_training_loader(dataset: NNUEDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Builds highly parallelized training batch execution pipelines."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_nnue_batch,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
