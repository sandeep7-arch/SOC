# nnue/replay_buffer.py

from __future__ import annotations

import random
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Experience:
    """
    Single self-play or training data state node.

    Optimized for NNUE training loops by storing pre-extracted integer feature indices
    instead of raw FEN text strings, eliminating CPU parsing bottlenecks during gradient descent.
    """
    # Pre-calculated sparse feature lists from FeatureEncoder
    white_features: List[int]
    black_features: List[int]

    # Value targets (e.g., +1.0 for White win, 0.0 for draw, -1.0 for Black win)
    target_value: float

    # Side-to-move identifier flag (True if White to move, False if Black to move)
    is_white_to_move: bool

    weight: float = 1.0
    game_id: Optional[str] = None
    ply: int = 0


class ReplayBuffer:
    """
    Experience Replay Buffer.
    Manages bounded collection tracking arrays mapping positions generated via self-play.

    Optimizations:
        - Bounded collections via deque limits memory leakage footprint.
        - Fast, uniform randomized selection sampling loops.
        - Pre-encoded training samples to saturate GPU tensor compute lanes.
    """

    def __init__(self, capacity: int = 1_000_000) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    # =====================================================
    # Add Experience Entry
    # =====================================================

    def add(
        self,
        white_features: List[int],
        black_features: List[int],
        target_value: float,
        is_white_to_move: bool,
        weight: float = 1.0,
        game_id: Optional[str] = None,
        ply: int = 0,
    ) -> None:
        """Appends a single structured feature layout into memory storage buffers."""
        self.buffer.append(
            Experience(
                white_features=white_features,
                black_features=black_features,
                target_value=target_value,
                is_white_to_move=is_white_to_move,
                weight=weight,
                game_id=game_id,
                ply=ply,
            )
        )

    # =====================================================
    # Add Whole Game Loop Data
    # =====================================================

    def add_game(
        self,
        game_positions: List[Tuple[List[int], List[int], bool]],
        result: float,
        game_id: Optional[str] = None,
    ) -> None:
        """
        Appends a collection of sequential moves matching a completed game.

        Parameters
        ----------
        game_positions: List of tuples containing:
            - white_features: List[int]
            - black_features: List[int]
            - is_white_to_move: bool
        result: Scalar float game value target outcome
        """
        total = max(1, len(game_positions))

        for ply, (w_feats, b_feats, is_w_move) in enumerate(game_positions):
            # Linearly scale position significance values as games near completion
            progress = (ply + 1) / total
            weight = 0.25 + 0.75 * progress

            self.add(
                white_features=w_feats,
                black_features=b_feats,
                target_value=result,
                is_white_to_move=is_w_move,
                weight=weight,
                game_id=game_id,
                ply=ply,
            )

    # =====================================================
    # Sampling Channel
    # =====================================================

    def sample(self, batch_size: int) -> List[Experience]:
        """Draws a random uniform slice of positions out of tracking queues."""
        actual_batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, actual_batch_size)

    # =====================================================
    # Housekeeping Utilities
    # =====================================================

    def clear(self) -> None:
        self.buffer.clear()

    def size(self) -> int:
        return len(self.buffer)

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    # =====================================================
    # Disk Persistence Serialization
    # =====================================================

    def save(self, path: str | Path) -> None:
        """Saves memory buffers to solid state drives via standard binary streams."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "capacity": self.capacity,
                    "buffer": list(self.buffer),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,  # Significant performance win for large arrays
            )

    @classmethod
    def load(self, path: str | Path) -> ReplayBuffer:
        """Loads historical training records from binary file fragments."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        constructed_buffer = self(capacity=data["capacity"])
        constructed_buffer.buffer.extend(data["buffer"])
        return constructed_buffer
