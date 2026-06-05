# nnue/accumulator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import chess
import torch

from .feature_encoder import FeatureEncoder
from .feature_transformer import FeatureTransformer


@dataclass(frozen=True)
class AccumulatorState:
    """
    Immutable snapshot record of both accumulators on the search unroll stack.
    Enables true O(1) undo recovery states when stepping backward through tree branches.
    """
    white_accumulator: torch.Tensor
    black_accumulator: torch.Tensor


class Accumulator:
    """
    Incremental NNUE accumulator manager tracking dual perspective states.

    Responsibilities:
    - Maintains separate, running layers for both White and Black perspectives.
    - Intercepts state modifications to perform O(1) incremental vector arithmetic updates.
    - Forces a full perspective reconstruction ONLY when a perspective king shifts coordinates.
    """

    def __init__(
        self,
        encoder: FeatureEncoder,
        transformer: FeatureTransformer,
        device: str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.transformer = transformer
        self.device = torch.device(device)

        self.board: chess.Board | None = None
        self.white_accumulator: torch.Tensor | None = None
        self.black_accumulator: torch.Tensor | None = None

        # Fixed operational stack history
        self.stack: List[AccumulatorState] = []

    # ==========================================================
    # Initialization
    # ==========================================================

    @torch.no_grad()
    def initialize(self, board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds both perspective accumulators from a clean baseline layout state."""
        self.board = board  # Keep a direct reference pointer (Zero cloning overhead)
        self.stack.clear()

        w_features, b_features = self.encoder.encode(self.board)

        w_tensor = torch.tensor(w_features, dtype=torch.long, device=self.device)
        b_tensor = torch.tensor(b_features, dtype=torch.long, device=self.device)

        self.white_accumulator = self.transformer(w_tensor)
        self.black_accumulator = self.transformer(b_tensor)

        return self.white_accumulator, self.black_accumulator

    # ==========================================================
    # State Access
    # ==========================================================

    def current_accumulators(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the current active structural layer evaluation states."""
        if self.white_accumulator is None or self.black_accumulator is None:
            raise RuntimeError("Accumulators have not been initialized.")
        return self.white_accumulator, self.black_accumulator

    # ==========================================================
    # Incremental Update Execution Core
    # ==========================================================

    @torch.no_grad()
    def push(self, move: chess.Move) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pushes a move into history and incrementally updates internal state parameters.
        Detects structural exceptions like king moves to re-anchor perspectives.
        """
        if self.board is None or self.white_accumulator is None or self.black_accumulator is None:
            raise RuntimeError("Accumulator referenced before initialization.")

        # Cache the current vectors onto our unroll state stack
        self.stack.append(
            AccumulatorState(
                white_accumulator=self.white_accumulator.clone(),
                black_accumulator=self.black_accumulator.clone(),
            )
        )

        # Identify key pieces before mutating the state layout
        moving_piece_type = self.board.piece_type_at(move.from_square)
        white_king_moved = (moving_piece_type == chess.KING and self.board.turn == chess.WHITE)
        black_king_moved = (moving_piece_type == chess.KING and self.board.turn == chess.BLACK)

        # Extract delta differences using our zero-clone mutation method
        (w_removed, w_added), (b_removed, b_added) = self.encoder.changed_features(self.board, move)

        # Mutate our primary reference state board forward
        self.board.push(move)

        # White Perspective Update Vector Logic
        if white_king_moved:
            w_features, _ = self.encoder.encode(self.board)
            w_tensor = torch.tensor(w_features, dtype=torch.long, device=self.device)
            self.white_accumulator = self.transformer(w_tensor)
        else:
            self.white_accumulator = self.transformer.update_accumulator(
                self.white_accumulator, w_removed, w_added
            )

        # Black Perspective Update Vector Logic
        if black_king_moved:
            _, b_features = self.encoder.encode(self.board)
            b_tensor = torch.tensor(b_features, dtype=torch.long, device=self.device)
            self.black_accumulator = self.transformer(b_tensor)
        else:
            self.black_accumulator = self.transformer.update_accumulator(
                self.black_accumulator, b_removed, b_added
            )

        return self.white_accumulator, self.black_accumulator

    # ==========================================================
    # Move Retraction Undo Layer
    # ==========================================================

    @torch.no_grad()
    def pop(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pops historical records out of cache memory to step backward in O(1) space."""
        if not self.stack or self.board is None:
            raise RuntimeError("Cannot pop state: History stack is empty.")

        # Step the board reference matrix back safely
        self.board.pop()

        # Unpack historical vectors
        state = self.stack.pop()
        self.white_accumulator = state.white_accumulator
        self.black_accumulator = state.black_accumulator

        return self.white_accumulator, self.black_accumulator

    # ==========================================================
    # Integrity Verification
    # ==========================================================

    @torch.no_grad()
    def verify(self) -> bool:
        """Compares current state accumulators against full reconstructions to prevent decay."""
        if self.board is None or self.white_accumulator is None or self.black_accumulator is None:
            return False

        cached_w, cached_b = self.white_accumulator.clone(), self.black_accumulator.clone()

        # Re-initialize from scratch to test correctness
        w_features, b_features = self.encoder.encode(self.board)
        rebuilt_w = self.transformer(torch.tensor(w_features, dtype=torch.long, device=self.device))
        rebuilt_b = self.transformer(torch.tensor(b_features, dtype=torch.long, device=self.device))

        w_match = torch.allclose(cached_w, rebuilt_w, atol=1e-4)
        b_match = torch.allclose(cached_b, rebuilt_b, atol=1e-4)

        # Restore tracking states safely
        self.white_accumulator, self.black_accumulator = cached_w, cached_b
        return w_match and b_match
