# nnue/feature_encoder.py

from __future__ import annotations

from typing import List, Tuple
import chess

from .config import CONFIG


class FeatureEncoder:
    """
    Stateless HalfKP-style feature encoder mapping board layouts to sparse index integers.

    Feature Layout Topology (Perspective Relative):
    - 64 King Squares × (2 Colors * 5 Piece Types * 64 Squares) = 40,960 features.
    - Kings are structural perspective coordinates and are excluded as piece features.

    Strict Constraints:
    - Automatically enforces vertical perspective square flipping for Black's perspective.
    - Separates White and Black perspective lists to keep parallel accumulator tracking pure.
    """

    # Map engine pieces to 0-4 (Excluding King)
    PIECE_TYPE_OFFSET = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }

    __slots__ = ("feature_dim",)

    def __init__(self) -> None:
        self.feature_dim = CONFIG.FEATURE_DIM

    # ==========================================================
    # Public API
    # ==========================================================

    def active_features(self, board: chess.Board) -> Tuple[List[int], List[int]]:
        """
        Returns active sparse feature indices separate for White and Black accumulators.

        Returns
        -------
        Tuple[List[int], List[int]]
            (white_perspective_features, black_perspective_features)
        """
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)

        if white_king_sq is None or black_king_sq is None:
            return [], []

        # White Accumulator View: Natural perspective mapping
        white_features = self._king_perspective_features(
            board,
            king_square=white_king_sq,
            perspective_color=chess.WHITE
        )

        # Black Accumulator View: Vertically flipped coordinate system
        black_features = self._king_perspective_features(
            board,
            king_square=black_king_sq,
            perspective_color=chess.BLACK
        )

        return white_features, black_features

    def encode(self, board: chess.Board) -> Tuple[List[int], List[int]]:
        """Alias coordinate mapper utility consumed by training and inference layers."""
        return self.active_features(board)

    def feature_count(self) -> int:
        return self.feature_dim

    # ==========================================================
    # Incremental Helpers (Optimized for Accumulator Updates)
    # ==========================================================

    def changed_features(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
        """
        Determine which features are removed and added for both accumulators.
        Uses inline push/pop mechanics to completely eliminate board cloning overhead.

        Returns
        -------
        Tuple[Tuple[WhiteRemoved, WhiteAdded], Tuple[BlackRemoved, BlackAdded]]
        """
        # 1. Capture the features of the current layout state
        white_before, black_before = self.active_features(board)

        # 2. Mutate the active engine board state directly (No cloning!)
        board.push(move)

        # 3. Capture the features of the new layout state
        white_after, black_after = self.active_features(board)

        # 4. Revert the engine board state back cleanly to protect history
        board.pop()

        # 5. Extract our difference masks
        white_removed = list(set(white_before) - set(white_after))
        white_added = list(set(white_after) - set(white_before))

        black_removed = list(set(black_before) - set(black_after))
        black_added = list(set(black_after) - set(black_before))

        return (white_removed, white_added), (black_removed, black_added)

    # ==========================================================
    # Internal Perspective Calculations
    # ==========================================================

    def _king_perspective_features(
        self,
        board: chess.Board,
        king_square: int,
        perspective_color: chess.Color,
    ) -> List[int]:
        """Generate HalfKP indices relative to a specific king's view axis."""
        features: List[int] = []

        # If evaluating from Black's perspective, flip the king's square vertically
        oriented_king_sq = king_square if perspective_color == chess.WHITE else (king_square ^ 56)

        king_offset = oriented_king_sq * (
            CONFIG.NUM_COLORS * CONFIG.NUM_PIECE_TYPES * CONFIG.NUM_SQUARES
        )

        for square, piece in board.piece_map().items():
            # Crucial: Exclude kings from the piece feature listing maps
            if piece.piece_type == chess.KING:
                continue

            # Determine relative color from the perspective player's viewpoint
            # If perspective is White: White=0, Black=1
            # If perspective is Black: Black=0, White=1
            is_opponent_color = (piece.color != perspective_color)
            color_idx = 1 if is_opponent_color else 0

            piece_idx = self.PIECE_TYPE_OFFSET[piece.piece_type]

            # Orient the piece square relative to the perspective color axis
            oriented_piece_sq = square if perspective_color == chess.WHITE else (square ^ 56)

            feature_index = (
                king_offset
                + (color_idx * CONFIG.NUM_PIECE_TYPES * CONFIG.NUM_SQUARES)
                + (piece_idx * CONFIG.NUM_SQUARES)
                + oriented_piece_sq
            )

            features.append(feature_index)

        return features

    # ==========================================================
    # Dense Conversion (Validation / Training Testing Hooks)
    # ==========================================================

    def to_dense(self, active_features: List[int]) -> torch.Tensor:
        """Convert sparse integer index structures into explicit tracking tensors."""
        import torch

        x = torch.zeros(self.feature_dim, dtype=torch.float32)
        x[active_features] = 1.0
        return x


# Singleton Instance
ENCODER = FeatureEncoder()
