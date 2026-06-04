# engine/zobrist.py

from __future__ import annotations

import random
from typing import Dict, Optional

import chess


class ZobristHasher:
    """
    Zobrist hashing implementation for chess positions.

    Purpose:
    - Repetition detection
    - Transposition table indexing
    - Position identity

    Characteristics:
    - Deterministic (seeded RNG)
    - Compatible with python-chess
    - Independent of search/evaluation logic
    - Uses 64-bit hash keys

    Notes:
    - Full-board hashing is implemented.
    - Incremental update hook is provided as an API
      but intentionally left conservative because
      board state updates are handled by python-chess.
    """

    __slots__ = (
        "_piece_keys",
        "_side_to_move_key",
        "_castling_keys",
        "_ep_file_keys",
    )

    DEFAULT_SEED = 2026

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        """
        Initialize deterministic Zobrist tables.

        Args:
            seed:
                RNG seed to ensure reproducible hashes.
        """
        rng = random.Random(seed)

        # -----------------------------------------------------
        # Piece-Square Keys
        #
        # 12 piece types:
        # White:
        #   P N B R Q K
        # Black:
        #   p n b r q k
        #
        # 64 squares each
        # -----------------------------------------------------

        self._piece_keys = [
            [rng.getrandbits(64) for _ in range(64)]
            for _ in range(12)
        ]

        # -----------------------------------------------------
        # Side To Move
        # -----------------------------------------------------

        self._side_to_move_key = rng.getrandbits(64)

        # -----------------------------------------------------
        # Castling Rights
        #
        # K Q k q
        # -----------------------------------------------------

        self._castling_keys = {
            "K": rng.getrandbits(64),
            "Q": rng.getrandbits(64),
            "k": rng.getrandbits(64),
            "q": rng.getrandbits(64),
        }

        # -----------------------------------------------------
        # En Passant File
        #
        # a-h => 8 files
        # -----------------------------------------------------

        self._ep_file_keys = [
            rng.getrandbits(64)
            for _ in range(8)
        ]

    # ---------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _piece_index(piece: chess.Piece) -> int:
        """
        Convert python-chess piece into index.

        Mapping:

        White:
            P=0 N=1 B=2 R=3 Q=4 K=5

        Black:
            p=6 n=7 b=8 r=9 q=10 k=11
        """

        offset = 0 if piece.color == chess.WHITE else 6

        return offset + {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }[piece.piece_type]

    # ---------------------------------------------------------
    # Full Hash Computation
    # ---------------------------------------------------------

    def hash_board(self, board: chess.Board) -> int:
        """
        Compute full Zobrist hash.

        Includes:
        - Piece placement
        - Side to move
        - Castling rights
        - En passant file

        Args:
            board:
                python-chess Board.

        Returns:
            64-bit integer hash.
        """

        h = 0

        # -----------------------------------------------------
        # Pieces
        # -----------------------------------------------------

        for square, piece in board.piece_map().items():
            piece_idx = self._piece_index(piece)
            h ^= self._piece_keys[piece_idx][square]

        # -----------------------------------------------------
        # Side To Move
        # -----------------------------------------------------

        if board.turn == chess.BLACK:
            h ^= self._side_to_move_key

        # -----------------------------------------------------
        # Castling Rights
        # -----------------------------------------------------

        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= self._castling_keys["K"]

        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self._castling_keys["Q"]

        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self._castling_keys["k"]

        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self._castling_keys["q"]

        # -----------------------------------------------------
        # En Passant
        # -----------------------------------------------------

        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            h ^= self._ep_file_keys[ep_file]

        return h

    # ---------------------------------------------------------
    # Incremental Update Hook
    # ---------------------------------------------------------

    def update_hash_incremental(
        self,
        current_hash: int,
        move: chess.Move,
        board_before_move: Optional[chess.Board] = None,
    ) -> int:
        """
        Incremental hash update hook.

        Current implementation intentionally falls
        back to full recomputation for correctness.

        Future search optimizations may replace this
        with true XOR-based incremental updates.

        Args:
            current_hash:
                Existing hash value.

            move:
                Move being applied.

            board_before_move:
                Board state before move execution.

        Returns:
            Updated hash value.
        """

        if board_before_move is None:
            raise ValueError(
                "board_before_move is required "
                "for incremental update fallback."
            )

        board_copy = board_before_move.copy(stack=False)
        board_copy.push(move)

        return self.hash_board(board_copy)