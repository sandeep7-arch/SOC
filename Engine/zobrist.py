# engine/zobrist.py
from __future__ import annotations

import random
import chess


class ZobristHasher:
    """
    Zobrist hashing mechanism for mapping unique chess layouts.
    
    Responsibilities:
    - Provides access to native, incrementally tracked 64-bit keys for TT cache logic.
    - Generates deterministically seeded data tables for validation testing.
    - Computes full-state position identity hashes from scratch for baseline testing.

    Strict Constraints:
    - Stateless execution model for manual hashing processes.
    - Zero state management or mutation ownership.
    """

    __slots__ = (
        "_piece_keys",
        "_side_to_move_key",
        "_castling_keys",
        "_ep_file_keys",
    )

    DEFAULT_SEED = 2026

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        """Initialize arrays filled with pseudo-random 64-bit integers."""
        rng = random.Random(seed)

        # 12 distinct piece options (6 White types, 6 Black types) x 64 squares
        self._piece_keys = [
            [rng.getrandbits(64) for _ in range(64)]
            for _ in range(12)
        ]

        # Side to move modifier flag
        self._side_to_move_key = rng.getrandbits(64)

        # Castling rights dictionary options
        self._castling_keys = {
            "K": rng.getrandbits(64),
            "Q": rng.getrandbits(64),
            "k": rng.getrandbits(64),
            "q": rng.getrandbits(64),
        }

        # En passant target files (Files A through H)
        self._ep_file_keys = [rng.getrandbits(64) for _ in range(8)]

    @staticmethod
    def get_native_hash(board: chess.Board) -> int:
        """
        Fetches the ultra-fast, incrementally updated hash tracking key 
        straight from the python-chess core. Use this inside search loops!
        """
        return board.zobrist_hash()

    def hash_board(self, board: chess.Board) -> int:
        """
        Manually compute a full 64-bit Zobrist hash from scratch.
        Excellent for sanity checks, testing, and debugging.
        """
        h = 0

        # Map active piece configurations
        for square, piece in board.piece_map().items():
            offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = offset + (piece.piece_type - 1)
            h ^= self._piece_keys[piece_idx][square]

        # Map active side to move color layout modification
        if board.turn == chess.BLACK:
            h ^= self._side_to_move_key

        # Map castling rights status variations
        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= self._castling_keys["K"]
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self._castling_keys["Q"]
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self._castling_keys["k"]
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self._castling_keys["q"]

        # Map en passant accessibility
        if board.ep_square is not None:
            h ^= self._ep_file_keys[chess.square_file(board.ep_square)]

        return h
