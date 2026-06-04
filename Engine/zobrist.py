# engine/zobrist.py
from __future__ import annotations

import random
from typing import Optional
import chess


class ZobristHasher:
    """
    Zobrist hashing interface for chess positions.
    
    Provides custom 64-bit fingerprint validation alongside native
    python-chess cached hash keys for use in the Transposition Table (TT).
    """

    __slots__ = (
        "_piece_keys",
        "_side_to_move_key",
        "_castling_keys",
        "_ep_file_keys",
    )

    DEFAULT_SEED = 2026

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        rng = random.Random(seed)

        # 12 piece types (6 White, 6 Black) x 64 squares
        self._piece_keys = [
            [rng.getrandbits(64) for _ in range(64)]
            for _ in range(12)
        ]

        # Turn descriptor key
        self._side_to_move_key = rng.getrandbits(64)

        # Castling rights map
        self._castling_keys = {
            "K": rng.getrandbits(64),
            "Q": rng.getrandbits(64),
            "k": rng.getrandbits(64),
            "q": rng.getrandbits(64),
        }

        # En passant target files
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

        # Map piece layouts
        for square, piece in board.piece_map().items():
            offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = offset + (piece.piece_type - 1)
            h ^= self._piece_keys[piece_idx][square]

        # Map turn
        if board.turn == chess.BLACK:
            h ^= self._side_to_move_key

        # Map castling rights
        if board.has_kingside_castling_rights(chess.WHITE): h ^= self._castling_keys["K"]
        if board.has_queenside_castling_rights(chess.WHITE): h ^= self._castling_keys["Q"]
        if board.has_kingside_castling_rights(chess.BLACK): h ^= self._castling_keys["k"]
        if board.has_queenside_castling_rights(chess.BLACK): h ^= self._castling_keys["q"]

        # Map en passant targeting
        if board.ep_square is not None:
            h ^= self._ep_file_keys[chess.square_file(board.ep_square)]

        return h
