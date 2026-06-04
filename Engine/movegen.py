from __future__ import annotations

import chess
from typing import List, Iterator, Union

from .move import Move


class MoveGenerator:
    """
    Legal move generation layer built on top of python-chess.

    Responsibilities:
    - Generate legal moves (raw objects for search, wrapped objects for APIs)
    - Distinguish capture vs. quiet moves for Quiescence Search and Move Ordering
    - Validate move legality

    Notes:
    - Optimized for alpha-beta search usage via Iterator methods.
    """

    __slots__ = ()

    # -------------------------------------------------------------
    # CORE ENGINE SEARCH METHODS (Blazing Fast, Zero Allocation)
    # -------------------------------------------------------------

    @staticmethod
    def raw_legal_moves(board: chess.Board) -> chess.LegalMoveGenerator:
        """
        Returns the raw, lazy legal move generator.
        Bypasses list allocation completely. Critical for Alpha-Beta search.
        """
        return board.legal_moves

    @staticmethod
    def raw_captures(board: chess.Board) -> Iterator[chess.Move]:
        """
        Yields raw capture moves one by one.
        Highly critical for Quiescence Search (searching only tactical captures).
        """
        for move in board.legal_moves:
            if board.is_capture(move):
                yield move

    @staticmethod
    def raw_quiet_moves(board: chess.Board) -> Iterator[chess.Move]:
        """
        Yields raw quiet (non-capture) moves one by one.
        """
        for move in board.legal_moves:
            if not board.is_capture(move):
                yield move
    @staticmethod
    def is_raw_castling(board: chess.Board, move: chess.Move) -> bool:
        """Fast bitboard-level check to see if a raw move is a castling move."""
        return board.is_castling(move)

    @staticmethod
    def is_raw_en_passant(board: chess.Board, move: chess.Move) -> bool:
        """Fast bitboard-level check to see if a raw move is en passant."""
        return board.is_en_passant(move)
    # -------------------------------------------------------------
    # BOUNDARY ZONE METHODS (Convenient API, Wrapped Objects)
    # -------------------------------------------------------------

    @staticmethod
    def generate_legal_moves(board: chess.Board) -> List[Move]:
        """Generate all legal moves wrapped as engine Move objects for the API."""
        return [Move(move) for move in board.legal_moves]

    @staticmethod
    def generate_captures(board: chess.Board) -> List[Move]:
        """Generate legal capture moves wrapped as engine Move objects."""
        return [Move(move) for move in board.legal_moves if board.is_capture(move)]

    @staticmethod
    def generate_quiet_moves(board: chess.Board) -> List[Move]:
        """Generate legal non-capture moves wrapped as engine Move objects."""
        return [Move(move) for move in board.legal_moves if not board.is_capture(move)]

    # -------------------------------------------------------------
    # Legality & Utilities
    # -------------------------------------------------------------

    @staticmethod
    def is_move_legal(board: chess.Board, move: Union[Move, chess.Move]) -> bool:
        """Check if a wrapped engine Move or a raw chess.Move is legal."""
        raw_move = move.chess_move if isinstance(move, Move) else move
        return raw_move in board.legal_moves

    @staticmethod
    def count_legal_moves(board: chess.Board) -> int:
        """
        Fast move count helper using python-chess's bitwise counting.
        Does zero object creation. Excellent for evaluation heuristics.
        """
        return board.legal_moves.count()
