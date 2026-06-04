# engine/movegen.py
from __future__ import annotations

import chess
from typing import List, Iterator
from .move import Move


class MoveGenerator:
    """
    The Single Source of Truth for move generation.
    
    Provides fast iterators for the engine search core, 
    and clean wrapped lists for the boundary/API layer.
    """
    __slots__ = ()

    # --- ENGINE CORE METHODS (Zero-Allocation Iterators) ---

    @staticmethod
    def raw_legal_moves(board: chess.Board) -> chess.LegalMoveGenerator:
        """Yields raw chess.Move objects lazily. Main search loop driver."""
        return board.legal_moves

    @staticmethod
    def raw_captures(board: chess.Board) -> Iterator[chess.Move]:
        """Yields raw capture moves lazily. Used in Quiescence Search."""
        for move in board.legal_moves:
            if board.is_capture(move):
                yield move

    @staticmethod
    def raw_quiet_moves(board: chess.Board) -> Iterator[chess.Move]:
        """Yields raw quiet moves lazily. Used in Move Ordering."""
        for move in board.legal_moves:
            if not board.is_capture(move):
                yield move

    # --- BOUNDARY / API METHODS (Wrapped Object Lists) ---

    @staticmethod
    def generate_legal_moves(board: chess.Board) -> List[Move]:
        """Returns a list of wrapped engine Move objects for the GUI/API."""
        return [Move(move) for move in board.legal_moves]

    @staticmethod
    def generate_captures(board: chess.Board) -> List[Move]:
        """Returns a list of wrapped capture moves."""
        return [Move(move) for move in board.legal_moves if board.is_capture(move)]

    # --- UTILITIES ---

    @staticmethod
    def count_legal_moves(board: chess.Board) -> int:
        """Fast bitwise legal move count without object allocation."""
        return board.legal_moves.count()
