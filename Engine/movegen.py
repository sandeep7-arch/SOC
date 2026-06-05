# engine/movegen.py
from __future__ import annotations

import chess
from typing import List, Iterator, Union

from .move import Move


class MoveGenerator:
    """
    Stateless move generation layer built on top of python-chess bitboards.

    Responsibilities:
    - Generates lazy move iterators for deep engine recursive searches.
    - Packs fully-allocated Move object arrays for API boundary consumers.
    - Segregates tactile captures from quiet positional alternatives.

    Strict Constraints:
    - Entirely stateless instance layer using __slots__ = () to block memory footprints.
    - Focuses solely on generation metrics; zero state mutations occur here.
    """

    __slots__ = ()

    # -------------------------------------------------------------
    # CORE ENGINE SEARCH METHODS (Blazing Fast, Zero Allocation)
    # -------------------------------------------------------------

    @staticmethod
    def raw_legal_moves(board: chess.Board) -> chess.LegalMoveGenerator:
        """
        Returns the raw, lazy legal move generator stream hook.
        Bypasses list array allocation entirely. Essential for Alpha-Beta search nodes.
        """
        return board.legal_moves

    @staticmethod
    def raw_captures(board: chess.Board) -> Iterator[chess.Move]:
        """
        Yields raw tactical capture moves sequentially out of the state bitmask.
        Highly critical to drive Quiescence Search pipelines without bloating the heap.
        """
        # Using a direct generator comprehension keeps this lazy and ultra-fast
        return (move for move in board.legal_moves if board.is_capture(move))

    @staticmethod
    def raw_quiet_moves(board: chess.Board) -> Iterator[chess.Move]:
        """
        Yields raw quiet (non-capture) moves sequentially out of the state bitmask.
        Optimized for progressive sorting models and late-move reduction steps.
        """
        return (move for move in board.legal_moves if not board.is_capture(move))

    @staticmethod
    def is_raw_castling(board: chess.Board, move: chess.Move) -> bool:
        """Fast bitmask coordinate check to see if a raw action is a castling maneuver."""
        return board.is_castling(move)

    @staticmethod
    def is_raw_en_passant(board: chess.Board, move: chess.Move) -> bool:
        """Fast bitmask coordinate check to see if a raw action targets an en passant square."""
        return board.is_en_passant(move)

    # -------------------------------------------------------------
    # BOUNDARY ZONE METHODS (Convenient API, Wrapped Objects)
    # -------------------------------------------------------------

    @staticmethod
    def generate_legal_moves(board: chess.Board) -> List[Move]:
        """Collects and packs all legal moves as wrapped object lists for the API layer."""
        return [Move(move) for move in board.legal_moves]

    @staticmethod
    def generate_captures(board: chess.Board) -> List[Move]:
        """Collects and packs only valid capture choices into an array list footprint."""
        return [Move(move) for move in board.legal_moves if board.is_capture(move)]

    @staticmethod
    def generate_quiet_moves(board: chess.Board) -> List[Move]:
        """Collects and packs only non-capture alternatives into an array list footprint."""
        return [Move(move) for move in board.legal_moves if not board.is_capture(move)]

    # -------------------------------------------------------------
    # Containment Verification & Analytics
    # -------------------------------------------------------------

    @staticmethod
    def is_move_legal(board: chess.Board, move: Union[Move, chess.Move]) -> bool:
        """Performs a direct bitboard containment test to confirm if a specific move option is legal."""
        raw_move = move.chess_move if isinstance(move, Move) else move
        return raw_move in board.legal_moves

    @staticmethod
    def count_legal_moves(board: chess.Board) -> int:
        """
        Performs optimized bitwise population count tracking of available legal moves.
        Zero object instantiations occur. Ideal for fast mobility evaluation features.
        """
        return board.legal_moves.count()
