# engine/movegen.py

from __future__ import annotations

import chess
from typing import List

from .move import Move


class MoveGenerator:
    """
    Legal move generation layer built on top of python-chess.

    Responsibilities:
    - Generate all legal moves
    - Generate capture moves
    - Generate quiet (non-capture) moves
    - Validate move legality
    - Generate check evasion moves

    Notes:
    - All legality comes from python-chess.
    - No custom chess rules are implemented.
    - Optimized for alpha-beta search usage.
    - Returns engine Move objects only.
    """

    __slots__ = ()

    # -------------------------------------------------------------
    # Legal Move Generation
    # -------------------------------------------------------------

    @staticmethod
    def generate_legal_moves(board: chess.Board) -> List[Move]:
        """
        Generate all legal moves.

        Args:
            board:
                python-chess Board instance.

        Returns:
            List of engine Move objects.
        """
        return [Move(move) for move in board.legal_moves]

    # -------------------------------------------------------------
    # Capture Generation
    # -------------------------------------------------------------

    @staticmethod
    def generate_captures(board: chess.Board) -> List[Move]:
        """
        Generate legal capture moves only.

        Useful for:
        - Quiescence Search
        - Move Ordering
        - Tactical Search

        Args:
            board:
                python-chess Board instance.

        Returns:
            List of capture moves.
        """
        captures = []

        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append(Move(move))

        return captures

    # -------------------------------------------------------------
    # Quiet Move Generation
    # -------------------------------------------------------------

    @staticmethod
    def generate_quiet_moves(board: chess.Board) -> List[Move]:
        """
        Generate legal non-capture moves.

        Args:
            board:
                python-chess Board instance.

        Returns:
            List of quiet moves.
        """
        quiet_moves = []

        for move in board.legal_moves:
            if not board.is_capture(move):
                quiet_moves.append(Move(move))

        return quiet_moves

    # -------------------------------------------------------------
    # Legality Check
    # -------------------------------------------------------------

    @staticmethod
    def is_move_legal(board: chess.Board, move: Move) -> bool:
        """
        Check if move is legal in current position.

        Args:
            board:
                python-chess Board instance.

            move:
                Engine Move object.

        Returns:
            True if legal.
        """
        return move.chess_move in board.legal_moves

    # -------------------------------------------------------------
    # Check Evasion Generation
    # -------------------------------------------------------------

    @staticmethod
    def get_check_evasions(board: chess.Board) -> List[Move]:
        """
        Generate moves that evade check.

        When not in check:
            Returns all legal moves.

        When in check:
            python-chess legal moves already
            contain only valid evasions.

        Args:
            board:
                python-chess Board instance.

        Returns:
            List of legal check-evasion moves.
        """
        return [Move(move) for move in board.legal_moves]

    # -------------------------------------------------------------
    # Utility Helpers
    # -------------------------------------------------------------

    @staticmethod
    def count_legal_moves(board: chess.Board) -> int:
        """
        Fast move count helper.

        Useful for:
        - Search diagnostics
        - Move ordering stats
        - Perft testing

        Args:
            board:
                python-chess Board instance.

        Returns:
            Number of legal moves.
        """
        return board.legal_moves.count()