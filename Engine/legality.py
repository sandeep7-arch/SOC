# engine/legality.py

from __future__ import annotations

import chess


class LegalityChecker:
    """
    High-level legality and game termination utilities.

    Purpose:
    - Provide a centralized interface for game-state validation.
    - Expose rule-related queries to search and UCI layers.
    - Delegate all rule logic to python-chess.

    Notes:
    - No custom chess rule implementation.
    - No move generation.
    - No search logic.
    - Stateless utility class.
    """

    __slots__ = ()

    # ---------------------------------------------------------
    # Check Status
    # ---------------------------------------------------------

    @staticmethod
    def in_check(board: chess.Board) -> bool:
        """
        Determine whether the side to move is in check.

        Args:
            board: python-chess Board

        Returns:
            True if side to move is in check.
        """
        return board.is_check()

    # ---------------------------------------------------------
    # Checkmate
    # ---------------------------------------------------------

    @staticmethod
    def is_checkmate(board: chess.Board) -> bool:
        """
        Determine whether the current position is checkmate.

        Args:
            board: python-chess Board

        Returns:
            True if checkmate.
        """
        return board.is_checkmate()

    # ---------------------------------------------------------
    # Stalemate
    # ---------------------------------------------------------

    @staticmethod
    def is_stalemate(board: chess.Board) -> bool:
        """
        Determine whether the current position is stalemate.

        Args:
            board: python-chess Board

        Returns:
            True if stalemate.
        """
        return board.is_stalemate()

    # ---------------------------------------------------------
    # Draw Detection
    # ---------------------------------------------------------

    @staticmethod
    def is_draw(board: chess.Board) -> bool:
        """
        Determine whether the position is drawn.

        Includes:
        - Stalemate
        - Insufficient material
        - Fifty-move rule
        - Threefold repetition
        - Other draw conditions recognized by python-chess

        Args:
            board: python-chess Board

        Returns:
            True if drawn.
        """
        return board.is_draw(claim_draw=True)

    # ---------------------------------------------------------
    # Material Draw
    # ---------------------------------------------------------

    @staticmethod
    def is_insufficient_material(board: chess.Board) -> bool:
        """
        Check insufficient mating material.

        Examples:
        - K vs K
        - K+B vs K
        - K+N vs K

        Args:
            board: python-chess Board

        Returns:
            True if insufficient material.
        """
        return board.is_insufficient_material()

    # ---------------------------------------------------------
    # Repetition
    # ---------------------------------------------------------

    @staticmethod
    def is_repetition(board: chess.Board) -> bool:
        """
        Check whether the current position is repeated.

        Uses python-chess repetition tracking.

        Args:
            board: python-chess Board

        Returns:
            True if current position has occurred before.
        """
        return board.is_repetition()

    # ---------------------------------------------------------
    # Fifty-Move Rule
    # ---------------------------------------------------------

    @staticmethod
    def is_fifty_move_rule(board: chess.Board) -> bool:
        """
        Check whether the fifty-move rule can be claimed.

        Args:
            board: python-chess Board

        Returns:
            True if fifty-move draw is claimable.
        """
        return board.can_claim_fifty_moves()

    # ---------------------------------------------------------
    # Combined Terminal Check
    # ---------------------------------------------------------

    @staticmethod
    def is_terminal(board: chess.Board) -> bool:
        """
        Determine whether the position is terminal.

        Includes:
        - Checkmate
        - Stalemate
        - Draws

        Args:
            board: python-chess Board

        Returns:
            True if game is over.
        """
        return board.is_game_over(claim_draw=True)

    # ---------------------------------------------------------
    # Result Helper
    # ---------------------------------------------------------

    @staticmethod
    def get_result(board: chess.Board) -> str:
        """
        Get official game result.

        Returns:
            "1-0"      -> White wins
            "0-1"      -> Black wins
            "1/2-1/2" -> Draw
            "*"        -> Game ongoing
        """
        return board.result(claim_draw=True)