# engine/legality.py
from __future__ import annotations

import chess


class LegalityChecker:
    """
    High-level rule analyst and match termination manager.

    Responsibilities:
    - Evaluates checks, checkmates, and stalemates.
    - Manages draw validation criteria (three-fold, material, 50-move rule).
    - Acts as the sole authoritative referee for both search trees and outer APIs.

    Strict Constraints:
    - Entirely stateless utility class; utilizing __slots__ = () to prevent allocations.
    - Contains absolute ownership of game-over queries across the engine.
    """

    __slots__ = ()

    # ---------------------------------------------------------
    # Primitive Condition Queries
    # ---------------------------------------------------------

    @staticmethod
    def in_check(board: chess.Board) -> bool:
        """Determine whether the active side to move is under check."""
        return board.is_check()

    @staticmethod
    def is_checkmate(board: chess.Board) -> bool:
        """Determine whether the current layout is checkmate."""
        return board.is_checkmate()

    @staticmethod
    def is_stalemate(board: chess.Board) -> bool:
        """Determine whether the current layout is stalemate."""
        return board.is_stalemate()

    # ---------------------------------------------------------
    # OPTIMIZED FOR ENGINE SEARCH CORE
    # ---------------------------------------------------------

    @staticmethod
    def is_search_terminal(board: chess.Board) -> bool:
        """
        A highly optimized terminal check designed for recursive search trees.
        
        Bypasses expensive historical repetition evaluations if foundational 
        game-ending conditions haven't been met, preserving engine velocity (Nps).
        """
        # 1. Evaluate primitive, instant lookups first (No allocation overhead)
        if not board.legal_moves:
            return True
            
        if board.halfmove_clock >= 100:  # 50 moves * 2 plies
            return True
            
        if board.is_insufficient_material():
            return True

        # 2. Heuristic bypass for three-fold repetition:
        # A 3-fold repetition is mathematically impossible if less than 8 plies have occurred.
        if len(board.move_stack) < 8:
            return False

        return board.is_repetition(count=3)

    # ---------------------------------------------------------
    # BOUNDARY ZONE INTERFACE METHODS
    # ---------------------------------------------------------

    @staticmethod
    def is_draw(board: chess.Board) -> bool:
        """Determine whether the position is drawn by strict rule or active claim."""
        return board.is_draw(claim_draw=True)

    @staticmethod
    def is_insufficient_material(board: chess.Board) -> bool:
        """Check for structural insufficient mating combinations on the board grids."""
        return board.is_insufficient_material()

    @staticmethod
    def is_repetition(board: chess.Board) -> bool:
        """Check whether the active layout matches a past identical state history."""
        return board.is_repetition()

    @staticmethod
    def is_fifty_move_rule(board: chess.Board) -> bool:
        """Check whether the fifty-move counter draw threshold can be claimed."""
        return board.can_claim_fifty_moves()

    @staticmethod
    def is_terminal(board: chess.Board) -> bool:
        """Determine completely whether the position is game over at the outer API boundary."""
        return board.is_game_over(claim_draw=True)

    @staticmethod
    def get_result(board: chess.Board) -> str:
        """Get the official tournament game outcome scoreboard string ("1-0", "0-1", "1/2-1/2", "*")."""
        return board.result(claim_draw=True)
