from __future__ import annotations

import chess


class LegalityChecker:
    """
    High-level legality and game termination utilities.
    
    Optimized for dual-use: Fast heuristic drops for alpha-beta search cores,
    and rigorous rule compliance for outer game boundaries.
    """

    __slots__ = ()

    @staticmethod
    def in_check(board: chess.Board) -> bool:
        """Determine whether the side to move is in check."""
        return board.is_check()

    @staticmethod
    def is_checkmate(board: chess.Board) -> bool:
        """Determine whether the current position is checkmate."""
        return board.is_checkmate()

    @staticmethod
    def is_stalemate(board: chess.Board) -> bool:
        """Determine whether the current position is stalemate."""
        return board.is_stalemate()

    # ---------------------------------------------------------
    # OPTIMIZED FOR ENGINE SEARCH CORE
    # ---------------------------------------------------------

    @staticmethod
    def is_search_terminal(board: chess.Board) -> bool:
        """
        A highly optimized terminal check designed for recursive search trees.
        
        Bypasses expensive repetition checks if basic game-ending conditions 
        haven't been met yet, protecting the engine's Nodes Per Second (Nps).
        """
        # 1. Quick check: fifty-move rule or structural stalemates/mates
        # board.is_variant_end() or checking if legal moves count is zero is incredibly cheap
        if not board.legal_moves:
            return True
            
        if board.halfmove_clock >= 100:  # 50 moves * 2 plies
            return True
            
        if board.is_insufficient_material():
            return True

        # 2. Only check repetition if the position history warrants it
        # (python-chess can quickly check if the current position has a repeat flag)
        return board.is_repetition(count=3)

    # ---------------------------------------------------------
    # BOUNDARY ZONE METHODS
    # ---------------------------------------------------------

    @staticmethod
    def is_draw(board: chess.Board) -> bool:
        """Determine whether the position is drawn by strict rule definition."""
        return board.is_draw(claim_draw=True)

    @staticmethod
    def is_insufficient_material(board: chess.Board) -> bool:
        """Check insufficient mating material."""
        return board.is_insufficient_material()

    @staticmethod
    def is_repetition(board: chess.Board) -> bool:
        """Check whether the current position has repeated."""
        return board.is_repetition()

    @staticmethod
    def is_fifty_move_rule(board: chess.Board) -> bool:
        """Check whether the fifty-move rule can be claimed."""
        return board.can_claim_fifty_moves()

    @staticmethod
    def is_terminal(board: chess.Board) -> bool:
        """Determine completely whether the position is game over at API boundary."""
        return board.is_game_over(claim_draw=True)

    @staticmethod
    def get_result(board: chess.Board) -> str:
        """Get official game result string ("1-0", "0-1", "1/2-1/2", "*")."""
        return board.result(claim_draw=True)
