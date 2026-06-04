# engine/state.py

from __future__ import annotations

import chess
from typing import Any, Dict, List, Optional


class GameState:
    """
    Extracts and manages game state information from a python-chess.Board.

    Responsibilities:
    - Halfmove clock tracking
    - Fullmove number tracking
    - Castling rights extraction
    - En passant square tracking
    - Check/checkmate detection
    - Repetition state interface

    Notes:
    - No move generation.
    - No search logic.
    - No evaluation logic.
    - Uses python-chess as the single source of truth.
    """

    def __init__(self, board: chess.Board) -> None:
        """
        Args:
            board:
                Active python-chess Board instance.
        """
        self._board = board

    # ------------------------------------------------------------------
    # Move Counters
    # ------------------------------------------------------------------

    @property
    def halfmove_clock(self) -> int:
        """
        Number of halfmoves since the last pawn move or capture.
        Used for fifty-move rule detection.
        """
        return self._board.halfmove_clock

    @property
    def fullmove_number(self) -> int:
        """
        Move number in the game.

        Starts at 1 and increments after Black moves.
        """
        return self._board.fullmove_number

    # ------------------------------------------------------------------
    # Castling Rights
    # ------------------------------------------------------------------

    @property
    def castling_rights(self) -> Dict[str, bool]:
        """
        Returns castling availability for both sides.
        """
        return {
            "white_kingside": self._board.has_kingside_castling_rights(
                chess.WHITE
            ),
            "white_queenside": self._board.has_queenside_castling_rights(
                chess.WHITE
            ),
            "black_kingside": self._board.has_kingside_castling_rights(
                chess.BLACK
            ),
            "black_queenside": self._board.has_queenside_castling_rights(
                chess.BLACK
            ),
        }

    # ------------------------------------------------------------------
    # En Passant
    # ------------------------------------------------------------------

    @property
    def en_passant_square(self) -> Optional[str]:
        """
        Returns en passant target square in algebraic notation.

        Example:
            'e3'

        Returns:
            None if no en passant target exists.
        """
        if self._board.ep_square is None:
            return None

        return chess.square_name(self._board.ep_square)

    # ------------------------------------------------------------------
    # Check / Mate Status
    # ------------------------------------------------------------------

    @property
    def is_check(self) -> bool:
        """
        Whether the current side to move is in check.
        """
        return self._board.is_check()

    @property
    def is_checkmate(self) -> bool:
        """
        Whether the position is checkmate.
        """
        return self._board.is_checkmate()

    @property
    def is_stalemate(self) -> bool:
        """
        Whether the position is stalemate.
        """
        return self._board.is_stalemate()

    # ------------------------------------------------------------------
    # Repetition Interface
    # ------------------------------------------------------------------

    def repetition_info(self) -> Dict[str, Any]:
        """
        Lightweight interface for repetition-related state.

        Actual repetition storage / TT logic belongs elsewhere.

        Returns:
            Dictionary describing current repetition status.
        """
        return {
            "is_repetition": self._board.is_repetition(),
            "can_claim_threefold": self._board.can_claim_threefold_repetition(),
            "can_claim_fifty_moves": self._board.can_claim_fifty_moves(),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_state(self) -> Dict[str, Any]:
        """
        Serialize state information into a dictionary.

        Returns:
            Dictionary suitable for logging,
            debugging, persistence, or IPC.
        """
        return {
            "halfmove_clock": self.halfmove_clock,
            "fullmove_number": self.fullmove_number,
            "castling_rights": self.castling_rights,
            "en_passant_square": self.en_passant_square,
            "is_check": self.is_check,
            "is_checkmate": self.is_checkmate,
            "is_stalemate": self.is_stalemate,
            "repetition": self.repetition_info(),
        }

    # ------------------------------------------------------------------
    # Loading State
    # ------------------------------------------------------------------

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state values where applicable.

        Important:
        The board remains the authoritative source.
        This method is intended for restoring metadata
        when reconstructing a board elsewhere.

        Args:
            state:
                Dictionary produced by serialize_state().
        """

        if "halfmove_clock" in state:
            self._board.halfmove_clock = int(state["halfmove_clock"])

        if "fullmove_number" in state:
            self._board.fullmove_number = int(state["fullmove_number"])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GameState("
            f"halfmove_clock={self.halfmove_clock}, "
            f"fullmove_number={self.fullmove_number}, "
            f"check={self.is_check}, "
            f"checkmate={self.is_checkmate})"
        )