# engine/fen.py

from __future__ import annotations

import chess


class FENHandler:
    """
    FEN parsing and serialization utilities.

    Responsibilities:
    - Parse FEN strings into python-chess boards
    - Export board positions as FEN
    - Validate FEN strings
    - Normalize FEN representations

    Notes:
    - Uses python-chess as the authoritative parser.
    - No search logic.
    - No evaluation logic.
    - Fully compatible with UCI position commands.
    """

    __slots__ = ()

    # ---------------------------------------------------------
    # Parsing
    # ---------------------------------------------------------

    @staticmethod
    def parse_fen(fen_str: str) -> chess.Board:
        """
        Parse a FEN string into a chess.Board.

        Args:
            fen_str:
                Valid FEN string.

        Returns:
            chess.Board instance.

        Raises:
            ValueError:
                If FEN is invalid.
        """
        try:
            return chess.Board(fen=fen_str)
        except ValueError as exc:
            raise ValueError(f"Invalid FEN: {fen_str}") from exc

    # ---------------------------------------------------------
    # Export
    # ---------------------------------------------------------

    @staticmethod
    def export_fen(board: chess.Board) -> str:
        """
        Export board state as FEN.

        Args:
            board:
                python-chess Board instance.

        Returns:
            Standard FEN string.
        """
        return board.fen()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @staticmethod
    def validate_fen(fen_str: str) -> bool:
        """
        Validate a FEN string.

        Args:
            fen_str:
                Candidate FEN string.

        Returns:
            True if valid.
            False otherwise.
        """
        try:
            chess.Board(fen=fen_str)
            return True
        except ValueError:
            return False

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------

    @staticmethod
    def normalize_fen(fen_str: str) -> str:
        """
        Normalize a FEN string.

        This ensures:
        - Consistent formatting
        - Proper spacing
        - Canonical python-chess representation

        Args:
            fen_str:
                Input FEN string.

        Returns:
            Normalized FEN string.

        Raises:
            ValueError:
                If FEN is invalid.
        """
        board = FENHandler.parse_fen(fen_str)
        return board.fen()

    # ---------------------------------------------------------
    # Starting Position Helper
    # ---------------------------------------------------------

    @staticmethod
    def starting_fen() -> str:
        """
        Return standard chess starting position FEN.

        Returns:
            Initial position FEN.
        """
        return chess.STARTING_FEN