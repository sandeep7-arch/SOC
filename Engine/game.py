# engine/game.py
from __future__ import annotations

import io
from typing import Optional

import chess
import chess.pgn

from .board import ChessBoard
from .legality import LegalityChecker
from .move import Move
from .movegen import MoveGenerator


class ChessGame:
    """
    High-level game controller.

    Responsibilities:
    - Manage overall game lifecycle
    - Coordinate underlying board state mutations
    - Apply and undo moves safely at the application boundary
    - Expose game validation stats and handle PGN generation

    Notes:
    - Acts as the main API interface for a GUI, CLI, or training environment.
    - Zero performance critical code runs here; this is entirely business logic.
    """

    def __init__(self) -> None:
        self.start_new_game()

    # ---------------------------------------------------------
    # Game Lifecycle
    # ---------------------------------------------------------

    def start_new_game(self) -> None:
        """Initialize a fresh chess game from the standard starting layout."""
        self._board_wrapper = ChessBoard()

    # ---------------------------------------------------------
    # Move Handling
    # ---------------------------------------------------------

    def apply_move(self, move: str | Move | chess.Move) -> bool:
        """
        Apply a move to the game.

        Args:
            move: Can be a UCI string ("e2e4"), wrapped engine Move, 
                  or raw chess.Move.

        Returns:
            True if the move was valid and applied, False otherwise.
        """
        # If it's our wrapped Move token, unwrap its raw chess.Move core
        if isinstance(move, Move):
            target_move = move.chess_move
        else:
            target_move = move

        # Delegate execution down to our optimized board state manager
        return self._board_wrapper.make_move(target_move)

    def undo_move(self) -> Optional[chess.Move]:
        """
        Undo the most recent move played.

        Returns:
            The raw chess.Move that was reverted, or None if stack is empty.
        """
        return self._board_wrapper.undo_move()

    # ---------------------------------------------------------
    # Accessors & State Metadata
    # ---------------------------------------------------------

    def get_board(self) -> ChessBoard:
        """Get the authoritative board wrapper object."""
        return self._board_wrapper

    def get_move_counters(self) -> tuple[int, int]:
        """
        Get current game progression metrics.
        Returns:
            tuple: (halfmove_clock, fullmove_number)
        """
        return (self._board_wrapper.halfmove_clock, self._board_wrapper.fullmove_number)

    def get_castling_rights(self) -> tuple[bool, bool, bool, bool]:
        """
        Get current castling metadata instantly.
        Returns:
            tuple: (White Kingside, White Queenside, Black Kingside, Black Queenside)
        """
        return self._board_wrapper.castling_rights_tuple

    # ---------------------------------------------------------
    # Move Generation Access (Boundary Zone)
    # ---------------------------------------------------------

    def get_legal_moves(self) -> list[Move]:
        """
        Fetch all legal moves packaged as clean, wrapped objects.
        Perfect for a UI dropdown or a human player move validation system.
        """
        return MoveGenerator.generate_legal_moves(self._board_wrapper.board)

    # ---------------------------------------------------------
    # Game Termination Rules
    # ---------------------------------------------------------

    def is_game_over(self) -> bool:
        """Check if the game has ended by any strict official chess rule."""
        return LegalityChecker.is_terminal(self._board_wrapper.board)

    def get_result(self) -> str:
        """Get the official game outcome score ("1-0", "0-1", "1/2-1/2", "*")."""
        return LegalityChecker.get_result(self._board_wrapper.board)

    # ---------------------------------------------------------
    # PGN Export
    # ---------------------------------------------------------

    def export_pgn(self) -> str:
        """
        Export the entire move history as an official, standard PGN record.

        Returns:
            A clean PGN format text string string.
        """
        game = chess.pgn.Game()
        game.headers["Event"] = "Engine Architecture Match"
        game.headers["Result"] = self.get_result()

        node = game
        # Step through the raw historical move log saved inside the stack
        for move in self._board_wrapper.board.move_stack:
            node = node.add_variation(move)

        exporter = chess.pgn.StringExporter(
            headers=True,
            variations=False,
            comments=False,
        )
        return game.accept(exporter)

    # ---------------------------------------------------------
    # Serialization Convenience
    # ---------------------------------------------------------

    def current_fen(self) -> str:
        """Returns the single-string FEN snapshot of the exact current position."""
        return self._board_wrapper.export_fen()

    def __repr__(self) -> str:
        return (
            f"ChessGame("
            f"result='{self.get_result()}', "
            f"game_over={self.is_game_over()}, "
            f"ply_count={len(self._board_wrapper.board.move_stack)})"
        )
