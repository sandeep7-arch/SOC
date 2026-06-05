# engine/game.py
from __future__ import annotations

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
    - Manage overall game lifecycle.
    - Coordinate underlying board state mutations via the board wrapper.
    - Route boundary system moves (UCI/wrapped string formats) to state primitives.
    - Expose unified status metadata for UI panels and export PGN sequences.

    Strict Constraints:
    - Zero algorithmic engine loops or math logic.
    - Acts entirely as a proxy boundary layer passing calls down to the core modules.
    """

    def __init__(self) -> None:
        self.start_new_game()

    # ---------------------------------------------------------
    # Game Lifecycle
    # ---------------------------------------------------------

    def start_new_game(self) -> None:
        """Initialize a fresh chess game environment from the standard starting layout."""
        self._board_wrapper = ChessBoard()

    # ---------------------------------------------------------
    # Move Handling
    # ---------------------------------------------------------

    def apply_move(self, move: str | Move | chess.Move) -> bool:
        """
        Apply a move to the active match state.

        Args:
            move: Can be a UCI string ("e2e4"), wrapped engine Move token, 
                  or raw chess.Move struct.

        Returns:
            True if the move was successfully validated and pushed, False otherwise.
        """
        # Unwrap our custom Move token if encountered at the boundary line
        if isinstance(move, Move):
            target_move: str | chess.Move = move.chess_move
        else:
            target_move = move

        # Delegate pure state alteration directly down to the board instance
        return self._board_wrapper.make_move(target_move)

    def undo_move(self) -> Optional[chess.Move]:
        """
        Undo the most recent move played on the state stack history.

        Returns:
            The raw chess.Move that was popped off, or None if the log is empty.
        """
        return self._board_wrapper.undo_move()

    # ---------------------------------------------------------
    # Accessors & State Metadata Proxy Routing
    # ---------------------------------------------------------

    def get_board(self) -> ChessBoard:
        """Get the authoritative board data-vault wrapper object."""
        return self._board_wrapper

    def get_move_counters(self) -> tuple[int, int]:
        """
        Fetch the current game metric values.
        Returns:
            tuple: (halfmove_clock, fullmove_number)
        """
        return (self._board_wrapper.halfmove_clock, self._board_wrapper.fullmove_number)

    def get_castling_rights(self) -> tuple[bool, bool, bool, bool]:
        """
        Fetch active castling permission flags instantly.
        Returns:
            tuple: (White Kingside, White Queenside, Black Kingside, Black Queenside)
        """
        return self._board_wrapper.castling_rights_tuple

    def get_active_turn(self) -> str:
        """Fetch human-readable text label of active side to move ("white"/"black")."""
        return self._board_wrapper.get_turn_color()

    # ---------------------------------------------------------
    # Move Generation Proxy Routing
    # ---------------------------------------------------------

    def get_legal_moves(self) -> list[Move]:
        """
        Fetch all legal variations wrapped cleanly into API Move objects.
        Ideal for external consumption like a visual match interface.
        """
        return MoveGenerator.generate_legal_moves(self._board_wrapper.board)

    # ---------------------------------------------------------
    # Game Termination Proxies (Delegated to LegalityChecker)
    # ---------------------------------------------------------

    def is_game_over(self) -> bool:
        """Queries the rule referee to evaluate if the match is officially finished."""
        return LegalityChecker.is_terminal(self._board_wrapper.board)

    def get_result(self) -> str:
        """Queries the referee to fetch the official game scoreboard outcome string."""
        return LegalityChecker.get_result(self._board_wrapper.board)

    # ---------------------------------------------------------
    # PGN Exporters
    # ---------------------------------------------------------

    def export_pgn(self) -> str:
        """
        Compiles the historical match log stack into an official PGN record.

        Returns:
            Standard structural PGN string data text.
        """
        game = chess.pgn.Game()
        game.headers["Event"] = "Engine Architecture Match"
        game.headers["Result"] = self.get_result()

        node = game
        # Extract past moves sequentially from our clean underlying array stack
        for move in self._board_wrapper.board.move_stack:
            node = node.add_variation(move)

        exporter = chess.pgn.StringExporter(
            headers=True,
            variations=False,
            comments=False,
        )
        return game.accept(exporter)

    # ---------------------------------------------------------
    # Convenience FEN Serialization
    # ---------------------------------------------------------

    def current_fen(self) -> str:
        """Returns the structural single-line FEN text string of the board state."""
        return self._board_wrapper.export_fen()

    def __repr__(self) -> str:
        return (
            f"ChessGame("
            f"result='{self.get_result()}', "
            f"game_over={self.is_game_over()}, "
            f"ply_count={len(self._board_wrapper.board.move_stack)})"
        )
