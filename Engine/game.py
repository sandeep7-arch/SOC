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
from .state import GameState


class ChessGame:
    """
    High-level game controller.

    Responsibilities:
    - Manage game lifecycle
    - Coordinate board state
    - Apply and undo moves
    - Expose game state
    - Detect game termination
    - Export PGN

    Notes:
    - No search logic
    - No evaluation logic
    - No AI logic
    - Uses python-chess as rule authority
    """

    def __init__(self) -> None:
        self.start_new_game()

    # ---------------------------------------------------------
    # Game Lifecycle
    # ---------------------------------------------------------

    def start_new_game(self) -> None:
        """
        Initialize a fresh chess game.
        """
        self._board_wrapper = ChessBoard()

    # ---------------------------------------------------------
    # Move Handling
    # ---------------------------------------------------------

    def apply_move(self, move: str | Move) -> bool:
        """
        Apply a move to the game.

        Args:
            move:
                Either:
                - UCI string ("e2e4")
                - engine.move.Move

        Returns:
            True if move was successfully applied.
            False otherwise.
        """

        if isinstance(move, Move):
            move_uci = move.to_uci()
        else:
            move_uci = move

        return self._board_wrapper.make_move(move_uci)

    def undo_move(self) -> Optional[chess.Move]:
        """
        Undo the most recent move.

        Returns:
            Undone move or None.
        """
        return self._board_wrapper.undo_move()

    # ---------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------

    def get_board(self) -> ChessBoard:
        """
        Get board wrapper.

        Returns:
            ChessBoard instance.
        """
        return self._board_wrapper

    def get_game_state(self) -> GameState:
        """
        Get current game state.

        Returns:
            GameState instance.
        """
        return GameState(self._board_wrapper.board)

    # ---------------------------------------------------------
    # Move Generation Access
    # ---------------------------------------------------------

    def get_legal_moves(self) -> list[Move]:
        """
        Convenience wrapper around MoveGenerator.

        Returns:
            List of legal engine Move objects.
        """
        return MoveGenerator.generate_legal_moves(
            self._board_wrapper.board
        )

    # ---------------------------------------------------------
    # Game Termination
    # ---------------------------------------------------------

    def is_game_over(self) -> bool:
        """
        Check if game is finished.

        Returns:
            True if game over.
        """
        return LegalityChecker.is_terminal(
            self._board_wrapper.board
        )

    def get_result(self) -> str:
        """
        Get official game result.

        Returns:
            "1-0"
            "0-1"
            "1/2-1/2"
            "*"
        """
        return LegalityChecker.get_result(
            self._board_wrapper.board
        )

    # ---------------------------------------------------------
    # PGN Export
    # ---------------------------------------------------------

    def export_pgn(self) -> str:
        """
        Export game as PGN.

        Returns:
            PGN string.
        """

        game = chess.pgn.Game()

        game.headers["Event"] = "RL Chess Engine Game"
        game.headers["Result"] = self.get_result()

        board = chess.Board()

        node = game

        for move in self._board_wrapper.board.move_stack:
            node = node.add_variation(move)
            board.push(move)

        exporter = chess.pgn.StringExporter(
            headers=True,
            variations=False,
            comments=False,
        )

        return game.accept(exporter)

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def current_fen(self) -> str:
        """
        Convenience FEN access.

        Returns:
            Current board FEN.
        """
        return self._board_wrapper.export_fen()

    def __repr__(self) -> str:
        return (
            f"ChessGame("
            f"result='{self.get_result()}', "
            f"game_over={self.is_game_over()})"
        )