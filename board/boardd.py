"""
engine/board.py

Board abstraction layer built on top of python-chess.

Responsibilities:
- Maintain board state
- Handle move execution
- Handle undo operations
- Provide legal moves
- Support FEN import/export
- Support search tree cloning

This module intentionally delegates all chess rule
validation to python-chess.
"""

from __future__ import annotations

import copy
from typing import List

import chess


class ChessBoard:
    """
    Thin wrapper around python-chess.Board.

    This class provides a stable interface for the rest
    of the engine (search, NNUE, RL, UCI).

    Internal Representation:
        python-chess.Board

    External Interface:
        ChessBoard
    """

    def __init__(self, fen: str | None = None) -> None:
        """
        Create a board.

        Parameters
        ----------
        fen : str | None
            Optional FEN string.
            If None, initialize standard start position.
        """

        if fen:
            self._board = chess.Board(fen)
        else:
            self._board = chess.Board()

    # --------------------------------------------------
    # FEN Handling
    # --------------------------------------------------

    def load_fen(self, fen: str) -> None:
        """
        Load position from FEN.

        Parameters
        ----------
        fen : str
            Valid FEN string.
        """

        self._board.set_fen(fen)

    def export_fen(self) -> str:
        """
        Export current position as FEN.

        Returns
        -------
        str
        """

        return self._board.fen()

    # --------------------------------------------------
    # Move Handling
    # --------------------------------------------------

    def make_move(self, move_uci: str) -> None:
        """
        Execute a move.

        Parameters
        ----------
        move_uci : str

        Examples
        --------
        e2e4
        g1f3
        e7e8q
        """

        move = chess.Move.from_uci(move_uci)

        if move not in self._board.legal_moves:
            raise ValueError(
                f"Illegal move: {move_uci}"
            )

        self._board.push(move)

    def undo_move(self) -> None:
        """
        Undo last move.

        Raises
        ------
        IndexError
            If move stack is empty.
        """

        if len(self._board.move_stack) == 0:
            raise IndexError(
                "Cannot undo. No moves available."
            )

        self._board.pop()

    # --------------------------------------------------
    # Move Generation
    # --------------------------------------------------

    def get_legal_moves(self) -> List[str]:
        """
        Return all legal moves in UCI format.

        Returns
        -------
        List[str]
        """

        return [
            move.uci()
            for move in self._board.legal_moves
        ]

    # --------------------------------------------------
    # Game Status
    # --------------------------------------------------

    def is_game_over(self) -> bool:
        """
        Check if game is finished.

        Returns
        -------
        bool
        """

        return self._board.is_game_over()

    def get_result(self) -> str:
        """
        Return game result.

        Examples
        --------
        "1-0"
        "0-1"
        "1/2-1/2"
        "*"

        Returns
        -------
        str
        """

        return self._board.result()

    def get_turn(self) -> str:
        """
        Return side to move.

        Returns
        -------
        str

        "white" or "black"
        """

        return (
            "white"
            if self._board.turn
            else "black"
        )

    # --------------------------------------------------
    # Cloning
    # --------------------------------------------------

    def clone(self) -> "ChessBoard":
        """
        Create a deep copy.

        Required by:
        - Alpha-Beta search
        - MCTS
        - RL rollouts

        Returns
        -------
        ChessBoard
        """

        cloned = ChessBoard()
        cloned._board = copy.deepcopy(
            self._board
        )

        return cloned

    # --------------------------------------------------
    # Utility Methods
    # --------------------------------------------------

    def move_count(self) -> int:
        """
        Number of moves played.

        Returns
        -------
        int
        """

        return len(self._board.move_stack)

    def reset(self) -> None:
        """
        Reset to initial position.
        """

        self._board.reset()

    def __str__(self) -> str:
        """
        Human-readable board.

        Returns
        -------
        str
        """

        return str(self._board)

    @property
    def board(self) -> chess.Board:
        """
        Read-only access to internal board.

        Useful for:
        - Evaluation
        - Feature extraction
        - Analysis modules

        Returns
        -------
        chess.Board
        """

        return self._board
    


board = ChessBoard()

print(board.get_turn())

board.make_move("e2e4")

print(board.export_fen())

print(board.get_legal_moves()[:10])

copy_board = board.clone()

board.undo_move()

print(board)