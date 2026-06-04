# engine/board.py

from __future__ import annotations

import chess
from typing import List, Optional


class ChessBoard:
    """
    Thin wrapper around python-chess.Board.

    Responsibilities:
    - Board state management
    - FEN import/export
    - Legal move generation
    - Move execution/undo
    - Game termination queries
    - Safe cloning for search trees

    Notes:
    - All move legality is delegated to python-chess.
    - Designed for integration with search/evaluation layers.
    """

    def __init__(self, fen: Optional[str] = None) -> None:
        """
        Initialize a new chess board.

        Args:
            fen:
                Optional FEN string. If None, initializes
                the standard starting position.
        """
        self._board = chess.Board(fen) if fen else chess.Board()

    # ------------------------------------------------------------------
    # Board Initialization / State Loading
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset board to the standard starting position.
        """
        self._board.reset()

    def load_fen(self, fen: str) -> None:
        """
        Load a position from a FEN string.

        Args:
            fen: Valid FEN representation.

        Raises:
            ValueError:
                If the FEN is invalid.
        """
        self._board.set_fen(fen)

    # ------------------------------------------------------------------
    # FEN Operations
    # ------------------------------------------------------------------

    def export_fen(self) -> str:
        """
        Export current position as FEN.

        Returns:
            Current board FEN.
        """
        return self._board.fen()

    # ------------------------------------------------------------------
    # Move Handling
    # ------------------------------------------------------------------

    def make_move(self, move_uci_str: str) -> bool:
        """
        Execute a legal move.

        Args:
            move_uci_str:
                Move in UCI format (e.g. 'e2e4', 'e7e8q').

        Returns:
            True if move was executed.
            False if move is illegal or invalid.
        """
        try:
            move = chess.Move.from_uci(move_uci_str)
        except ValueError:
            return False

        if move not in self._board.legal_moves:
            return False

        self._board.push(move)
        return True

    def undo_move(self) -> Optional[chess.Move]:
        """
        Undo the most recent move.

        Returns:
            The undone chess.Move object,
            or None if no move exists.
        """
        if not self._board.move_stack:
            return None

        return self._board.pop()

    # ------------------------------------------------------------------
    # Move Generation
    # ------------------------------------------------------------------

    def get_legal_moves(self) -> List[str]:
        """
        Generate all legal moves in UCI format.

        Returns:
            List of legal UCI move strings.
        """
        return [move.uci() for move in self._board.legal_moves]

    # ------------------------------------------------------------------
    # Game State Queries
    # ------------------------------------------------------------------

    def is_game_over(self) -> bool:
        """
        Check whether the game has ended.

        Returns:
            True if terminal position.
        """
        return self._board.is_game_over()

    def get_result(self) -> str:
        """
        Return game result.

        Returns:
            '1-0'   -> White wins
            '0-1'   -> Black wins
            '1/2-1/2' -> Draw
            '*'     -> Game not finished
        """
        return self._board.result(claim_draw=True)

    def get_turn(self) -> str:
        """
        Get side to move.

        Returns:
            'white' or 'black'
        """
        return "white" if self._board.turn == chess.WHITE else "black"

    # ------------------------------------------------------------------
    # Search Support
    # ------------------------------------------------------------------

    def clone(self) -> "ChessBoard":
        """
        Create an independent deep copy of the board.

        Essential for:
        - Alpha-beta search
        - Minimax
        - Monte Carlo rollouts
        - Parallel search workers

        Returns:
            New ChessBoard instance with identical state.
        """
        cloned = ChessBoard()
        cloned._board = self._board.copy(stack=True)
        return cloned

    # ------------------------------------------------------------------
    # Internal Access
    # ------------------------------------------------------------------

    @property
    def board(self) -> chess.Board:
        """
        Read-only access to underlying python-chess board.

        Useful for integration with:
        - NNUE evaluators
        - Search modules
        - UCI handlers

        Returns:
            Internal chess.Board object.
        """
        return self._board

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return str(self._board)

    def __repr__(self) -> str:
        return f"ChessBoard(fen='{self.export_fen()}')"